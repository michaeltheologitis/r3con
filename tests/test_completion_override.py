"""Bring-your-own-connection: `completion=` replaces litellm.completion everywhere.

The convention every peer library has (smolagents' ``ApiModel(client=…)``,
pydantic-ai's ``OpenAIProvider(openai_client=…)``, the OpenAI Agents SDK's
``openai_client=``): the caller may already own the connection. Because r3con's
transport is a single litellm function rather than a client object, ours is the
callable itself.

These tests pin the two things that make it useful: that it reaches *every* stage, and
that it never leaks into the provider request as an argument.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r3con.config import RunConfig  # noqa: E402
from r3con.runtime.llm import litellm_chat_completion, litellm_chat_completion_full  # noqa: E402


def _reply(text: str = "ok") -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(role="assistant", content=text), finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


def test_completion_callable_is_used_instead_of_litellm() -> None:
    seen: dict[str, Any] = {}

    def fake(**request: Any) -> Any:
        seen.update(request)
        return _reply("from my own connection")

    out = litellm_chat_completion(
        system_prompt="s", user_prompt="u", model="openai/whatever", completion=fake,
    )
    assert out == "from my own connection"
    assert seen["model"] == "openai/whatever"
    assert seen["messages"][0]["role"] == "system"


def test_completion_is_not_forwarded_into_the_provider_request() -> None:
    """It must bind to the named parameter, not ride along in **kwargs — a stray
    `completion=` in the request would be rejected by the provider."""
    seen: dict[str, Any] = {}

    def fake(**request: Any) -> Any:
        seen.update(request)
        return _reply()

    litellm_chat_completion_full(system_prompt="s", user_prompt="u", model="m", completion=fake)
    assert "completion" not in seen
    # the things that SHOULD be there still are
    assert seen["model"] == "m" and "messages" in seen and "num_retries" in seen


def test_default_is_litellm_completion() -> None:
    """Omitting it must not change behaviour: the default is litellm's own function."""
    import litellm

    calls: list[dict[str, Any]] = []
    original = litellm.completion
    litellm.completion = lambda **kw: (calls.append(kw), _reply())[1]  # type: ignore[assignment]
    try:
        litellm_chat_completion(system_prompt="s", user_prompt="u", model="m")
    finally:
        litellm.completion = original  # type: ignore[assignment]
    assert len(calls) == 1 and calls[0]["model"] == "m"


def test_it_reaches_every_stage_of_a_whole_run() -> None:
    """The point of the parameter: one connection, used by all three stages — not just
    the one you happened to look at."""
    from r3con.pipeline import run_pipeline
    from r3con.runs import TaskLogger

    models_seen: list[str] = []
    schema_src = "from pydantic import BaseModel\nclass Row(BaseModel):\n    a: str | None\nclass Parse(BaseModel):\n    rows: list[Row]"

    def router_like(**request: Any) -> Any:
        """Stand-in for `litellm.Router.completion` — same (model, messages, **kwargs)."""
        models_seen.append(request["model"])
        if request.get("response_format"):            # the parsing stage wants JSON
            return _reply('{"rows": []}')
        if "<schema>" in str(request["messages"][0]["content"]).lower() or "Pydantic" in str(
            request["messages"][0]["content"]
        ):                                            # the schema stage wants a schema
            return _reply(f"Thought: fine\n<schema>\n{schema_src}\n</schema>")
        return _reply("<code>\nfinal_answer('answered via my own connection')\n</code>")

    cfg = RunConfig(
        model="my-router-group", relevance_rounds=1,
        prompts={k: "v1" for k in ("relevance", "structuring/schema", "structuring/parsing", "reasoning")},
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = run_pipeline(
            task="what happened?", documents=["a document", "another document"], config=cfg,
            task_logger=TaskLogger("run", root=Path(tmp)), completion=router_like,
        )
    assert out.answer == "answered via my own connection"
    # every stage went through the supplied callable, and none through litellm
    assert len(models_seen) >= 5, models_seen          # 2 relevance + schema + 2 parsing + reasoning
    assert set(models_seen) == {"my-router-group"}


def test_completion_is_transport_and_stays_out_of_the_run_identity() -> None:
    cfg = RunConfig(
        model="m",
        prompts={k: "v1" for k in ("relevance", "structuring/schema", "structuring/parsing", "reasoning")},
    )
    assert "completion" not in cfg.label()
    assert "completion" not in cfg.model_dump()


if __name__ == "__main__":
    tests = [
        test_completion_callable_is_used_instead_of_litellm,
        test_completion_is_not_forwarded_into_the_provider_request,
        test_default_is_litellm_completion,
        test_it_reaches_every_stage_of_a_whole_run,
        test_completion_is_transport_and_stays_out_of_the_run_identity,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
