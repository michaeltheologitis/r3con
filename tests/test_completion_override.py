"""Bring-your-own-connection: `completion=` replaces litellm.completion everywhere.

The convention every peer library has (smolagents' ``ApiModel(client=…)``,
pydantic-ai's ``OpenAIProvider(openai_client=…)``, the OpenAI Agents SDK's
``openai_client=``): the caller may already own the connection. Because r3con's
transport is a single litellm function rather than a client object, ours is the
callable itself.

These tests pin that it reaches *every* stage of a run, and that it stays out of the
run's identity; tests/test_llm.py pins it at the call itself.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from r3con.config import RunConfig


def _reply(text: str = "ok") -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(role="assistant", content=text),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1, total_tokens=2),
    )


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
        if request.get("response_format"):  # the parsing stage wants JSON
            return _reply('{"rows": []}')
        if "<schema>" in str(
            request["messages"][0]["content"]
        ).lower() or "Pydantic" in str(
            request["messages"][0]["content"]
        ):  # the schema stage wants a schema
            return _reply(f"Thought: fine\n<schema>\n{schema_src}\n</schema>")
        return _reply("<code>\nfinal_answer('answered via my own connection')\n</code>")

    cfg = RunConfig(
        model="my-router-group",
        relevance_rounds=1,
        prompts={
            k: "v1"
            for k in (
                "relevance",
                "structuring/schema",
                "structuring/parsing",
                "reasoning",
            )
        },
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = run_pipeline(
            task="what happened?",
            documents=["a document", "another document"],
            config=cfg,
            task_logger=TaskLogger("run", root=Path(tmp)),
            completion=router_like,
        )
    assert out.answer == "answered via my own connection"
    # every stage went through the supplied callable, and none through litellm
    assert len(models_seen) >= 5, (
        models_seen
    )  # 2 relevance + schema + 2 parsing + reasoning
    assert set(models_seen) == {"my-router-group"}


def test_completion_is_transport_and_stays_out_of_the_run_identity() -> None:
    cfg = RunConfig(
        model="m",
        prompts={
            k: "v1"
            for k in (
                "relevance",
                "structuring/schema",
                "structuring/parsing",
                "reasoning",
            )
        },
    )
    assert "completion" not in cfg.label()
    assert "completion" not in cfg.model_dump()


if __name__ == "__main__":
    tests = [
        test_it_reaches_every_stage_of_a_whole_run,
        test_completion_is_transport_and_stays_out_of_the_run_identity,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
