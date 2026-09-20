"""Tests for `r3con.stages.structuring.schema.propose_schema` — retry loop and schema extraction.

`litellm_chat_completion` (the real LLM call) is monkeypatched with a scripted
fake so these stay offline.
Run with:  uv run python tests/test_schema.py
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from r3con.stages.structuring import schema as schema_mod
from r3con.stages.structuring.parsing import SchemaError


VALID_SCHEMA = """
from pydantic import BaseModel, Field

class Move(BaseModel):
    time: int = Field(description="t")
    new_location: str = Field(description="loc")

class Parse(BaseModel):
    moves: list[Move]
""".strip()

INVALID_NO_PARSE = """
from pydantic import BaseModel

class Foo(BaseModel):
    x: int
""".strip()

INVALID_SYNTAX = "class Parse(:"


@contextlib.contextmanager
def _patched_llm(fake: Callable[..., str]) -> Iterator[list[str]]:
    """Swap ``schema_mod.litellm_chat_completion`` for a scripted fake.

    The fake is called with all kwargs `litellm_chat_completion` would receive;
    it returns a string (or raises). Recorded prompts list is yielded so tests
    can assert on what the schema proposal fed back into the model.
    """
    original = schema_mod.litellm_chat_completion
    recorded_user_prompts: list[str] = []

    def wrapper(**kwargs: Any) -> str:
        recorded_user_prompts.append(kwargs["user_prompt"])
        return fake(**kwargs)

    schema_mod.litellm_chat_completion = wrapper  # type: ignore[assignment]
    try:
        yield recorded_user_prompts
    finally:
        schema_mod.litellm_chat_completion = original  # type: ignore[assignment]


def test_first_attempt_succeeds() -> None:
    def fake(**_: Any) -> str:
        return VALID_SCHEMA

    with _patched_llm(fake) as prompts:
        result = schema_mod.propose_schema(task="Where is the cake?", model="x", prompt_version="v1")

    assert result.schema_code == VALID_SCHEMA
    assert result.parse_cls.__name__ == "Parse"
    assert len(result.attempts) == 1
    assert result.attempts[0].error is None
    assert len(prompts) == 1
    assert "Where is the cake?" in prompts[0]
    assert "Previous attempt" not in prompts[0]


def test_retries_after_invalid_attempt() -> None:
    """First attempt missing Parse class → second attempt fixes it → returns success."""
    outputs = iter([INVALID_NO_PARSE, VALID_SCHEMA])

    def fake(**_: Any) -> str:
        return next(outputs)

    with _patched_llm(fake) as prompts:
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1", max_attempts=3)

    assert result.schema_code == VALID_SCHEMA
    assert len(result.attempts) == 2
    assert result.attempts[0].error is not None
    assert "Parse" in result.attempts[0].error  # fed back to model
    assert result.attempts[1].error is None
    # Second prompt should mention the previous attempt and the error
    assert "Previous attempt" in prompts[1]
    assert "did not define" in prompts[1] or "Parse" in prompts[1]


def test_exhausts_attempts_then_raises() -> None:
    """All max_attempts attempts invalid → SchemaError surfaces."""
    def fake(**_: Any) -> str:
        return INVALID_NO_PARSE

    with _patched_llm(fake):
        try:
            schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1", max_attempts=3)
        except SchemaError as e:
            assert "exhausted 3 attempts" in str(e)
        else:
            raise AssertionError("expected SchemaError")


def test_max_attempts_zero_rejected() -> None:
    try:
        schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1", max_attempts=0)
    except ValueError as e:
        assert "max_attempts" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_extracts_schema_tag_body() -> None:
    """Primary path: the prompts ask the model for ``<schema>...</schema>``."""
    wrapped = f"<schema>\n{VALID_SCHEMA}\n</schema>"
    def fake(**_: Any) -> str:
        return wrapped

    with _patched_llm(fake):
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")

    assert result.schema_code == VALID_SCHEMA


def test_extracts_schema_tag_with_thought_preamble() -> None:
    """The prompts ask for a `Thought:` snippet before the `<schema>` block —
    the schema body is extracted from the tag, and the Thought snippet
    is captured separately onto the attempt for logging.
    """
    text = (
        "Thought: This is a quick rationale for the schema design.\n"
        f"\n<schema>\n{VALID_SCHEMA}\n</schema>"
    )
    def fake(**_: Any) -> str:
        return text

    with _patched_llm(fake):
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")

    assert result.schema_code == VALID_SCHEMA
    assert result.attempts[-1].thought == "This is a quick rationale for the schema design."


def test_thought_captured_after_thought_marker_before_fence() -> None:
    """When the model uses a markdown fence instead of <schema> tags,
    the Thought snippet still lives before the fence and is captured.
    """
    text = (
        "Thought: One row per occurrence; inference counts the list.\n\n"
        f"```python\n{VALID_SCHEMA}\n```\n"
    )
    def fake(**_: Any) -> str:
        return text

    with _patched_llm(fake):
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")

    assert result.schema_code == VALID_SCHEMA
    assert result.attempts[-1].thought == "One row per occurrence; inference counts the list."


def test_thought_is_none_when_marker_absent() -> None:
    """Bare schema with no Thought: marker — attempt.thought is None."""
    text = f"<schema>\n{VALID_SCHEMA}\n</schema>"
    def fake(**_: Any) -> str:
        return text

    with _patched_llm(fake):
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")

    assert result.attempts[-1].thought is None


def test_schema_tag_takes_precedence_over_fence() -> None:
    """When both ``<schema>`` and ```` ```python ``` ```` are present, prefer the tag."""
    inner_python = VALID_SCHEMA
    inner_fenced = "from pydantic import BaseModel\nclass Other(BaseModel):\n    x: int\nclass Parse(BaseModel):\n    others: list[Other]"
    text = (
        f"```python\n{inner_fenced}\n```\n\n"
        f"<schema>\n{inner_python}\n</schema>"
    )
    def fake(**_: Any) -> str:
        return text

    with _patched_llm(fake):
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")

    # The tag's body was extracted, not the fenced block's.
    assert result.schema_code == VALID_SCHEMA


def test_strips_python_fence_as_fallback() -> None:
    """Fallback when no ``<schema>`` tag — markdown ``` ```python ``` ``` fence."""
    fenced = f"```python\n{VALID_SCHEMA}\n```"
    def fake(**_: Any) -> str:
        return fenced

    with _patched_llm(fake):
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")

    assert result.schema_code == VALID_SCHEMA  # fence stripped


def test_strips_python_fence_after_thought_preamble() -> None:
    """Fallback case where the model emits a ``Thought:`` preamble before a
    markdown fence (i.e. it forgot the ``<schema>`` tags but still wrapped
    the code). The fence-stripper searches anywhere in the response, not
    just at the very start.
    """
    text = (
        "Thought: One row per occurrence; inference counts the list.\n\n"
        f"```python\n{VALID_SCHEMA}\n```\n"
    )
    def fake(**_: Any) -> str:
        return text

    with _patched_llm(fake):
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")

    assert result.schema_code == VALID_SCHEMA


def test_strips_bare_fence_as_fallback() -> None:
    """Fallback when no ``<schema>`` tag — bare ``` ``` ``` fence (no language tag)."""
    fenced = f"```\n{VALID_SCHEMA}\n```"
    def fake(**_: Any) -> str:
        return fenced

    with _patched_llm(fake):
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")

    assert result.schema_code == VALID_SCHEMA


def test_preserves_unfenced_output() -> None:
    """Final fallback: raw Python with no tags and no fences is returned unchanged."""
    def fake(**_: Any) -> str:
        return VALID_SCHEMA

    with _patched_llm(fake):
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")

    assert result.schema_code == VALID_SCHEMA


def test_retry_includes_invalid_syntax_error() -> None:
    """Syntax-error feedback also feeds back into the next prompt."""
    outputs = iter([INVALID_SYNTAX, VALID_SCHEMA])

    def fake(**_: Any) -> str:
        return next(outputs)

    with _patched_llm(fake) as prompts:
        result = schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")

    assert result.schema_code == VALID_SCHEMA
    # The retry prompt should include the syntax error feedback
    assert "Previous attempt" in prompts[1]
    assert "SyntaxError" in prompts[1] or "failed to execute" in prompts[1]


def test_relevance_states_rendered_into_system_prompt() -> None:
    """When relevance snippets are passed, propose_schema renders them into the system prompt's
    '## Task-conditioned document summaries' block; with none, the block is omitted."""
    seen: dict[str, str] = {}

    def fake(**kwargs: Any) -> str:
        seen["system"] = kwargs["system_prompt"]
        return VALID_SCHEMA

    original = schema_mod.litellm_chat_completion
    schema_mod.litellm_chat_completion = fake  # type: ignore[assignment]
    try:
        schema_mod.propose_schema(
            task="Q?", model="x", prompt_version="v1",
            relevance_snippets=["Doc A is about whales.", "Doc B is about ships."],
        )
        # Distinctive prose of the injected block (the in-context examples contain a
        # "Task-conditioned document summaries:" label, so key on the block's own text).
        assert "In order to know what the documents contain" in seen["system"]
        assert "Doc A is about whales." in seen["system"]
        assert "Doc B is about ships." in seen["system"]
        # No relevance snippets → no injected block.
        schema_mod.propose_schema(task="Q?", model="x", prompt_version="v1")
        assert "In order to know what the documents contain" not in seen["system"]
    finally:
        schema_mod.litellm_chat_completion = original  # type: ignore[assignment]


def test_llm_kwargs_forwarded() -> None:
    """Extra kwargs (seed, api_base, …) reach `litellm_chat_completion`."""
    seen: dict[str, Any] = {}

    def fake(**kwargs: Any) -> str:
        seen.update(kwargs)
        return VALID_SCHEMA

    with _patched_llm(fake):
        schema_mod.propose_schema(task="Q?", model="m", prompt_version="v1", seed=42, api_base="http://x")

    assert seen.get("seed") == 42
    assert seen.get("api_base") == "http://x"
    assert seen.get("model") == "m"


if __name__ == "__main__":
    tests = [
        test_first_attempt_succeeds,
        test_retries_after_invalid_attempt,
        test_exhausts_attempts_then_raises,
        test_max_attempts_zero_rejected,
        test_extracts_schema_tag_body,
        test_extracts_schema_tag_with_thought_preamble,
        test_thought_captured_after_thought_marker_before_fence,
        test_thought_is_none_when_marker_absent,
        test_schema_tag_takes_precedence_over_fence,
        test_strips_python_fence_as_fallback,
        test_strips_python_fence_after_thought_preamble,
        test_strips_bare_fence_as_fallback,
        test_preserves_unfenced_output,
        test_retry_includes_invalid_syntax_error,
        test_relevance_states_rendered_into_system_prompt,
        test_llm_kwargs_forwarded,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
