"""Tests for `r3con.runtime.llm` — strict-object schema rewriting + the
num_retries forwarding into litellm.completion.

Run with:  uv run python tests/test_llm.py
"""

from __future__ import annotations

from pydantic import BaseModel

import r3con.runtime.llm as llm_mod
from r3con.runtime.llm import (
    _enforce_strict_objects,
    litellm_chat_completion,
    litellm_chat_completion_full,
)
from r3con.settings import settings


def test_sets_additional_properties_false_on_object() -> None:
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    out = _enforce_strict_objects(schema)
    assert out["additionalProperties"] is False


def test_required_includes_every_property_key() -> None:
    """OpenAI strict mode demands every property key appears in ``required``."""
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
        "required": ["a"],  # pydantic emitted only 'a' because 'b' is optional
    }
    out = _enforce_strict_objects(schema)
    assert set(out["required"]) == {"a", "b"}


def test_required_added_when_missing() -> None:
    """Objects without a ``required`` key get one synthesized from their properties."""
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    out = _enforce_strict_objects(schema)
    assert out["required"] == ["x"]


def test_objects_without_properties_left_alone() -> None:
    """Schemas that declare ``type: object`` with no ``properties`` (e.g. an empty
    free-form dict) get neither ``additionalProperties`` nor ``required`` injected.
    """
    schema = {"type": "object"}
    out = _enforce_strict_objects(schema)
    assert "required" not in out
    assert "additionalProperties" not in out


def test_recurses_into_defs() -> None:
    schema = {
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
            }
        },
        "type": "object",
        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
    }
    out = _enforce_strict_objects(schema)
    assert out["$defs"]["Inner"]["additionalProperties"] is False
    assert out["$defs"]["Inner"]["required"] == ["a"]


def test_recurses_into_array_items() -> None:
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"k": {"type": "string"}}},
            }
        },
    }
    out = _enforce_strict_objects(schema)
    inner = out["properties"]["items"]["items"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["k"]


# ---------- num_retries forwarding ----------


def _capture_completion():
    """Monkeypatch litellm.completion to record kwargs; returns (restore, captured)."""
    captured: dict = {}

    def fake(**kw):
        captured.update(kw)
        return "dummy-response"

    orig = llm_mod.litellm.completion
    llm_mod.litellm.completion = fake
    return (lambda: setattr(llm_mod.litellm, "completion", orig)), captured


def test_num_retries_forwarded_from_settings() -> None:
    """Every call passes num_retries=settings.LLM_NUM_RETRIES (so a transient
    vLLM blip is retried instead of aborting the task)."""
    restore, captured = _capture_completion()
    try:
        out = litellm_chat_completion_full(
            system_prompt="s", user_prompt="u", model="hosted_vllm/x", run=None
        )
    finally:
        restore()
    assert out == "dummy-response"
    assert captured.get("num_retries") == settings.LLM_NUM_RETRIES
    assert settings.LLM_NUM_RETRIES == 10  # the configured default


def test_num_retries_caller_override_wins() -> None:
    """A caller-supplied num_retries beats the settings default (setdefault)."""
    restore, captured = _capture_completion()
    try:
        litellm_chat_completion_full(
            system_prompt="s", user_prompt="u", model="hosted_vllm/x", run=None, num_retries=2
        )
    finally:
        restore()
    assert captured.get("num_retries") == 2


# ---------- empty-structured-output re-roll ----------


class _Msg:
    def __init__(self, content):
        self.role = "assistant"
        self.content = content


class _Choice:
    def __init__(self, content):
        self.message = _Msg(content)
        self.finish_reason = "stop"


class _Resp:
    def __init__(self, content):
        self.choices = [_Choice(content)]
        self.usage = None


def _sequence_completion(contents):
    """Monkeypatch litellm.completion to return each ``contents[i]`` in turn, recording
    each call's kwargs. Returns (restore, seen_kwargs_list)."""
    seq = list(contents)
    seen: list[dict] = []

    def fake(**kw):
        seen.append(kw)
        return _Resp(seq.pop(0))

    orig = llm_mod.litellm.completion
    llm_mod.litellm.completion = fake
    return (lambda: setattr(llm_mod.litellm, "completion", orig)), seen


class _Out(BaseModel):
    x: int


def test_empty_structured_output_rerolls_then_succeeds() -> None:
    """An empty structured-output response is re-rolled (not raised), and the seed is
    perturbed each retry so a pinned-seed re-roll actually differs."""
    restore, seen = _sequence_completion(["", "", '{"x": 7}'])
    try:
        out = litellm_chat_completion(
            system_prompt="s", user_prompt="u", model="hosted_vllm/x", schema=_Out, seed=0
        )
    finally:
        restore()
    assert isinstance(out, _Out) and out.x == 7
    assert len(seen) == 3  # 2 empty re-rolls then success
    assert [c.get("seed") for c in seen] == [0, 1, 2]  # perturbed, deterministic


def test_empty_structured_output_raises_after_max_rerolls() -> None:
    restore, seen = _sequence_completion([""] * 20)
    try:
        litellm_chat_completion(
            system_prompt="s", user_prompt="u", model="x", schema=_Out, seed=0, max_empty_retries=3
        )
    except ValueError as e:
        assert "empty" in str(e).lower()
    else:
        raise AssertionError("expected ValueError after exhausting re-rolls")
    finally:
        restore()
    assert len(seen) == 4  # 1 initial + 3 re-rolls


def test_non_schema_empty_is_not_rerolled() -> None:
    """A non-structured call returning empty text is returned as-is (the caller
    decides) — only structured-output empties are re-rolled."""
    restore, seen = _sequence_completion([""])
    try:
        out = litellm_chat_completion(system_prompt="s", user_prompt="u", model="x", schema=None)
    finally:
        restore()
    assert out == "" and len(seen) == 1


def test_structured_success_first_try_does_not_reroll() -> None:
    restore, seen = _sequence_completion(['{"x": 1}'])
    try:
        out = litellm_chat_completion(system_prompt="s", user_prompt="u", model="x", schema=_Out, seed=5)
    finally:
        restore()
    assert out.x == 1 and len(seen) == 1
    assert seen[0].get("seed") == 5  # unperturbed on the first try


def test_sampling_params_reach_completion_request() -> None:
    """The leaf-level proof that a sampling preset's generation params actually ride
    into litellm.completion — top-level OpenAI params land at the top, and
    extra_body (the vLLM-specific knobs incl. chat_template_kwargs) passes straight
    through. This is what makes `run_pipeline(run_config=...)` end-to-end real."""
    restore, captured = _capture_completion()
    try:
        litellm_chat_completion_full(
            system_prompt="s", user_prompt="u", model="hosted_vllm/x", run=None,
            temperature=0.7, top_p=0.8, presence_penalty=1.5,
            extra_body={"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}},
        )
    finally:
        restore()
    assert captured["temperature"] == 0.7
    assert captured["top_p"] == 0.8
    assert captured["presence_penalty"] == 1.5
    assert captured["extra_body"]["top_k"] == 20
    assert captured["extra_body"]["chat_template_kwargs"] == {"enable_thinking": False}


if __name__ == "__main__":
    tests = [
        test_sets_additional_properties_false_on_object,
        test_required_includes_every_property_key,
        test_required_added_when_missing,
        test_objects_without_properties_left_alone,
        test_recurses_into_defs,
        test_recurses_into_array_items,
        test_num_retries_forwarded_from_settings,
        test_num_retries_caller_override_wins,
        test_empty_structured_output_rerolls_then_succeeds,
        test_empty_structured_output_raises_after_max_rerolls,
        test_non_schema_empty_is_not_rerolled,
        test_structured_success_first_try_does_not_reroll,
        test_sampling_params_reach_completion_request,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
