"""Tests for `r3con.stages.reasoning` — the stage-3 glue around the CodeAct loop.

The generic CodeAct loop itself is tested in ``test_codeact.py``; this file covers
the reasoning-specific part: ``reason`` renders the ``reasoning`` prompt correctly
(the parse view and the corpus-wide relevance state in the system message; the task
wrapped in `<task>` tags in the user message), binds the parse as the sandbox
variable ``parse``, and stamps each record with its source document.

``reason`` delegates the LLM call to ``r3con.runtime.codeact``, which is where it
is patched.

Run with:  uv run python tests/test_reasoning.py
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from r3con.runtime import codeact
from r3con.stages import reasoning as reasoning_mod
from r3con.stages.reasoning import reason

RELEVANCE_STATES = ["Doc 1 is a 1923 court opinion.", "Doc 2 cites the petitioner Smith."]


@contextlib.contextmanager
def _patched_llm(fake: Callable[..., str]) -> Iterator[list[list[dict[str, str]]]]:
    """Swap the LLM call (which lives in `runtime.codeact`) for a scripted fake."""
    original = codeact.litellm_chat_completion
    seen: list[list[dict[str, str]]] = []

    def wrapper(**kwargs: Any) -> str:
        if kwargs.get("messages") is not None:
            seen.append([dict(m) for m in kwargs["messages"]])
        return fake(**kwargs)

    codeact.litellm_chat_completion = wrapper  # type: ignore[assignment]
    try:
        yield seen
    finally:
        codeact.litellm_chat_completion = original  # type: ignore[assignment]


def _final(value: str) -> str:
    return f"Thought: commit.\n<code>\nfinal_answer({value!r})\n</code>"


# ----- reason -----


def test_reason_binds_parse_and_commits() -> None:
    response = "Thought: sum.\n<code>\nfinal_answer(sum(it['n'] for it in parse['items']))\n</code>"

    def fake(**_: Any) -> str:
        return response

    with _patched_llm(fake):
        r = reason(
            task="sum?", schema_code="class Parse(BaseModel): items: list[Item]",
            parsed={"items": [{"n": 1}, {"n": 2}, {"n": 3}]}, model="m", prompt_version="v1",
        )
    assert r.answer == "6"
    assert r.terminated_by == "final_answer"


def test_user_message_wraps_task_in_tags() -> None:
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as msgs:
        reason(
            task="What did Alice say?", schema_code="class Parse(BaseModel): statements: list[S]",
            parsed={"statements": [{"speaker": "Alice", "text": "hi"}]}, model="m", prompt_version="v1",
        )
    assert msgs[0][1]["content"] == "Input:\n<task>\nWhat did Alice say?\n</task>"


def test_system_prompt_contains_every_parse_field() -> None:
    """Every top-level field of the parse reaches the model — including one that is
    empty, which is itself information (nothing was found for it)."""
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as msgs:
        reason(
            task="?", schema_code="class Parse(BaseModel): supports: list[X]",
            parsed={"supports": [{"ok": True}], "oppositions": []}, model="m", prompt_version="v1",
        )
    sys_msg = msgs[0][0]["content"]
    assert "supports" in sys_msg
    assert "oppositions" in sys_msg


def test_system_prompt_includes_document_relevance_states() -> None:
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as msgs:
        reason(task="?", schema_code="class Parse(BaseModel): x: list[X]",
                      parsed={"x": [{"y": 1}]}, relevance_states=RELEVANCE_STATES, model="m", prompt_version="v1")
    sys_msg = msgs[0][0]["content"]
    assert "## Document summaries" in sys_msg
    assert "1923 court opinion" in sys_msg
    assert "petitioner Smith" in sys_msg


def test_system_prompt_omits_relevance_block_when_none() -> None:
    def fake(**_: Any) -> str:
        return _final("ok")

    # The heading and "### Document N" both appear in the prompt's in-context examples, so
    # the only sound test is a sentinel that could only have come from the rendered block.
    sentinel = "ZZQX-relevance-sentinel-42"
    with _patched_llm(fake) as ml_present:
        reason(task="?", schema_code="class Parse(BaseModel): x: list[X]",
                      parsed={"x": [{"y": 1}]}, relevance_states=[sentinel], model="m", prompt_version="v1")
    assert sentinel in ml_present[0][0]["content"]

    with _patched_llm(fake) as ml_none:
        reason(task="?", schema_code="class Parse(BaseModel): x: list[X]",
                      parsed={"x": [{"y": 1}]}, relevance_states=None, model="m", prompt_version="v1")
    with _patched_llm(fake) as ml_empty:
        reason(task="?", schema_code="class Parse(BaseModel): x: list[X]",
                      parsed={"x": [{"y": 1}]}, relevance_states=[], model="m", prompt_version="v1")
    for ml in (ml_none, ml_empty):
        assert sentinel not in ml[0][0]["content"]


def test_sample_block_keeps_unicode_readable() -> None:
    """Non-ASCII parse values reach the model (and the logs) as readable text, not
    \\uXXXX escapes — across scalar, list-record, and dict fields."""
    from r3con.stages.reasoning import _sample_record_per_field

    out = _sample_record_per_field({
        "判决文书1_result": "判决结果4",                 # scalar
        "rows": [{"label": "应付账款", "amount": 12}],     # list of records
        "meta": {"机构": "上海"},                          # dict
    })
    assert "判决结果4" in out and "应付账款" in out and "上海" in out
    assert "\\u" not in out


def test_codeact_system_prompt_keeps_unicode_readable() -> None:
    """End-to-end: a CJK parse renders to a codeact system prompt with literal CJK."""
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as msgs:
        reason(task="哪个判决结果?", schema_code="class Parse(BaseModel): 判决文书1_result: str",
                      parsed={"判决文书1_result": "判决结果4"}, model="m", prompt_version="v1")
    sys_msg = msgs[0][0]["content"]
    assert "判决结果4" in sys_msg and "\\u5224" not in sys_msg


def test_reason_surfaces_relevance_states_for_alias_resolution() -> None:
    parse = {"actions": [
        {"actor": "Mike", "action": "destroyed the bridge"},
        {"actor": "Mike", "action": "burned the library"},
        {"actor": "Jenny", "action": "watched"},
    ]}
    relevance_states = ["A 1992 noir. Mike is also referred to as 'The Destroyer' throughout."]
    response = (
        "Thought: summaries say The Destroyer is Mike.\n<code>\n"
        "m = [r for r in parse['actions'] if r['actor'] == 'Mike']\n"
        "final_answer(f'The Destroyer (Mike) performed {len(m)} actions.')\n</code>"
    )

    def fake(**_: Any) -> str:
        return response

    with _patched_llm(fake) as msgs:
        r = reason(task="How many actions did The Destroyer perform?",
                          schema_code="...", parsed=parse, relevance_states=relevance_states, model="m", prompt_version="v1")
    assert r.terminated_by == "final_answer"
    assert "2 actions" in r.answer
    assert "Mike is also referred to as 'The Destroyer'" in msgs[0][0]["content"]


def test_reason_tags_records_with_source_document() -> None:
    """The same source-document tag reaches the codeact agent (in the sample block /
    bound parse), regardless of prompt version."""
    def fake(**_: Any) -> str:
        return _final("ok")

    with _patched_llm(fake) as msgs:
        reason(task="?", schema_code="class Parse(BaseModel): docs: list[D]",
                      parsed={"docs": [{"a": 1}]}, source_docs={"docs": [4]},  # index 4 → Document 5
                      model="m", prompt_version="v1")
    assert '"document": 5' in msgs[0][0]["content"]


def test_prompt_drops_schema_section_and_conditional_hedge() -> None:
    """The prompt no longer prints the proposed schema (we show the full parse instead,
    and the schema lacked the injected `document` field), and it drops the conditional
    'if the parse was too large' hedge — the small-parse case shows the whole parse with
    no such caveat."""
    def fake(**_: Any) -> str:
        return _final("ok")

    schema = "class Rec(BaseModel):\n    UNIQUE_SCHEMA_MARKER: str\nclass Parse(BaseModel): records: list[Rec]"
    parsed = {"records": [{"name": f"name_{i}", "value": i} for i in range(5)]}
    with _patched_llm(fake) as msgs:
        reason(task="?", schema_code=schema, parsed=parsed, model="m", prompt_version="v1")
    sys = msgs[0][0]["content"]
    assert "UNIQUE_SCHEMA_MARKER" not in sys  # the schema source is no longer embedded
    assert "## Schema used to extract" not in sys
    assert "If the parse was too large" not in sys  # the conditional hedge is gone


def test_shows_whole_small_parse_and_document_note() -> None:
    """The WHOLE small parse is embedded, and the per-record `document` field is explained."""
    def fake(**_: Any) -> str:
        return _final("ok")

    parsed = {"records": [{"name": f"name_{i}", "value": i} for i in range(8)]}
    with _patched_llm(fake) as msgs:
        reason(task="?", schema_code="class Parse(BaseModel): records: list[R]",
                      parsed=parsed, source_docs={"records": list(range(8))}, model="m", prompt_version="v1")
    sys = msgs[0][0]["content"]
    assert "name_0" in sys and "name_7" in sys      # the WHOLE parse
    assert "`document`" in sys                       # the document-field explanation
    assert '"document": 1' in sys                    # id-injection still stamps records


def test_falls_back_to_samples_for_huge_parse() -> None:
    """A parse over REASONING_PARSE_MAX_TOKS still falls back to one sample per field + the
    prominent "this is only a SAMPLE" note."""
    def fake(**_: Any) -> str:
        return _final("ok")

    parsed = {"records": [{"name": f"name_{i}", "blob": f"item {i}: descriptive text {i * 7}"}
                          for i in range(1500)]}
    with _patched_llm(fake) as msgs:
        reason(task="?", schema_code="class Parse(BaseModel): records: list[R]",
                      parsed=parsed, model="m", prompt_version="v1")
    sys = msgs[0][0]["content"]
    assert "name_0" in sys and "name_1499" not in sys
    assert "only a SAMPLE" in sys and "`parse` variable" in sys


if __name__ == "__main__":
    tests = [
        test_reason_binds_parse_and_commits,
        test_user_message_wraps_task_in_tags,
        test_system_prompt_contains_every_parse_field,
        test_system_prompt_includes_document_relevance_states,
        test_system_prompt_omits_relevance_block_when_none,
        test_prompt_drops_schema_section_and_conditional_hedge,
        test_shows_whole_small_parse_and_document_note,
        test_falls_back_to_samples_for_huge_parse,
        test_sample_block_keeps_unicode_readable,
        test_codeact_system_prompt_keeps_unicode_readable,
        test_reason_surfaces_relevance_states_for_alias_resolution,
        test_reason_tags_records_with_source_document,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
