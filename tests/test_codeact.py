"""Tests for `r3con.runtime.codeact` — the generic CodeAct loop + helpers.

Each turn the LLM emits a CodeAct-style ``Thought:`` plan plus a
``<code>...</code>`` block; ``run_codeact`` executes the code, captures
``print`` output, and feeds it back as ``<observation>...</observation>`` on
the next user message. State (variable bindings) persists across turns. The
agent commits via ``final_answer(x)`` from inside a ``<code>`` block; the loop
terminates when ``final_answer`` is called or ``max_turns`` is exhausted.

``litellm_chat_completion`` is monkeypatched (where it is imported, in the
codeact module) with a scripted fake so these stay offline. The fake receives
``messages=`` (the full conversation history).

`_run` is a thin shim that calls ``run_codeact`` with a dummy system prompt and
binds the test's parse dict as the sandbox variable ``parse`` — mirroring what
``reason`` does, minus the prompt rendering (which is tested in
``test_reasoning.py``).

Run with:  uv run python tests/test_codeact.py
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from r3con.runtime import codeact
from r3con.runtime.codeact import (
    CodeActTurn,
    CodeExecutionError,
    ExecutionResult,
    CodeActResult,
    DEFAULT_EXEC_TIMEOUT_S,
    _clip_assistant_response,
    _extract_code_from_response,
    _format_observation,
    _supports_stop_parameter,
    run_codeact,
)
from r3con.settings import settings


_SYS = "You are a CodeAct agent. The parse is bound as the variable `parse`."


@contextlib.contextmanager
def _patched_llm(fake: Callable[..., str]) -> Iterator[list[list[dict[str, str]]]]:
    """Swap `codeact.litellm_chat_completion` for a scripted fake.

    Yields a list of the message-lists each call observed; tests can assert
    on the conversation history the LLM saw.
    """
    original = codeact.litellm_chat_completion
    seen: list[list[dict[str, str]]] = []

    def wrapper(**kwargs: Any) -> str:
        if "messages" in kwargs and kwargs["messages"] is not None:
            seen.append([dict(m) for m in kwargs["messages"]])
        return fake(**kwargs)

    codeact.litellm_chat_completion = wrapper  # type: ignore[assignment]
    try:
        yield seen
    finally:
        codeact.litellm_chat_completion = original  # type: ignore[assignment]


def _run(
    *,
    parsed: Any = None,
    question: str = "?",
    model: str = "m",
    max_turns: int = settings.REASONING_MAX_TURNS,
    timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    run: Any = None,
    **llm_kwargs: Any,
) -> CodeActResult:
    """Call run_codeact with a dummy system prompt, binding `parsed` as `parse`."""
    return run_codeact(
        system_prompt=_SYS,
        user_message=question,
        model=model,
        variables={"parse": parsed} if parsed is not None else None,
        max_turns=max_turns,
        timeout_s=timeout_s,
        run=run,
        **llm_kwargs,
    )


def _final(value: str) -> str:
    return f"Thought: commit.\n<code>\nfinal_answer({value!r})\n</code>"


def _print_then_continue(expr: str) -> str:
    return f"Thought: peeking.\n<code>\nprint({expr})\n</code>"


def _is_synthesis_call(kwargs: dict[str, Any]) -> bool:
    """True for the post-max_turns synthesis call — its last user message is the
    'give your final answer' prompt, recognizable by a distinctive phrase."""
    msgs = kwargs.get("messages") or [{}]
    return "out of code turns" in (msgs[-1].get("content", "") or "").lower()


# ----- _extract_code_from_response -----


def test_extract_code_block_with_surrounding_prose() -> None:
    response = (
        "Thought: I need to filter records where x is set.\n"
        "<code>\n"
        "print(parse['x'])\n"
        "</code>\n"
        "That should do it."
    )
    assert _extract_code_from_response(response) == "print(parse['x'])"


def test_extract_code_block_strips_inner_whitespace() -> None:
    response = "<code>\n\n  x = 1\n  print(x)\n\n</code>"
    out = _extract_code_from_response(response)
    assert out == "x = 1\n  print(x)"


def test_extract_concatenates_multiple_code_blocks() -> None:
    response = (
        "First we set up:\n"
        "<code>\nx = 10\n</code>\n"
        "Then we compute:\n"
        "<code>\nprint(x * 2)\n</code>"
    )
    assert _extract_code_from_response(response) == "x = 10\n\nprint(x * 2)"


def test_extract_falls_back_to_markdown_fence() -> None:
    response = "Here is the program:\n```python\nprint(parse['x'])\n```"
    assert _extract_code_from_response(response) == "print(parse['x'])"


def test_extract_markdown_fence_strips_code_language_label() -> None:
    """A ```code fence must not leak the literal word 'code' into the program."""
    response = "```code\nrecs = parse['x']\nprint(len(recs))\n```"
    assert _extract_code_from_response(response) == "recs = parse['x']\nprint(len(recs))"


def test_extract_missing_block_raises_with_actionable_message() -> None:
    response = "I think the answer is just 42."
    try:
        _extract_code_from_response(response)
    except CodeExecutionError as e:
        assert "<code>" in str(e) and "</code>" in str(e)
        assert not e.timed_out
    else:
        raise AssertionError("expected CodeExecutionError when no code block is present")


def test_extract_accepts_bare_code_without_tags_or_fence() -> None:
    assert _extract_code_from_response("final_answer('z')") == "final_answer('z')"


def test_extract_accepts_bare_multiline_code() -> None:
    out = _extract_code_from_response("x = 1\nprint(x)")
    assert out == "x = 1\nprint(x)"


def test_extract_bare_data_literal_is_no_code() -> None:
    """A bare data-literal 'answer' (raw JSON / dict / list / string / number, no <code>)
    is NOT code — it routes to the no-code reminder instead of being silently executed to
    '(no output)'. (Tag-less real code like final_answer(...) is still accepted above.)"""
    for bare in (
        '{"《判决文书1》":"判决结果5","《判决文书2》":"判决结果6"}',  # the reported pattern
        "['a', 'b', 'c']",
        "'just a string answer'",
        "42",
    ):
        try:
            _extract_code_from_response(bare)
        except CodeExecutionError as e:
            assert "<code>" in e.message and "final_answer(" in e.message
            assert not e.timed_out
        else:
            raise AssertionError(f"expected no-code for bare literal {bare!r}")


def test_extract_no_code_message_reminds_run_and_commit() -> None:
    """Any no-code response gets ONE message reminding both how to run code and
    how to commit via final_answer() inside a <code> block."""
    for prose in (
        "My final answer is 42.",
        "I really don't know.",
        "1. Thórir\n2. GUDRúN\n3. Amleth",
    ):
        try:
            _extract_code_from_response(prose)
        except CodeExecutionError as e:
            msg = str(e)
            assert "<code>" in msg, f"{prose!r}: missing <code> guidance"
            assert "final_answer(" in msg, f"{prose!r}: missing final_answer reminder"
            assert not e.timed_out
        else:
            raise AssertionError(f"expected CodeExecutionError for {prose!r}")


# ----- _supports_stop_parameter -----


def test_supports_stop_parameter_true_for_standard_model() -> None:
    assert _supports_stop_parameter("openai/gpt-4o") is True


def test_supports_stop_parameter_false_for_reasoning_model() -> None:
    assert _supports_stop_parameter("openai/gpt-5") is False


def test_supports_stop_parameter_unknown_model_is_quiet_and_false() -> None:
    import io

    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        result = _supports_stop_parameter("totally-not-a-real-model-xyz-9000")
    assert result is False
    assert buf.getvalue() == ""


# ----- _clip_assistant_response -----


def test_clip_strips_hallucinated_tail_after_code_close() -> None:
    resp = (
        "Thought: go.\n<code>\nprint(x)\n</code>\n"
        "<observation>\nFAKE 999\n</observation>\n"
        "final_answer('WRONG')"
    )
    out = _clip_assistant_response(resp)
    assert "FAKE 999" not in out
    assert "WRONG" not in out
    assert "print(x)" in out
    assert out.rstrip().endswith("</code>")


def test_clip_truncates_fabricated_observation_without_code() -> None:
    resp = "The answer is five.\n<observation>\nbogus\n</observation>"
    out = _clip_assistant_response(resp)
    assert "bogus" not in out
    assert "</code>" not in out
    assert out.strip() == "The answer is five."


def test_clip_reappends_unclosed_code_block() -> None:
    resp = "Thought: commit.\n<code>\nfinal_answer('ok')"
    out = _clip_assistant_response(resp)
    assert out.rstrip().endswith("</code>")
    assert _extract_code_from_response(out) == "final_answer('ok')"


def test_clip_leaves_clean_prose_untouched() -> None:
    resp = "I think the answer is 42."
    assert _clip_assistant_response(resp) == "I think the answer is 42."


# ----- _format_observation -----


def test_format_observation_wraps_text_in_tags() -> None:
    out = _format_observation("hello")
    assert out.startswith("<observation>")
    assert out.endswith("</observation>")
    assert "hello" in out


def test_format_observation_handles_empty_input() -> None:
    out = _format_observation("")
    assert "no output" in out.lower()


# ----- run_codeact: single-turn happy path -----


def test_run_commits_on_first_turn_via_final_answer() -> None:
    response = _final("the-answer")

    def fake(**_: Any) -> str:
        return response

    with _patched_llm(fake) as message_lists:
        r = _run(parsed={"x": "the-answer"}, question="What is x?")

    assert isinstance(r, CodeActResult)
    assert r.answer == "the-answer"
    assert r.terminated_by == "final_answer"
    assert len(r.turns) == 1
    assert r.turns[0].is_final_answer
    assert r.turns[0].error is None
    assert len(message_lists) == 1
    assert [m["role"] for m in message_lists[0]] == ["system", "user"]


# ----- run_codeact: multi-turn explore-then-commit -----


def test_run_observation_is_fed_back_on_next_turn() -> None:
    responses = iter([_print_then_continue("parse['x']"), _final("done")])

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake) as message_lists:
        r = _run(parsed={"x": "spotted-value"}, max_turns=5)

    assert r.answer == "done"
    assert r.terminated_by == "final_answer"
    assert len(r.turns) == 2
    assert r.turns[0].is_final_answer is False
    assert r.turns[1].is_final_answer is True
    second_call = message_lists[1]
    roles = [m["role"] for m in second_call]
    assert roles == ["system", "user", "assistant", "user"]
    assert "<observation>" in second_call[-1]["content"]
    assert "spotted-value" in second_call[-1]["content"]


def test_run_executor_state_persists_across_turns() -> None:
    responses = iter(
        [
            "Thought: stash.\n<code>\nrows = ['a', 'b', 'c']\nprint('stashed')\n</code>",
            "Thought: use stashed.\n<code>\nfinal_answer(', '.join(rows))\n</code>",
        ]
    )

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake):
        r = _run(parsed={}, max_turns=4)

    assert r.answer == "a, b, c"
    assert r.terminated_by == "final_answer"
    assert len(r.turns) == 2


# ----- run_codeact: error recovery -----


def test_run_recovers_when_first_response_has_no_code_block() -> None:
    responses = iter(["I'm not sure, but I think the answer is 42.", _final("recovered")])

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake) as message_lists:
        r = _run(parsed={"x": "ok"}, max_turns=3)

    assert r.answer == "recovered"
    assert len(r.turns) == 2
    assert r.turns[0].code is None
    assert r.turns[0].error is not None
    assert "<code>" in r.turns[0].error
    second_call_user = message_lists[1][-1]["content"]
    assert "<observation>" in second_call_user
    assert "<code>" in second_call_user


def test_run_reminds_when_response_is_bare_json_answer() -> None:
    """The reported pattern: the model 'answers' with a bare JSON object (no <code>).
    Instead of running it to '(no output)', the loop feeds back the no-code reminder, and
    the model then commits."""
    responses = iter(['{"《判决文书1》":"判决结果5"}', _final("done")])

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake):
        r = _run(parsed={"x": "ok"}, max_turns=3)

    assert r.answer == "done"
    assert r.turns[0].code is None and r.turns[0].error is not None
    assert "(no output)" not in r.turns[0].observation        # NOT silently run to no-op
    assert "final_answer(" in r.turns[0].observation           # the informative reminder
    assert "<code>" in r.turns[0].observation


def test_run_recovers_from_runtime_error() -> None:
    responses = iter(
        ["Thought: oops.\n<code>\nraise RuntimeError('boom')\n</code>", _final("recovered")]
    )

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake) as message_lists:
        r = _run(parsed={"x": "ok"}, max_turns=3)

    assert r.answer == "recovered"
    assert len(r.turns) == 2
    assert r.turns[0].error is not None
    assert "boom" in r.turns[0].error
    assert "boom" in message_lists[1][-1]["content"]


def test_run_recovers_from_timeout() -> None:
    responses = iter(
        ["Thought: hang.\n<code>\nwhile True:\n    pass\n</code>", _final("recovered")]
    )

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake) as message_lists:
        r = _run(parsed={"x": "ok"}, max_turns=2, timeout_s=1.0)

    assert r.answer == "recovered"
    assert "timed out" in (r.turns[0].error or "").lower()
    assert "timed out" in message_lists[1][-1]["content"].lower()


# ----- run_codeact: termination -----


def test_run_max_turns_synthesizes_final_answer() -> None:
    """On max_turns exhaustion, run_codeact makes one plain LLM call over the
    whole conversation and returns its prose answer — not the last raw
    observation. The synthesis call drops the stop sequence (it's prose)."""
    synth_kwargs: dict[str, Any] = {}

    def fake(**kwargs: Any) -> str:
        if _is_synthesis_call(kwargs):
            synth_kwargs.update(kwargs)
            return "Putting it together, the answer is 42."
        return _print_then_continue("'still-working'")

    with _patched_llm(fake):
        r = _run(parsed={}, model="openai/gpt-4o", max_turns=2)

    assert r.terminated_by == "max_turns"
    assert r.answer == "Putting it together, the answer is 42."
    assert len(r.turns) == 2  # the synthesis call is not recorded as a turn
    assert "stop" not in synth_kwargs


def test_run_max_turns_falls_back_to_last_observation_when_synthesis_empty() -> None:
    """If the synthesis call returns nothing, fall back to the last non-empty
    observation — never regress to nothing."""

    def fake(**kwargs: Any) -> str:
        if _is_synthesis_call(kwargs):
            return "   "  # synthesis declines / empty
        return _print_then_continue("'still-looking'")

    with _patched_llm(fake):
        r = _run(parsed={}, max_turns=3)

    assert r.terminated_by == "max_turns"
    assert r.answer == "still-looking"
    assert all(not t.is_final_answer for t in r.turns)


def test_run_max_turns_synthesis_empty_and_no_observations_says_cannot_determine() -> None:
    def fake(**kwargs: Any) -> str:
        if _is_synthesis_call(kwargs):
            return ""
        return "Thought: nop.\n<code>\npass\n</code>"

    with _patched_llm(fake):
        r = _run(parsed={}, max_turns=2)

    assert r.terminated_by == "max_turns"
    assert "Cannot determine" in r.answer or "max_turns" in r.answer


def test_run_falls_back_to_markdown_fence() -> None:
    response = "```python\nfinal_answer('z')\n```"

    def fake(**_: Any) -> str:
        return response

    with _patched_llm(fake):
        r = _run(parsed={"x": "z"})
    assert r.answer == "z"
    assert r.terminated_by == "final_answer"


def test_run_max_turns_zero_rejected() -> None:
    try:
        _run(parsed={}, max_turns=0)
    except ValueError as e:
        assert "max_turns" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_run_binds_variables_into_program() -> None:
    code = "final_answer(sum(it['n'] for it in parse['items']))"
    response = f"Thought: sum.\n<code>\n{code}\n</code>"

    def fake(**_: Any) -> str:
        return response

    with _patched_llm(fake):
        r = _run(parsed={"items": [{"n": 1}, {"n": 2}, {"n": 3}]}, question="sum?")
    assert r.answer == "6"


def test_run_final_answer_value_is_stringified() -> None:
    response = "Thought: numeric.\n<code>\nfinal_answer(42)\n</code>"

    def fake(**_: Any) -> str:
        return response

    with _patched_llm(fake):
        r = _run(parsed={})
    assert r.answer == "42"


# ----- run_codeact: stop-sequence gating + client-side truncation -----


def test_run_passes_stop_sequences_when_model_supports_it() -> None:
    captured: dict[str, Any] = {}

    def fake(**kwargs: Any) -> str:
        captured.update(kwargs)
        return _final("ok")

    with _patched_llm(fake):
        _run(parsed={"x": 1}, model="openai/gpt-4o")
    assert captured.get("stop") == ["</code>", "<observation>"]


def test_run_omits_stop_when_model_does_not_support_it() -> None:
    captured: dict[str, Any] = {}

    def fake(**kwargs: Any) -> str:
        captured.update(kwargs)
        return _final("ok")

    with _patched_llm(fake):
        _run(parsed={"x": 1}, model="openai/gpt-5")
    assert "stop" not in captured


def test_run_strips_hallucinated_tail_so_no_false_commit() -> None:
    responses = iter(
        [
            "Thought: peek.\n<code>\nprint(parse['x'])\n</code>\n"
            "<observation>\nFAKE 999\n</observation>\n"
            "final_answer('WRONG')",
            _final("RIGHT"),
        ]
    )

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake) as message_lists:
        r = _run(parsed={"x": "REALVAL"}, max_turns=5)

    assert r.answer == "RIGHT"
    assert r.turns[0].is_final_answer is False
    assert "WRONG" not in r.turns[0].response
    assert "FAKE 999" not in r.turns[0].response
    assert "REALVAL" in r.turns[0].observation
    assert "FAKE 999" not in r.turns[0].observation
    joined = "".join(m["content"] for m in message_lists[1])
    assert "WRONG" not in joined
    assert "FAKE 999" not in joined


def test_run_commits_from_stop_truncated_response() -> None:
    def fake(**_: Any) -> str:
        return "Thought: commit.\n<code>\nfinal_answer('ok')"  # no closing tag

    with _patched_llm(fake):
        r = _run(parsed={"x": 1}, max_turns=3)

    assert r.answer == "ok"
    assert r.terminated_by == "final_answer"
    assert len(r.turns) == 1
    assert r.turns[0].response.rstrip().endswith("</code>")


def test_run_feeds_commit_hint_when_response_is_final_answer_prose() -> None:
    responses = iter(["The final answer is four.", _final("4")])

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake) as message_lists:
        r = _run(parsed={"x": 1}, max_turns=3)

    assert r.answer == "4"
    assert "final_answer(" in (r.turns[0].error or "")
    assert "final_answer(" in message_lists[1][-1]["content"]


def test_run_commits_from_bare_code_without_tags() -> None:
    def fake(**_: Any) -> str:
        return "final_answer('bare')"

    with _patched_llm(fake):
        r = _run(parsed={"x": 1})
    assert r.answer == "bare"
    assert r.terminated_by == "final_answer"


def test_run_repairs_final_answer_assignment_shadowing() -> None:
    def fake(**_: Any) -> str:
        return "Thought: oops.\n<code>\nfinal_answer = 10\nfinal_answer(99)\n</code>"

    with _patched_llm(fake):
        r = _run(parsed={}, max_turns=1)

    assert r.terminated_by == "final_answer"
    assert r.answer == "99"


def test_run_records_raw_response_before_clipping() -> None:
    raw = (
        "Thought: go.\n<code>\nprint(parse['x'])\n</code>\n"
        "<observation>\nFAKE\n</observation>\n"
        "final_answer('WRONG')"
    )
    responses = iter([raw, _final("RIGHT")])

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake):
        r = _run(parsed={"x": "v"}, max_turns=5)

    t0 = r.turns[0]
    assert "FAKE" not in t0.response and "WRONG" not in t0.response
    assert t0.raw_response == raw


# ----- run_codeact: realistic-parse reflection scenarios -----

KILL_EVENTS_PARSE = {
    "liz_kill_events": [
        {"killer": "Liz", "victim": "Hashke", "victim_gender": "M",
         "night_ordinal": "first night she lost control",
         "quote": "Liz surges toward him with impossible speed..."},
        {"killer": "Liz", "victim": "Granger", "victim_gender": "M",
         "night_ordinal": "first night she lost control",
         "quote": "WHAM, Granger tumbles backward..."},
        {"killer": "Liz", "victim": "third man", "victim_gender": "M",
         "night_ordinal": "first night she lost control",
         "quote": "rip his lower jaw clean away."},
        {"killer": "Granger", "victim": "unknown victim", "victim_gender": "",
         "night_ordinal": "",
         "quote": "Granger swings again..."},
    ]
}


def test_run_real_parse_explore_then_filter_then_commit() -> None:
    responses = iter([
        "Thought: peek at the schema before filtering.\n"
        "<code>\n"
        "print(sorted(parse.keys()))\n"
        "print(parse['liz_kill_events'][0])\n"
        "</code>",
        "Thought: now filter to killer=Liz and night_ordinal contains 'first night'.\n"
        "<code>\n"
        "matches = [r for r in parse['liz_kill_events']\n"
        "           if r['killer'].strip().lower() == 'liz'\n"
        "           and 'first night' in r['night_ordinal'].lower()]\n"
        "print(len(matches), 'matches')\n"
        "</code>",
        "Thought: 3 matches — commit.\n"
        "<code>\n"
        "final_answer(f'{len(matches)} men')\n"
        "</code>",
    ])

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake) as message_lists:
        r = _run(
            parsed=KILL_EVENTS_PARSE,
            question="How many men did Liz kill the first night she lost control?",
            max_turns=5,
        )

    assert r.terminated_by == "final_answer"
    assert r.answer == "3 men"
    assert len(r.turns) == 3
    assert "3 matches" in r.turns[1].observation
    last_user = [m for m in message_lists[2] if m["role"] == "user"][-1]
    assert "3 matches" in last_user["content"]


def test_run_real_parse_filter_too_strict_then_refine() -> None:
    responses = iter([
        "Thought: filter exactly on night_ordinal.\n"
        "<code>\n"
        "matches = [r for r in parse['liz_kill_events']\n"
        "           if r['night_ordinal'] == 'first night she lost control'\n"
        "           and r['killer'] == 'Liz']\n"
        "print('exact:', len(matches))\n"
        "</code>",
        "Thought: empty — loosen the night_ordinal match.\n"
        "<code>\n"
        "matches = [r for r in parse['liz_kill_events']\n"
        "           if 'first night' in r['night_ordinal'].lower()\n"
        "           and r['killer'].lower() == 'liz']\n"
        "final_answer(f'{len(matches)} men')\n"
        "</code>",
    ])

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake):
        r = _run(
            parsed=KILL_EVENTS_PARSE,
            question="How many men did Liz kill the first night?",
            max_turns=5,
        )

    assert r.terminated_by == "final_answer"
    assert r.answer == "3 men"
    assert len(r.turns) == 2
    assert "exact:" in r.turns[0].observation


def test_run_real_parse_date_arithmetic_via_defensive_try_except() -> None:
    """Regression guard for the executor's return-in-except handling."""
    parse = {
        "israel_protests_eruption_events": [
            {"event_description": "large-scale street protests across Israel",
             "date_str": "11 February 2023",
             "quote": "...145,000 people protested in Tel Aviv..."},
        ],
        "shekel_four_year_low_events": [
            {"event_description": "shekel dropped to a four-year low",
             "date_str": "March 20, 2023",
             "quote": "...the shekel dropped to a four-year low"},
        ],
    }
    code = (
        "from datetime import datetime\n"
        "def parse_date(s):\n"
        "    for fmt in ['%d %B %Y', '%B %d, %Y']:\n"
        "        try:\n"
        "            return datetime.strptime(s, fmt)\n"
        "        except:\n"
        "            pass\n"
        "    return None\n"
        "d1 = parse_date(parse['israel_protests_eruption_events'][0]['date_str'])\n"
        "d2 = parse_date(parse['shekel_four_year_low_events'][0]['date_str'])\n"
        "if d1 is None or d2 is None:\n"
        "    final_answer('Cannot determine — date parse failed.')\n"
        "else:\n"
        "    final_answer(f'{(d2 - d1).days} days')"
    )
    response = f"Thought: parse both dates and subtract.\n<code>\n{code}\n</code>"

    def fake(**_: Any) -> str:
        return response

    with _patched_llm(fake):
        r = _run(
            parsed=parse,
            question="How many days elapsed between the protests and the shekel low?",
            max_turns=2,
        )

    assert r.terminated_by == "final_answer"
    assert r.answer == "37 days", (
        f"Got {r.answer!r}. If this says 'Cannot determine' the executor's "
        f"return/except handling regressed — see python_executor.py header."
    )


def test_run_real_parse_loop_records_per_turn() -> None:
    responses = iter([
        "Thought: peek.\n<code>\nprint(len(parse['liz_kill_events']))\n</code>",
        "Thought: commit.\n<code>\nfinal_answer(str(len(parse['liz_kill_events'])))\n</code>",
    ])

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake):
        r = _run(parsed=KILL_EVENTS_PARSE, question="how many records?", max_turns=5)

    assert len(r.turns) == 2
    t1 = r.turns[0]
    assert t1.code is not None
    assert t1.execution is not None
    assert t1.is_final_answer is False
    assert t1.error is None
    assert "4" in t1.observation
    t2 = r.turns[1]
    assert t2.is_final_answer is True
    assert t2.error is None


# ----- run_codeact: tools (extra sandbox callables) -----


def test_run_exposes_tools_in_the_sandbox() -> None:
    """A callable passed via `tools=` is invocable from inside the loop's code."""

    def shout(s: str) -> str:
        return s.upper()

    response = "Thought: use the tool.\n<code>\nfinal_answer(shout('hi'))\n</code>"

    def fake(**_: Any) -> str:
        return response

    with _patched_llm(fake):
        r = run_codeact(
            system_prompt=_SYS,
            user_message="?",
            model="m",
            tools={"shout": shout},
        )
    assert r.answer == "HI"
    assert r.terminated_by == "final_answer"


def test_run_tool_persists_and_composes_across_turns() -> None:
    """A tool is usable across turns just like any binding (state persists)."""

    def add(a: int, b: int) -> int:
        return a + b

    responses = iter(
        [
            "Thought: stash.\n<code>\ntotal = add(2, 3)\nprint(total)\n</code>",
            "Thought: commit.\n<code>\nfinal_answer(add(total, 10))\n</code>",
        ]
    )

    def fake(**_: Any) -> str:
        return next(responses)

    with _patched_llm(fake):
        r = run_codeact(
            system_prompt=_SYS, user_message="?", model="m",
            tools={"add": add}, max_turns=4,
        )
    assert r.answer == "15"


def test_run_rejects_final_answer_as_tool_name() -> None:
    """`final_answer` is reserved as the terminator; passing it as a tool errors
    before any LLM call."""
    try:
        run_codeact(
            system_prompt=_SYS,
            user_message="?",
            model="m",
            tools={"final_answer": lambda x=None: x},
        )
    except ValueError as e:
        assert "final_answer" in str(e)
    else:
        raise AssertionError("expected ValueError for the reserved tool name")


# ----- CodeActTurn shape -----


def test_codeact_turn_dataclass_holds_per_turn_record() -> None:
    t = CodeActTurn(
        response="r",
        raw_response="r-raw",
        code="print(1)",
        execution=ExecutionResult(stdout="1\n"),
        observation="<observation>\n1\n</observation>",
        error=None,
        is_final_answer=False,
    )
    assert t.code == "print(1)"
    assert t.raw_response == "r-raw"
    assert t.is_final_answer is False


if __name__ == "__main__":
    tests = [
        test_extract_code_block_with_surrounding_prose,
        test_extract_code_block_strips_inner_whitespace,
        test_extract_concatenates_multiple_code_blocks,
        test_extract_falls_back_to_markdown_fence,
        test_extract_markdown_fence_strips_code_language_label,
        test_extract_missing_block_raises_with_actionable_message,
        test_extract_accepts_bare_code_without_tags_or_fence,
        test_extract_accepts_bare_multiline_code,
        test_extract_bare_data_literal_is_no_code,
        test_extract_no_code_message_reminds_run_and_commit,
        test_supports_stop_parameter_true_for_standard_model,
        test_supports_stop_parameter_false_for_reasoning_model,
        test_supports_stop_parameter_unknown_model_is_quiet_and_false,
        test_clip_strips_hallucinated_tail_after_code_close,
        test_clip_truncates_fabricated_observation_without_code,
        test_clip_reappends_unclosed_code_block,
        test_clip_leaves_clean_prose_untouched,
        test_format_observation_wraps_text_in_tags,
        test_format_observation_handles_empty_input,
        test_run_commits_on_first_turn_via_final_answer,
        test_run_observation_is_fed_back_on_next_turn,
        test_run_executor_state_persists_across_turns,
        test_run_recovers_when_first_response_has_no_code_block,
        test_run_reminds_when_response_is_bare_json_answer,
        test_run_recovers_from_runtime_error,
        test_run_recovers_from_timeout,
        test_run_max_turns_synthesizes_final_answer,
        test_run_max_turns_falls_back_to_last_observation_when_synthesis_empty,
        test_run_max_turns_synthesis_empty_and_no_observations_says_cannot_determine,
        test_run_falls_back_to_markdown_fence,
        test_run_max_turns_zero_rejected,
        test_run_binds_variables_into_program,
        test_run_final_answer_value_is_stringified,
        test_run_passes_stop_sequences_when_model_supports_it,
        test_run_omits_stop_when_model_does_not_support_it,
        test_run_strips_hallucinated_tail_so_no_false_commit,
        test_run_commits_from_stop_truncated_response,
        test_run_feeds_commit_hint_when_response_is_final_answer_prose,
        test_run_commits_from_bare_code_without_tags,
        test_run_repairs_final_answer_assignment_shadowing,
        test_run_records_raw_response_before_clipping,
        test_run_real_parse_explore_then_filter_then_commit,
        test_run_real_parse_filter_too_strict_then_refine,
        test_run_real_parse_date_arithmetic_via_defensive_try_except,
        test_run_real_parse_loop_records_per_turn,
        test_run_exposes_tools_in_the_sandbox,
        test_run_tool_persists_and_composes_across_turns,
        test_run_rejects_final_answer_as_tool_name,
        test_codeact_turn_dataclass_holds_per_turn_record,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
