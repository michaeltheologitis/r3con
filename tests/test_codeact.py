import pytest

from r3con.runtime.codeact import (
    CodeExecutionError,
    _clip_assistant_response,
    _extract_code_from_response,
    _supports_stop_parameter,
    run_codeact,
)

MODEL = "openai/gpt-6-luna"
SYSTEM = "You are a CodeAct agent. The parse is bound as the variable `parse`."
ITEMS = {"items": [{"n": 1}, {"n": 2}, {"n": 3}]}
CANNOT_DETERMINE = (
    "Cannot determine — the reasoning loop hit max_turns with no committed answer."
)


def solve(llm, *replies, **options):
    """``run_codeact`` over SYSTEM, the model answering with ``replies`` in turn."""
    options = {"user_message": "?", "model": MODEL, **options}
    return run_codeact(
        system_prompt=SYSTEM, completion=llm.replies(*replies), **options
    )


def final(value: object) -> str:
    return f"Thought: commit.\n<code>\nfinal_answer({value!r})\n</code>"


def prints(expression: str) -> str:
    return f"Thought: peeking.\n<code>\nprint({expression})\n</code>"


def observation_fed_back(llm, turn: int) -> str:
    """The last message of the request that followed ``turn`` (1-based)."""
    return llm.requests[turn]["messages"][-1]["content"]


@pytest.mark.parametrize(
    ("response", "code"),
    [
        ("Thought: x.\n<code>\nprint(parse['x'])\n</code>\nDone.", "print(parse['x'])"),
        ("<code>\n\n  x = 1\n  print(x)\n\n</code>", "x = 1\n  print(x)"),
        (
            "<code>\nx = 10\n</code>\nThen:\n<code>\nprint(x)\n</code>",
            "x = 10\n\nprint(x)",
        ),
        ("Here:\n```python\nprint(parse['x'])\n```", "print(parse['x'])"),
        (
            "```code\nrecs = parse['x']\nprint(recs)\n```",
            "recs = parse['x']\nprint(recs)",
        ),
        ("```\nprint(1)\n```", "print(1)"),
        ("final_answer('z')", "final_answer('z')"),
        ("x = 1\nprint(x)", "x = 1\nprint(x)"),
    ],
)
def test_code_is_taken_from_tags_then_fences_then_the_bare_response(response, code):
    assert _extract_code_from_response(response) == code


@pytest.mark.parametrize(
    "response",
    [
        '{"《判决文书1》":"判决结果5","《判决文书2》":"判决结果6"}',
        "['a', 'b', 'c']",
        "'just a string answer'",
        "42",
        "My final answer is 42.",
        "I really don't know.",
        "1. Thórir\n2. GUDRúN\n3. Amleth",
    ],
)
def test_a_bare_literal_or_prose_is_refused_with_how_to_run_and_commit(response):
    with pytest.raises(CodeExecutionError) as refused:
        _extract_code_from_response(response)
    assert "<code>" in refused.value.message
    assert "final_answer(" in refused.value.message
    assert not refused.value.timed_out


@pytest.mark.parametrize(
    ("response", "clipped"),
    [
        (
            "Thought: go.\n<code>\nprint(x)\n</code>\n<observation>\n1\n</observation>",
            "Thought: go.\n<code>\nprint(x)\n\n</code>",
        ),
        (
            "The answer is five.\n<observation>\nbogus\n</observation>",
            "The answer is five.\n",
        ),
        (
            "Thought: commit.\n<code>\nfinal_answer('ok')",
            "Thought: commit.\n<code>\nfinal_answer('ok')\n</code>",
        ),
        ("I think the answer is 42.", "I think the answer is 42."),
    ],
)
def test_a_response_is_cut_at_its_first_stop_sequence_and_its_code_reclosed(
    response, clipped
):
    assert _clip_assistant_response(response) == clipped


@pytest.mark.parametrize(
    ("model", "supported"),
    [("openai/gpt-4o", True), ("openai/gpt-5", False), ("my-router-group", False)],
)
def test_stop_support_comes_from_litellms_parameter_map_without_a_word(
    model, supported, capfd
):
    assert _supports_stop_parameter(model) is supported
    assert capfd.readouterr() == ("", "")


def test_a_first_turn_commit_ends_the_loop(llm):
    result = solve(llm, final("the-answer"))
    assert (result.answer, result.terminated_by) == ("the-answer", "final_answer")
    assert len(result.turns) == 1
    assert result.turns[0].is_final_answer
    assert result.turns[0].error is None
    assert [m["role"] for m in llm.requests[0]["messages"]] == ["system", "user"]


@pytest.mark.parametrize(
    ("response", "answer"),
    [
        ("```python\nfinal_answer('z')\n```", "z"),
        ("Thought: commit.\n<code>\nfinal_answer('ok')", "ok"),
        ("final_answer('bare')", "bare"),
        ("<code>\nfinal_answer = 10\nfinal_answer(99)\n</code>", "99"),
        ("<code>\nfinal_answer(42)\n</code>", "42"),
        ("<code>\nfinal_answer(sum(it['n'] for it in parse['items']))\n</code>", "6"),
    ],
)
def test_a_commit_is_run_whatever_shape_its_code_arrives_in(llm, response, answer):
    result = solve(llm, response, variables={"parse": ITEMS}, max_turns=1)
    assert (result.answer, result.terminated_by) == (answer, "final_answer")


def test_what_a_turn_prints_is_fed_back_as_the_next_observation(llm):
    parse = {"x": "spotted-value"}
    result = solve(llm, prints("parse['x']"), final("done"), variables={"parse": parse})
    assert result.answer == "done"
    printed, committed = result.turns
    assert printed.code == "print(parse['x'])"
    assert printed.execution.stdout == "spotted-value\n"
    assert printed.error is None
    assert not printed.is_final_answer
    assert committed.is_final_answer
    roles = [m["role"] for m in llm.requests[1]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    fed_back = observation_fed_back(llm, 1)
    assert fed_back == printed.observation
    assert fed_back.startswith("<observation>\nspotted-value")
    assert fed_back.endswith("</observation>")


def test_a_turn_that_prints_nothing_is_told_so(llm):
    solve(llm, "<code>\nx = 1\n</code>", final("done"))
    assert observation_fed_back(llm, 1) == "<observation>\n(no output)\n</observation>"


def test_bindings_persist_from_one_turn_to_the_next(llm):
    stash = "<code>\nrows = ['a', 'b', 'c']\nprint('stashed')\n</code>"
    result = solve(llm, stash, "<code>\nfinal_answer(', '.join(rows))\n</code>")
    assert result.answer == "a, b, c"
    assert len(result.turns) == 2


@pytest.mark.parametrize(
    ("first", "told"),
    [
        ("I'm not sure, but I think the answer is 42.", "final_answer("),
        ("The final answer is four.", "final_answer("),
        ('{"《判决文书1》":"判决结果5"}', "final_answer("),
        ("Thought: oops.\n<code>\nraise RuntimeError('boom')\n</code>", "boom"),
        ("Thought: hang.\n<code>\nwhile True:\n    pass\n</code>", "timed out"),
    ],
)
def test_a_turn_that_fails_is_told_why_and_the_loop_goes_on(llm, first, told):
    result = solve(llm, first, final("recovered"), max_turns=2, timeout_s=1.0)
    assert result.answer == "recovered"
    assert told in result.turns[0].error
    assert told in observation_fed_back(llm, 1)


@pytest.mark.parametrize(
    ("turn", "synthesis", "answer"),
    [
        (prints("'working'"), "The answer is 42.", "The answer is 42."),
        (prints("'still-looking'"), "   ", "still-looking"),
        ("<code>\npass\n</code>", "", CANNOT_DETERMINE),
    ],
)
def test_running_out_of_turns_asks_once_for_an_answer_in_prose(
    llm, turn, synthesis, answer
):
    result = solve(llm, turn, turn, synthesis, model="openai/gpt-4o", max_turns=2)
    assert (result.answer, result.terminated_by) == (answer, "max_turns")
    assert len(result.turns) == 2
    synthesis_request = llm.requests[2]
    assert "out of code turns" in synthesis_request["messages"][-1]["content"]
    assert "stop" in llm.requests[0]
    assert "stop" not in synthesis_request


def test_max_turns_below_one_is_refused_before_any_request(llm):
    with pytest.raises(ValueError, match="max_turns must be >= 1"):
        solve(llm, max_turns=0)
    assert llm.requests == []


@pytest.mark.parametrize(
    ("model", "stop"),
    [("openai/gpt-4o", ["</code>", "<observation>"]), ("openai/gpt-5", None)],
)
def test_stop_sequences_are_sent_only_to_models_that_take_them(llm, model, stop):
    solve(llm, final("ok"), model=model)
    assert llm.requests[0].get("stop") == stop


def test_a_hallucinated_observation_and_commit_are_cut_and_never_resent(llm):
    raw = (
        "Thought: peek.\n<code>\nprint(parse['x'])\n</code>\n"
        "<observation>\nFAKE 999\n</observation>\nfinal_answer('WRONG')"
    )
    result = solve(llm, raw, final("RIGHT"), variables={"parse": {"x": "REALVAL"}})
    first = result.turns[0]
    assert result.answer == "RIGHT"
    assert not first.is_final_answer
    assert first.raw_response == raw
    assert "FAKE 999" not in first.response
    assert "WRONG" not in first.response
    assert "REALVAL" in first.observation
    resent = "".join(m["content"] for m in llm.requests[1]["messages"])
    assert "FAKE 999" not in resent
    assert "WRONG" not in resent


def test_a_tool_is_callable_from_the_sandbox_across_turns(llm):
    result = solve(
        llm,
        "<code>\ntotal = add(2, 3)\nprint(total)\n</code>",
        "<code>\nfinal_answer(shout(f'{add(total, 10)} ok'))\n</code>",
        tools={"add": lambda a, b: a + b, "shout": str.upper},
    )
    assert result.answer == "15 OK"


def test_final_answer_is_refused_as_a_tool_name_before_any_request(llm):
    with pytest.raises(ValueError, match="'final_answer' is reserved"):
        solve(llm, tools={"final_answer": lambda x=None: x})
    assert llm.requests == []
