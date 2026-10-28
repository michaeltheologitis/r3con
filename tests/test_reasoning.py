import json
import logging

import litellm
import pytest

from r3con import settings
from r3con.config import REASONING_PATHS
from r3con.prompts import load_prompt
from r3con.stages.reasoning import _sample_record_per_field, reason
from r3con.stages.relevance import render_relevance

MODEL = "openai/gpt-6-luna"
PROMPTS = {"reasoning/direct": "v1", "reasoning/coding": "v1"}
PATHS = pytest.mark.parametrize("path", REASONING_PATHS)
COMMIT = "Thought: commit.\n<code>\nfinal_answer('ok')\n</code>"
READ_IN_PARTS = "Some documents were too long to read whole and were read in parts"
PRINTS = "Thought: peek.\n<code>\nprint(1)\n</code>"
SAMPLE = "only a SAMPLE"
# 100 records, 1,609 tokens as JSON: the coding path's first turn takes about 3,900
# tokens with the whole parse and about 2,400 with one sample record per field; the
# direct prompt about 3,450.
ROWS = {"rows": [{"who": f"Site {i:03d} logged nothing unusual"} for i in range(100)]}
LAST_ROW = '"who": "Site 099 logged nothing unusual"'


def reasoning_prompt(llm, path="coding", **kwargs) -> str:
    """The system prompt ``reason`` sends on ``path``, on the coding path its first
    turn's."""
    kwargs.setdefault("task", "?")
    kwargs.setdefault("schema_code", "class Parse(BaseModel): rows: list[Row]")
    kwargs.setdefault("parsed", {"rows": [{"who": "Halloran"}]})
    kwargs.setdefault("model", MODEL)
    reply = COMMIT if path == "coding" else "ok"
    reason(prompts=PROMPTS, path=path, completion=llm.replies(reply), **kwargs)
    return llm.requests[-1]["messages"][0]["content"]


def ask(llm, *replies, path="direct", **kwargs):
    """``reason`` over ROWS on ``path``, ``llm`` answering with ``replies``."""
    kwargs.setdefault("model", MODEL)
    return reason(
        task="Who?",
        schema_code="",
        parsed=ROWS,
        prompts=PROMPTS,
        path=path,
        completion=llm.replies(*replies),
        **kwargs,
    )


def first_turn_prompts(llm) -> list[str]:
    return [
        r["messages"][0]["content"] for r in llm.requests if len(r["messages"]) == 2
    ]


def test_the_parse_is_bound_in_the_sandbox_and_the_commit_returned(llm):
    summing = "<code>\nfinal_answer(sum(it['n'] for it in parse['items']))\n</code>"
    result = reason(
        task="sum?",
        schema_code="class Parse(BaseModel): items: list[Item]",
        parsed={"items": [{"n": 1}, {"n": 2}, {"n": 3}]},
        model=MODEL,
        prompts=PROMPTS,
        path="coding",
        completion=llm.replies(summing),
    )
    assert (result.answer, result.path, result.terminated_by) == (
        "6",
        "coding",
        "final_answer",
    )


def test_on_the_direct_path_one_call_answers_and_its_reply_is_the_answer(llm):
    result = ask(llm, "  Site 099 logged nothing unusual.\n")
    assert result.answer == "Site 099 logged nothing unusual."
    assert (result.path, result.fallback) == ("direct", None)
    assert (result.turns, result.terminated_by) == ((), "reply")
    [request] = llm.requests
    assert request["messages"] == [
        {
            "role": "system",
            "content": load_prompt(
                "reasoning/direct", version="v1", parse_block=json.dumps(ROWS, indent=2)
            ),
        },
        {"role": "user", "content": "Input:\n<task>\nWho?\n</task>"},
    ]


def test_an_empty_direct_reply_is_sent_again(llm):
    assert ask(llm, "", "Halloran.").answer == "Halloran."
    first, again = llm.requests
    assert again["messages"] == first["messages"]


@PATHS
def test_the_user_message_is_the_task_in_tags(llm, path):
    reasoning_prompt(llm, path, task="What did Alice say?", parsed={"rows": []})
    user = llm.requests[0]["messages"][1]["content"]
    assert user == "Input:\n<task>\nWhat did Alice say?\n</task>"


@PATHS
def test_every_parse_field_reaches_the_prompt_even_an_empty_one(llm, path):
    prompt = reasoning_prompt(
        llm, path, parsed={"supports": [{"ok": True}], "oppositions": []}
    )
    assert '"supports"' in prompt
    assert '"oppositions": []' in prompt


@PATHS
@pytest.mark.parametrize(
    ("snippets", "shown"), [(["ZZQX-note-42"], True), (None, False), ([], False)]
)
def test_the_relevance_notes_are_in_the_prompt_only_when_given(
    llm, path, snippets, shown
):
    # The heading also appears in the prompt's examples, so look for the note itself.
    prompt = reasoning_prompt(
        llm, path, parsed={"x": [{"y": 1}]}, relevance_snippets=snippets
    )
    assert ("ZZQX-note-42" in prompt) is shown
    assert ("## Document summaries\n\nA short, per-document" in prompt) is shown


@PATHS
def test_each_record_is_stamped_with_its_one_based_source_document(llm, path):
    prompt = reasoning_prompt(
        llm, path, parsed={"docs": [{"a": 1}]}, source_docs={"docs": [4]}
    )
    assert '"document": 5' in prompt


@PATHS
def test_a_small_parse_is_shown_whole_and_the_schema_source_is_not(llm, path):
    schema = "class Rec(BaseModel):\n    UNIQUE_SCHEMA_MARKER: str"
    parsed = {"records": [{"name": f"name_{i}"} for i in range(8)]}
    prompt = reasoning_prompt(
        llm,
        path,
        schema_code=schema,
        parsed=parsed,
        source_docs={"records": list(range(8))},
    )
    assert "name_0" in prompt
    assert "name_7" in prompt
    assert "`document`" in prompt
    assert '"document": 1' in prompt
    assert "UNIQUE_SCHEMA_MARKER" not in prompt
    assert "## Schema used to extract" not in prompt
    assert "If the parse was too large" not in prompt


@PATHS
def test_a_small_parses_prompt_is_its_paths_template_with_the_parse_as_json(llm, path):
    notes = ["Northgate logged 5.", "Riverside logged 6."]
    prompt = reasoning_prompt(
        llm,
        path,
        parsed={"rows": [{"who": "Halloran"}, {"who": "Merrow"}]},
        source_docs={"rows": [0, 1]},
        relevance_snippets=notes,
    )
    stamped = {
        "rows": [{"document": 1, "who": "Halloran"}, {"document": 2, "who": "Merrow"}]
    }
    assert prompt == load_prompt(
        f"reasoning/{path}",
        version="v1",
        relevance=render_relevance(notes),
        parse_block=json.dumps(stamped, indent=2, ensure_ascii=False),
    )


def test_a_huge_parse_is_shown_as_one_sample_per_field_and_says_so(llm):
    parsed = {
        "records": [
            {"name": f"name_{i}", "blob": f"item {i}: descriptive text {i * 7}"}
            for i in range(1500)
        ]
    }
    prompt = reasoning_prompt(llm, parsed=parsed)
    assert "name_0" in prompt
    assert "name_1499" not in prompt
    assert "only a SAMPLE" in prompt
    assert "`parse` variable" in prompt


@pytest.mark.parametrize(("path", "whole"), [("direct", True), ("coding", False)])
def test_only_the_coding_path_shows_samples_of_a_parse_over_the_flood_guard(
    llm, monkeypatch, path, whole
):
    monkeypatch.setattr(settings, "REASONING_PARSE_MAX_TOKS", 1_000)
    prompt = reasoning_prompt(llm, path, parsed=ROWS)
    assert (LAST_ROW in prompt) is whole
    assert (SAMPLE in prompt) is not whole


@PATHS
def test_non_ascii_values_reach_the_prompt_as_text(llm, path):
    prompt = reasoning_prompt(
        llm, path, task="哪个判决结果?", parsed={"判决": ["判决结果4"]}
    )
    assert "判决结果4" in prompt
    assert "\\u5224" not in prompt


@pytest.mark.parametrize(
    ("parse", "shown"),
    [
        ({"判决文书1_result": "判决结果4"}, '`判决文书1_result` (scalar): "判决结果4"'),
        ({"rows": [{"label": "应付账款"}]}, '"label": "应付账款"'),
        ({"meta": {"机构": "上海"}}, '"机构": "上海"'),
    ],
)
def test_the_sample_view_keeps_non_ascii_values_readable(parse, shown):
    sample = _sample_record_per_field(parse)
    assert shown in sample
    assert "\\u" not in sample


@PATHS
@pytest.mark.parametrize(
    ("snippets", "in_parts"),
    [([["part one", "part two"], "whole"], True), (["one", "two"], False)],
    ids=["with-a-part", "without"],
)
def test_the_prompt_says_documents_were_read_in_parts_only_when_a_summary_is_a_part(
    llm, path, snippets, in_parts
):
    prompt = reasoning_prompt(llm, path, relevance_snippets=snippets)
    assert (READ_IN_PARTS in prompt) is in_parts
    assert ("### Document 1.1\npart one\n\n### Document 1.2" in prompt) is in_parts


@pytest.mark.parametrize(("tokens", "whole"), [(4_000, False), (1_000_000, True)])
def test_the_whole_parse_is_shown_only_when_the_first_turn_fits_the_line(
    llm, window, caplog, tokens, whole
):
    caplog.set_level(logging.INFO, logger="r3con.reasoning")
    prompt = reasoning_prompt(llm, parsed=ROWS, model=window(tokens))
    assert (LAST_ROW in prompt) is whole
    assert (SAMPLE in prompt) is not whole
    if whole:
        assert prompt == load_prompt(
            "reasoning/coding", version="v1", parse_block=json.dumps(ROWS, indent=2)
        )
        assert caplog.messages == []
    else:
        assert caplog.messages == [
            (
                "the whole parse (about 1,609 tokens) would put reasoning's first turn "
                "at about 3,905 tokens, over the 3,400-token line; showing one sample "
                "record per field"
            )
        ]


def test_a_first_turn_refused_with_the_whole_parse_is_sent_once_more_with_samples(
    llm, caplog, refusal
):
    caplog.set_level(logging.WARNING, logger="r3con.reasoning")
    result = ask(llm, refusal(), PRINTS, COMMIT, path="coding")
    assert result.answer == "ok"
    whole, samples = first_turn_prompts(llm)
    assert LAST_ROW in whole and SAMPLE not in whole
    assert SAMPLE in samples and LAST_ROW not in samples
    assert llm.requests[-1]["messages"][0]["content"] == samples
    assert caplog.messages == [
        (
            "reasoning's first turn was refused as too long with the whole parse "
            "(about 1,609 tokens); sending it again with one sample record per field"
        )
    ]


def test_a_later_turn_refused_is_raised_as_it_is(llm, refusal):
    refused = refusal()
    with pytest.raises(litellm.ContextWindowExceededError) as failure:
        reasoning_prompt(llm.replies(PRINTS, refused), parsed=ROWS)
    assert failure.value is refused
    assert len(llm.requests) == 2


def test_a_first_turn_refused_with_samples_is_raised(llm, refusal):
    whole, samples = refusal(), refusal()
    with pytest.raises(litellm.ContextWindowExceededError) as failure:
        reasoning_prompt(llm.replies(whole, samples, COMMIT), parsed=ROWS)
    assert failure.value is samples
    assert [SAMPLE in prompt for prompt in first_turn_prompts(llm)] == [False, True]


def test_a_direct_prompt_over_the_line_is_not_sent_and_the_coding_path_answers(
    llm, window, caplog
):
    caplog.set_level(logging.INFO, logger="r3con.reasoning")
    result = ask(llm, COMMIT, model=window(4_000))
    assert (result.answer, result.path) == ("ok", "coding")
    assert result.fallback == {"cause": "estimate", "estimate": 3_452, "line": 3_400}
    [request] = llm.requests
    assert SAMPLE in request["messages"][0]["content"]
    assert caplog.messages == [
        (
            "reasoning's direct prompt would take about 3,452 tokens, over the "
            "3,400-token line; reasoning with code instead"
        )
    ]


def test_a_direct_prompt_refused_as_too_long_goes_to_the_coding_path_with_samples(
    llm, caplog
):
    # The direct prompt takes about 3,450 tokens, the coding one with samples 2,400.
    caplog.set_level(logging.WARNING, logger="r3con.reasoning")
    result = ask(llm.refuses_over(tokens=3_000), COMMIT)
    assert (result.answer, result.path) == ("ok", "coding")
    assert result.fallback is not None and result.fallback["cause"] == "refusal"
    assert "maximum context length is 3000 tokens" in result.fallback["error"]
    direct, coding = (request["messages"][0]["content"] for request in llm.requests)
    assert "final_answer(" not in direct and LAST_ROW in direct
    assert "final_answer(" in coding and SAMPLE in coding and LAST_ROW not in coding
    assert caplog.messages == [
        "reasoning's direct prompt was refused as too long; reasoning with code instead"
    ]
