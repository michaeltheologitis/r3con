import json
import logging

import litellm
import pytest

from r3con.notes import Budget, NotesTooLong
from r3con.prompts import load_prompt
from r3con.runtime.llm import count_tokens
from r3con.splitting import Splits
from r3con.stages.reasoning import _sample_record_per_field, check_first_turn, reason
from r3con.stages.relevance import render_relevance

MODEL = "openai/gpt-6-luna"
COMMIT = "Thought: commit.\n<code>\nfinal_answer('ok')\n</code>"
READ_IN_PARTS = "Some documents were too long to read whole and were read in parts"
PRINTS = "Thought: peek.\n<code>\nprint(1)\n</code>"
SAMPLE = "only a SAMPLE"
# 100 records, 1,609 tokens as JSON: the first turn takes about 3,900 tokens with
# the whole parse and about 2,400 with one sample record per field.
ROWS = {"rows": [{"who": f"Site {i:03d} logged nothing unusual"} for i in range(100)]}
LAST_ROW = '"who": "Site 099 logged nothing unusual"'


def reasoning_prompt(llm, version="v1", **kwargs) -> str:
    """The system prompt ``reason`` sends on its first turn."""
    kwargs.setdefault("task", "?")
    kwargs.setdefault("schema_code", "class Parse(BaseModel): rows: list[Row]")
    kwargs.setdefault("parsed", {"rows": [{"who": "Halloran"}]})
    kwargs.setdefault("model", MODEL)
    reason(prompt_version=version, completion=llm.replies(COMMIT), **kwargs)
    return llm.requests[-1]["messages"][0]["content"]


def too_long() -> litellm.ContextWindowExceededError:
    return litellm.ContextWindowExceededError(
        message="This model's maximum context length is 4000 tokens.",
        model=MODEL,
        llm_provider="openai",
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
        prompt_version="v1",
        completion=llm.replies(summing),
    )
    assert (result.answer, result.terminated_by) == ("6", "final_answer")


def test_the_user_message_is_the_task_in_tags(llm):
    reasoning_prompt(llm, task="What did Alice say?", parsed={"rows": []})
    user = llm.requests[0]["messages"][1]["content"]
    assert user == "Input:\n<task>\nWhat did Alice say?\n</task>"


def test_every_parse_field_reaches_the_prompt_even_an_empty_one(llm):
    prompt = reasoning_prompt(
        llm, parsed={"supports": [{"ok": True}], "oppositions": []}
    )
    assert '"supports"' in prompt
    assert '"oppositions": []' in prompt


@pytest.mark.parametrize(
    ("snippets", "shown"), [(["ZZQX-note-42"], True), (None, False), ([], False)]
)
def test_the_relevance_notes_are_in_the_prompt_only_when_given(llm, snippets, shown):
    # The heading also appears in the prompt's examples, so look for the note itself.
    prompt = reasoning_prompt(
        llm, parsed={"x": [{"y": 1}]}, relevance_snippets=snippets
    )
    assert ("ZZQX-note-42" in prompt) is shown
    assert ("## Document summaries\n\nA short, per-document" in prompt) is shown


def test_each_record_is_stamped_with_its_one_based_source_document(llm):
    prompt = reasoning_prompt(
        llm, parsed={"docs": [{"a": 1}]}, source_docs={"docs": [4]}
    )
    assert '"document": 5' in prompt


def test_a_small_parse_is_shown_whole_and_the_schema_source_is_not(llm):
    schema = "class Rec(BaseModel):\n    UNIQUE_SCHEMA_MARKER: str"
    parsed = {"records": [{"name": f"name_{i}"} for i in range(8)]}
    prompt = reasoning_prompt(
        llm, schema_code=schema, parsed=parsed, source_docs={"records": list(range(8))}
    )
    assert "name_0" in prompt
    assert "name_7" in prompt
    assert "`document`" in prompt
    assert '"document": 1' in prompt
    assert "UNIQUE_SCHEMA_MARKER" not in prompt
    assert "## Schema used to extract" not in prompt
    assert "If the parse was too large" not in prompt


def test_a_small_parses_prompt_is_the_v1_template_with_the_parse_as_json(llm):
    notes = ["Northgate logged 5.", "Riverside logged 6."]
    prompt = reasoning_prompt(
        llm,
        parsed={"rows": [{"who": "Halloran"}, {"who": "Merrow"}]},
        source_docs={"rows": [0, 1]},
        relevance_snippets=notes,
    )
    stamped = {
        "rows": [{"document": 1, "who": "Halloran"}, {"document": 2, "who": "Merrow"}]
    }
    assert prompt == load_prompt(
        "reasoning",
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


def test_non_ascii_values_reach_the_prompt_as_text(llm):
    prompt = reasoning_prompt(llm, task="哪个判决结果?", parsed={"判决": ["判决结果4"]})
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


@pytest.mark.parametrize("snippets", [None, [], ["Northgate logged 5.", ""]])
def test_without_a_part_v2_is_the_v1_prompt_byte_for_byte(llm, snippets):
    v1 = reasoning_prompt(llm, "v1", relevance_snippets=snippets)
    assert reasoning_prompt(llm, "v2", relevance_snippets=snippets) == v1


@pytest.mark.parametrize(
    ("version", "snippets", "sentence", "part_heading"),
    [
        ("v2", [["part one", "part two"], "whole"], True, True),
        ("v2", ["one", "two"], False, False),
        ("v1", [["part one", "part two"], "whole"], False, True),
    ],
    ids=["v2-with-a-part", "v2-without", "v1-with-a-part"],
)
def test_v2_says_documents_were_read_in_parts_only_when_a_summary_is_a_part(
    llm, version, snippets, sentence, part_heading
):
    prompt = reasoning_prompt(llm, version, relevance_snippets=snippets)
    assert (READ_IN_PARTS in prompt) is sentence
    assert ("### Document 1.1\npart one\n\n### Document 1.2" in prompt) is part_heading


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
            "reasoning", version="v1", parse_block=json.dumps(ROWS, indent=2)
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
    llm, caplog
):
    caplog.set_level(logging.WARNING, logger="r3con.reasoning")
    result = reason(
        task="?",
        schema_code="",
        parsed=ROWS,
        model=MODEL,
        prompt_version="v1",
        completion=llm.replies(too_long(), PRINTS, COMMIT),
    )
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


def test_a_later_turn_refused_is_raised_as_it_is(llm):
    refused = too_long()
    with pytest.raises(litellm.ContextWindowExceededError) as failure:
        reasoning_prompt(llm.replies(PRINTS, refused), parsed=ROWS)
    assert failure.value is refused
    assert len(llm.requests) == 2


@pytest.mark.parametrize(
    "snippets", [None, ["Northgate logged 5."]], ids=["no-notes", "no-budget"]
)
def test_a_first_turn_refused_with_samples_hands_over_its_notes(llm, snippets):
    whole, samples = too_long(), too_long()
    with pytest.raises(litellm.ContextWindowExceededError) as failure:
        reasoning_prompt(
            llm.replies(whole, samples, COMMIT),
            parsed=ROWS,
            relevance_snippets=snippets,
        )
    assert failure.value is samples
    assert [SAMPLE in prompt for prompt in first_turn_prompts(llm)] == [False, True]


@pytest.mark.parametrize("n_notes", [40, 1])
def test_the_first_turn_without_a_parse_hands_over_notes_over_the_line(window, n_notes):
    notes = [" ".join(["word"] * 60)] * n_notes
    budget = Budget(Splits([], model=window(4_000)))
    if n_notes == 1:
        check_first_turn(
            task="?", relevance_snippets=notes, prompt_version="v2", budget=budget
        )
        return
    with pytest.raises(NotesTooLong) as handed:
        check_first_turn(
            task="?", relevance_snippets=notes, prompt_version="v2", budget=budget
        )
    without_a_parse = load_prompt(
        "reasoning", version="v2", task="?", relevance=render_relevance(notes)
    )
    assert (handed.value.call, handed.value.notes) == ("reasoning", notes)
    assert handed.value.estimate == count_tokens(without_a_parse) + count_tokens(
        "Input:\n<task>\n?\n</task>"
    )
