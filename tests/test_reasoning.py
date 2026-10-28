import json
import logging
import re

import litellm
import pytest

from r3con import settings
from r3con.config import REASONING_PATHS
from r3con.prompts import load_prompt
from r3con.runs import StageRun, TaskLogger
from r3con.stages.reasoning import _sample_record_per_field, reason
from r3con.stages.relevance import render_relevance

MODEL = "openai/gpt-6-luna"
PROMPTS = {"reasoning/direct": "v2", "reasoning/coding": "v2"}
PATHS = pytest.mark.parametrize("path", REASONING_PATHS)
IDS = [f"d0c{n:03x}" for n in range(10)]
READ_IN_PARTS = "Some documents were too long to read whole and were read in parts"
PRINTS = "Thought: peek.\n<code>\nprint(1)\n</code>"
SAMPLE = "only a SAMPLE"
NUMBERED = re.compile(r"Document \d|\"document\": \d")
# 100 records, 1,609 tokens as JSON: the coding path's first turn takes about 4,200
# tokens with the whole parse and about 2,700 with one sample record per field; the
# direct request about 3,920 with its reply's schema.
ROWS = {"rows": [{"who": f"Site {i:03d} logged nothing unusual"} for i in range(100)]}
LAST_ROW = '"who": "Site 099 logged nothing unusual"'
TWO_DOCUMENTS = {
    "relevance_snippets": ["Halloran logged 5.", ["Merrow, part one.", "Part two."]],
    "parsed": {"rows": [{"who": "Halloran"}, {"who": "Merrow"}, {"who": "Merrow"}]},
    "source_docs": {"rows": [0, 1, 1]},
    "document_ids": IDS[:2],
}
EMPTY_ANSWER = {"answer": " \n", "provenance": []}


def reasoning_prompt(llm, path="coding", **kwargs) -> str:
    """The system prompt ``reason`` sends on ``path``, on the coding path its first
    turn's."""
    kwargs.setdefault("task", "?")
    kwargs.setdefault("schema_code", "class Parse(BaseModel): rows: list[Row]")
    kwargs.setdefault("parsed", {"rows": [{"who": "Halloran"}]})
    kwargs.setdefault("model", MODEL)
    snippets = kwargs.get("relevance_snippets")
    kwargs.setdefault("document_ids", IDS[: len(snippets)] if snippets else IDS)
    reply = citing(path, [])
    reason(prompts=PROMPTS, path=path, completion=llm.replies(reply), **kwargs)
    return llm.requests[-1]["messages"][0]["content"]


def ask(llm, *replies, path="direct", **kwargs):
    """``reason`` over ROWS on ``path``, ``llm`` answering with ``replies``."""
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("document_ids", IDS)
    return reason(
        task="Who?",
        schema_code="",
        parsed=ROWS,
        prompts=PROMPTS,
        path=path,
        completion=llm.replies(*replies),
        **kwargs,
    )


def citing(path: str, provenance: list) -> str | dict:
    """A reply on ``path`` that answers "ok" and cites ``provenance``."""
    if path == "direct":
        return {"answer": "ok", "provenance": provenance}
    return f"Thought: commit.\n<code>\nfinal_answer('ok', provenance={provenance!r})\n</code>"


def first_turn_prompts(llm) -> list[str]:
    return [
        r["messages"][0]["content"] for r in llm.requests if len(r["messages"]) == 2
    ]


def summaries(prompt: str) -> str:
    """The prompt's summaries and parse, without its examples'."""
    return prompt.split("## Rules", 1)[0]


def test_the_parse_is_bound_in_the_sandbox_and_the_commit_returned(llm):
    summing = "<code>\nfinal_answer(sum(it['n'] for it in parse['items']))\n</code>"
    result = reason(
        task="sum?",
        schema_code="class Parse(BaseModel): items: list[Item]",
        parsed={"items": [{"n": 1}, {"n": 2}, {"n": 3}]},
        model=MODEL,
        prompts=PROMPTS,
        path="coding",
        document_ids=[],
        completion=llm.replies(summing),
    )
    assert (result.answer, result.path, result.terminated_by) == (
        "6",
        "coding",
        "final_answer",
    )


def test_on_the_direct_path_one_call_answers_with_the_documents_it_used(llm):
    reply = {"answer": "  Site 099 logged nothing unusual.\n", "provenance": [IDS[1]]}
    result = ask(llm, reply)
    assert result.answer == "Site 099 logged nothing unusual."
    assert result.provenance == [IDS[1]]
    assert (result.path, result.fallback) == ("direct", None)
    assert (result.turns, result.terminated_by) == ((), "reply")
    [request] = llm.requests
    assert request["messages"] == [
        {
            "role": "system",
            "content": load_prompt(
                "reasoning/direct", version="v2", parse_block=json.dumps(ROWS, indent=2)
            ),
        },
        {"role": "user", "content": "Input:\n<task>\nWho?\n</task>"},
    ]


def test_the_direct_request_asks_for_the_answer_and_the_shown_documents_ids(llm):
    reasoning_prompt(llm, "direct", **TWO_DOCUMENTS)
    [request] = llm.requests
    response_format = request["response_format"]["json_schema"]
    assert (response_format["name"], response_format["strict"]) == (
        "ReasoningAnswer",
        True,
    )
    schema = response_format["schema"]
    assert schema["required"] == ["answer", "provenance"]
    assert schema["properties"]["provenance"]["items"]["enum"] == IDS[:2]
    assert "default" not in schema["properties"]["provenance"]


def test_a_direct_reply_without_provenance_cites_nothing(llm):
    result = ask(llm, {"answer": "Halloran."})
    assert (result.answer, result.provenance, result.path) == (
        "Halloran.",
        [],
        "direct",
    )
    assert len(llm.requests) == 1


def test_an_empty_direct_reply_is_sent_again(llm):
    result = ask(llm, "", {"answer": "Halloran.", "provenance": []})
    assert result.answer == "Halloran."
    first, again = llm.requests
    assert again["messages"] == first["messages"]


def test_an_empty_answer_is_sent_again_with_the_validators_error(llm, tmp_path):
    run = StageRun(stage="reasoning", task_logger=TaskLogger("run", root=tmp_path))
    result = ask(llm, EMPTY_ANSWER, {"answer": "Halloran.", "provenance": []}, run=run)
    assert result.answer == "Halloran."
    assert [step.kind for step in run.steps] == ["direct", "direct-retry-1"]
    first, again = llm.requests
    assert again["messages"][0] == first["messages"][0]
    user = again["messages"][1]["content"]
    assert user.startswith(
        "Input:\n<task>\nWho?\n</task>\n\n---\n\n"
        "Your previous response failed schema validation:\n\n"
        "ValidationError: 1 validation error for ReasoningAnswer\nanswer\n"
        "  Value error, the answer is empty"
    )
    assert user.endswith(
        "Answer the same task again, fixing only the offending field(s) so the output "
        "validates against the schema. Output the corrected JSON."
    )


@pytest.mark.parametrize(
    ("flood_guard", "whole"), [(16_000, True), (1_000, False)], ids=["whole", "samples"]
)
def test_direct_replies_that_never_fit_go_to_the_coding_path_with_the_view_it_picks(
    llm, monkeypatch, caplog, flood_guard, whole
):
    monkeypatch.setattr(settings, "REASONING_PARSE_MAX_TOKS", flood_guard)
    caplog.set_level(logging.WARNING, logger="r3con.reasoning")
    bad = [{"provenance": []}, "not JSON", EMPTY_ANSWER]
    result = ask(llm, *bad, citing("coding", []))
    assert (result.answer, result.path) == ("ok", "coding")
    assert result.fallback == {
        "cause": "invalid",
        "error": "answer: Value error, the answer is empty",
    }
    *direct, coding = llm.requests
    assert len(direct) == settings.REASONING_MAX_ATTEMPTS
    assert (LAST_ROW in coding["messages"][0]["content"]) is whole
    assert caplog.messages[-1] == (
        "reasoning's direct reply did not fit its schema in 3 send(s); reasoning with "
        "code instead"
    )


def test_a_reasoning_attempt_cap_set_in_settings_bounds_the_direct_sends(
    llm, monkeypatch
):
    monkeypatch.setattr(settings, "REASONING_MAX_ATTEMPTS", 2)
    result = ask(llm, "not JSON", "{", citing("coding", []))
    assert result.fallback is not None and result.fallback["cause"] == "invalid"
    assert result.fallback["error"].startswith("reply: Invalid JSON")
    assert len(llm.requests) == 3


def test_fewer_than_one_attempt_is_refused_before_any_request(llm):
    with pytest.raises(ValueError, match=r"max_attempts must be >= 1, got 0\."):
        ask(llm, max_attempts=0)
    assert llm.requests == []


@PATHS
@pytest.mark.parametrize(
    ("cited", "kept", "dropped"),
    [
        ([IDS[1], IDS[0]], [IDS[1], IDS[0]], None),
        ([IDS[1], IDS[0], IDS[1]], [IDS[1], IDS[0]], None),
        ([f"{IDS[1]}.2", f"{IDS[1]}.1", IDS[0]], [IDS[1], IDS[0]], None),
        (
            ["ffffff", IDS[0], "Document 2", "ffffff.2"],
            [IDS[0]],
            "['ffffff', 'Document 2', 'ffffff']",
        ),
        ([], [], None),
    ],
    ids=["in-order", "repeats", "parts", "unknown", "none"],
)
def test_only_the_shown_documents_ids_are_kept_in_order_once_each(
    llm, caplog, path, cited, kept, dropped
):
    caplog.set_level(logging.WARNING, logger="r3con.relevance")
    result = reason(
        task="?",
        schema_code="",
        model=MODEL,
        prompts=PROMPTS,
        path=path,
        completion=llm.replies(citing(path, cited)),
        **TWO_DOCUMENTS,
    )
    assert result.provenance == kept
    warning = f"{path}: dropped provenance id(s) the call did not show: {dropped}"
    assert caplog.messages == ([warning] if dropped else [])


@pytest.mark.parametrize(
    ("indices", "uncovered"), [([0, 2, 3], "[2, 3]"), ([-1, 1], "[-1]")]
)
def test_records_from_a_document_without_an_id_are_refused_before_any_request(
    llm, indices, uncovered
):
    message = (
        f"source_docs refers to document index(es) {uncovered}, but document_ids "
        "names 2 document(s): give one id per document."
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        reason(
            task="?",
            schema_code="",
            parsed={"rows": [{"who": "x"} for _ in indices]},
            source_docs={"rows": indices},
            document_ids=IDS[:2],
            model=MODEL,
            prompts=PROMPTS,
            path="direct",
            completion=llm,
        )
    assert llm.requests == []


def test_a_coding_answer_at_the_turn_cap_cites_nothing(llm):
    result = ask(llm, PRINTS, "The answer is ok.", path="coding", max_turns=1)
    assert (result.provenance, result.terminated_by) == ([], "max_turns")


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
def test_the_prompt_names_each_document_by_its_id_and_never_by_number(llm, path):
    prompt = summaries(reasoning_prompt(llm, path, **TWO_DOCUMENTS))
    assert re.findall(r"^### Document ID: (\S+)$", prompt, re.MULTILINE) == [
        IDS[0],
        f"{IDS[1]}.1",
        f"{IDS[1]}.2",
    ]
    stamps = re.findall(r'"document": (.+?),', prompt)
    assert stamps == [f'"{IDS[0]}"', f'"{IDS[1]}"', f'"{IDS[1]}"']
    assert NUMBERED.search(prompt) is None


def test_the_sandboxs_parse_carries_the_ids_the_prompt_shows(llm):
    reads_a_stamp = "<code>\nfinal_answer(parse['rows'][1]['document'])\n</code>"
    result = reason(
        task="?",
        schema_code="",
        model=MODEL,
        prompts=PROMPTS,
        path="coding",
        completion=llm.replies(reads_a_stamp),
        **TWO_DOCUMENTS,
    )
    assert result.answer == IDS[1]


def test_a_sample_view_shows_the_records_ids(llm, monkeypatch):
    monkeypatch.setattr(settings, "REASONING_PARSE_MAX_TOKS", 1_000)
    rows = {"rows": [dict(row) for row in ROWS["rows"]]}
    prompt = reasoning_prompt(
        llm, parsed=rows, source_docs={"rows": [3] * 100}, document_ids=IDS
    )
    assert SAMPLE in prompt
    assert f'"document": "{IDS[3]}"' in prompt
    assert NUMBERED.search(summaries(prompt)) is None


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_ids_and_the_provenance_schema_do_not_depend_on_the_prompt_version(
    llm, version
):
    reason(
        task="?",
        schema_code="",
        model=MODEL,
        prompts={**PROMPTS, "reasoning/direct": version},
        path="direct",
        completion=llm.replies(citing("direct", [])),
        **TWO_DOCUMENTS,
    )
    [request] = llm.requests
    assert request["response_format"]["json_schema"]["name"] == "ReasoningAnswer"
    assert (
        f"### Document ID: {IDS[0]}\nHalloran logged 5."
        in (request["messages"][0]["content"])
    )


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
    assert f'"document": "{IDS[0]}"' in prompt
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
        "rows": [
            {"document": IDS[0], "who": "Halloran"},
            {"document": IDS[1], "who": "Merrow"},
        ]
    }
    assert prompt == load_prompt(
        f"reasoning/{path}",
        version="v2",
        relevance=render_relevance(notes, IDS[:2]),
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
    parts = f"### Document ID: {IDS[0]}.1\npart one\n\n### Document ID: {IDS[0]}.2"
    assert (parts in prompt) is in_parts


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
            "reasoning/coding", version="v2", parse_block=json.dumps(ROWS, indent=2)
        )
        assert caplog.messages == []
    else:
        assert caplog.messages == [
            (
                "the whole parse (about 1,609 tokens) would put reasoning's first turn "
                "at about 4,195 tokens, over the 3,400-token line; showing one sample "
                "record per field"
            )
        ]


def test_a_first_turn_refused_with_the_whole_parse_is_sent_once_more_with_samples(
    llm, caplog, refusal
):
    caplog.set_level(logging.WARNING, logger="r3con.reasoning")
    result = ask(llm, refusal(), PRINTS, citing("coding", []), path="coding")
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
        reasoning_prompt(llm.replies(whole, samples, citing("coding", [])), parsed=ROWS)
    assert failure.value is samples
    assert [SAMPLE in prompt for prompt in first_turn_prompts(llm)] == [False, True]


def test_a_direct_prompt_over_the_line_is_not_sent_and_the_coding_path_answers(
    llm, window, caplog
):
    caplog.set_level(logging.INFO, logger="r3con.reasoning")
    result = ask(llm, citing("coding", []), model=window(4_000))
    assert (result.answer, result.path) == ("ok", "coding")
    assert result.fallback == {"cause": "estimate", "estimate": 3_913, "line": 3_400}
    [request] = llm.requests
    assert SAMPLE in request["messages"][0]["content"]
    assert caplog.messages == [
        (
            "reasoning's direct prompt would take about 3,913 tokens, over the "
            "3,400-token line; reasoning with code instead"
        )
    ]


def test_a_direct_prompt_refused_as_too_long_goes_to_the_coding_path_with_samples(
    llm, caplog
):
    # The direct request takes about 3,920 tokens, the coding one with samples 2,700.
    caplog.set_level(logging.WARNING, logger="r3con.reasoning")
    result = ask(llm.refuses_over(tokens=3_000), citing("coding", []))
    assert (result.answer, result.path) == ("ok", "coding")
    assert result.fallback is not None and result.fallback["cause"] == "refusal"
    assert "maximum context length is 3000 tokens" in result.fallback["error"]
    direct, coding = (request["messages"][0]["content"] for request in llm.requests)
    assert "final_answer(" not in direct and LAST_ROW in direct
    assert "final_answer(" in coding and SAMPLE in coding and LAST_ROW not in coding
    assert caplog.messages == [
        "reasoning's direct prompt was refused as too long; reasoning with code instead"
    ]
