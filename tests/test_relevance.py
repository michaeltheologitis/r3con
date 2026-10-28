import logging
import re
from collections import Counter

import pytest
from pydantic import ValidationError

from r3con import settings
from r3con.prompts import load_prompt
from r3con.runs import StageRun, TaskLogger
from r3con.stages.relevance import (
    RelevantContext,
    SnippetRecord,
    draw_document_ids,
    join_parts,
    render_relevance,
    surface_relevance,
)

MODEL = "openai/gpt-6-luna"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
TASK = "Which filing reports higher revenue?"
PROMPTS = {"relevance/first-round": "v1", "relevance/later-rounds": "v1"}
IDS = ["0a0a0a", "1b1b1b", "2c2c2c"]
NOTE = re.compile(r"N\(\w+,r\d\)")
MALFORMED = {"summary": "A reply without its provenance."}


def paragraphs(n: int) -> str:
    """``n`` paragraphs of about 195 characters, P0 to P{n-1}, separated by blank
    lines."""
    return "\n\n".join(f"P{i}. " + "word " * 38 for i in range(n))


# About 3,900 characters: refused whole beside the first-round prompt under a
# 6,500-character limit, and read in halves, from P0 and P10, beside either prompt.
LONG = paragraphs(20)
READ_IN_HALVES = 6_500
# About 7,900 characters: read whole beside the first-round prompt under an
# 11,500-character limit, and refused whole beside the later-rounds prompt.
LONGER = paragraphs(40)
READ_WHOLE_ONLY_IN_ROUND_1 = 11_500


def notes_by_round():
    """A relevance reply naming the document, by its text up to the first full stop,
    and the round it was read in."""
    reads = Counter()

    def note(request) -> str:
        document = request["messages"][1]["content"]
        reads[document] += 1
        return f"N({document.split('.')[0]},r{reads[document]})"

    return note


def surface(llm, documents, rounds, *, model=MODEL, reply=None, **kwargs):
    kwargs.setdefault("document_ids", IDS[: len(documents)])
    return surface_relevance(
        task=TASK,
        documents=documents,
        model=model,
        prompts=PROMPTS,
        rounds=rounds,
        completion=llm.answers(relevance=reply or notes_by_round()),
        **kwargs,
    )


def by_round(result: RelevantContext, field: str) -> list[list[list]]:
    """``field`` of every snippet record, by round, document and part."""
    return [
        [[getattr(record, field) for record in parts] for parts in per_doc]
        for per_doc in result.records
    ]


def reads_of(llm, document) -> list[dict]:
    return [r for r in llm.requests if r["messages"][1]["content"] == document]


def allowed_ids(request) -> dict:
    """The JSON schema of each id a request's reply may cite."""
    schema = request["response_format"]["json_schema"]["schema"]
    return schema["properties"]["provenance"]["items"]


def first_round_prompt() -> str:
    return load_prompt("relevance/first-round", version="v1", task=TASK)


@pytest.mark.parametrize(("n_docs", "rounds"), [(3, 1), (3, 2), (3, 3), (2, 3), (1, 3)])
def test_each_round_reads_a_document_with_only_the_others_last_notes(
    llm, n_docs, rounds
):
    documents = [f"DOC{d}" for d in range(n_docs)]
    surface(llm, documents, rounds, workers=4)
    for document in documents:
        reads = reads_of(llm, document)
        assert len(reads) == rounds
        for k, read in enumerate(reads, start=1):
            seen = set(NOTE.findall(read["messages"][0]["content"]))
            others = {f"N({d},r{k - 1})" for d in documents if d != document}
            assert seen == (others if k > 1 else set())


@pytest.mark.parametrize("rounds", [1, 3])
def test_the_last_rounds_notes_are_returned_aligned_with_the_documents(llm, rounds):
    documents = ["DOC0", "DOC1", "DOC2"]
    result = surface(llm, documents, rounds, workers=4)
    assert result.snippets == [f"N({d},r{rounds})" for d in documents]
    assert by_round(result, "summary") == [
        [[f"N({d},r{k})"] for d in documents] for k in range(1, rounds + 1)
    ]
    assert by_round(result, "id") == [[[id_] for id_ in IDS]] * rounds


@pytest.mark.parametrize(("documents", "rounds"), [([], 3), (["a", "b"], 0)])
def test_no_documents_or_no_rounds_means_no_notes_and_no_request(
    llm, documents, rounds
):
    result = surface(llm, documents, rounds)
    assert (result.records, result.snippets, result.provenance) == ([], [], [])
    assert llm.requests == []


def test_the_system_prompt_holds_the_task_and_the_user_message_the_document(llm):
    surface(llm, ["AAA", "BBB"], rounds=2, workers=1)
    first, second = llm.requests[0]["messages"], llm.requests[2]["messages"]
    assert [first[1]["content"], second[1]["content"]] == ["AAA", "AAA"]
    task = f"<task>\n{TASK}\n</task>"
    assert task in first[0]["content"]
    assert task in second[0]["content"]
    assert "summaries of the other documents" not in first[0]["content"]
    assert "summaries of the other documents" in second[0]["content"]


@pytest.mark.parametrize(
    ("snippets", "rendered"),
    [
        (None, ""),
        ([], ""),
        (
            ["first note", ""],
            (
                "### Document 1\nfirst note\n\n"
                "### Document 2\n(no relevant summary for this task)"
            ),
        ),
        (
            [["a", ""], "c"],
            (
                "### Document 1.1\na\n\n"
                "### Document 1.2\n(no relevant summary for this task)\n\n"
                "### Document 2\nc"
            ),
        ),
    ],
)
def test_the_notes_render_as_one_labelled_section_per_document(snippets, rendered):
    assert render_relevance(snippets) == rendered


@pytest.mark.parametrize(
    ("snippet", "joined"),
    [("a note", "a note"), (["a", " ", "b "], "a\n\nb"), (["", ""], "")],
)
def test_a_split_documents_notes_join_in_order_without_the_empty_ones(snippet, joined):
    assert join_parts(snippet) == joined


def test_a_refused_document_is_read_in_parts_and_keeps_one_note_per_part(llm):
    documents = ["DOC0", LONG, "DOC2"]
    result = surface(llm.refuses_over(READ_IN_HALVES), documents, rounds=2, model=QWEN)
    assert result.snippets == ["N(DOC0,r2)", ["N(P0,r2)", "N(P10,r2)"], "N(DOC2,r2)"]
    assert by_round(result, "summary")[0][1] == ["N(P0,r1)", "N(P10,r1)"]
    halves = [LONG[: LONG.index("P10. ")], LONG[LONG.index("P10. ") :]]
    reads = [r["messages"][1]["content"] for r in llm.requests]
    assert [read for read in reads if read.startswith("P")] == [LONG, *halves, *halves]


def test_a_parts_later_round_sees_the_other_documents_notes_but_never_its_own_parts(
    llm,
):
    llm.refuses_over(READ_IN_HALVES)
    surface(llm, ["DOC0", LONG, "DOC2"], rounds=2, model=QWEN)
    round_2 = {
        request["messages"][1]["content"].split(".")[0]: request["messages"][0]
        for request in llm.requests
        if "summaries of the other documents" in request["messages"][0]["content"]
    }
    seen = {
        reader: set(NOTE.findall(prompt["content"]))
        for reader, prompt in round_2.items()
    }
    assert seen == {
        "P0": {"N(DOC0,r1)", "N(DOC2,r1)"},
        "P10": {"N(DOC0,r1)", "N(DOC2,r1)"},
        "DOC0": {"N(P0,r1)", "N(P10,r1)", "N(DOC2,r1)"},
        "DOC2": {"N(DOC0,r1)", "N(P0,r1)", "N(P10,r1)"},
    }
    assert (
        "### Summary ID: 1b1b1b.1\nN(P0,r1)\n\n### Summary ID: 1b1b1b.2\nN(P10,r1)"
    ) in round_2["DOC0"]["content"]
    assert not any("### Document" in prompt["content"] for prompt in round_2.values())


def test_drawn_document_ids_are_distinct_codes_of_six_hex_characters():
    ids = draw_document_ids(5_000)
    assert len(set(ids)) == 5_000
    assert all(re.fullmatch(r"[0-9a-f]{6}", id_) for id_ in ids)


@pytest.mark.parametrize(
    "document_ids",
    [["0a0a0a"], ["0a0a0a", "1b1b1b", "2c2c2c"], ["0a0a0a", "0a0a0a"]],
    ids=["too-few", "too-many", "repeated"],
)
def test_ids_that_are_not_one_distinct_id_per_document_are_refused(llm, document_ids):
    with pytest.raises(ValueError, match="one distinct id per document"):
        surface(llm, ["DOC0", "DOC1"], rounds=2, document_ids=document_ids)
    assert llm.requests == []


def test_round_one_sends_the_first_round_prompt_and_reads_plain_text(llm):
    result = surface(llm, ["DOC0", "DOC1"], rounds=1)
    for request in llm.requests:
        assert request["messages"][0]["content"] == first_round_prompt()
        assert "response_format" not in request
    assert result.records == [
        [
            [SnippetRecord("0a0a0a", "N(DOC0,r1)", [])],
            [SnippetRecord("1b1b1b", "N(DOC1,r1)", [])],
        ]
    ]


def test_a_later_round_shows_the_others_under_their_ids_and_asks_for_their_ids(llm):
    surface(llm, ["DOC0", "DOC1", "DOC2"], rounds=2, workers=1)
    [request] = reads_of(llm, "DOC0")[1:]
    assert request["messages"][0]["content"] == load_prompt(
        "relevance/later-rounds",
        version="v1",
        task=TASK,
        other_snippets=(
            "### Summary ID: 1b1b1b\nN(DOC1,r1)\n\n### Summary ID: 2c2c2c\nN(DOC2,r1)"
        ),
    )
    schema = request["response_format"]["json_schema"]
    assert (schema["name"], schema["strict"]) == ("RelevanceSummary", True)
    assert list(schema["schema"]["properties"]) == ["summary", "provenance"]
    assert schema["schema"]["properties"]["summary"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert allowed_ids(request) == {"enum": ["1b1b1b", "2c2c2c"], "type": "string"}


def test_a_later_rounds_snippet_keeps_the_ids_it_cites(llm, shown_ids):
    def reply(request):
        document = request["messages"][1]["content"]
        if "response_format" not in request:
            return f"{document} in round 1."
        cited = shown_ids(request)
        return {"summary": f"  {document} in round 2.\n", "provenance": cited}

    result = surface(llm, ["DOC0", "DOC1", "DOC2"], rounds=2, reply=reply)
    assert result.snippets == [
        "DOC0 in round 2.",
        "DOC1 in round 2.",
        "DOC2 in round 2.",
    ]
    assert by_round(result, "provenance") == [
        [[[]], [[]], [[]]],
        [[["1b1b1b", "2c2c2c"]], [["0a0a0a", "2c2c2c"]], [["0a0a0a", "1b1b1b"]]],
    ]
    assert result.provenance == [
        ["1b1b1b", "2c2c2c"],
        ["0a0a0a", "2c2c2c"],
        ["0a0a0a", "1b1b1b"],
    ]


def test_a_cited_id_the_call_did_not_show_is_dropped_with_a_warning(llm, caplog):
    def reply(request):
        if "response_format" not in request:
            return "A snippet."
        return {"summary": "A snippet.", "provenance": ["1b1b1b", "0a0a0a", "1b1b1b"]}

    with caplog.at_level(logging.WARNING, logger="r3con.relevance"):
        result = surface(llm, ["DOC0", "DOC1"], rounds=2, reply=reply)
    assert result.provenance == [["1b1b1b"], ["0a0a0a"]]
    assert (
        "relevance-r2-d0: dropped provenance id(s) the call did not show: ['0a0a0a']"
        in caplog.text
    )
    assert (
        "relevance-r2-d1: dropped provenance id(s) the call did not show: "
        "['1b1b1b', '1b1b1b']" in caplog.text
    )


def test_a_reply_that_does_not_fit_its_schema_is_sent_again(llm, tmp_path):
    llm.replies("Round 1.", "Round 1.", MALFORMED, MALFORMED)
    run = StageRun(stage="relevance", task_logger=TaskLogger("run", root=tmp_path))
    reply = "Round 2."
    result = surface(llm, ["DOC0", "DOC1"], rounds=2, reply=reply, workers=1, run=run)
    assert result.snippets == ["Round 2.", "Round 2."]
    assert [step.kind for step in run.steps] == [
        "relevance-r1-d0",
        "relevance-r1-d1",
        "relevance-r2-d0",
        "relevance-r2-d0-retry-1",
        "relevance-r2-d0-retry-2",
        "relevance-r2-d1",
    ]
    resent = [request["messages"] for request in reads_of(llm, "DOC0")[1:]]
    assert resent == [resent[0]] * 3


@pytest.mark.parametrize("attempts", [1, 2])
def test_the_last_malformed_reply_is_raised_once_the_attempts_run_out(
    llm, monkeypatch, attempts
):
    monkeypatch.setattr(settings, "RELEVANCE_MAX_ATTEMPTS", attempts)

    def reply(request):
        return MALFORMED if "response_format" in request else "A snippet."

    with pytest.raises(ValidationError, match="provenance"):
        surface(llm, ["DOC0", "DOC1"], rounds=2, reply=reply, workers=1)
    assert len(reads_of(llm, "DOC0")) == 1 + attempts


def test_a_null_summary_is_kept_empty_and_neither_shown_nor_citable(llm, shown_ids):
    reads = Counter()

    def reply(request):
        document = request["messages"][1]["content"]
        reads[document] += 1
        if "response_format" not in request:
            return f"{document} in round {reads[document]}."
        summary = (
            None if document == "DOC1" else f"{document} in round {reads[document]}."
        )
        return {"summary": summary, "provenance": []}

    result = surface(llm, ["DOC0", "DOC1", "DOC2"], rounds=3, reply=reply, workers=1)
    assert by_round(result, "summary")[1] == [
        ["DOC0 in round 2."],
        [""],
        ["DOC2 in round 2."],
    ]
    round_3 = reads_of(llm, "DOC0")[2]
    assert shown_ids(round_3) == ["2c2c2c"]
    assert allowed_ids(round_3) == {"enum": ["2c2c2c"], "type": "string"}


@pytest.mark.parametrize(
    ("documents", "empty_first"), [(["DOC0"], False), (["DOC0", "DOC1"], True)]
)
def test_a_later_round_with_nothing_to_show_reads_like_round_one(
    llm, documents, empty_first
):
    def reply(request):
        document = request["messages"][1]["content"]
        if document == "DOC1" and "response_format" not in request and empty_first:
            return ""
        return f"Notes on {document}."

    result = surface(llm, documents, rounds=2, reply=reply)
    [_, round_2] = reads_of(llm, "DOC0")
    assert round_2["messages"][0]["content"] == first_round_prompt()
    assert "response_format" not in round_2
    assert result.records[1][0] == [SnippetRecord("0a0a0a", "Notes on DOC0.", [])]


def test_a_parts_ids_are_numbered_within_each_round(llm, shown_ids):
    llm.refuses_over(READ_WHOLE_ONLY_IN_ROUND_1)
    result = surface(llm, ["DOC0", LONGER], rounds=3, model=QWEN, workers=1)
    assert [ids[1] for ids in by_round(result, "id")] == [
        ["1b1b1b"],
        ["1b1b1b.1", "1b1b1b.2"],
        ["1b1b1b.1", "1b1b1b.2"],
    ]
    _, round_2, round_3 = reads_of(llm, "DOC0")
    assert shown_ids(round_2) == ["1b1b1b"]
    assert shown_ids(round_3) == ["1b1b1b.1", "1b1b1b.2"]


@pytest.mark.parametrize(
    ("provenance", "joined"),
    [
        ([["b", "c"]], ["b", "c"]),
        ([["b", "c"], ["c", "d", "b"]], ["b", "c", "d"]),
        ([[], ["d.2", "d.1"]], ["d.2", "d.1"]),
    ],
)
def test_a_split_documents_provenance_joins_in_order_without_repeats(
    provenance, joined
):
    parts = [SnippetRecord(f"a.{k}", "x", ids) for k, ids in enumerate(provenance, 1)]
    assert RelevantContext(records=[[parts]]).provenance == [joined]
