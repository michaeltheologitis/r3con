import logging
import re
from collections import Counter

import pytest
from pydantic import ValidationError

from r3con import settings
from r3con.runs import StageRun, TaskLogger
from r3con.stages.relevance import (
    RelevantContext,
    SnippetRecord,
    draw_document_ids,
    join_parts,
    relevance_snippet,
    render_relevance,
    surface_relevance,
    trail_quotes,
)
from tests.fakes import is_later_round

MODEL = "openai/gpt-6-luna"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
TASK = "Which filing reports higher revenue?"
PROMPTS = {"relevance/first-round": "v2", "relevance/later-rounds": "v3"}
IDS = ["0a0a0a", "1b1b1b", "2c2c2c"]
NOTE = re.compile(r"N\(\w+,r\d\)")
MALFORMED = {"provenance_ids": ["1b1b1b"]}
NONE = (None, None, None)
QUOTES_LINE = (
    "- `quotes`: the sentences of this document that `summary` is based on, copied "
    "exactly."
)


def paragraphs(n: int) -> str:
    """``n`` paragraphs of about 245 characters, P0 to P{n-1}, separated by blank
    lines."""
    return "\n\n".join(f"P{i}. " + "word " * 48 for i in range(n))


# About 4,900 characters: refused whole beside the first-round prompt under a
# 7,800-character limit, and read in halves, from P0 and P10, beside either prompt.
LONG = paragraphs(20)
READ_IN_HALVES = 7_800
# About 9,900 characters: read whole beside the first-round prompt under a
# 14,000-character limit, and refused whole beside the later-rounds prompt.
LONGER = paragraphs(40)
READ_WHOLE_ONLY_IN_ROUND_1 = 14_000


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


def sends_of(llm, document) -> list[dict]:
    """Every request reading ``document``, resends with feedback included."""
    return [r for r in llm.requests if r["messages"][1]["content"].startswith(document)]


def allowed_ids(request) -> dict:
    """The JSON schema of each id a request's reply may cite."""
    schema = request["response_format"]["json_schema"]["schema"]
    return schema["properties"]["provenance_ids"]["items"]


def quote(text: str, start: int | None, end: int | None, match: str | None) -> dict:
    return {"text": text, "start": start, "end": end, "match": match, "score": None}


def check_strict_schema(request, fields: list[str]) -> None:
    """Checks ``request`` asks, in strict mode, for a ``RelevanceSummary`` of
    ``fields``, in order, each required and none with a default: the summary a string
    or null, and the quotes strings."""
    sent = request["response_format"]["json_schema"]
    assert (sent["name"], sent["strict"]) == ("RelevanceSummary", True)
    schema = sent["schema"]
    assert list(schema["properties"]) == schema["required"] == fields
    assert not any("default" in field for field in schema["properties"].values())
    assert schema["properties"]["summary"]["anyOf"] == [
        {"type": "string"},
        {"type": "null"},
    ]
    assert schema["properties"]["quotes"]["items"] == {"type": "string"}


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
    ("snippets", "ids", "rendered"),
    [
        (None, None, ""),
        ([], None, ""),
        ([], IDS, ""),
        (
            ["first note", ""],
            None,
            (
                "### Document 1\nfirst note\n\n"
                "### Document 2\n(no relevant summary for this task)"
            ),
        ),
        (
            [["a", ""], "c"],
            None,
            (
                "### Document 1.1\na\n\n"
                "### Document 1.2\n(no relevant summary for this task)\n\n"
                "### Document 2\nc"
            ),
        ),
        (
            [["a", ""], "c"],
            IDS[:2],
            (
                "### Document ID: 0a0a0a.1\na\n\n"
                "### Document ID: 0a0a0a.2\n(no relevant summary for this task)\n\n"
                "### Document ID: 1b1b1b\nc"
            ),
        ),
    ],
)
def test_the_notes_render_as_one_labelled_section_per_document(snippets, ids, rendered):
    assert render_relevance(snippets, ids) == rendered


def test_notes_and_ids_that_do_not_pair_up_are_refused():
    with pytest.raises(ValueError):
        render_relevance(["one", "two"], IDS[:1])


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


def test_round_one_sends_the_first_round_prompt_and_asks_for_a_summary_and_quotes(
    llm, sections
):
    result = surface(llm, ["DOC0", "DOC1"], rounds=1)
    for request in llm.requests:
        prompt = request["messages"][0]["content"]
        assert f"<task>\n{TASK}\n</task>" in prompt
        shown = sections(prompt)
        assert list(shown) == ["## Output"]
        assert shown["## Output"].startswith("Reply in two fields:")
        assert shown["## Output"].endswith(QUOTES_LINE)
        check_strict_schema(request, ["summary", "quotes"])
    assert result.records == [
        [
            [
                SnippetRecord(
                    id="0a0a0a", summary="N(DOC0,r1)", quotes=[], provenance=[]
                )
            ],
            [
                SnippetRecord(
                    id="1b1b1b", summary="N(DOC1,r1)", quotes=[], provenance=[]
                )
            ],
        ]
    ]


def test_a_later_round_shows_the_others_under_their_ids_and_asks_for_their_ids(
    llm, sections
):
    surface(llm, ["DOC0", "DOC1", "DOC2"], rounds=2, workers=1)
    [request] = reads_of(llm, "DOC0")[1:]
    prompt = request["messages"][0]["content"]
    assert f"<task>\n{TASK}\n</task>" in prompt
    shown = sections(prompt)
    others = "## Task-conditioned summaries of the other documents (previous round)"
    assert list(shown) == [others, "## Output"]
    assert shown[others].endswith(
        "### Summary ID: 1b1b1b\nN(DOC1,r1)\n\n### Summary ID: 2c2c2c\nN(DOC2,r1)"
    )
    output = shown["## Output"]
    assert output.startswith("Reply in three fields:")
    assert (
        f"{QUOTES_LINE}\n- `provenance_ids`: The IDs of the other summaries" in output
    )
    check_strict_schema(request, ["summary", "quotes", "provenance_ids"])
    assert allowed_ids(request) == {"enum": ["1b1b1b", "2c2c2c"], "type": "string"}


@pytest.mark.parametrize(
    "reply",
    [
        {"summary": "A snippet."},
        {"summary": "A snippet.", "quotes": None, "provenance_ids": None},
    ],
    ids=["left-out", "null"],
)
def test_a_reply_without_quotes_or_provenance_ids_quotes_and_cites_nothing(llm, reply):
    result = surface(llm, ["DOC0", "DOC1"], rounds=2, reply=reply)
    assert result.snippets == ["A snippet.", "A snippet."]
    assert by_round(result, "quotes") == [[[[]], [[]]], [[[]], [[]]]]
    assert result.provenance == [[], []]
    assert len(llm.requests) == 4


@pytest.mark.parametrize(
    "pinned",
    [
        {"relevance/first-round": "v1"},
        {"relevance/later-rounds": "v1"},
        {"relevance/later-rounds": "v2"},
        {},
    ],
    ids=["first-round-v1", "later-rounds-v1", "later-rounds-v2", "default"],
)
def test_the_reply_schema_does_not_depend_on_the_prompt_version(llm, pinned):
    def reply(request):
        if not is_later_round(request):
            return "A snippet."
        return {"summary": "A snippet.", "provenance_ids": ["1b1b1b"]}

    result = surface_relevance(
        task=TASK,
        documents=["DOC0", "DOC1"],
        document_ids=IDS[:2],
        model=MODEL,
        prompts={**PROMPTS, **pinned},
        rounds=2,
        completion=llm.answers(relevance=reply),
    )
    assert result.provenance == [["1b1b1b"], []]
    for k, request in enumerate(llm.requests):
        schema = request["response_format"]["json_schema"]["schema"]
        fields = ["summary", "quotes", "provenance_ids"]
        assert list(schema["properties"]) == fields[: 2 if k < 2 else 3]


def test_a_later_rounds_snippet_keeps_the_ids_it_cites(llm, shown_ids):
    def reply(request):
        document = request["messages"][1]["content"]
        if not is_later_round(request):
            return f"{document} in round 1."
        cited = shown_ids(request)
        return {"summary": f"  {document} in round 2.\n", "provenance_ids": cited}

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
        if not is_later_round(request):
            return "A snippet."
        return {
            "summary": "A snippet.",
            "provenance_ids": ["1b1b1b", "0a0a0a", "1b1b1b"],
        }

    with caplog.at_level(logging.WARNING, logger="r3con.provenance"):
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


@pytest.mark.parametrize(
    ("rounds", "replies", "kinds"),
    [
        (
            1,
            [MALFORMED, MALFORMED],
            [
                "relevance-r1-d0",
                "relevance-r1-d0-retry-1",
                "relevance-r1-d0-retry-2",
                "relevance-r1-d1",
            ],
        ),
        (
            2,
            ["A snippet.", "A snippet.", MALFORMED, MALFORMED],
            [
                "relevance-r1-d0",
                "relevance-r1-d1",
                "relevance-r2-d0",
                "relevance-r2-d0-retry-1",
                "relevance-r2-d0-retry-2",
                "relevance-r2-d1",
            ],
        ),
    ],
    ids=["round-1", "round-2"],
)
def test_a_reply_that_does_not_fit_its_schema_is_sent_again(
    llm, tmp_path, rounds, replies, kinds
):
    llm.replies(*replies)
    run = StageRun(stage="relevance", task_logger=TaskLogger("run", root=tmp_path))
    reply = "Fitted."
    result = surface(
        llm, ["DOC0", "DOC1"], rounds=rounds, reply=reply, workers=1, run=run
    )
    assert result.snippets == ["Fitted.", "Fitted."]
    assert [step.kind for step in run.steps] == kinds
    first, *resends = sends_of(llm, "DOC0")[-3:]
    assert first["messages"][1]["content"] == "DOC0"
    for resend in resends:
        assert resend["messages"][0] == first["messages"][0]
        user = resend["messages"][1]["content"]
        assert user.startswith(
            "DOC0\n\n---\n\nYour previous response failed schema validation:\n\n"
            "ValidationError: 1 validation error for RelevanceSummary\nsummary\n"
            "  Field required"
        )
        assert user.endswith(
            "Write the summary of the same document again, fixing only the offending "
            "field(s) so the output validates against the schema. Output the corrected "
            "JSON."
        )


@pytest.mark.parametrize("attempts", [1, 2])
def test_the_last_malformed_reply_is_raised_once_the_attempts_run_out(
    llm, monkeypatch, attempts
):
    monkeypatch.setattr(settings, "RELEVANCE_MAX_ATTEMPTS", attempts)

    def reply(request):
        return MALFORMED if is_later_round(request) else "A snippet."

    with pytest.raises(ValidationError, match="summary"):
        surface(llm, ["DOC0", "DOC1"], rounds=2, reply=reply, workers=1)
    assert len(sends_of(llm, "DOC0")) == 1 + attempts


def test_fewer_than_one_attempt_is_refused_before_any_request(llm, monkeypatch):
    message = r"max_attempts must be >= 1, got 0\. Override via"
    with pytest.raises(ValueError, match=message):
        surface(llm, ["DOC0", "DOC1"], rounds=2, max_attempts=0)
    with pytest.raises(ValueError, match=message):
        relevance_snippet(
            task=TASK,
            document="DOC0",
            other_snippets={"1b1b1b": "A snippet."},
            model=MODEL,
            prompts=PROMPTS,
            max_attempts=0,
            completion=llm,
        )
    monkeypatch.setattr(settings, "RELEVANCE_MAX_ATTEMPTS", 0)
    with pytest.raises(ValueError, match=message):
        surface(llm, ["DOC0", "DOC1"], rounds=2)
    assert llm.requests == []


@pytest.mark.parametrize("empty", [None, " \n"], ids=["null", "blank"])
def test_an_empty_summary_is_kept_empty_without_quotes_and_neither_shown_nor_citable(
    llm, shown_ids, empty
):
    reads = Counter()

    def reply(request):
        document = request["messages"][1]["content"]
        reads[document] += 1
        summary = f"{document} in round {reads[document]}."
        if document == "DOC1" and is_later_round(request):
            summary = empty
        return {"summary": summary, "quotes": [document]}

    result = surface(llm, ["DOC0", "DOC1", "DOC2"], rounds=3, reply=reply, workers=1)
    assert by_round(result, "summary")[1] == [
        ["DOC0 in round 2."],
        [""],
        ["DOC2 in round 2."],
    ]
    assert by_round(result, "quotes")[1] == [
        [[quote("DOC0", 0, 4, "exact")]],
        [[]],
        [[quote("DOC2", 0, 4, "exact")]],
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
        if document == "DOC1" and not is_later_round(request) and empty_first:
            return ""
        return f"Notes on {document}."

    result = surface(llm, documents, rounds=2, reply=reply)
    [round_1, round_2] = reads_of(llm, "DOC0")
    assert round_2["messages"][0] == round_1["messages"][0]
    assert round_2["response_format"] == round_1["response_format"]
    assert not is_later_round(round_2)
    assert result.records[1][0] == [
        SnippetRecord(id="0a0a0a", summary="Notes on DOC0.", quotes=[], provenance=[])
    ]


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
    parts = [
        SnippetRecord(id=f"a.{k}", summary="x", quotes=[], provenance=ids)
        for k, ids in enumerate(provenance, 1)
    ]
    assert RelevantContext(records=[[parts]]).provenance == [joined]


def test_each_rounds_quotes_are_kept_where_they_are_in_the_document_they_quote(llm):
    documents = [
        "Revenue rose to $5M. Costs fell.",
        "Sales were \u201cflat\u201d \u2014 at $3M.",
    ]
    quotes = {
        (documents[0], 1): ["Revenue rose to $5M."],
        (documents[0], 2): ["costs fell.", "Profit doubled."],
        (documents[1], 1): ['Sales were "flat" - at $3M.'],
        (documents[1], 2): [documents[1]],
    }

    def reply(request):
        document = request["messages"][1]["content"]
        round_ = 2 if is_later_round(request) else 1
        return {"summary": "A note.", "quotes": quotes[document, round_]}

    result = surface(llm, documents, rounds=2, reply=reply)
    assert by_round(result, "quotes") == [
        [
            [[quote("Revenue rose to $5M.", 0, 20, "exact")]],
            [[quote('Sales were "flat" - at $3M.', 0, 27, "normalized")]],
        ],
        [
            [
                [
                    quote("costs fell.", 21, 32, "normalized"),
                    quote("Profit doubled.", *NONE),
                ]
            ],
            [[quote(documents[1], 0, 27, "exact")]],
        ],
    ]


def test_a_parts_quotes_are_kept_where_they_are_in_its_whole_document(llm):
    def quoting_its_first_paragraph(request):
        part = request["messages"][1]["content"]
        return {"summary": "A note.", "quotes": [part.split("\n\n", 1)[0]]}

    llm.refuses_over(READ_IN_HALVES)
    documents = ["DOC0", LONG]
    result = surface(
        llm, documents, rounds=2, model=QWEN, reply=quoting_its_first_paragraph
    )
    first, second = LONG.split("\n\n")[0], LONG.split("\n\n")[10]
    second_at = LONG.index("P10. ")
    halves = [
        [quote(first, 0, len(first), "exact")],
        [quote(second, second_at, second_at + len(second), "exact")],
    ]
    assert [quotes[1] for quotes in by_round(result, "quotes")] == [halves, halves]


A, B, C = IDS


def snippet(id_: str, *quotes: dict, cites: tuple[str, ...] = ()) -> SnippetRecord:
    return SnippetRecord(
        id=id_, summary=f"Notes on {id_}.", quotes=list(quotes), provenance=list(cites)
    )


def at(start: int, end: int, text: str = "A sentence.") -> dict:
    """A quote found in its document from ``start`` to ``end``."""
    return quote(text, start, end, "exact")


def quotes_behind(rounds, summary_ids=(), parse_ids=(), parses=None) -> list:
    """The answer's quotes (:func:`trail_quotes`) over ``rounds``, each a round's
    snippets, the answer citing ``summary_ids`` and ``parse_ids``, and each document's
    parse the ids in ``parses`` (none by default)."""
    relevance = RelevantContext(
        records=[
            [[s for s in round_ if s.id.partition(".")[0] == id_] for id_ in IDS]
            for round_ in rounds
        ]
    )
    provenance = {"summary_ids": list(summary_ids), "parse_ids": list(parse_ids)}
    return trail_quotes(relevance, provenance, parses or [[], [], []], IDS)


def places(quotes: list) -> list[tuple]:
    return [(q["document"], q["start"], q["end"]) for q in quotes]


def test_the_answer_quotes_the_final_round_summaries_it_cites():
    final = [
        snippet(A, at(0, 9, "A said x.")),
        snippet(B, at(3, 9)),
        snippet(C, at(4, 20, "C, it said, saw y.")),
    ]
    assert quotes_behind([final], summary_ids=[A, C]) == [
        {"document": A, "start": 0, "end": 9, "text": "A said x."},
        {"document": C, "start": 4, "end": 20, "text": "C, it said, saw y."},
    ]


def test_the_answer_quotes_the_earlier_summaries_its_summaries_cite_round_by_round():
    first = [snippet(A, at(0, 5)), snippet(B, at(10, 15)), snippet(C, at(20, 25))]
    second = [
        snippet(A, at(30, 35)),
        snippet(B, at(40, 45), cites=(C,)),
        snippet(C, at(50, 55)),
    ]
    third = [snippet(A, at(60, 65), cites=(B,)), snippet(B), snippet(C)]
    quotes = quotes_behind([first, second, third], summary_ids=[A])
    assert places(quotes) == [(A, 60, 65), (B, 40, 45), (C, 20, 25)]


def test_the_answer_quotes_what_the_parses_it_cites_cite_and_their_trail():
    first = [snippet(A, at(0, 5)), snippet(B, at(10, 15)), snippet(C, at(20, 25))]
    second = [
        snippet(A, at(30, 35)),
        snippet(B, at(40, 45), cites=(C,)),
        snippet(C, at(50, 55)),
    ]
    quotes = quotes_behind([first, second], parse_ids=[A], parses=[[B], [], []])
    assert places(quotes) == [(B, 40, 45), (C, 20, 25)]


def test_a_cited_parse_that_cites_nothing_adds_no_quotes():
    final = [snippet(A, at(0, 5)), snippet(B, at(10, 15)), snippet(C, at(20, 25))]
    assert quotes_behind([final], parse_ids=[A, B]) == []


def test_a_part_is_cited_by_its_part_id_and_quoted_under_its_documents_id():
    first = [
        snippet(A, at(0, 5)),
        snippet(f"{B}.1", at(10, 15)),
        snippet(f"{B}.2", at(5_000, 5_005)),
        snippet(C, at(20, 25)),
    ]
    second = [
        snippet(A, at(30, 35), cites=(f"{B}.2",)),
        snippet(f"{B}.1", at(40, 45)),
        snippet(f"{B}.2", at(5_040, 5_045)),
        snippet(C, at(50, 55)),
    ]
    quotes = quotes_behind([first, second], summary_ids=[A, f"{B}.1"])
    assert places(quotes) == [(A, 30, 35), (B, 40, 45), (B, 5_000, 5_005)]


def test_a_quote_not_found_in_its_document_is_left_out():
    final = [
        snippet(A, quote("Not in A.", *NONE), at(0, 5)),
        snippet(B, quote("Not in B.", *NONE)),
        snippet(C),
    ]
    assert places(quotes_behind([final], summary_ids=[A, B])) == [(A, 0, 5)]


def test_a_place_quoted_more_than_once_on_the_trail_is_given_once_as_first_met():
    first = [snippet(A, at(0, 9, "As round 1 copied it.")), snippet(B), snippet(C)]
    second = [
        snippet(A, at(0, 9, "As round 2 copied it.")),
        snippet(B, at(0, 9), cites=(A,)),
        snippet(C, cites=(A,)),
    ]
    quotes = quotes_behind([first, second], summary_ids=[A, B, C])
    assert quotes == [
        {"document": A, "start": 0, "end": 9, "text": "As round 2 copied it."},
        {"document": B, "start": 0, "end": 9, "text": "A sentence."},
    ]


def test_quotes_come_in_document_order_then_by_where_they_start():
    final = [
        snippet(A, at(50, 60), at(0, 10)),
        snippet(B, at(5, 8)),
        snippet(C, at(1, 2)),
    ]
    quotes = quotes_behind([final], summary_ids=[C, B, A])
    assert places(quotes) == [(A, 0, 10), (A, 50, 60), (B, 5, 8), (C, 1, 2)]


@pytest.mark.parametrize("n_rounds", [0, 2])
def test_an_answer_that_cites_nothing_has_no_quotes(n_rounds):
    a_round = [snippet(A, at(0, 5)), snippet(B, at(10, 15)), snippet(C, at(20, 25))]
    assert quotes_behind([a_round] * n_rounds) == []
