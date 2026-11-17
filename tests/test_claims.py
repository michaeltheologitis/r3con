import json
import logging
import shutil
import threading

import pytest

from r3con import Answer, Cancelled, ask, resume, run, trace
from r3con.config import CLAIMS_STAGES, EVIDENCE_LEVELS
from tests.fakes import (
    ANSWERING,
    TWO_LINE_MEMOS,
    bad_key,
    citing,
    claims_reply,
    first_line,
    in_json,
    is_later_round,
    quoting_a_line_a_round,
    shown_evidence,
    stage_of,
)

QUESTION = "Who?"
HALLORAN = "The Halloran memo logs two sites."
MERROW = "The Merrow memo logs two sites."
ANSWER = f"{HALLORAN} {MERROW}"
SENTENCE_SYSTEM = """You will be given a question and an answer that is supported by a collection of quotes. Your job is to extract the exact claims made in the answer, word for word (each sentence, list item or table row), along with the IDs of the quotes that support them, if any.
Return exactly one JSON object with the key `claims`, where each item has `text` and `quote_ids`. Output only that JSON.
Preserve the original wording of each claim exactly as it appears in the answer. Include every claim from the answer once, in the same order it appears. Base quote_ids only on the provided quotes; if no quote supports a claim, use an empty array. Do not guess or infer unsupported quote IDs.
## Example
<question>
Which bakery sold more loaves in March, and by how much?
</question>
<answer>
Rye & Co. sold more loaves in March: 1,240 against Hearth's 980, a difference of 260. Neither report gives prices.
</answer>
<quotes>
### Quote ID: q1
Loaves sold in March: 1,240.
### Quote ID: q2
All flour is milled on site.
### Quote ID: q3
Loaves sold in March: 980.
### Quote ID: q4
The shop is open seven days a week.
</quotes>
Reply: {"claims": [{"text": "Rye & Co. sold more loaves in March: 1,240 against Hearth's 980, a difference of 260.", "quote_ids": ["q1", "q3"]}, {"text": "Neither report gives prices.", "quote_ids": []}]}"""
DOCUMENT_SYSTEM = """You will be given a question and an answer that is supported by a collection of document summaries. Your job is to extract the exact claims made in the answer, word for word (each sentence, list item or table row), along with the IDs of the documents whose summaries support them, if any.
Return exactly one JSON object with the key `claims`, where each item has `text` and `document_ids`. Output only that JSON.
Preserve the original wording of each claim exactly as it appears in the answer. Include every claim from the answer once, in the same order it appears. Base document_ids only on the provided summaries; if no summary supports a claim, use an empty array. Do not guess or infer unsupported document IDs.
## Example
<question>
Which trial sites reported adverse events?
</question>
<answer>
- Site A: two mild headaches.
- Site C: one case of nausea.
- Site B reported none.
</answer>
<summaries>
### Document ID: 3f9a
Site A's safety log for the trial: two adverse events, both mild headaches.
### Document ID: 7c21
Site B's monthly report: enrollment figures, and a note that no adverse events occurred.
### Document ID: b604
Site C's safety summary: one adverse event, a case of nausea, resolved the same day.
</summaries>
Reply: {"claims": [{"text": "Site A: two mild headaches.", "document_ids": ["3f9a"]}, {"text": "Site C: one case of nausea.", "document_ids": ["b604"]}, {"text": "Site B reported none.", "document_ids": ["7c21"]}]}"""
# The trail of an ANSWER over TWO_LINE_MEMOS quotes each memo's line in its order.
SENTENCE_USER = """<question>
Who?
</question>
<answer>
The Halloran memo logs two sites. The Merrow memo logs two sites.
</answer>
<quotes>
### Quote ID: q1
Halloran memo
### Quote ID: q2
The Halloran memo logs two sites.
### Quote ID: q3
Merrow memo
### Quote ID: q4
The Merrow memo logs two sites.
### Quote ID: q5
The Calder memo logs two sites.
</quotes>"""


@pytest.fixture
def quoting_llm(answering_llm, shown_ids):
    """``answering_llm``, each summary quoting a line of its memo a round and citing
    the first summary shown, and the answer ANSWER, citing every summary."""
    return answering_llm.answers(
        relevance=quoting_a_line_a_round(shown_ids),
        reasoning=citing(ANSWER, slice(None)),
    )


def claimed(llm, tmp_path, *claims, **options) -> Answer:
    """A run of QUESTION over TWO_LINE_MEMOS whose claims reply is ``claims``, each
    its text and the ids that support it."""
    llm.answers(claims=claims_reply(*claims))
    logs = tmp_path / "logs"
    return run(QUESTION, TWO_LINE_MEMOS, completion=llm, logs_dir=logs, **options)


def place(quote: dict) -> dict:
    """Where ``quote`` is, as a claim names it."""
    return {key: quote[key] for key in ("document", "start", "end")}


def reply_schema(ids_key: str, title: str, ids: list[str]) -> dict:
    """The strict schema a claims request sends for a reply citing ``ids``."""
    claim = {
        "properties": {
            "text": {"title": "Text", "type": "string"},
            ids_key: {
                "items": {"enum": ids, "type": "string"},
                "title": title,
                "type": "array",
            },
        },
        "required": ["text", ids_key],
        "title": "Claim",
        "type": "object",
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "Claims",
            "schema": {
                "$defs": {"Claim": claim},
                "properties": {
                    "claims": {
                        "items": {"$ref": "#/$defs/Claim"},
                        "title": "Claims",
                        "type": "array",
                    }
                },
                "title": "Claims",
                "type": "object",
                "additionalProperties": False,
                "required": ["claims"],
            },
            "strict": True,
        },
    }


def test_a_sentence_level_request_is_its_prompt_then_the_question_answer_and_quotes(
    quoting_llm, tmp_path
):
    run(QUESTION, TWO_LINE_MEMOS, completion=quoting_llm, logs_dir=tmp_path)
    [request] = quoting_llm.requests_for("claims")
    assert request["messages"] == [
        {"role": "system", "content": SENTENCE_SYSTEM},
        {"role": "user", "content": SENTENCE_USER},
    ]
    quote_ids = ["q1", "q2", "q3", "q4", "q5"]
    assert request["response_format"] == reply_schema(
        "quote_ids", "Quote Ids", quote_ids
    )


def test_a_document_level_request_is_its_prompt_then_the_question_answer_and_summaries(
    answering_llm, tmp_path
):
    answering_llm.answers(reasoning=citing(ANSWER, slice(None)))
    result = run(
        QUESTION,
        TWO_LINE_MEMOS,
        evidence_level="document",
        completion=answering_llm,
        logs_dir=tmp_path,
    )
    halloran, merrow, calder = result.document_ids
    [request] = answering_llm.requests_for("claims")
    assert request["messages"] == [
        {"role": "system", "content": DOCUMENT_SYSTEM},
        {
            "role": "user",
            "content": f"""<question>
Who?
</question>
<answer>
The Halloran memo logs two sites. The Merrow memo logs two sites.
</answer>
<summaries>
### Document ID: {halloran}
Notes on Halloran memo.
### Document ID: {merrow}
Notes on Merrow memo.
### Document ID: {calder}
Notes on Calder memo.
</summaries>""",
        },
    ]
    assert request["response_format"] == reply_schema(
        "document_ids", "Document Ids", result.document_ids
    )


def test_claims_are_asked_of_reasonings_model_with_its_params(quoting_llm, tmp_path):
    params = {"temperature": 0.3, "seed": 7}
    claimed(quoting_llm, tmp_path, (ANSWER, []), model="openai/gpt-4o", params=params)
    [reasoning] = quoting_llm.requests_for("reasoning")
    [claims] = quoting_llm.requests_for("claims")
    assert claims["model"] == reasoning["model"] == "openai/gpt-4o"
    assert (claims["temperature"], claims["seed"]) == (0.3, 7)


def test_claims_are_the_answers_own_words_in_its_order_with_what_supports_them(
    quoting_llm, tmp_path
):
    result = claimed(quoting_llm, tmp_path, (MERROW, ["q5", "q3"]), (HALLORAN, ["q2"]))
    quotes = result.provenance["quotes"]
    halloran, merrow, calder = result.document_ids
    merrow_starts = len(HALLORAN) + 1
    assert result.provenance["claims"] == [
        {
            "text": HALLORAN,
            "start": 0,
            "end": len(HALLORAN),
            "documents": [halloran],
            "quotes": [place(quotes[1])],
        },
        {
            "text": MERROW,
            "start": merrow_starts,
            "end": len(ANSWER),
            "documents": [merrow, calder],
            "quotes": [place(quotes[2]), place(quotes[4])],
        },
    ]
    saved = json.loads((result.run_dir / "claims" / "result.json").read_text())
    assert saved["claims"] == result.provenance["claims"]


def test_a_sentence_the_answer_repeats_is_claimed_where_each_claim_makes_it(
    quoting_llm, tmp_path
):
    repeated = f"{ANSWER} {HALLORAN}"
    quoting_llm.answers(reasoning=citing(repeated, slice(None)))
    result = claimed(
        quoting_llm, tmp_path, (HALLORAN, ["q2"]), (MERROW, []), (HALLORAN, ["q1"])
    )
    claims = result.provenance["claims"]
    starts = [0, len(HALLORAN) + 1, repeated.rindex(HALLORAN)]
    assert [claim["start"] for claim in claims] == starts
    assert [claim["text"] for claim in claims] == [HALLORAN, MERROW, HALLORAN]


def test_an_id_the_call_did_not_show_is_dropped_with_a_warning(
    quoting_llm, tmp_path, caplog
):
    caplog.set_level(logging.WARNING, logger="r3con")
    result = claimed(quoting_llm, tmp_path, (HALLORAN, ["q9", "q2", "a1b2c3"]))
    [claim] = result.provenance["claims"]
    assert claim["quotes"] == [place(result.provenance["quotes"][1])]
    assert (
        "claims: dropped provenance id(s) the call did not show: ['q9', 'a1b2c3']"
    ) in caplog.text


def test_a_claim_the_answer_does_not_make_is_dropped_with_a_warning(
    quoting_llm, tmp_path, caplog
):
    caplog.set_level(logging.WARNING, logger="r3con")
    result = claimed(
        quoting_llm,
        tmp_path,
        (HALLORAN, []),
        ("Calder logs nothing.", ["q5"]),
        (" ", []),
        (MERROW, []),
    )
    assert [claim["text"] for claim in result.provenance["claims"]] == [
        HALLORAN,
        MERROW,
    ]
    assert (
        "claims: dropped a claim not found in the answer: 'Calder logs nothing.'"
    ) in caplog.text


@pytest.mark.parametrize(
    ("altered", "found"),
    [
        ("the merrow memo logs two sites", MERROW.removesuffix(".")),
        ("The Merrow memo lists two sites.", MERROW),
    ],
    ids=["normalized", "fuzzy"],
)
def test_a_claim_altered_slightly_is_found_and_keeps_the_answers_words(
    quoting_llm, tmp_path, altered, found
):
    result = claimed(quoting_llm, tmp_path, (altered, ["q4"]))
    [claim] = result.provenance["claims"]
    assert claim["text"] == found == ANSWER[claim["start"] : claim["end"]]
    assert claim["quotes"] == [place(result.provenance["quotes"][3])]


@pytest.mark.parametrize("level", EVIDENCE_LEVELS)
def test_an_answer_whose_trail_holds_no_evidence_makes_no_claims_call(
    answering_llm, tmp_path, level
):
    result = run(
        QUESTION,
        TWO_LINE_MEMOS,
        evidence_level=level,
        completion=answering_llm,
        logs_dir=tmp_path,
    )
    assert (result.provenance["quotes"], result.provenance["documents"]) == ([], [])
    assert answering_llm.requests_for("claims") == []
    assert result.provenance["claims"] == []
    folder = result.run_dir / "claims"
    assert json.loads((folder / "calls.json").read_text()) == []
    assert json.loads((folder / "result.json").read_text())["claims"] == []


@pytest.mark.parametrize(
    ("reply", "sends"), [(bad_key(), 1), ("not JSON", 3)], ids=["refused", "invalid"]
)
def test_a_claims_call_that_fails_leaves_no_claims_and_the_run_finishes(
    quoting_llm, tmp_path, caplog, events_in, reply, sends
):
    caplog.set_level(logging.WARNING, logger="r3con")
    quoting_llm.answers(claims=reply)
    result = run(QUESTION, TWO_LINE_MEMOS, completion=quoting_llm, logs_dir=tmp_path)
    assert result.answer == ANSWER
    assert result.provenance["claims"] == []
    assert len(quoting_llm.requests_for("claims")) == sends
    assert "claims: the answer has none, as their call failed: " in caplog.text
    events = events_in(result.run_dir)
    assert [event["type"] for event in events[-3:]] == [
        "claims_started",
        "claims_finished",
        "run_finished",
    ]
    assert events[-2]["claims"] == []
    assert not (result.run_dir / "claims" / "error.txt").exists()


def test_claims_turned_off_make_no_call_and_leave_no_trace_but_the_label(
    quoting_llm, tmp_path, events_in
):
    result = run(
        QUESTION,
        TWO_LINE_MEMOS,
        claims=False,
        completion=quoting_llm,
        logs_dir=tmp_path,
    )
    assert result.provenance["quotes"]
    assert quoting_llm.requests_for("claims") == []
    assert result.provenance["claims"] == trace(result)["claims"] == []
    assert not (result.run_dir / "claims").exists()
    assert not any(e["type"].startswith("claims_") for e in events_in(result.run_dir))
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert (manifest["config"]["claims"], manifest["config"]["overrides"]) == (
        False,
        {"claims": False},
    )


def test_a_document_level_request_shows_each_round_1_summary_or_the_latest_instead(
    answering_llm, grown_registry, window, tmp_path
):
    def summary(request) -> str:
        line = first_line(request)
        if is_later_round(request):
            return f"Round 2 notes on {line}."
        return "" if line == "Merrow memo" else f"Round 1 notes on {line}."

    def claiming_a_part_and_merrow(request) -> dict:
        shown = shown_evidence(request)
        return claims_reply((ANSWER, [shown[-1], shown[2]]))(request)

    answering_llm.answers(
        relevance=summary,
        reasoning=citing(ANSWER, slice(None)),
        claims=claiming_a_part_and_merrow,
    )
    documents = ["Halloran memo", grown_registry(40_000), "Merrow memo"]
    result = run(
        QUESTION,
        documents,
        model=window(5_000),
        evidence_level="document",
        completion=answering_llm,
        logs_dir=tmp_path,
    )
    halloran, registry, merrow = result.document_ids
    relevance = json.loads((result.run_dir / "relevance" / "result.json").read_text())
    round_1 = {s["id"]: s["summary"] for s in relevance["rounds"][0]["snippets"]}
    parts = [id_ for id_ in round_1 if id_.startswith(f"{registry}.")]
    assert len(parts) > 1
    shown = [
        (halloran, "Round 1 notes on Halloran memo."),
        *((part, round_1[part]) for part in parts),
        (merrow, "Round 2 notes on Merrow memo."),
    ]
    [request] = answering_llm.requests_for("claims")
    summaries = request["messages"][1]["content"].split("\n<summaries>\n", 1)[1]
    assert summaries == "".join(
        f"### Document ID: {id_}\n{text}\n" for id_, text in shown
    ) + ("</summaries>")
    assert all(text.startswith("Round 1 notes") for _, text in shown[1:-1])
    [claim] = result.provenance["claims"]
    assert (claim["documents"], claim["quotes"]) == ([registry, merrow], [])


def test_trace_gives_the_saved_claims_whose_quotes_its_rules_trails_quote(
    quoting_llm, tmp_path
):
    result = claimed(quoting_llm, tmp_path, (HALLORAN, ["q2"]), (MERROW, ["q4"]))
    assert result.provenance["claims"]
    for rule in ("through-summaries", "parses-light-documents"):
        traced = trace(result.run_dir, rule)
        assert traced["claims"] == result.provenance["claims"]
        places = [place(quote) for quote in traced["quotes"]]
        assert all(q in places for c in traced["claims"] for q in c["quotes"])


def test_a_folder_saved_before_claims_traces_with_none(quoting_llm, tmp_path):
    result = claimed(quoting_llm, tmp_path, (HALLORAN, ["q2"]))
    shutil.rmtree(result.run_dir / "claims")
    assert trace(result)["claims"] == []


def test_an_ask_finds_the_claims_of_its_own_answer_on_its_own_model(
    quoting_llm, tmp_path, events_in
):
    result = claimed(quoting_llm, tmp_path, (ANSWER, ["q2"]))
    quoting_llm.answers(
        reasoning=citing(MERROW, slice(None)), claims=claims_reply((MERROW, ["q4"]))
    )
    asked = ask(result, "Who else?", model="openai/gpt-4o", completion=quoting_llm)
    [_, request] = quoting_llm.requests_for("claims")
    assert request["model"] == "openai/gpt-4o"
    user = request["messages"][1]["content"]
    assert user.startswith(f"<question>\nWho else?\n</question>\n<answer>\n{MERROW}\n")
    [claim] = asked.provenance["claims"]
    assert (claim["text"], claim["documents"]) == (MERROW, [result.document_ids[1]])
    assert trace(asked)["claims"] == asked.provenance["claims"]
    [finished] = [e for e in events_in(asked.run_dir) if e["type"] == "claims_finished"]
    assert finished["claims"] == asked.provenance["claims"]
    assert result.provenance["claims"] == trace(result)["claims"] != []


def test_a_run_that_failed_resumes_through_to_its_claims(quoting_llm, tmp_path):
    quoting_llm.answers(parsing=bad_key())
    with pytest.raises(Exception, match="bad key"):
        claimed(quoting_llm, tmp_path, (HALLORAN, ["q2"]))
    [run_dir] = (tmp_path / "logs").iterdir()
    quoting_llm.answers(parsing=ANSWERING["parsing"])
    resumed = resume(run_dir, TWO_LINE_MEMOS, completion=quoting_llm)
    [claim] = resumed.provenance["claims"]
    assert claim["text"] == HALLORAN
    assert trace(run_dir)["claims"] == resumed.provenance["claims"]


# Cancelled as reasoning's call answers, claims never start; as the claims call
# answers, with a reply to send again, no other is sent.
@pytest.mark.parametrize(
    ("stage", "reply", "claims_sent"),
    [("reasoning", citing(ANSWER, slice(None)), 0), ("claims", "not JSON", 1)],
)
def test_a_run_cancelled_once_reasoning_answers_fails_in_claims_and_resumes_into_them(
    quoting_llm, tmp_path, events_in, stage, reply, claims_sent
):
    cancel = threading.Event()

    def cancelling(request):
        cancel.set()
        return reply(request) if callable(reply) else reply

    quoting_llm.answers(**{stage: cancelling})
    logs = tmp_path / "logs"
    with pytest.raises(Cancelled):
        run(
            QUESTION,
            TWO_LINE_MEMOS,
            completion=quoting_llm,
            logs_dir=logs,
            cancel=cancel,
        )
    [run_dir] = logs.iterdir()
    assert len(quoting_llm.requests_for("claims")) == claims_sent
    failed = events_in(run_dir)[-1]
    assert (failed["type"], failed["stage"]) == ("run_failed", "claims")
    assert (run_dir / "claims" / "error.txt").is_file()
    quoting_llm.answers(
        reasoning=citing(ANSWER, slice(None)), claims=claims_reply((HALLORAN, ["q2"]))
    )
    sent = len(quoting_llm.requests)
    events = []
    resumed = resume(
        run_dir, TWO_LINE_MEMOS, completion=quoting_llm, on_event=events.append
    )
    assert [stage_of(request) for request in quoting_llm.requests[sent:]] == ["claims"]
    assert resumed.answer == ANSWER
    assert [claim["text"] for claim in resumed.provenance["claims"]] == [HALLORAN]
    [reused] = [e["reused"] for e in events if e["type"] == "run_resumed"]
    assert reused["reasoning"] == 1
    assert "reasoning_finished" in [e["type"] for e in events]
    calls = json.loads((run_dir / "claims" / "calls.json").read_text())
    assert len(calls) == claims_sent + 1
    assert not (run_dir / "claims" / "error.txt").exists()
    with pytest.raises(ValueError, match="this run finished"):
        resume(run_dir, TWO_LINE_MEMOS, completion=quoting_llm)


def as_before_claims(manifest: dict) -> None:
    """A run's manifest as r3con wrote one before it made claims."""
    del manifest["config"]["claims"]
    for stage in CLAIMS_STAGES:
        del manifest["config"]["prompts"][stage]
        del manifest["sources"]["prompts"][stage]


def test_a_run_made_before_claims_is_asked_without_them_and_stays_finished(
    quoting_llm, tmp_path
):
    result = claimed(quoting_llm, tmp_path, (HALLORAN, ["q2"]))
    shutil.rmtree(result.run_dir / "claims")
    in_json("manifest.json", as_before_claims)(result.run_dir)
    sent = len(quoting_llm.requests)
    asked = ask(result, "Who else?", completion=quoting_llm)
    assert [stage_of(r) for r in quoting_llm.requests[sent:]] == ["reasoning"]
    assert asked.provenance["claims"] == []
    assert not (asked.run_dir / "claims").exists()
    with pytest.raises(ValueError, match="this run finished"):
        resume(result.run_dir, TWO_LINE_MEMOS, completion=quoting_llm)
