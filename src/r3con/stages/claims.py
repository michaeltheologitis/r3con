"""After reasoning: link each claim of the answer to the evidence that supports it.

An answer alone does not say which of its sentences rests on what, so one more call,
to reasoning's model with its params, is shown the question, the answer, and the
evidence on the answer's trail, each piece under an id. It copies out the answer's
claims, each sentence, list item or table row, with the ids of the evidence that
supports each, and each claim is then found in the answer. The evidence is the trail's
quotes at the ``"sentence"`` evidence level, and at ``"document"`` each trail
document's round-1 relevance summary, the one written from that document alone.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from operator import itemgetter
from typing import Any, NamedTuple, cast

import jinja2
from pydantic import BaseModel, create_model

from r3con.config import EvidenceLevel, PromptStage
from r3con.logging_setup import get_logger
from r3con.masking import one_line
from r3con.prompts import load_prompt
from r3con.provenance import Claim, NamedQuote, Provenance, QuotePlace
from r3con.quotes import locate_quotes
from r3con.runs import StageRun
from r3con.runtime.llm import schema_as_sent, send_until_valid
from r3con.splitting import Splits
from r3con.stages.citing import cited, ids_field, lenient_list
from r3con.stages.relevance import RelevantContext, SnippetRecord

_log = get_logger("claims")

_KIND = "claims"
_REWRITE = "Extract the same answer's claims again"
_USER_MESSAGE = jinja2.Template(
    "<question>\n{{ question }}\n</question>\n"
    "<answer>\n{{ answer }}\n</answer>\n"
    "<{{ tag }}>\n"
    "{% for id, piece in evidence.items() %}"
    "### {{ heading }}: {{ id }}\n{{ piece.text }}\n"
    "{% endfor %}"
    "</{{ tag }}>"
)


class _Piece(NamedTuple):
    """A piece of evidence shown: its text, its document, and a quote's place."""

    text: str
    document: str
    quote: QuotePlace | None


class _Level(NamedTuple):
    """What varies by evidence level: the evidence, how it is shown, and its ids."""

    evidence: Callable[[Provenance, RelevantContext, Sequence[str]], dict[str, _Piece]]
    tag: str
    heading: str
    ids_key: str


def find_claims(
    *,
    question: str,
    answer: str,
    provenance: Provenance,
    relevance: RelevantContext,
    document_ids: Sequence[str],
    evidence_level: EvidenceLevel,
    prompts: Mapping[PromptStage, str],
    model: str,
    max_attempts: int,
    splits: Splits,
    run: StageRun | None = None,
    **llm_kwargs: Any,
) -> list[Claim]:
    """The claims ``answer`` makes, each with the evidence that supports it, in the
    answer's order. It never fails the run: with no evidence to show, a request over
    the model's line, or a call that fails or is cancelled, there are none."""
    level = _LEVELS[evidence_level]
    evidence = level.evidence(provenance, relevance, document_ids)
    if not evidence:
        _log.info("claims: the answer's trail holds no evidence, so no call")
        return []
    system_prompt = load_prompt(
        "claims", version=prompts["claims"], evidence_level=evidence_level
    )
    user_message = _USER_MESSAGE.render(
        question=question,
        answer=answer,
        tag=level.tag,
        heading=level.heading,
        evidence=evidence,
    )
    fields: dict[str, Any] = {
        "text": (str, ...),
        level.ids_key: ids_field(list(evidence)),
    }
    schema = create_model(
        "Claims", claims=lenient_list(create_model("Claim", **fields))
    )
    schema_json = json.dumps(schema_as_sent(schema))
    if estimate := splits.over_line(system_prompt, user_message, schema_json):
        _log.warning(
            "claims: none, as their request would take about %s tokens, over the "
            "%s-token line",
            f"{estimate:,}",
            f"{splits.line:,}",
        )
        return []
    try:
        reply = send_until_valid(
            system_prompt,
            user_message,
            schema=schema,
            rewrite=_REWRITE,
            kind=_KIND,
            max_attempts=max_attempts,
            model=model,
            run=run,
            **llm_kwargs,
        )
    # A cancel included: once reasoning has answered, the run finishes with it.
    except Exception as error:  # noqa: BLE001
        _log.warning("claims: none, as their call failed: %s", one_line(error))
        return []
    found = cast(BaseModel, reply).model_dump()["claims"]
    return _claims(found, answer, evidence, level.ids_key)


def _quotes(
    provenance: Provenance, _relevance: RelevantContext, _ids: Sequence[str]
) -> dict[str, _Piece]:
    """The trail's quotes, as ``q1``, ``q2``, … in order."""
    return {
        f"q{n}": _Piece(
            _quote_text(quote),
            quote["document"],
            QuotePlace(
                document=quote["document"], start=quote["start"], end=quote["end"]
            ),
        )
        for n, quote in enumerate(provenance["quotes"], start=1)
    }


def _quote_text(quote: NamedQuote) -> str:
    """The quote in the document's words, or the note's when elided past some."""
    if quote["match"] == "elided" or quote["document_text"] is None:
        return quote["text"]
    return quote["document_text"]


def _summaries(
    provenance: Provenance, relevance: RelevantContext, document_ids: Sequence[str]
) -> dict[str, _Piece]:
    """Each trail document's summaries (:func:`_shown_summaries`), in order."""
    index = {id_: doc for doc, id_ in enumerate(document_ids)}
    pieces: dict[str, _Piece] = {}
    for document in provenance["documents"]:
        id_ = document["document"]
        notes = _shown_summaries(relevance.records, index[id_])
        pieces |= {part: _Piece(note, id_, None) for part, note in notes.items()}
    return pieces


def _shown_summaries(
    records: Sequence[Sequence[Sequence[SnippetRecord]]], doc: int
) -> dict[str, str]:
    """Document ``doc``'s non-empty round-1 summaries by id, else its latest round's."""
    for round_ in [*records[:1], *reversed(records[1:])]:
        notes = {r.id: r.summary.strip() for r in round_[doc] if r.summary.strip()}
        if notes:
            return notes
    return {}


_LEVELS: dict[EvidenceLevel, _Level] = {
    "sentence": _Level(_quotes, "quotes", "Quote ID", "quote_ids"),
    "document": _Level(_summaries, "summaries", "Document ID", "document_ids"),
}


def _claims(
    found: Sequence[Mapping[str, Any]],
    answer: str,
    evidence: dict[str, _Piece],
    ids_key: str,
) -> list[Claim]:
    """The reply's claims placed in ``answer``, each with the evidence it cites."""
    order = {id_: n for n, id_ in enumerate(evidence)}
    shown = frozenset(order)
    claims: list[Claim] = []
    for start, end, claim in _placed(found, answer):
        ids = sorted(cited(claim[ids_key], shown, _KIND), key=order.__getitem__)
        pieces = [evidence[id_] for id_ in ids]
        claims.append(
            Claim(
                text=answer[start:end],
                start=start,
                end=end,
                documents=list(dict.fromkeys(piece.document for piece in pieces)),
                quotes=[piece.quote for piece in pieces if piece.quote is not None],
            )
        )
    return claims


def _placed(
    claims: Sequence[Mapping[str, Any]], answer: str
) -> list[tuple[int, int, Mapping[str, Any]]]:
    """Each claim's span in ``answer``, sorted, or the claim dropped with a warning."""
    placed: list[tuple[int, int, Mapping[str, Any]]] = []
    misses: list[tuple[str, Mapping[str, Any]]] = []
    cursor = 0
    # Verbatim from the last one's end, so a sentence said twice lands where it is.
    for claim in claims:
        text = claim["text"].strip()
        start = answer.find(text, cursor) if text else -1
        if start >= 0:
            cursor = start + len(text)
            placed.append((start, cursor, claim))
        elif text:
            misses.append((text, claim))
        else:
            _log.warning("claims: dropped a blank claim")
    # The rest in one call, which normalizes the answer once.
    located = locate_quotes([text for text, _ in misses], answer)
    for (text, claim), quote in zip(misses, located, strict=True):
        start, end = quote["start"], quote["end"]
        if start is None or end is None:
            _log.warning("claims: dropped a claim not found in the answer: %r", text)
        elif any(start < other_end and other < end for other, other_end, _ in placed):
            _log.warning("claims: dropped a claim that overlaps another: %r", text)
        else:
            placed.append((start, end, claim))
    return sorted(placed, key=itemgetter(0))
