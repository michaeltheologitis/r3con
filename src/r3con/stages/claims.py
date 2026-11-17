"""After reasoning: the answer's claims, each with the evidence that supports it.

One more call, to the model reasoning called with its params, is sent the question, the
answer, and the evidence on the answer's trail, each piece under an id, and replies
with the answer's claims, each one of its sentences, list items or table rows, word
for word, with the ids of the evidence that supports it:

- at the ``"sentence"`` evidence level, the trail's quotes (``provenance["quotes"]``),
  in order, as ``q1``, ``q2``, …, each in the document's own words, or as the model
  copied it when those are not kept;
- at the ``"document"`` level, the round-1 relevance summary of each document on the
  trail (``provenance["documents"]``), in order, under its id: the one written from
  that document alone. When it is empty, the document's latest summary that is not
  empty stands in for it, and a document with none is left out. A document read in
  parts in that round shows each part's, under ``<id>.<k>``, and a claim citing one
  cites the document.

The reply's ids are constrained to those shown, and any other is dropped with a
warning (:func:`~r3con.stages.citing.cited`). Each claim is then found in the answer
(:func:`_place`) and keeps the answer's own words there; one not found is dropped with
a warning. The claims are given in the order of the answer.

Claims never fail a run. With no evidence to show, no call is sent; when the call fails
after its resends, or no reply fits its schema, the answer has no claims and a warning
says why. Only :class:`~r3con.runtime.llm.Cancelled` is raised, as for any stage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import itemgetter
from typing import Any, NamedTuple, cast

import jinja2
from pydantic import BaseModel, create_model

from r3con.config import CLAIMS_STAGE, EvidenceLevel, PromptStage
from r3con.logging_setup import get_logger
from r3con.masking import one_line
from r3con.prompts import load_prompt
from r3con.provenance import Claim, Provenance, QuotePlace
from r3con.quotes import locate_quotes
from r3con.runs import StageRun
from r3con.runtime.llm import Cancelled, send_until_valid
from r3con.stages.citing import cited, ids_field, lenient_list
from r3con.stages.relevance import RelevantContext, SnippetRecord

_log = get_logger("claims")

_KIND = "claims"
_REWRITE = "Extract the same answer's claims again"
_USER_MESSAGE = jinja2.Template(
    "<question>\n{{ question }}\n</question>\n"
    "<answer>\n{{ answer }}\n</answer>\n"
    "<{{ tag }}>\n"
    "{% for id, text in evidence.items() %}### {{ heading }}: {{ id }}\n{{ text }}\n"
    "{% endfor %}"
    "</{{ tag }}>"
)


class _Level(NamedTuple):
    """How a level's evidence is shown: the tag around it, the heading over each
    piece, and the reply's field for the ids that support a claim."""

    tag: str
    heading: str
    ids: str


_LEVELS: dict[EvidenceLevel, _Level] = {
    "sentence": _Level("quotes", "Quote ID", "quote_ids"),
    "document": _Level("summaries", "Document ID", "document_ids"),
}


class _Evidence(NamedTuple):
    """What the call is shown, by the id each piece is shown under, in the order
    shown: its text, its document's id, and, for a quote, its place."""

    texts: dict[str, str]
    documents: dict[str, str]
    quotes: dict[str, QuotePlace]


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
    run: StageRun | None = None,
    **llm_kwargs: Any,
) -> list[Claim]:
    """The claims ``answer`` to ``question`` makes, each with the evidence on its trail
    that supports it, as the module says, in the order of the answer.

    ``provenance`` is the answer's, ``relevance`` the run's relevant context, whose
    documents ``document_ids`` names in order, and ``prompts`` pins the claims prompt
    of ``evidence_level``. A reply that does not fit its schema is sent again with the
    validator's error, up to ``max_attempts`` sends.

    Raises:
        Cancelled: once ``cancel``, among ``llm_kwargs``, is set.
    """
    if evidence_level == "sentence":
        evidence = _quotes(provenance)
    else:
        evidence = _summaries(provenance, relevance.records, document_ids)
    if not evidence.texts:
        _log.info("claims: the answer's trail holds no evidence to show, so no call")
        return []
    level = _LEVELS[evidence_level]
    stage = CLAIMS_STAGE[evidence_level]
    user_message = _USER_MESSAGE.render(
        question=question,
        answer=answer,
        tag=level.tag,
        heading=level.heading,
        evidence=evidence.texts,
    )
    fields: dict[str, Any] = {
        "text": (str, ...),
        level.ids: ids_field(list(evidence.texts)),
    }
    claims = lenient_list(create_model("Claim", **fields))
    schema = create_model("Claims", claims=claims)
    try:
        reply = send_until_valid(
            load_prompt(stage, version=prompts[stage]),
            user_message,
            schema=schema,
            rewrite=_REWRITE,
            kind=_KIND,
            max_attempts=max_attempts,
            model=model,
            run=run,
            **llm_kwargs,
        )
    except Cancelled:
        raise
    except Exception as error:  # noqa: BLE001 — claims add to an answer, never fail it
        _log.warning(
            "claims: the answer has none, as their call failed: %s", one_line(error)
        )
        return []
    found = cast(BaseModel, reply).model_dump()["claims"]
    return _located(found, answer, evidence, level.ids)


def _quotes(provenance: Provenance) -> _Evidence:
    """The trail's quotes, as ``q1``, ``q2``, … in order."""
    ids = [f"q{n}" for n in range(1, len(provenance["quotes"]) + 1)]
    pairs = list(zip(ids, provenance["quotes"], strict=True))
    return _Evidence(
        texts={id_: quote["document_text"] or quote["text"] for id_, quote in pairs},
        documents={id_: quote["document"] for id_, quote in pairs},
        quotes={
            id_: QuotePlace(document=q["document"], start=q["start"], end=q["end"])
            for id_, q in pairs
        },
    )


def _summaries(
    provenance: Provenance,
    records: Sequence[Sequence[Sequence[SnippetRecord]]],
    document_ids: Sequence[str],
) -> _Evidence:
    """Each trail document's summaries, as :func:`_shown_summaries` picks them, under
    their ids, in order; ``records`` are every round's, each document's in the order of
    ``document_ids``."""
    index = {id_: doc for doc, id_ in enumerate(document_ids)}
    texts: dict[str, str] = {}
    documents: dict[str, str] = {}
    for document in provenance["documents"]:
        shown = _shown_summaries(records, index[document["document"]])
        texts |= shown
        documents |= dict.fromkeys(shown, document["document"])
    return _Evidence(texts=texts, documents=documents, quotes={})


def _shown_summaries(
    records: Sequence[Sequence[Sequence[SnippetRecord]]], doc: int
) -> dict[str, str]:
    """Document ``doc``'s round-1 summaries that are not empty, one per part, by id;
    when there are none, those of the latest round that has one; else none."""
    for round_ in [*records[:1], *reversed(records[1:])]:
        notes = {r.id: r.summary.strip() for r in round_[doc] if r.summary.strip()}
        if notes:
            return notes
    return {}


def _located(
    claims: Sequence[Mapping[str, Any]],
    answer: str,
    evidence: _Evidence,
    ids: str,
) -> list[Claim]:
    """Each of the reply's ``claims`` found in ``answer`` (:func:`_place`), with the
    ``evidence`` its ``ids`` field cites, sorted by where it starts. One not found is
    dropped with a warning."""
    order = {id_: n for n, id_ in enumerate(evidence.texts)}
    located: list[Claim] = []
    after = 0
    for claim in claims:
        place = _place(claim["text"], answer, after)
        if place is None:
            _log.warning(
                "claims: dropped a claim not found in the answer: %r", claim["text"]
            )
            continue
        start, end = place
        after = end
        supporting = sorted(cited(claim[ids], order, _KIND), key=order.__getitem__)
        located.append(
            Claim(
                text=answer[start:end],
                start=start,
                end=end,
                documents=list(
                    dict.fromkeys(evidence.documents[id_] for id_ in supporting)
                ),
                quotes=[
                    evidence.quotes[id_] for id_ in supporting if id_ in evidence.quotes
                ],
            )
        )
    return sorted(located, key=itemgetter("start"))


def _place(text: str, answer: str, after: int) -> tuple[int, int] | None:
    """Where ``text`` is in ``answer``: verbatim from ``after`` on, the end of the
    claim before it, so that a sentence the answer repeats is found where the claim
    is; else verbatim anywhere; else where :func:`r3con.quotes.locate_quotes` finds
    it; ``None`` when it is not found, or blank."""
    text = text.strip()
    if not text:
        return None
    start = answer.find(text, after)
    if start < 0:
        start = answer.find(text)
    if start >= 0:
        return start, start + len(text)
    found = locate_quotes([text], answer)[0]
    if found["start"] is None or found["end"] is None:
        return None
    return found["start"], found["end"]
