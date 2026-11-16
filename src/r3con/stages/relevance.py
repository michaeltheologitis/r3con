"""Stage 1: surfacing relevance — the per-document, task-conditioned relevance snippets.

A relevance snippet is **not** a generic summary. Each one is written *for a specific
task* and is conditioned on it — the note one document contributes toward the task,
read in light of the rest of the corpus. The point is cross-document reasoning: to use
one document for the task you often need to know what the others say. The per-document
snippets taken together are the **relevant context**.

A document too long for the model's window is read in parts (:mod:`r3con.splitting`),
and its snippet is then one note per part. Cross-corpus awareness is built over
``rounds`` **synchronous** rounds:

- **Round 1** — each document's relevance snippet is written from the task and the
  document alone (no other document is visible), with the ``relevance/first-round``
  prompt. The reply is structured: the ``summary``, or null, and its ``quotes``, the
  sentences of the document it is based on.
- **Round k ≥ 2** — each document's snippet is rewritten given the task and the
  **previous round's** snippets of the OTHER documents (never the document's own prior
  snippet — the document itself is in the user message), with the
  ``relevance/later-rounds`` prompt. Only the immediately-previous round is fed in, not
  the whole history. The reply adds ``provenance_ids``, the ids of the shown snippets
  it used. With nothing to show, the first-round prompt is used instead.

After each round from 2 on, a document whose snippet in it is empty, every part's, is
**pruned** (:attr:`RelevantContext.pruned`): no later round reads it, its snippet there
empty, and parsing does not parse it (:mod:`r3con.stages.structuring.parsing`), the
final round's included. An empty round-1 snippet alone prunes nothing, since round 2
may bring the document back once the others are known; with fewer than two rounds
nothing is pruned. The other documents' snippets in the round it went empty in may
still cite its snippet of the round before, so the answer's trail can reach it through
them.

Relevance reads at one of two evidence levels (``evidence_level``). At
``"sentence"``, the default, each reply quotes the sentences of the document its summary
is based on, as above. At ``"document"``, neither the prompts nor the reply schema
mention quotes, so every snippet keeps none, and the answer's trail ends at documents.

Each quote is looked for in the text its call read (:func:`r3con.quotes.locate_quotes`)
and kept with its position in the whole document and the document's words there, or
with neither when it is not found; a null or empty summary keeps no quotes. Later stages
read the summaries alone; the quotes are kept for inspection and for the quotes behind
an answer (:mod:`r3con.provenance`).

Every document has an id for the run (:func:`draw_document_ids`), and each snippet
carries its document's id, or ``<id>.<k>`` for part ``k`` of a document read in parts in
that round. The model sees ids only on the other documents' snippets, never the id of
the document it is reading.

Round k reads the *frozen* round-(k-1) set, so within a round the per-document calls
are independent and fan out in parallel (bounded by
:func:`r3con.settings.active_doc_workers`); a provider that signs in on its first
call gets one call at a time until one has signed in (:mod:`r3con.runtime.llm`).
Downstream stages consume the relevant context, the snippets of
:attr:`RelevantContext.context_round` (``RelevantContext.snippets``); earlier rounds are
kept for inspection and for the answer's trail back through them
(:mod:`r3con.provenance`).
"""

from __future__ import annotations

import random
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel

from r3con import settings
from r3con.config import EvidenceLevel, PromptStage
from r3con.logging_setup import get_logger
from r3con.parallel import parallel_map
from r3con.prompts import load_prompt
from r3con.quotes import Quote, locate_quotes
from r3con.runs import Emit, StageRun, no_events
from r3con.settings import active_doc_workers
from r3con.splitting import Splits
from r3con.stages.citing import (
    CitingRequest,
    citing_schema,
    lenient_list,
    read_citing,
)

_log = get_logger("relevance")

_ID_HEX_DIGITS = 6
_REWRITE = "Write the summary of the same document again"

Snippet = str | list[str]
"""A document's relevance snippet, or, for a document read in parts, one per part."""


@dataclass(frozen=True)
class SnippetRecord:
    """One relevance snippet with its id: a whole document's, or one part's.

    ``summary`` is the snippet's text, ``quotes`` the sentences of the document it is
    based on, each where it was found in the whole document (none at the document
    evidence level), and ``provenance`` the ids of the previous round's snippets it
    used: empty in round 1, and whenever it used none.
    """

    id: str
    summary: str
    quotes: list[Quote]
    provenance: list[str]


@dataclass
class RelevantContext:
    """Output of :func:`surface_relevance` — the relevant context.

    ``records[k][d]`` is document ``d``'s snippet in round ``k+1``, one record per part
    (one for a document read whole). Earlier rounds are kept for inspection and for
    the answer's trail (:mod:`r3con.provenance`); the round of :attr:`context_round` is
    what downstream stages consume, as :attr:`snippets`. ``pruned_after`` maps each
    document the run pruned, by index, to the round after which it was pruned (the
    module's rule), a copy of the mapping given: empty with fewer than two rounds.
    """

    records: list[list[list[SnippetRecord]]]
    pruned_after: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.pruned_after = dict(self.pruned_after)

    @property
    def pruned(self) -> frozenset[int]:
        """The documents the run pruned, by index."""
        return frozenset(self.pruned_after)

    @property
    def all_pruned(self) -> bool:
        """Whether the run pruned every document (:attr:`context_round`)."""
        n_docs = len(self.records[0]) if self.records else 0
        return 0 < n_docs == len(self.pruned_after)

    @property
    def context_round(self) -> int:
        """The round, from 1, whose snippets are the relevant context, which every
        later stage reads: the final one, or round 1 when the run pruned every
        document. Nothing relevant is then left in the end, so there is nothing to
        structure: the schema is one of no field, proposed without a call, nothing is
        parsed, and reasoning answers from what the first reads found rather than from
        nothing. 0 when no round ran."""
        return 1 if self.all_pruned else len(self.records)

    @property
    def snippets(self) -> list[Snippet]:
        """The relevant context: each document's snippet in :attr:`context_round`,
        aligned 1:1 with the input ``documents``: its text, or, for a document read in
        parts, one per part. Empty when no round ran."""
        return [
            parts[0].summary if len(parts) == 1 else [part.summary for part in parts]
            for parts in self._context()
        ]

    @property
    def provenance(self) -> list[list[str]]:
        """The ids each document's snippet in :attr:`context_round` used, aligned with
        :attr:`snippets`, a document read in parts having its parts' ids combined
        (:func:`combine_parts`)."""
        return [
            combine_parts(part.provenance for part in parts)
            for parts in self._context()
        ]

    def _context(self) -> list[list[SnippetRecord]]:
        return self.records[self.context_round - 1] if self.records else []


def _pruned(round_: Sequence[Sequence[SnippetRecord]]) -> frozenset[int]:
    """The documents ``round_``'s snippets prune, by index: those whose snippet there is
    empty, every part's."""
    return frozenset(
        d for d, parts in enumerate(round_) if not any(r.summary for r in parts)
    )


def draw_document_ids(n: int) -> list[str]:
    """``n`` distinct random ids of 6 hex characters, one per document of a run, so
    identical documents still get distinct ids."""
    space = 16**_ID_HEX_DIGITS
    return [f"{x:0{_ID_HEX_DIGITS}x}" for x in random.sample(range(space), n)]


def relevance_call(round_: int) -> str:
    """The name of round ``round_``'s calls, as ``calls.json`` and ``splits.json`` give
    it."""
    return f"relevance-r{round_}"


def part_ids(document_id: str, n_parts: int) -> list[str]:
    """The ids of a document's snippets in one round: its id when read whole, else
    ``<id>.<k>`` for part ``k``, from 1."""
    if n_parts == 1:
        return [document_id]
    return [f"{document_id}.{k}" for k in range(1, n_parts + 1)]


def join_parts(snippet: Snippet) -> str:
    """A document's snippet as one note: a split document's part notes, stripped, in
    order, joined by a blank line, with the empty ones left out."""
    if isinstance(snippet, str):
        return snippet
    return "\n\n".join(note.strip() for note in snippet if note.strip())


def render_relevance(
    relevance_snippets: Sequence[Snippet] | None,
    document_ids: Sequence[str] | None = None,
) -> str:
    """Render the relevance snippets into the labeled block a prompt embeds as the
    relevant context.

    Each document gets a ``### Document N`` heading, and each part of a document read in
    parts a ``### Document N.k`` heading, ``k`` from 1. With ``document_ids``, one per
    snippet, the headings name each document, and each part, by its id instead
    (:func:`part_ids`): ``### Document ID: <id>``. A note with nothing relevant renders
    as ``(no relevant summary for this task)``. An empty / ``None`` list renders to
    ``""`` so the consuming prompt omits the block.

    Raises:
        ValueError: if ``document_ids`` is not one id per snippet.
    """
    numbers = [str(n) for n in range(1, len(relevance_snippets or ()) + 1)]
    names, label = (
        (numbers, "Document")
        if document_ids is None
        else (document_ids, "Document ID:")
    )
    nothing = "(no relevant summary for this task)"
    return "\n\n".join(
        f"### {label} {name}\n{note.strip() or nothing}"
        for name, note in _headed_notes(relevance_snippets, names)
    )


def citable_ids(
    relevance_snippets: Sequence[Snippet] | None, document_ids: Sequence[str]
) -> list[str]:
    """The ids of the notes :func:`render_relevance` shows under ``document_ids`` that
    a reply may cite, in order: every note's but an empty one's, as a later relevance
    round shows only non-empty snippets.

    Raises:
        ValueError: if ``document_ids`` is not one id per snippet.
    """
    return [
        id_
        for id_, note in _headed_notes(relevance_snippets, document_ids)
        if note.strip()
    ]


def _headed_notes(
    relevance_snippets: Sequence[Snippet] | None, names: Sequence[str]
) -> Iterator[tuple[str, str]]:
    """Each note of ``relevance_snippets``, in order, with the name heading it: its
    document's from ``names``, or ``<name>.<k>`` for part ``k`` of a document read in
    parts (:func:`part_ids`); nothing for an empty / ``None`` list.

    Raises:
        ValueError: if ``names`` is not one name per snippet.
    """
    if not relevance_snippets:
        return
    for name, snippet in zip(names, relevance_snippets, strict=True):
        notes = snippet if isinstance(snippet, list) else [snippet]
        for part, note in zip(part_ids(name, len(notes)), notes, strict=True):
            yield part, note or ""


def _render_other_snippets(shown: Mapping[str, str]) -> str:
    """The shown snippets, each under its ``### Summary ID: <id>`` heading, separated
    by blank lines."""
    return "\n\n".join(f"### Summary ID: {id_}\n{s}" for id_, s in shown.items())


def _reply_schema(
    citable: Sequence[str] | None, evidence_level: EvidenceLevel
) -> type[BaseModel]:
    """A reply's schema: its ``summary``, or null, then, at the sentence level, its
    ``quotes`` (:func:`lenient_list` of strings), then, unless ``citable`` is ``None``,
    its ``provenance_ids`` of those ids."""
    quotes = {"quotes": lenient_list(str)} if evidence_level == "sentence" else {}
    return citing_schema(
        "RelevanceSummary", citable, summary=(str | None, ...), **quotes
    )


def _request(
    task: str,
    shown: dict[str, str],
    prompts: Mapping[PromptStage, str],
    evidence_level: EvidenceLevel,
) -> CitingRequest:
    """A document's request in one round, given the OTHER documents' non-empty snippets
    it shows, by id: the first-round prompt and its ``{summary, quotes}`` reply when
    there are none, else the later-rounds prompt and its ``{summary, quotes,
    provenance_ids}`` reply. At the document ``evidence_level``, neither prompt nor
    reply has quotes."""
    quotes = evidence_level == "sentence"
    if not shown:
        prompt = load_prompt(
            "relevance/first-round",
            version=prompts["relevance/first-round"],
            task=task,
            quotes=quotes,
        )
        return CitingRequest(prompt, _reply_schema(None, evidence_level), [])
    prompt = load_prompt(
        "relevance/later-rounds",
        version=prompts["relevance/later-rounds"],
        task=task,
        other_snippets=_render_other_snippets(shown),
        quotes=quotes,
    )
    ids = list(shown)
    return CitingRequest(prompt, _reply_schema(ids, evidence_level), ids)


def combine_parts(ids_per_part: Iterable[Sequence[str]]) -> list[str]:
    """A document's ids from those of its parts: a document read in parts has its
    parts' ids combined, in order, without repeats."""
    return list(dict.fromkeys(id_ for ids in ids_per_part for id_ in ids))


def _read(
    request: CitingRequest, part: str, *, kind: str, max_attempts: int, **llm_call: Any
) -> tuple[str, list[str], list[str]]:
    """``part`` (a document or one part of one) read under ``request`` by
    :func:`~r3con.stages.citing.read_citing`: its snippet, stripped, its quotes,
    none for an empty snippet or a reply without quotes, and the shown ids it used."""
    reply, provenance = read_citing(
        request,
        part,
        rewrite=_REWRITE,
        kind=kind,
        max_attempts=max_attempts,
        **llm_call,
    )
    summary = (reply.summary or "").strip()
    return summary, getattr(reply, "quotes", []) if summary else [], provenance


def _report_snippets(
    emit: Emit, round_: int, document_id: str, records: list[SnippetRecord]
) -> None:
    """Emit ``relevance_finished`` for each of a document's final ``records`` in round
    ``round_``, its ``part`` the ``k`` of the record's id ``<id>.<k>``, or ``None`` for
    a document read whole."""
    for record in records:
        k = record.id.partition(".")[2]
        emit(
            "relevance_finished",
            round=round_,
            doc_id=document_id,
            part=int(k) if k else None,
            summary=record.summary,
            quotes=record.quotes,
            provenance=record.provenance,
        )


def _attempts(max_attempts: int | None) -> int:
    """``max_attempts``, or ``settings.RELEVANCE_MAX_ATTEMPTS`` for ``None``.

    Raises:
        ValueError: if it is not positive.
    """
    if max_attempts is None:
        max_attempts = settings.RELEVANCE_MAX_ATTEMPTS
    if max_attempts < 1:
        raise ValueError(
            f"max_attempts must be >= 1, got {max_attempts}. "
            f"Override via settings.RELEVANCE_MAX_ATTEMPTS (current default: "
            f"{settings.RELEVANCE_MAX_ATTEMPTS})."
        )
    return max_attempts


def relevance_snippet(
    *,
    task: str,
    document: str,
    other_snippets: Mapping[str, str],
    model: str,
    prompts: Mapping[PromptStage, str],
    evidence_level: EvidenceLevel = "sentence",
    run: StageRun | None = None,
    kind: str = "relevance",
    max_attempts: int | None = None,
    **llm_kwargs: Any,
) -> tuple[str, list[Quote], list[str]]:
    """Surface one ``document``'s relevance snippet for the task, given the OTHER
    documents' previous-round snippets by id (empty in round 1).

    The task + the other documents' snippets live in the system prompt; the document is
    the user message. ``prompts`` pins the version of ``relevance/first-round`` and
    ``relevance/later-rounds`` (a run config's ``prompts``). With no non-empty snippet
    to show, the call uses the first-round prompt and a structured ``{summary, quotes}``
    reply; otherwise the later-rounds prompt and a ``{summary, quotes,
    provenance_ids}`` reply. At the ``"document"`` ``evidence_level``, neither prompt
    nor reply has quotes, and none are returned. A reply that does not fit its schema
    is sent again with the validator's error, up to ``max_attempts`` sends (``None``
    reads ``settings.RELEVANCE_MAX_ATTEMPTS``).

    Returns the snippet, stripped — an empty string is allowed (the document contributes
    nothing relevant to the task) — its quotes, each located in ``document``
    (:func:`r3con.quotes.locate_quotes`), none for an empty snippet, and the shown ids
    it used.

    Raises:
        ValueError: if ``max_attempts`` is not positive.
    """
    attempts = _attempts(max_attempts)
    shown = {id_: s.strip() for id_, s in other_snippets.items() if s.strip()}
    summary, quotes, provenance = _read(
        _request(task, shown, prompts, evidence_level),
        document,
        kind=kind,
        max_attempts=attempts,
        model=model,
        run=run,
        **llm_kwargs,
    )
    return summary, locate_quotes(quotes, document), provenance


def surface_relevance(
    *,
    task: str,
    documents: list[str],
    document_ids: list[str],
    model: str,
    prompts: Mapping[PromptStage, str],
    evidence_level: EvidenceLevel = "sentence",
    rounds: int = 2,
    run: StageRun | None = None,
    workers: int | None = None,
    splits: Splits | None = None,
    max_attempts: int | None = None,
    emit: Emit = no_events,
    saved: Mapping[tuple[int, int], list[SnippetRecord]] = MappingProxyType({}),
    recorded: Mapping[int, int] = MappingProxyType({}),
    recorded_through: int = 1,
    **llm_kwargs: Any,
) -> RelevantContext:
    """Surface the relevant context over ``rounds`` synchronous rounds.

    Round 1 writes each document's snippet from the task + document alone; each later
    round rewrites every document but the pruned ones given the *previous round's*
    snippets of the OTHER documents (never its own). Returns a
    :class:`RelevantContext` holding every round's snippet records. ``rounds=1`` =
    independent round-1 only; ``rounds < 1`` = no relevance snippets at all (no records
    — a deliberate "no-relevance" run). An empty ``documents`` → empty results.

    ``document_ids`` names each document, one distinct id each
    (:func:`draw_document_ids`). ``prompts``, ``evidence_level`` and
    ``max_attempts`` are as in :func:`relevance_snippet`.

    Each document is read through ``splits`` (built over ``documents``; ``None`` builds
    one), so a document too long for the model's window is read in parts, one snippet
    per part, and stays split in later rounds. A part never sees its own document's
    other parts: the others' block holds the other documents only, a split one's
    snippets listed separately under their part ids.

    A pruned document (:attr:`RelevantContext.pruned`) is not read in a later round: its
    snippet there is empty, under the ids it had in the round before, and reports
    nothing.

    ``emit`` reports ``relevance_started`` before each document is read in a round, and
    ``relevance_finished`` for each of its final parts once it is read; once a round
    that prunes is in, ``documents_pruned`` (:func:`r3con.pipeline.run_pipeline` lists
    their fields). It is called from worker threads.

    ``saved`` holds snippets an earlier attempt at the run read, by round and document
    index (:func:`r3con.pipeline.resume_pipeline`): such a document is not read again in
    that round, and reports nothing. ``recorded`` maps each document that attempt
    pruned after a round up to ``recorded_through`` to that round, as it recorded it:
    up to that round, those are the documents pruned, whatever their snippets say, and
    no ``documents_pruned`` is reported again.

    Raises:
        ValueError: if ``document_ids`` is not one distinct id per document, or
            ``max_attempts`` is not positive.
    """
    if len(document_ids) != len(documents) or len(set(document_ids)) != len(documents):
        raise ValueError(
            "document_ids must be one distinct id per document: got "
            f"{len(set(document_ids))} distinct of {len(document_ids)} for "
            f"{len(documents)} document(s)."
        )
    attempts = _attempts(max_attempts)
    # rounds < 1 = "no relevance snippets at all" (a deliberate rounds=0 re-run); an empty
    # document corpus is likewise empty. Both short-circuit BEFORE round 1 runs (the
    # unconditional first run_round below) — a `return`, not a fall-through.
    if rounds < 1 or not documents:
        return RelevantContext(records=[])

    max_workers = workers if workers is not None else active_doc_workers()
    reader = splits if splits is not None else Splits(documents, model=model)

    def run_round(
        prev: list[list[SnippetRecord]], round_idx: int, skipped: Collection[int]
    ) -> list[list[SnippetRecord]]:
        """Rewrite every document's relevance snippet in parallel, but the ``skipped``
        ones'. ``prev`` is the frozen previous-round set (no snippets in round 1)."""
        _log.info(
            "round %d/%d · %d doc(s) (≤%d parallel)",
            round_idx,
            rounds,
            len(documents) - len(skipped),
            max_workers,
        )
        shown = {r.id: r.summary for parts in prev for r in parts if r.summary}

        def one(i: int, _document: str) -> list[SnippetRecord]:
            if i in skipped:
                return [SnippetRecord(r.id, "", [], []) for r in prev[i]]
            if (records := saved.get((round_idx, i))) is not None:
                return records
            # Others-only: document i sees the previous round's snippets of the OTHER
            # documents, never its own (the document itself is the user message).
            own = {record.id for record in prev[i]}
            others = {id_: s for id_, s in shown.items() if id_ not in own}
            request = _request(task, others, prompts, evidence_level)
            emit("relevance_started", round=round_idx, doc_id=document_ids[i])
            replies = reader.read_in_parts(
                i,
                call=relevance_call(round_idx),
                rest=request.system_prompt,
                schema=request.schema,
                send=lambda part, kind: _read(
                    request,
                    part,
                    kind=kind,
                    max_attempts=attempts,
                    model=model,
                    run=run,
                    **llm_kwargs,
                ),
            )
            ids = part_ids(document_ids[i], len(replies))
            records = [
                SnippetRecord(
                    id=id_,
                    summary=summary,
                    quotes=locate_quotes(quotes, part, start),
                    provenance=provenance,
                )
                for id_, part, start, (summary, quotes, provenance) in zip(
                    ids, reader.parts(i), reader.starts(i), replies, strict=True
                )
            ]
            _report_snippets(emit, round_idx, document_ids[i], records)
            return records

        result = parallel_map(one, documents, max_workers=max_workers)
        n_nonempty = sum(1 for parts in result if any(r.summary for r in parts))
        _log.info(
            "round %d/%d done · %d/%d doc(s) had a relevant snippet",
            round_idx,
            rounds,
            n_nonempty,
            len(documents),
        )
        return result

    pruned: dict[int, int] = {}
    records = [run_round([[] for _ in documents], 1, pruned.keys())]
    for r in range(2, rounds + 1):
        # only the previous round feeds in
        records.append(run_round(records[-1], r, pruned.keys()))
        if r <= recorded_through:
            pruned |= {d: k for d, k in recorded.items() if k == r}
            continue
        newly = sorted(_pruned(records[-1]) - pruned.keys())
        if newly:
            emit("documents_pruned", round=r, doc_ids=[document_ids[i] for i in newly])
        pruned |= dict.fromkeys(newly, r)
    return RelevantContext(records=records, pruned_after=pruned)
