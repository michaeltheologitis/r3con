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
  prompt, as plain text.
- **Round k ≥ 2** — each document's snippet is rewritten given the task and the
  **previous round's** snippets of the OTHER documents (never the document's own prior
  snippet — the document itself is in the user message), with the
  ``relevance/later-rounds`` prompt. Only the immediately-previous round is fed in, not
  the whole history. The reply is structured: the ``summary``, or null, and its
  ``provenance_ids``, the ids of the shown snippets it used. With nothing to show, the
  first-round prompt is used instead.

Every document has an id for the run (:func:`draw_document_ids`), and each snippet
carries its document's id, or ``<id>.<k>`` for part ``k`` of a document read in parts in
that round. The model sees ids only on the other documents' snippets, never the id of
the document it is reading.

Round k reads the *frozen* round-(k-1) set, so within a round the per-document calls
are independent and fan out in parallel (bounded by
:func:`r3con.settings.active_doc_workers`). Downstream stages consume the
**final-round** snippets (``RelevantContext.snippets``); earlier rounds are kept only
for inspection.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, NamedTuple, TypeVar, cast

from pydantic import BaseModel, Field, create_model

from r3con import settings
from r3con.config import PromptStage
from r3con.logging_setup import get_logger
from r3con.parallel import parallel_map
from r3con.prompts import load_prompt
from r3con.runs import Emit, StageRun, no_events
from r3con.runtime.llm import send_until_valid
from r3con.settings import active_doc_workers
from r3con.splitting import Splits

_log = get_logger("relevance")

_ID_HEX_DIGITS = 6
_REWRITE = "Write the summary of the same document again"

_ReplyT = TypeVar("_ReplyT", bound=BaseModel)

Snippet = str | list[str]
"""A document's relevance snippet, or, for a document read in parts, one per part."""


@dataclass(frozen=True)
class SnippetRecord:
    """One relevance snippet with its id: a whole document's, or one part's.

    ``summary`` is the snippet's text, and ``provenance`` the ids of the previous
    round's snippets it used: empty in round 1, and whenever it used none.
    """

    id: str
    summary: str
    provenance: list[str]


@dataclass
class RelevantContext:
    """Output of :func:`surface_relevance` — the relevant context.

    ``records[k][d]`` is document ``d``'s snippet in round ``k+1``, one record per part
    (one for a document read whole). Earlier rounds are kept for inspection; the final
    round is what downstream stages consume, as :attr:`snippets`.
    """

    records: list[list[list[SnippetRecord]]]

    @property
    def snippets(self) -> list[Snippet]:
        """The final round's snippet of each document, aligned 1:1 with the input
        ``documents``: its text, or, for a document read in parts, one per part. Empty
        when no round ran."""
        return [
            parts[0].summary if len(parts) == 1 else [part.summary for part in parts]
            for parts in self._final_round()
        ]

    @property
    def provenance(self) -> list[list[str]]:
        """The ids each document's final-round snippet used, aligned with
        :attr:`snippets`, a document read in parts having its parts' ids combined
        (:func:`combine_parts`)."""
        return [
            combine_parts(part.provenance for part in parts)
            for parts in self._final_round()
        ]

    def _final_round(self) -> list[list[SnippetRecord]]:
        return self.records[-1] if self.records else []


class _Reply(BaseModel):
    summary: str | None
    provenance_ids: list[str]


class _Request(NamedTuple):
    """What a document's calls in one round send beside it: the system prompt, the
    reply's schema (``None`` for plain text), and the snippets shown, by id."""

    system_prompt: str
    schema: type[_Reply] | None
    shown: dict[str, str]


def draw_document_ids(n: int) -> list[str]:
    """``n`` distinct random ids of 6 hex characters, one per document of a run, so
    identical documents still get distinct ids."""
    space = 16**_ID_HEX_DIGITS
    return [f"{x:0{_ID_HEX_DIGITS}x}" for x in random.sample(range(space), n)]


def _part_ids(document_id: str, n_parts: int) -> list[str]:
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
    """Render the final-round relevance snippets into the labeled block a prompt embeds
    as the relevant context.

    Each document gets a ``### Document N`` heading, and each part of a document read in
    parts a ``### Document N.k`` heading, ``k`` from 1. With ``document_ids``, one per
    snippet, the headings name each document, and each part, by its id instead
    (:func:`_part_ids`): ``### Document ID: <id>``. A note with nothing relevant renders
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
    parts (:func:`_part_ids`); nothing for an empty / ``None`` list.

    Raises:
        ValueError: if ``names`` is not one name per snippet.
    """
    if not relevance_snippets:
        return
    for name, snippet in zip(names, relevance_snippets, strict=True):
        notes = snippet if isinstance(snippet, list) else [snippet]
        for part, note in zip(_part_ids(name, len(notes)), notes, strict=True):
            yield part, note or ""


def _render_other_snippets(shown: Mapping[str, str]) -> str:
    """The shown snippets, each under its ``### Summary ID: <id>`` heading, separated
    by blank lines."""
    return "\n\n".join(f"### Summary ID: {id_}\n{s}" for id_, s in shown.items())


def list_of_ids(shown_ids: Sequence[str]) -> Any:
    """The type of a reply's list of the ids a call shows. The shown ids reach only the
    JSON schema, as an ``enum``, or, when there are none, as any string, since an empty
    enum is refused; the reply itself is read as plain strings and checked by
    :func:`cited`."""
    if not shown_ids:
        return list[str]
    return list[Annotated[str, Field(json_schema_extra={"enum": list(shown_ids)})]]


def provenance_schema(
    name: str, base: type[_ReplyT], shown_ids: Sequence[str]
) -> type[_ReplyT]:
    """``base``, named ``name``, with its ``provenance_ids`` a :func:`list_of_ids` the
    call shows. A reply without ``provenance_ids`` cites nothing, though the strict
    schema sent requires it."""
    # A default_factory stays out of the schema; strict mode refuses a default.
    return create_model(
        name,
        __base__=base,
        provenance_ids=(list_of_ids(shown_ids), Field(default_factory=list)),
    )


def _request(
    task: str, shown: dict[str, str], prompts: Mapping[PromptStage, str]
) -> _Request:
    """A document's request in one round, given the OTHER documents' non-empty snippets
    it shows, by id: the first-round prompt and plain text when there are none, else
    the later-rounds prompt and its ``{summary, provenance_ids}`` reply."""
    if not shown:
        version = prompts["relevance/first-round"]
        prompt = load_prompt("relevance/first-round", version=version, task=task)
        return _Request(prompt, None, shown)
    prompt = load_prompt(
        "relevance/later-rounds",
        version=prompts["relevance/later-rounds"],
        task=task,
        other_snippets=_render_other_snippets(shown),
    )
    schema = provenance_schema("RelevanceSummary", _Reply, list(shown))
    return _Request(prompt, schema, shown)


def combine_parts(ids_per_part: Iterable[Sequence[str]]) -> list[str]:
    """A document's ids from those of its parts: a document read in parts has its
    parts' ids combined, in order, without repeats."""
    return list(dict.fromkeys(id_ for ids in ids_per_part for id_ in ids))


def cited(provenance: Sequence[str], shown: Collection[str], kind: str) -> list[str]:
    """The ids a reply cites, in order and without repeats, less those the call did not
    show, which are dropped with a warning."""
    unshown = [id_ for id_ in provenance if id_ not in shown]
    if unshown:
        _log.warning(
            "%s: dropped provenance id(s) the call did not show: %s", kind, unshown
        )
    return list(dict.fromkeys(id_ for id_ in provenance if id_ in shown))


def _read(
    request: _Request, part: str, *, kind: str, max_attempts: int, **llm_call: Any
) -> tuple[str, list[str]]:
    """``part`` (a document or one part of one) read under ``request`` by
    :func:`send_until_valid`: its snippet, stripped, and the shown ids it used."""
    reply = send_until_valid(
        request.system_prompt,
        part,
        schema=request.schema,
        rewrite=_REWRITE,
        kind=kind,
        max_attempts=max_attempts,
        **llm_call,
    )
    if request.schema is None:
        return cast(str, reply).strip(), []
    reply = cast(_Reply, reply)
    summary = (reply.summary or "").strip()
    return summary, cited(reply.provenance_ids, request.shown, kind)


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
    run: StageRun | None = None,
    kind: str = "relevance",
    max_attempts: int | None = None,
    **llm_kwargs: Any,
) -> tuple[str, list[str]]:
    """Surface one ``document``'s relevance snippet for the task, given the OTHER
    documents' previous-round snippets by id (empty in round 1).

    The task + the other documents' snippets live in the system prompt; the document is
    the user message. ``prompts`` pins the version of ``relevance/first-round`` and
    ``relevance/later-rounds`` (a run config's ``prompts``). With no non-empty snippet
    to show, the call uses the first-round prompt and reads back plain text; otherwise
    the later-rounds prompt and a structured ``{summary, provenance_ids}`` reply. A reply
    that does not fit its schema is sent again with the validator's error, up to
    ``max_attempts`` sends (``None`` reads ``settings.RELEVANCE_MAX_ATTEMPTS``).

    Returns the snippet, stripped — an empty string is allowed (the document contributes
    nothing relevant to the task) — and the shown ids it used.

    Raises:
        ValueError: if ``max_attempts`` is not positive.
    """
    attempts = _attempts(max_attempts)
    shown = {id_: s.strip() for id_, s in other_snippets.items() if s.strip()}
    return _read(
        _request(task, shown, prompts),
        document,
        kind=kind,
        max_attempts=attempts,
        model=model,
        run=run,
        **llm_kwargs,
    )


def surface_relevance(
    *,
    task: str,
    documents: list[str],
    document_ids: list[str],
    model: str,
    prompts: Mapping[PromptStage, str],
    rounds: int = 2,
    run: StageRun | None = None,
    workers: int | None = None,
    splits: Splits | None = None,
    max_attempts: int | None = None,
    emit: Emit = no_events,
    **llm_kwargs: Any,
) -> RelevantContext:
    """Surface the relevant context over ``rounds`` synchronous rounds.

    Round 1 writes each document's snippet from the task + document alone; each later
    round rewrites every document given the *previous round's* snippets of the OTHER
    documents (never its own). Returns a :class:`RelevantContext` holding every round's
    snippet records. ``rounds=1`` = independent round-1 only; ``rounds < 1`` = no
    relevance snippets at all (no records — a deliberate "no-relevance" run). An empty
    ``documents`` → empty results.

    ``document_ids`` names each document, one distinct id each
    (:func:`draw_document_ids`). ``prompts`` and ``max_attempts`` are as in
    :func:`relevance_snippet`.

    Each document is read through ``splits`` (built over ``documents``; ``None`` builds
    one), so a document too long for the model's window is read in parts, one snippet
    per part, and stays split in later rounds. A part never sees its own document's
    other parts: the others' block holds the other documents only, a split one's
    snippets listed separately under their part ids.

    ``emit`` reports ``relevance_started`` before each document is read in a round, and
    ``relevance_finished`` for each of its final parts once it is read
    (:func:`r3con.pipeline.run_pipeline` lists their fields). It is called from worker
    threads.

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
        prev: list[list[SnippetRecord]], round_idx: int
    ) -> list[list[SnippetRecord]]:
        """Rewrite every document's relevance snippet in parallel. ``prev`` is the frozen
        previous-round set (no snippets in round 1)."""
        _log.info(
            "round %d/%d · %d doc(s) (≤%d parallel)",
            round_idx,
            rounds,
            len(documents),
            max_workers,
        )
        shown = {r.id: r.summary for parts in prev for r in parts if r.summary}

        def one(i: int, _document: str) -> list[SnippetRecord]:
            # Others-only: document i sees the previous round's snippets of the OTHER
            # documents, never its own (the document itself is the user message).
            own = {record.id for record in prev[i]}
            others = {id_: s for id_, s in shown.items() if id_ not in own}
            request = _request(task, others, prompts)
            emit("relevance_started", round=round_idx, doc_id=document_ids[i])
            replies = reader.read_in_parts(
                i,
                call=f"relevance-r{round_idx}",
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
            records = [
                SnippetRecord(id=id_, summary=summary, provenance=provenance)
                for id_, (summary, provenance) in zip(
                    _part_ids(document_ids[i], len(replies)), replies, strict=True
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

    records = [run_round([[] for _ in documents], 1)]
    for r in range(2, rounds + 1):
        # only the previous round feeds in
        records.append(run_round(records[-1], r))
    return RelevantContext(records=records)
