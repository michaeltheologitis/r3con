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
- **Round k ≥ 2** — each document's state is rewritten given the task and the
  **previous round's** snippets of the OTHER documents (never the document's own prior
  state — the document itself is in the user message), with the
  ``relevance/later-rounds`` prompt. Only the immediately-previous round is fed in, not
  the whole history. The reply is structured: the ``summary``, or null, and its
  ``provenance``, the ids of the shown snippets it used. With nothing to show, the
  first-round prompt is used instead.

Every document has a random id for the run (:func:`draw_document_ids`), and each note
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

import json
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, create_model

from r3con.logging_setup import get_logger
from r3con.parallel import parallel_map
from r3con.prompts import load_prompt
from r3con.runs import StageRun
from r3con.runtime.llm import litellm_chat_completion
from r3con.settings import active_doc_workers
from r3con.splitting import Splits

_log = get_logger("relevance")

# OpenAI Structured Outputs: "up to 1000 enum values across all enum properties".
MAX_ALLOWED_IDS = 1000
_DOCUMENT_ID_BYTES = 3

Snippet = str | list[str]
"""A document's relevance snippet, or, for a document read in parts, one per part."""


@dataclass(frozen=True)
class Note:
    """One relevance snippet with its id: a whole document's, or one part's.

    ``provenance`` holds the ids of the previous round's snippets it used: empty in
    round 1, and whenever nothing was shown or nothing shown was used.
    """

    id: str
    summary: str
    provenance: list[str]


@dataclass
class RelevantContext:
    """Output of :func:`surface_relevance` — the relevant context.

    - ``snippets`` — the final-round per-document relevance snippets, aligned 1:1 with the
      input ``documents`` (``snippets[d]`` is the note document ``d`` contributes toward
      the task, or, for a document read in parts, the list of its parts' notes). This is
      what downstream stages consume.
    - ``rounds`` — every round's per-document snippets, in order: ``rounds[k]`` is round
      ``k+1`` (so ``rounds[-1] is snippets``). Kept for inspection of the refinement;
      not fed downstream.
    - ``document_ids`` — each document's id in this run, aligned with ``documents``.
    - ``notes`` — ``rounds`` with ids and provenance: ``notes[k][d]`` is document
      ``d``'s notes in round ``k+1``, one per part.
    """

    snippets: list[Snippet]
    rounds: list[list[Snippet]]
    document_ids: list[str]
    notes: list[list[list[Note]]]


def draw_document_ids(n: int) -> list[str]:
    """``n`` distinct random ids of 6 hex characters, one per document of a run. An id
    already drawn is drawn again, so identical documents still get distinct ids."""
    ids: dict[str, None] = {}
    while len(ids) < n:
        ids[secrets.token_hex(_DOCUMENT_ID_BYTES)] = None
    return list(ids)


def _part_ids(document_id: str, n_parts: int) -> list[str]:
    """The ids of a document's notes in one round: its id when read whole, else
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


def join_provenance(notes: Sequence[Note]) -> list[str]:
    """A document's provenance: its parts' cited ids, in order, without repeats."""
    return list(dict.fromkeys(cited for note in notes for cited in note.provenance))


def render_relevance(relevance_snippets: Sequence[Snippet] | None) -> str:
    """Render the final-round relevance snippets into the labeled block a prompt embeds
    as the relevant context.

    Each document gets a ``### Document N`` heading, and each part of a document read in
    parts a ``### Document N.k`` heading, ``k`` from 1. A note with nothing relevant
    renders as ``(no relevant summary for this task)``. An empty / ``None`` list renders
    to ``""`` so the consuming prompt omits the block.
    """
    nothing = "(no relevant summary for this task)"
    sections: list[str] = []
    for n, snippet in enumerate(relevance_snippets or [], start=1):
        if isinstance(snippet, list):
            sections += [
                f"### Document {n}.{k}\n{(note or '').strip() or nothing}"
                for k, note in enumerate(snippet, start=1)
            ]
        else:
            sections.append(f"### Document {n}\n{(snippet or '').strip() or nothing}")
    return "\n\n".join(sections)


def _shown(other_snippets: Mapping[str, str]) -> dict[str, str]:
    """The other documents' snippets a call shows, by id, stripped: the empty ones are
    left out, so they cannot be cited."""
    return {id_: s.strip() for id_, s in other_snippets.items() if s.strip()}


def _render_other_states(shown: Mapping[str, str]) -> str:
    """The shown snippets, each under its ``### Summary ID: <id>`` heading, separated
    by blank lines."""
    return "\n\n".join(f"### Summary ID: {id_}\n{s}" for id_, s in shown.items())


def _summary_schema(shown_ids: Sequence[str]) -> type[BaseModel]:
    """The later-rounds reply: the summary, or null, then the shown ids it used. Past
    :data:`MAX_ALLOWED_IDS` the ids are plain strings, checked once the reply is in."""
    allowed = Literal[tuple(shown_ids)] if len(shown_ids) <= MAX_ALLOWED_IDS else str
    return create_model(
        "RelevanceSummary",
        summary=(str | None, ...),
        provenance=(list[allowed], ...),
    )


def _request(
    task: str,
    shown: Mapping[str, str],
    first_round_version: str,
    later_rounds_version: str,
) -> tuple[str, type[BaseModel] | None]:
    """The system prompt and the reply's schema: the first-round prompt and no schema
    when nothing is shown, else the later-rounds prompt and :func:`_summary_schema`."""
    if not shown:
        prompt = load_prompt(
            "relevance/first-round", version=first_round_version, task=task
        )
        return prompt, None
    prompt = load_prompt(
        "relevance/later-rounds",
        version=later_rounds_version,
        task=task,
        other_snippets=_render_other_states(shown),
    )
    return prompt, _summary_schema(list(shown))


def _keep_shown(cited: list[str], shown: Mapping[str, str], kind: str) -> list[str]:
    """``cited`` without the ids the call did not show, each dropped with a warning."""
    unshown = [id_ for id_ in cited if id_ not in shown]
    if unshown:
        _log.warning(
            "%s: dropped provenance id(s) the call did not show: %s", kind, unshown
        )
    return [id_ for id_ in cited if id_ in shown]


def relevance_snippet(
    *,
    task: str,
    document: str,
    other_snippets: Mapping[str, str],
    model: str,
    first_round_version: str,
    later_rounds_version: str,
    run: StageRun | None = None,
    kind: str = "relevance",
    **llm_kwargs: Any,
) -> tuple[str, list[str]]:
    """Surface one ``document``'s relevance snippet for the task, given the OTHER
    documents' previous-round snippets by id (empty in round 1).

    The task + the other documents' snippets live in the system prompt; the document is
    the user message. With no non-empty snippet to show, the call uses the first-round
    prompt and reads back plain text; otherwise the later-rounds prompt and a structured
    ``{summary, provenance}`` reply, whose ids may only be those shown.

    Returns the snippet, stripped — an empty string is allowed (the document contributes
    nothing relevant to the task) — and the shown ids it used.
    """
    shown = _shown(other_snippets)
    system_prompt, schema = _request(
        task, shown, first_round_version, later_rounds_version
    )
    reply = litellm_chat_completion(
        system_prompt=system_prompt,
        user_prompt=document,
        model=model,
        schema=schema,
        run=run,
        kind=kind,
        **llm_kwargs,
    )
    if schema is None:
        return cast(str, reply).strip(), []
    fields = cast(BaseModel, reply).model_dump()
    return (fields["summary"] or "").strip(), _keep_shown(
        fields["provenance"], shown, kind
    )


def surface_relevance(
    *,
    task: str,
    documents: list[str],
    model: str,
    first_round_version: str,
    later_rounds_version: str,
    rounds: int = 2,
    document_ids: list[str] | None = None,
    run: StageRun | None = None,
    workers: int | None = None,
    splits: Splits | None = None,
    **llm_kwargs: Any,
) -> RelevantContext:
    """Surface the relevant context over ``rounds`` synchronous rounds.

    Round 1 writes each document's state from the task + document alone; each later
    round rewrites every document given the *previous round's* snippets of the OTHER
    documents (never its own). Returns a :class:`RelevantContext` (`.snippets` =
    last-round per-document, `.rounds` = all rounds, `.notes` = all rounds with ids and
    provenance). ``rounds=1`` = independent round-1 only; ``rounds < 1`` = no relevance
    snippets at all (`.snippets` / `.rounds` / `.notes` empty — a deliberate
    "no-relevance" run). An empty ``documents`` → empty results.

    ``document_ids`` names each document (``None`` draws them, :func:`draw_document_ids`).

    Each document is read through ``splits`` (built over ``documents``; ``None`` builds
    one), so a document too long for the model's window is read in parts, one note per
    part, and stays split in later rounds. A part never sees its own document's other
    parts: the others' block holds the other documents only, a split one's notes listed
    separately under their part ids.
    """
    ids = draw_document_ids(len(documents)) if document_ids is None else document_ids
    # rounds < 1 = "no relevance snippets at all" (a deliberate rounds=0 re-run); an empty
    # document corpus is likewise empty. Both short-circuit BEFORE round 1 runs (the
    # unconditional run_round(None, 1) below) — a `return`, not a fall-through.
    if rounds < 1 or not documents:
        return RelevantContext(snippets=[], rounds=[], document_ids=ids, notes=[])

    max_workers = workers if workers is not None else active_doc_workers()
    reader = splits if splits is not None else Splits(documents, model=model)

    def run_round(prev: list[list[Note]] | None, round_idx: int) -> list[list[Note]]:
        """Rewrite every document's relevance snippet in parallel. ``prev`` is the frozen
        previous-round note set (``None`` in round 1)."""
        _log.info(
            "round %d/%d · %d doc(s) (≤%d parallel)",
            round_idx,
            rounds,
            len(documents),
            max_workers,
        )

        def one(i: int, _document: str) -> list[Note]:
            # Others-only: document i sees the previous round's notes of the OTHER
            # documents (j != i), never its own (the document itself is the user message).
            others = {
                note.id: note.summary
                for j, notes in enumerate(prev or [])
                if j != i
                for note in notes
            }
            system_prompt, schema = _request(
                task, _shown(others), first_round_version, later_rounds_version
            )
            schema_text = (
                "" if schema is None else json.dumps(schema.model_json_schema())
            )
            replies = reader.read_in_parts(
                i,
                call=f"relevance-r{round_idx}",
                rest=system_prompt + schema_text,
                send=lambda part, kind: relevance_snippet(
                    task=task,
                    document=part,
                    other_snippets=others,
                    model=model,
                    first_round_version=first_round_version,
                    later_rounds_version=later_rounds_version,
                    run=run,
                    kind=kind,
                    **llm_kwargs,
                ),
            )
            return [
                Note(id=id_, summary=summary, provenance=provenance)
                for id_, (summary, provenance) in zip(
                    _part_ids(ids[i], len(replies)), replies, strict=True
                )
            ]

        result = parallel_map(one, documents, max_workers=max_workers)
        n_nonempty = sum(1 for notes in result if any(n.summary for n in notes))
        _log.info(
            "round %d/%d done · %d/%d doc(s) had a relevant state",
            round_idx,
            rounds,
            n_nonempty,
            len(documents),
        )
        return result

    all_notes: list[list[list[Note]]] = [run_round(None, 1)]
    for r in range(2, rounds + 1):
        # only the previous round feeds in
        all_notes.append(run_round(all_notes[-1], r))

    all_rounds = [[_snippet(notes) for notes in per_doc] for per_doc in all_notes]
    return RelevantContext(
        snippets=all_rounds[-1], rounds=all_rounds, document_ids=ids, notes=all_notes
    )


def _snippet(notes: list[Note]) -> Snippet:
    """A document's notes as its snippet: the text, or one per part."""
    if len(notes) == 1:
        return notes[0].summary
    return [note.summary for note in notes]
