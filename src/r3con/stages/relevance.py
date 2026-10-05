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
  document alone (no other document is visible).
- **Round k ≥ 2** — each document's state is rewritten given the task and the
  **previous round's** snippets of the OTHER documents (never the document's own prior
  state — the document itself is in the user message). Only the immediately-previous
  round is fed in, not the whole history.

Round k reads the *frozen* round-(k-1) set, so within a round the per-document calls
are independent and fan out in parallel (bounded by
:func:`r3con.settings.active_doc_workers`). Downstream stages consume the
**final-round** snippets (``RelevantContext.snippets``); earlier rounds are kept only
for inspection.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

from r3con.logging_setup import get_logger
from r3con.parallel import parallel_map
from r3con.prompts import load_prompt
from r3con.runs import StageRun
from r3con.runtime.llm import litellm_chat_completion
from r3con.settings import active_doc_workers
from r3con.splitting import Splits

_log = get_logger("relevance")

# Separator between the other documents' relevance snippets in the cross-conditioning block.
_OTHER_SEP = "\n\n---\n\n"

Snippet = str | list[str]
"""A document's relevance snippet, or, for a document read in parts, one per part."""


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
    """

    snippets: list[Snippet]
    rounds: list[list[Snippet]]


def join_parts(snippet: Snippet) -> str:
    """A document's snippet as one note: a split document's part notes, stripped, in
    order, joined by a blank line, with the empty ones left out."""
    if isinstance(snippet, str):
        return snippet
    return "\n\n".join(note.strip() for note in snippet if note.strip())


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


def _render_other_states(other_snippets: list[str]) -> str:
    """Concatenate the OTHER documents' relevance snippets for the cross-conditioning
    block. Empty / whitespace-only entries are dropped; an empty list renders to ``""``
    so the prompt omits the block (round 1, or a one-document corpus)."""
    parts = [s.strip() for s in other_snippets if s and s.strip()]
    return _OTHER_SEP.join(parts)


def _system_prompt(task: str, other_snippets: list[str], prompt_version: str) -> str:
    """The relevance system prompt: the task, and the other documents' notes."""
    return load_prompt(
        "relevance",
        version=prompt_version,
        task=task,
        other_snippets=_render_other_states(other_snippets),
    )


def relevance_snippet(
    *,
    task: str,
    document: str,
    other_snippets: list[str],
    model: str,
    prompt_version: str,
    run: StageRun | None = None,
    kind: str = "relevance",
    **llm_kwargs: Any,
) -> str:
    """Surface one ``document``'s relevance snippet for the task, given the OTHER
    documents' previous-round snippets (empty list in round 1).

    The task + the (possibly empty) other-documents block live in the system prompt;
    the document is the user message. Returns the state text, stripped — an empty
    string is allowed (the document contributes nothing relevant to the task).
    """
    return cast(
        str,
        litellm_chat_completion(
            system_prompt=_system_prompt(task, other_snippets, prompt_version),
            user_prompt=document,
            model=model,
            run=run,
            kind=kind,
            **llm_kwargs,
        ),
    ).strip()


def surface_relevance(
    *,
    task: str,
    documents: list[str],
    model: str,
    prompt_version: str,
    rounds: int = 2,
    run: StageRun | None = None,
    workers: int | None = None,
    splits: Splits | None = None,
    **llm_kwargs: Any,
) -> RelevantContext:
    """Surface the relevant context over ``rounds`` synchronous rounds.

    Round 1 writes each document's state from the task + document alone; each later
    round rewrites every document given the *previous round's* snippets of the OTHER
    documents (never its own). Returns a :class:`RelevantContext` (`.snippets` =
    last-round per-document, `.rounds` = all rounds). ``rounds=1`` = independent
    round-1 only; ``rounds < 1`` = no relevance snippets at all (`.snippets` / `.rounds`
    empty — a deliberate "no-relevance" run). An empty ``documents`` → empty results.

    Each document is read through ``splits`` (built over ``documents``; ``None`` builds
    one), so a document too long for the model's window is read in parts, one note per
    part, and stays split in later rounds. A part never sees its own document's other
    parts: the others' block holds the other documents only, a split one's notes joined.
    """
    # rounds < 1 = "no relevance snippets at all" (a deliberate rounds=0 re-run); an empty
    # document corpus is likewise empty. Both short-circuit BEFORE round 1 runs (the
    # unconditional run_round(None, 1) below) — a `return`, not a fall-through.
    if rounds < 1 or not documents:
        return RelevantContext(snippets=[], rounds=[])

    max_workers = workers if workers is not None else active_doc_workers()
    reader = splits if splits is not None else Splits(documents, model=model)

    def run_round(prev: list[Snippet] | None, round_idx: int) -> list[Snippet]:
        """Rewrite every document's relevance snippet in parallel. ``prev`` is the frozen
        previous-round state set (``None`` in round 1)."""
        _log.info(
            "round %d/%d · %d doc(s) (≤%d parallel)",
            round_idx,
            rounds,
            len(documents),
            max_workers,
        )

        def one(i: int, _document: str) -> Snippet:
            # Others-only: document i sees the previous round's snippets of the OTHER
            # documents (j != i), never its own (the document itself is the user message).
            others = (
                []
                if prev is None
                else [join_parts(s) for j, s in enumerate(prev) if j != i]
            )
            notes = reader.read_in_parts(
                i,
                call=f"relevance-r{round_idx}",
                rest=_system_prompt(task, others, prompt_version),
                send=lambda part, kind: relevance_snippet(
                    task=task,
                    document=part,
                    other_snippets=others,
                    model=model,
                    prompt_version=prompt_version,
                    run=run,
                    kind=kind,
                    **llm_kwargs,
                ),
            )
            return notes[0] if len(notes) == 1 else notes

        result = parallel_map(one, documents, max_workers=max_workers)
        n_nonempty = sum(1 for s in result if join_parts(s).strip())
        _log.info(
            "round %d/%d done · %d/%d doc(s) had a relevant state",
            round_idx,
            rounds,
            n_nonempty,
            len(documents),
        )
        return result

    all_rounds: list[list[Snippet]] = [run_round(None, 1)]
    for r in range(2, rounds + 1):
        # only the previous round feeds in
        all_rounds.append(run_round(all_rounds[-1], r))

    return RelevantContext(snippets=all_rounds[-1], rounds=all_rounds)
