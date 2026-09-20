"""Stage 1: surfacing relevance — the per-document, task-conditioned relevance snippets.

A relevance snippet is **not** a generic summary. Each one is written *for a specific
task* and is conditioned on it — the note one document contributes toward the task,
read in light of the rest of the corpus. The point is cross-document reasoning: to use
one document for the task you often need to know what the others say. The per-document
snippets taken together are the **relevant context**.

Documents are assumed to each fit in context (no chunking). Cross-corpus awareness is
built over ``rounds`` **synchronous** rounds:

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

from dataclasses import dataclass
from typing import Any, cast

from r3con.logging_setup import get_logger
from r3con.parallel import parallel_map
from r3con.prompts import load_prompt
from r3con.runs import StageRun
from r3con.runtime.llm import litellm_chat_completion
from r3con.settings import active_doc_workers

_log = get_logger("relevance")

# Separator between the other documents' relevance snippets in the cross-conditioning block.
_OTHER_SEP = "\n\n---\n\n"


@dataclass
class RelevantContext:
    """Output of :func:`surface_relevance` — the relevant context.

    - ``snippets`` — the final-round per-document relevance snippets, aligned 1:1 with the
      input ``documents`` (``snippets[d]`` is the note document ``d`` contributes toward
      the task). This is what downstream stages consume.
    - ``rounds`` — every round's per-document snippets, in order: ``rounds[k]`` is round
      ``k+1`` (so ``rounds[-1] is snippets``). Kept for inspection of the refinement;
      not fed downstream.
    """

    snippets: list[str]
    rounds: list[list[str]]


def render_relevance(relevance_snippets: list[str] | None, doc_ids: list[str] | None = None) -> str:
    """Render the final-round relevance snippets into the labeled block that stages 2
    and 3 embed as the relevant context.

    Each document gets a ``### Document N`` (or its ``doc_ids`` label) heading. A
    document with nothing relevant renders as ``(no relevant summary for this task)``.
    An empty / ``None`` list renders to ``""`` so the consuming prompt omits the block.
    """
    if not relevance_snippets:
        return ""
    parts: list[str] = []
    for i, s in enumerate(relevance_snippets):
        text = (s or "").strip()
        label = doc_ids[i] if (doc_ids and i < len(doc_ids)) else f"Document {i + 1}"
        parts.append(f"### {label}\n{text}" if text else f"### {label}\n(no relevant summary for this task)")
    return "\n\n".join(parts)


def _render_other_states(other_snippets: list[str]) -> str:
    """Concatenate the OTHER documents' relevance snippets for the cross-conditioning
    block. Empty / whitespace-only entries are dropped; an empty list renders to ``""``
    so the prompt omits the block (round 1, or a one-document corpus)."""
    parts = [s.strip() for s in other_snippets if s and s.strip()]
    return _OTHER_SEP.join(parts)


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
    system_prompt = load_prompt(
        "relevance", version=prompt_version, task=task, other_snippets=_render_other_states(other_snippets)
    )
    return cast(
        str,
        litellm_chat_completion(
            system_prompt=system_prompt, user_prompt=document,
            model=model, run=run, kind=kind, **llm_kwargs,
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
    **llm_kwargs: Any,
) -> RelevantContext:
    """Surface the relevant context over ``rounds`` synchronous rounds.

    Round 1 writes each document's state from the task + document alone; each later
    round rewrites every document given the *previous round's* snippets of the OTHER
    documents (never its own). Returns a :class:`RelevantContext` (`.snippets` =
    last-round per-document, `.rounds` = all rounds). ``rounds=1`` = independent
    round-1 only; ``rounds < 1`` = no relevance snippets at all (`.snippets` / `.rounds`
    empty — a deliberate "no-relevance" run). An empty ``documents`` → empty results.
    """
    # rounds < 1 = "no relevance snippets at all" (a deliberate rounds=0 re-run); an empty
    # document corpus is likewise empty. Both short-circuit BEFORE round 1 runs (the
    # unconditional run_round(None, 1) below) — a `return`, not a fall-through.
    if rounds < 1 or not documents:
        return RelevantContext(snippets=[], rounds=[])

    max_workers = workers if workers is not None else active_doc_workers()

    def run_round(prev: list[str] | None, round_idx: int) -> list[str]:
        """Rewrite every document's relevance snippet in parallel. ``prev`` is the frozen
        previous-round state set (``None`` in round 1)."""
        _log.info("round %d/%d · %d doc(s) (≤%d parallel)", round_idx, rounds, len(documents), max_workers)

        def one(i: int, doc: str) -> str:
            # Others-only: document i sees the previous round's snippets of the OTHER
            # documents (j != i), never its own (the document itself is the user message).
            others = [] if prev is None else [s for j, s in enumerate(prev) if j != i]
            return relevance_snippet(
                task=task, document=doc, other_snippets=others, model=model,
                prompt_version=prompt_version, run=run, kind=f"relevance-r{round_idx}-d{i}", **llm_kwargs,
            )

        result = parallel_map(one, documents, max_workers=max_workers)
        n_nonempty = sum(1 for s in result if s and s.strip())
        _log.info("round %d/%d done · %d/%d doc(s) had a relevant state",
                  round_idx, rounds, n_nonempty, len(documents))
        return result

    all_rounds: list[list[str]] = [run_round(None, 1)]
    for r in range(2, rounds + 1):
        all_rounds.append(run_round(all_rounds[-1], r))  # only the previous round feeds in

    return RelevantContext(snippets=all_rounds[-1], rounds=all_rounds)
