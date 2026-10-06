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

Given the run's notes budget (:mod:`r3con.notes`), a round whose notes would not fit a
later request is read again under a word budget, each note asked to stay under W words,
and every round after it is read under the same W.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from r3con.logging_setup import get_logger
from r3con.notes import MIN_NOTE_WORDS, Budget, NotesTooLong
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
    - ``words`` — the word budget each round's notes were read under, aligned with
      ``rounds``: ``None`` for a round read without one.
    """

    snippets: list[Snippet]
    rounds: list[list[Snippet]]
    words: list[int | None] = field(default_factory=list)


def join_parts(snippet: Snippet) -> str:
    """A document's snippet as one note: a split document's part notes, stripped, in
    order, joined by a blank line, with the empty ones left out."""
    if isinstance(snippet, str):
        return snippet
    return "\n\n".join(note.strip() for note in snippet if note.strip())


def note_texts(snippets: Sequence[Snippet]) -> list[str]:
    """The notes a rendered block of ``snippets`` carries, in order: each document's
    note, and each part's of a document read in parts, stripped, without the empty
    ones."""
    notes = (note for s in snippets for note in ([s] if isinstance(s, str) else s))
    return [note.strip() for note in notes if note.strip()]


def takes_word_budget(prompt_version: str) -> bool:
    """Whether the relevance prompt at ``prompt_version`` can ask each note to stay
    under a number of words: whether it renders differently when given one."""
    return _system_prompt("", [], prompt_version, MIN_NOTE_WORDS) != _system_prompt(
        "", [], prompt_version
    )


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


def _system_prompt(
    task: str,
    other_snippets: list[str],
    prompt_version: str,
    max_words: int | None = None,
) -> str:
    """The relevance system prompt: the task, the other documents' notes, and the
    words the note may take, if any."""
    return load_prompt(
        "relevance",
        version=prompt_version,
        task=task,
        other_snippets=_render_other_states(other_snippets),
        max_words=max_words,
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
    max_words: int | None = None,
    **llm_kwargs: Any,
) -> str:
    """Surface one ``document``'s relevance snippet for the task, given the OTHER
    documents' previous-round snippets (empty list in round 1).

    The task + the (possibly empty) other-documents block live in the system prompt;
    the document is the user message. With ``max_words``, a prompt that can (v2) asks
    the note to stay under that many words. Returns the state text, stripped — an empty
    string is allowed (the document contributes nothing relevant to the task).
    """
    return cast(
        str,
        litellm_chat_completion(
            system_prompt=_system_prompt(
                task, other_snippets, prompt_version, max_words
            ),
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
    budget: Budget | None = None,
    reread: RelevantContext | None = None,
    final_check: Callable[[list[Snippet]], None] | None = None,
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

    With the run's ``budget``, every round measures its requests before sending them,
    and a round whose requests do not fit because the previous round's notes are the
    bigger part of them reads that round again under the budget's word budget first
    (and the rounds before it, if need be); every round after is read under it too.
    ``final_check`` measures the final notes against the biggest request they will ride
    in later, raising ``NotesTooLong`` when they do not fit: it is run after the last
    round, which is read again until it passes, and beside a round that did not fit.
    ``reread`` is a context read before, under a budget since lowered: its last round is
    read again under the budget, its earlier rounds kept. Without a budget, no round is
    read again, and splitting alone applies.

    Raises:
        litellm.ContextWindowExceededError: when the notes cannot be read short enough
            to fit (from :meth:`r3con.notes.Budget.shorten`), or when more parts cannot
            help a document (:mod:`r3con.splitting`).
        ValueError: for ``reread`` or ``final_check`` without a ``budget``, before any
            request.
    """
    if budget is None and (reread is not None or final_check is not None):
        raise ValueError(
            "reread= and final_check= belong to a run's notes budget; pass budget= too."
        )
    # rounds < 1 = "no relevance snippets at all" (a deliberate rounds=0 re-run); an empty
    # document corpus is likewise empty. Both short-circuit before any round is read.
    if rounds < 1 or not documents:
        return RelevantContext(snippets=[], rounds=[])

    max_workers = workers if workers is not None else active_doc_workers()
    reader = splits if splits is not None else Splits(documents, model=model)

    def run_round(prev: list[Snippet] | None, round_idx: int) -> list[Snippet]:
        """Rewrite every document's relevance snippet in parallel. ``prev`` is the frozen
        previous-round state set (``None`` in round 1)."""
        max_words = budget.words if budget is not None else None
        call = f"relevance-r{round_idx}"
        if max_words is not None:
            call += f"-w{max_words}"
        _log.info(
            "round %d/%d · %d doc(s) (≤%d parallel)%s",
            round_idx,
            rounds,
            len(documents),
            max_workers,
            "" if max_words is None else f", each note under {max_words} words",
        )

        def request(i: int) -> tuple[list[str], str, list[str]]:
            """Document i's view of the others: their notes, the rest it is sent with,
            and the notes inside that rest."""
            # Others-only: document i sees the previous round's snippets of the OTHER
            # documents (j != i), never its own (the document itself is the user message).
            others = (
                []
                if prev is None
                else [join_parts(s) for j, s in enumerate(prev) if j != i]
            )
            rest = _system_prompt(task, others, prompt_version, max_words)
            return others, rest, note_texts(others)

        if budget is not None:
            reader.measure(
                range(len(documents)), call=call, rests=lambda i: request(i)[1:]
            )

        def one(i: int, _document: str) -> Snippet:
            others, rest, notes = request(i)
            parts = reader.read_in_parts(
                i,
                call=call,
                rest=rest,
                notes=notes if budget is not None else (),
                send=lambda part, kind: relevance_snippet(
                    task=task,
                    document=part,
                    other_snippets=others,
                    model=model,
                    prompt_version=prompt_version,
                    run=run,
                    kind=kind,
                    max_words=max_words,
                    **llm_kwargs,
                ),
            )
            return parts[0] if len(parts) == 1 else parts

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

    # The rounds read so far, and the word budget each was read under; only the
    # previous round feeds the next.
    read: list[list[Snippet]] = [] if reread is None else list(reread.rounds[:-1])
    words: list[int | None] = (
        [] if reread is None else [*reread.words, *[None] * len(read)][: len(read)]
    )

    def read_round(r: int) -> None:
        """Read round ``r``, dropping any later round; when its requests do not fit
        for the previous round's notes, read that round again shorter first."""
        while True:
            calls_before = len(run.steps) if run is not None else 0
            try:
                notes = run_round(read[r - 2] if r > 1 else None, r)
            except NotesTooLong as trouble:
                if budget is None:
                    raise
                later = _trouble(final_check, read[r - 2])
                budget.shorten(
                    trouble,
                    round_idx=r - 1,
                    discarded=(len(run.steps) if run is not None else 0) - calls_before,
                    also=[] if later is None else [later],
                )
                read_round(r - 1)
                continue
            read[r - 1 :] = [notes]
            words[r - 1 :] = [budget.words if budget is not None else None]
            return

    for r in range(len(read) + 1, rounds + 1):
        read_round(r)
    if budget is not None:
        while (trouble := _trouble(final_check, read[-1])) is not None:
            budget.shorten(trouble, round_idx=rounds)
            read_round(rounds)
    return RelevantContext(snippets=read[-1], rounds=read, words=words)


def _trouble(
    check: Callable[[list[Snippet]], None] | None, notes: list[Snippet]
) -> NotesTooLong | None:
    """The ``NotesTooLong`` that ``check`` raises for ``notes``, if any."""
    if check is None:
        return None
    try:
        check(notes)
    except NotesTooLong as trouble:
        return trouble
    return None
