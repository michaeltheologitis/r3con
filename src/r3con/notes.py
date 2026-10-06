"""The notes' budget: how many words each relevance note may take in one run.

Every document's note rides in every round-2 relevance call, the schema call, every
parse call and reasoning's first turn, so the notes grow with the collection. When a
request does not fit the model's window and the notes it carries are the bigger part of
it, :mod:`r3con.splitting` hands it here by raising :class:`NotesTooLong` instead of
cutting the document beside them. :meth:`Budget.shorten` then halves the words each
note may take, W, as many times as the estimate needs before anything is read again
(once per refusal where the window is unknown), and the caller reads again the round
that wrote the notes, every note asked to stay under W words. W is one value per run
and only ever falls; no note is asked for fewer than ``MIN_NOTE_WORDS``. When even that
cannot fit, the run stops with ``litellm.ContextWindowExceededError``.

``NotesTooLong`` is r3con's own exception, not a ``ContextWindowExceededError``, so
nothing that cuts a document on a refusal takes it for one. ``r3con.run`` catches it
only between its stages and the loops that read notes again; a stop is the only way a
notes trouble leaves a run. Every event is recorded in ``notes.json`` in the run folder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING, Any, NoReturn

from litellm.exceptions import ContextWindowExceededError

from r3con.logging_setup import get_logger
from r3con.runtime.llm import count_tokens

if TYPE_CHECKING:
    from r3con.runs import TaskLogger
    from r3con.splitting import Splits

_log = get_logger("notes")

MIN_NOTE_WORDS = 10

_OUTGROWN = "The relevant context has outgrown the model's window."


@dataclass(eq=False, repr=False)
class NotesTooLong(Exception):
    """A request that does not fit the model's window while the notes it carries are
    the bigger part of it: bigger than its document part, or all it carries beside its
    own prompt (the schema call, reasoning's first turn). Not a
    ``ContextWindowExceededError``, so nothing that cuts a document takes it for one.

    - ``call``: the request, a kind (``relevance-r2-d3``, ``parse-d0c1``) or a stage
      (``schema``, ``reasoning``); ``model``: the run's model string.
    - ``cause``: ``"estimate"`` (not sent) or ``"refusal"`` (refused, ``refusal``).
    - ``estimate``: the request's tokens; ``notes``: the notes it carries, one text per
      document or part, without the empty ones; ``notes_tokens``: their
      :func:`count_notes`.
    - ``room``: what the notes may take for the request to fit, with a known window and
      an estimate; ``None`` after a refusal.
    """

    message: str
    _: KW_ONLY
    model: str
    call: str
    cause: str
    estimate: int
    notes: Sequence[str]
    notes_tokens: int
    room: int | None = None
    refusal: ContextWindowExceededError | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)
        self.notes = list(self.notes)
        self.add_note(
            f"r3con: the notes {self.call} carries (about {self.notes_tokens:,} tokens) "
            "are the bigger part of a request that does not fit; hand this to the "
            "budget's shorten and read their round again, as r3con.run does."
        )


def count_notes(notes: Sequence[str]) -> int:
    """The tokens ``notes`` take in a request: each with the paragraph break every
    rendering puts after a note, since a note's last token merges with it."""
    return sum(count_tokens(note + "\n\n") for note in notes)


class Budget:
    """How many words each relevance note may take in one run (``words``, ``None``
    until a request needs it), and the record of every time the notes were read again
    shorter, written to ``notes.json`` with a ``task_logger``.

    ``splits`` is the run's :class:`r3con.splitting.Splits`, whose window and line the
    record names and whose ``check_notes`` the stages without a document call.
    ``can_shorten`` is whether the run's relevance prompt can ask for a length; without
    it every :meth:`shorten` stops the run. Used from one thread.
    """

    splits: Splits
    words: int | None

    def __init__(
        self,
        splits: Splits,
        *,
        can_shorten: bool = True,
        task_logger: TaskLogger | None = None,
    ) -> None:
        self.splits = splits
        self.words = None
        self._can_shorten = can_shorten
        self._task_logger = task_logger
        self._record: dict[str, Any] = {**splits.window, "events": []}

    def shorten(
        self,
        trouble: NotesTooLong,
        *,
        round_idx: int,
        discarded: int = 0,
        also: Sequence[NotesTooLong] = (),
    ) -> int:
        """Set and return the words each note may take, for reading round
        ``round_idx`` (the round that wrote ``trouble``'s notes) again.

        W is ``words`` halved, or the notes' mean halved when no budget is set, as many
        times as it takes for the notes, each cut at W words, to fit the room of
        ``trouble`` and of every request in ``also`` (the requests the same notes, or
        later rounds' notes read under W, ride in later); once, after a refusal. Never
        under ``MIN_NOTE_WORDS``, and always under the last W. ``discarded`` is how many
        of the stage's accepted calls are thrown away.

        Raises:
            litellm.ContextWindowExceededError: the stop, when no W can fit or the
                prompt cannot ask for one: the provider's refusal, or r3con's own error
                with ``trouble``'s message, with a note that says why, raised from
                ``None``.
        """
        rooms = [(r, r.room) for r in (trouble, *also) if r.room is not None]
        binding = min(rooms, key=lambda pair: pair[1])[0] if rooms else trouble
        mean = _mean_words(trouble.notes)
        start = self.words if self.words is not None else mean
        levels = [
            w
            for w in _halvings(start)
            if w < mean and all(_tokens_at(r.notes, w) <= room for r, room in rooms)
        ]
        if not levels or not self._can_shorten:
            self._stop(trouble, binding, discarded, round_idx)
        self.words = levels[0]
        self._record_event(trouble, binding, discarded, "read again", round_idx)
        _log.warning(_read_again(trouble, self.splits.line, round_idx, self.words))
        return self.words

    def _stop(
        self,
        trouble: NotesTooLong,
        binding: NotesTooLong,
        discarded: int,
        round_idx: int,
    ) -> NoReturn:
        """Record the stop and raise the provider's refusal, or, when nothing was sent,
        r3con's own error, with a note that says why reading shorter cannot help."""
        self._record_event(trouble, binding, discarded, "stop", round_idx)
        stop = trouble.refusal
        if stop is None:
            stop = ContextWindowExceededError(
                message=str(trouble), model=trouble.model, llm_provider="r3con"
            )
        stop.add_note(f"r3con: {self._why_not(trouble, binding)} {_OUTGROWN}")
        raise stop from None

    def _why_not(self, trouble: NotesTooLong, binding: NotesTooLong) -> str:
        """Why reading the notes shorter cannot help, for the stop's note: the prompt
        cannot ask for it (``trouble`` names the request that did not fit), or the
        request that binds (``binding``) would not fit even then."""
        if not self._can_shorten:
            return (
                f"{trouble.call} carries the notes of {len(trouble.notes)} documents "
                f"(about {trouble.notes_tokens:,} tokens), the bigger part of a request "
                "that does not fit, and the relevance prompt this run pins cannot ask "
                "for shorter notes (the shipped v2 can)."
            )
        n, room = len(binding.notes), binding.room
        if room is not None and room <= 0:
            return (
                f"what {binding.call} sends beside its notes fills the "
                f"{self.splits.line:,}-token line by itself, so reading them shorter "
                "cannot help."
            )
        at_floor = _tokens_at(binding.notes, MIN_NOTE_WORDS)
        if room is not None and at_floor > room:
            return (
                f"even at {MIN_NOTE_WORDS} words each, the notes of {n} documents "
                f"would take about {at_floor:,} tokens, over the {room:,} that "
                f"{binding.call} leaves them, so reading them shorter cannot help."
            )
        sent = (
            "was refused as too long"
            if binding.refusal is not None
            else "is estimated over the line"
        )
        return (
            f"{binding.call} {sent}, and the notes of {n} documents it carries are "
            f"already about {_mean_words(binding.notes)} words each, with "
            f"{MIN_NOTE_WORDS} the fewest a note is asked for, so reading them shorter "
            "cannot help."
        )

    def _record_event(
        self,
        trouble: NotesTooLong,
        binding: NotesTooLong,
        discarded: int,
        action: str,
        round_idx: int,
    ) -> None:
        self._record["events"].append(
            {
                "call": trouble.call,
                "cause": trouble.cause,
                "estimate": trouble.estimate,
                "notes": trouble.notes_tokens,
                "room": binding.room,
                "sized_for": binding.call,
                "discarded": discarded,
                "action": action,
                "round": round_idx,
                "words": self.words if action == "read again" else None,
                "error": None if trouble.refusal is None else str(trouble.refusal),
            }
        )
        if self._task_logger is not None:
            self._task_logger.write_json("notes", self._record)


def _halvings(start: int) -> list[int]:
    """``start`` halved, then halved again, each raised to ``MIN_NOTE_WORDS``, until
    the first that is ``MIN_NOTE_WORDS``; only those under ``start``."""
    levels: list[int] = []
    words = start
    while not levels or levels[-1] > MIN_NOTE_WORDS:
        words //= 2
        levels.append(max(words, MIN_NOTE_WORDS))
    return [w for w in levels if w < start]


def _mean_words(notes: Sequence[str]) -> int:
    return sum(len(note.split()) for note in notes) // max(len(notes), 1)


def _tokens_at(notes: Sequence[str], words: int) -> int:
    """The tokens ``notes`` would take with each cut at its ``words``-th word: a
    stand-in for the notes a model writes under that budget."""
    return count_notes([" ".join(note.split()[:words]) for note in notes])


def _read_again(trouble: NotesTooLong, line: int | None, round_idx: int, w: int) -> str:
    """The warning for reading round ``round_idx`` again under ``w`` words."""
    n, notes_tokens = len(trouble.notes), f"{trouble.notes_tokens:,}"
    if trouble.refusal is not None:
        return (
            f"{trouble.call} was refused as too long; the notes of {n} documents it "
            f"carries (about {notes_tokens} tokens) are the bigger part, so round "
            f"{round_idx} is read again with each note under {w} words"
        )
    return (
        f"{trouble.call}: the notes of {n} documents come to about {notes_tokens} "
        f"tokens, and the request with them to about {trouble.estimate:,}, over the "
        f"{line:,}-token line; reading round {round_idx} again with each note under "
        f"{w} words"
    )
