"""Reading a document too long for the model's window in parts.

A stage sends each document whole, as the user message of one call. When that request
does not fit the model's input window, the document is cut in 2 at the break nearest its
middle, then every part in 2 again, and so on, until every part's request fits. The parts
never overlap and always rejoin to the document, and a part keeps its document's index,
so the caller's numbering never moves.

Two things say a request does not fit:

- **The estimate**, when litellm's model map gives the model's input window: a request
  estimated over the window less ``settings.WINDOW_MARGIN_PERCENT`` is split before it
  is sent. The estimate counts cl100k_base tokens, from the vocabulary litellm ships.
- **A refusal**: the provider raises ``litellm.ContextWindowExceededError``. It splits
  whatever the estimate said.

Splitting stops when more parts cannot help, because what is sent beside the document
(the prompt and the other documents' notes) has outgrown the window. The run then
raises ``litellm.ContextWindowExceededError`` with a note naming the document.

The bigger of what a request carries is halved. When a caller passes the notes a
request carries and they are bigger than its document part (or the request has no
document, ``check_notes``), the request is handed to :mod:`r3con.notes` by raising
:class:`r3con.notes.NotesTooLong`, which reads the notes again shorter, instead of
being cut.
"""

from __future__ import annotations

import functools
import itertools
import re
import threading
from collections.abc import Callable, Iterable, Sequence
from typing import Any, NoReturn, TypeVar

import litellm
from litellm.exceptions import ContextWindowExceededError

from r3con import settings
from r3con.logging_setup import get_logger
from r3con.notes import NotesTooLong, count_notes
from r3con.runs import TaskLogger
from r3con.runtime.llm import count_tokens, quiet_litellm

_log = get_logger("splitting")

T = TypeVar("T")


class Splits:
    """Where each document of one run is cut into parts, kept for the rest of the run.

    The window is ``model``'s ``max_input_tokens`` in litellm's model map, looked up
    once; ``line`` is that window less ``margin_percent``. Both are ``None`` when the
    map does not know the model. With a ``task_logger``, every split and stop is
    written to ``splits.json`` in the run folder as it happens.

    Different documents may be read at once, from different threads, but one document
    by one thread at a time: a document's cuts are not locked, and the record and its
    file are written under a lock.
    """

    model: str
    max_input_tokens: int | None
    margin_percent: int
    line: int | None

    def __init__(
        self,
        documents: Sequence[str],
        *,
        model: str,
        margin_percent: int | None = None,
        task_logger: TaskLogger | None = None,
    ) -> None:
        """``margin_percent=None`` reads ``settings.WINDOW_MARGIN_PERCENT``.

        Raises:
            ValueError: for a margin that is not an integer from 0 to 99, as the
                settings snapshot refuses one.
        """
        if margin_percent is None:
            margin_percent = settings.WINDOW_MARGIN_PERCENT
        settings.check_cap("window_margin_percent", margin_percent)
        self._documents = list(documents)
        self._cuts: list[list[int]] = [[] for _ in self._documents]
        self.model = model
        self._task_logger = task_logger
        self._lock = threading.Lock()
        self.margin_percent = margin_percent
        self.max_input_tokens = _max_input_tokens(model)
        self.line = (
            None
            if self.max_input_tokens is None
            else self.max_input_tokens * (100 - margin_percent) // 100
        )
        self._record: dict[str, Any] = {
            "model": model,
            "max_input_tokens": self.max_input_tokens,
            "margin_percent": margin_percent,
            "line": self.line,
            "documents": {},
        }

    def parts(self, doc: int) -> list[str]:
        """``documents[doc]`` cut at its cuts so far: ``[documents[doc]]`` until it is
        split."""
        text = self._documents[doc]
        bounds = itertools.pairwise([0, *self._cuts[doc], len(text)])
        return [text[start:end] for start, end in bounds]

    def over_line(self, *texts: str) -> int | None:
        """The tokens of ``texts`` together, when the window is known and they are over
        the line; otherwise ``None``. Texts no larger than the line in UTF-8 bytes are
        not counted."""
        if self.line is None or sum(map(_utf8_size, texts)) <= self.line:
            return None
        tokens = sum(map(count_tokens, texts))
        return tokens if tokens > self.line else None

    def read_in_parts(
        self,
        doc: int,
        *,
        call: str,
        rest: str,
        send: Callable[[str, str], T],
        notes: Sequence[str] = (),
    ) -> list[T]:
        """One result of ``send(part, kind)`` per part of ``documents[doc]``, in order,
        after splitting it as far as it must.

        ``call`` names the call (``"relevance-r2"``, ``"parse"``); a part's ``kind`` is
        ``{call}-d{doc}`` while the document is whole and ``{call}-d{doc}c{k}`` once it
        is split. ``rest`` is everything sent beside a part, and ``notes`` the notes
        inside it, one text per document or part. ``send`` raises
        ``litellm.ContextWindowExceededError`` when the provider refuses a part; any
        other exception, a ``NotesTooLong`` included, passes through untouched.

        With a known window: while the rest is under the line, halve until every part's
        estimate is under it, then send; a refusal halves again and starts over. With an
        unknown window: send, and halve on a refusal. Either way the parts accepted
        before a refusal are discarded and the level is read again. With ``notes``, a
        request that does not fit while its notes are bigger than its part, or whose
        rest alone is over the line, is not cut: it is handed over.

        Raises:
            r3con.notes.NotesTooLong: only with ``notes``, for that request, before
                sending it or from its refusal.
            litellm.ContextWindowExceededError: with a note naming the document, when
                more parts cannot help: the rest alone is over the line; with an
                unknown window, a refused part is shorter than the rest; or a part of
                one character still does not fit.
        """
        # The rest's and the notes' tokens: counted once, and only when the request's
        # size in bytes cannot settle whether it fits.
        count_rest = functools.cache(functools.partial(count_tokens, rest))
        count_rest_notes = functools.cache(functools.partial(count_notes, notes))
        while True:
            if self.line is not None:
                self._split_to_fit(
                    doc, call, rest, count_rest, notes, count_rest_notes, self.line
                )
            parts = self.parts(doc)
            results: list[T] = []
            for k, part in enumerate(parts):
                kind = _kind(call, doc, k, len(parts))
                try:
                    results.append(send(part, kind))
                except ContextWindowExceededError as refusal:
                    self._split_on_refusal(
                        doc, kind, k, part, count_rest, notes, count_rest_notes, refusal
                    )
                    break
            else:
                self._record_parts(doc, call, len(parts))
                return results

    def measure(
        self,
        docs: Iterable[int],
        *,
        call: str,
        rests: Callable[[int], tuple[str, Sequence[str]]],
    ) -> None:
        """Before a stage sends anything, with a known window: cut each of ``docs`` by
        estimate as :meth:`read_in_parts` would, ``rests(doc)`` giving the rest it is
        sent with and the notes inside it.

        Raises:
            r3con.notes.NotesTooLong: once every document is measured, the one with
                the least room of those whose notes must be handed over.
            litellm.ContextWindowExceededError: R2's stops, at once.
        """
        if self.line is None:
            return
        troubles: list[NotesTooLong] = []
        for doc in docs:
            rest, notes = rests(doc)
            try:
                self._split_to_fit(
                    doc,
                    call,
                    rest,
                    functools.cache(functools.partial(count_tokens, rest)),
                    notes,
                    functools.cache(functools.partial(count_notes, notes)),
                    self.line,
                )
            except NotesTooLong as trouble:
                troubles.append(trouble)
        if troubles:
            raise min(troubles, key=lambda trouble: trouble.room or 0)

    def check_notes(
        self,
        *,
        call: str,
        request: Sequence[str],
        notes: Sequence[str],
        refusal: ContextWindowExceededError | None = None,
    ) -> None:
        """Hand over the notes of a request that carries no document, ``call`` (the
        schema call, reasoning's first turn), when it does not fit: ``request`` is its
        texts, ``notes`` the notes inside them. A request without notes is never
        handed over.

        Raises:
            r3con.notes.NotesTooLong: without ``refusal``, when the request is over the
                line; with it, from it.
        """
        if not notes:
            return
        if refusal is not None:
            estimate = sum(map(count_tokens, request))
            raise self._notes_too_long(
                call, estimate, notes, count_notes(notes), refusal
            ) from refusal
        if (estimate := self.over_line(*request)) is not None:
            raise self._notes_too_long(call, estimate, notes, count_notes(notes))

    def _split_to_fit(
        self,
        doc: int,
        call: str,
        rest: str,
        count_rest: Callable[[], int],
        notes: Sequence[str],
        count_rest_notes: Callable[[], int],
        line: int,
    ) -> None:
        """Before sending, with a known window: stop if the rest alone is over the
        line, or hand its notes over when it carries any; otherwise halve until every
        part's request is estimated under it, handing the notes over instead when they
        are bigger than the part over the line."""
        rest_size = _utf8_size(rest)
        if rest_size > line and count_rest() > line:
            parts = self.parts(doc)
            kind = _kind(call, doc, 0, len(parts))
            estimate = count_rest() + count_tokens(parts[0])
            if notes:
                raise self._notes_too_long(kind, estimate, notes, count_rest_notes())
            self._stop(
                doc,
                self._event(kind, "estimate", estimate, count_rest(), 0),
                f"the prompt and notes sent with it are about {count_rest():,} tokens, "
                f"over the {line:,}-token line by themselves",
            )
        while (over := self._first_over(doc, rest_size, count_rest, line)) is not None:
            k, estimate = over
            parts = self.parts(doc)
            kind = _kind(call, doc, k, len(parts))
            if notes and count_rest_notes() > estimate - count_rest():
                raise self._notes_too_long(kind, estimate, notes, count_rest_notes())
            event = self._event(kind, "estimate", estimate, count_rest(), 0)
            if len(parts[k]) == 1:
                self._stop(doc, event, _no_room(count_rest()))
            n_parts = self._split(doc, event)
            _log.warning(
                "%s: documents[%d] (Document %d) is estimated at %s tokens, over the "
                "%s-token line; reading it in %d parts",
                kind,
                doc,
                doc + 1,
                f"{estimate:,}",
                f"{line:,}",
                n_parts,
            )

    def _first_over(
        self, doc: int, rest_size: int, count_rest: Callable[[], int], line: int
    ) -> tuple[int, int] | None:
        """The first part of ``documents[doc]`` whose request is estimated over
        ``line``, with that estimate. A request no larger than the line in UTF-8 bytes
        is under it without counting, since every token is at least one byte."""
        for k, part in enumerate(self.parts(doc)):
            if rest_size + _utf8_size(part) <= line:
                continue
            estimate = count_rest() + count_tokens(part)
            if estimate > line:
                return k, estimate
        return None

    def _split_on_refusal(
        self,
        doc: int,
        kind: str,
        k: int,
        part: str,
        count_rest: Callable[[], int],
        notes: Sequence[str],
        count_rest_notes: Callable[[], int],
        refusal: ContextWindowExceededError,
    ) -> None:
        """Halve after the provider refused part ``k``, or stop where more parts cannot
        help: a part of one character, or, with an unknown window, a part shorter than
        the rest. Notes bigger than the part are handed over instead, from the
        refusal."""
        part_tokens = count_tokens(part)
        if notes and count_rest_notes() > part_tokens:
            raise self._notes_too_long(
                kind, count_rest() + part_tokens, notes, count_rest_notes(), refusal
            ) from refusal
        event = self._event(
            kind, "refusal", count_rest() + part_tokens, count_rest(), k
        )
        if self.line is None and part_tokens < count_rest():
            self._stop(
                doc,
                event,
                f"the part is about {part_tokens:,} tokens and the prompt and notes "
                f"sent with it about {count_rest():,}",
                refusal,
            )
        if len(part) == 1:
            self._stop(doc, event, _no_room(count_rest()), refusal)
        n_parts = self._split(doc, event, refusal)
        _log.warning(
            "%s: documents[%d] (Document %d) was refused as too long; reading it in %d "
            "parts",
            kind,
            doc,
            doc + 1,
            n_parts,
        )

    def _split(
        self,
        doc: int,
        event: dict[str, Any],
        refusal: ContextWindowExceededError | None = None,
    ) -> int:
        """Halve every part of ``documents[doc]`` longer than one character, record the
        split, and return the number of parts."""
        starts = [0, *self._cuts[doc]]
        cuts = [
            start + halve(part)
            for start, part in zip(starts, self.parts(doc))
            if len(part) > 1
        ]
        self._cuts[doc] = sorted([*self._cuts[doc], *cuts])
        self._record_event(doc, event, "split", refusal)
        return len(self._cuts[doc]) + 1

    def _stop(
        self,
        doc: int,
        event: dict[str, Any],
        clause: str,
        refusal: ContextWindowExceededError | None = None,
    ) -> NoReturn:
        """Record the stop and raise the provider's ``refusal``, or, when nothing was
        sent, r3con's own error, with a note that says why more parts cannot help
        (``clause``)."""
        self._record_event(doc, event, "stop", refusal)
        stop = self._not_sent(event) if refusal is None else refusal
        stop.add_note(
            f"r3con: reading documents[{doc}] (Document {doc + 1}) in more parts cannot "
            f"help in {event['call']}: {clause}. The relevant context has outgrown the "
            "model's window."
        )
        raise stop

    def _not_sent(self, event: dict[str, Any]) -> ContextWindowExceededError:
        """The error for ``event``'s request, which r3con stopped before sending."""
        return ContextWindowExceededError(
            message=self._estimated(event["call"], event["estimate"]),
            model=self.model,
            llm_provider="r3con",
        )

    def _estimated(self, call: str, estimate: int) -> str:
        """Why ``call``, estimated at ``estimate`` tokens, was not sent."""
        return (
            f"r3con estimated {call} at {estimate:,} tokens, over the {self.line:,}-token "
            f"line (the {self.max_input_tokens:,}-token input window litellm's model "
            f"map gives {self.model}, less {self.margin_percent}%); it was not sent."
        )

    def _notes_too_long(
        self,
        call: str,
        estimate: int,
        notes: Sequence[str],
        notes_tokens: int,
        refusal: ContextWindowExceededError | None = None,
    ) -> NotesTooLong:
        """The hand-over of ``call``'s notes: by estimate, with the room they have
        under the line; by refusal, with none."""
        if refusal is None:
            message = self._estimated(call, estimate)
        else:
            message = (
                f"{call} was refused as too long, and the notes it carries (about "
                f"{notes_tokens:,} tokens) are the bigger part of it."
            )
        room = (
            None
            if refusal is not None or self.line is None
            else self.line - (estimate - notes_tokens)
        )
        return NotesTooLong(
            message,
            model=self.model,
            call=call,
            cause="estimate" if refusal is None else "refusal",
            estimate=estimate,
            notes=notes,
            notes_tokens=notes_tokens,
            room=room,
            refusal=refusal,
        )

    @staticmethod
    def _event(
        kind: str, cause: str, estimate: int, rest: int, discarded: int
    ) -> dict[str, Any]:
        return {
            "call": kind,
            "cause": cause,
            "estimate": estimate,
            "rest": rest,
            "discarded": discarded,
        }

    def _record_event(
        self,
        doc: int,
        event: dict[str, Any],
        action: str,
        refusal: ContextWindowExceededError | None,
    ) -> None:
        """Record a split or stop with the parts it leaves and the provider's message,
        if a refusal caused it."""
        n_parts = len(self._cuts[doc]) + 1
        error = None if refusal is None else str(refusal)
        with self._lock:
            entry = self._record["documents"].setdefault(
                str(doc), {"cuts": [], "parts": {}, "events": []}
            )
            entry["cuts"] = list(self._cuts[doc])
            entry["events"].append(
                {**event, "action": action, "parts": n_parts, "error": error}
            )
            self._write()

    def _record_parts(self, doc: int, call: str, n_parts: int) -> None:
        """Record how many parts ``call`` read, for a document already recorded."""
        with self._lock:
            entry = self._record["documents"].get(str(doc))
            if entry is None:
                return
            entry["parts"][call] = n_parts
            self._write()

    def _write(self) -> None:
        if self._task_logger is not None:
            self._task_logger.write_json("splits", self._record)


_BREAKS = (
    re.compile(r"\n\s*\n\s*"),  # paragraph break
    re.compile(r"\n"),  # line break
    re.compile(r"(?<=[.!?])\s+|(?<=[。！？])"),  # sentence end
    re.compile(r"\s+"),  # any space
)


def halve(text: str) -> int:
    """The offset at which ``text`` is cut in two: after the break nearest its middle.

    Each kind of break in ``_BREAKS`` is looked for, in order, in the middle half of
    ``text`` only, so each half is at least a quarter of it; the first kind found wins,
    and the break nearest the middle (the earlier on a tie) is the cut. With no break
    at all in the middle half, the cut is the middle.

    Raises:
        ValueError: for a text of fewer than 2 characters, which cannot be cut.
    """
    n = len(text)
    if n < 2:
        raise ValueError(f"a text of {n} character(s) cannot be cut in two")
    low, high, middle = max(1, n // 4), min(n - 1, n - n // 4), n // 2
    for pattern in _BREAKS:
        cuts = [m.end() for m in pattern.finditer(text) if low <= m.end() <= high]
        if cuts:
            return min(cuts, key=lambda cut: (abs(cut - middle), cut))
    return middle


def _max_input_tokens(model: str) -> int | None:
    """``model``'s input window in litellm's model map, or ``None`` when the map does
    not give one. The lookup prints nothing, whatever the model string."""
    try:
        with quiet_litellm():
            window = litellm.get_model_info(model).get("max_input_tokens")
    except Exception:  # noqa: BLE001 — litellm before 1.104 raises a bare Exception for a model its map lacks
        window = None
    if window is None:
        _log.info(
            "litellm's model map gives no input window for %s; r3con fits a request "
            "to it only when the model refuses one",
            model,
        )
    return window


def _utf8_size(text: str) -> int:
    """``text``'s size in UTF-8 bytes, which no count of its tokens exceeds."""
    return len(text.encode("utf-8"))


def _kind(call: str, doc: int, k: int, n_parts: int) -> str:
    """The kind a call records for part ``k`` of ``n_parts`` of document ``doc``."""
    return f"{call}-d{doc}" if n_parts == 1 else f"{call}-d{doc}c{k}"


def _no_room(rest_tokens: int) -> str:
    return (
        "the part is one character, and the prompt and notes sent with it (about "
        f"{rest_tokens:,} tokens) leave it no room"
    )
