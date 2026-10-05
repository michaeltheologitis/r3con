"""Reading a document too long for the model's window in parts.

A stage sends each document whole, as the user message of one call. When that request
does not fit the model's input window, the document is cut in 2 at the break nearest its
middle, then every part in 2 again, and so on, until every part's request fits. The parts
never overlap and always rejoin to the document, and the caller's numbering never moves:
only the relevant context's headings say ``Document N.1``, ``N.2``.

Two things say a request does not fit:

- **The estimate**, when litellm's model map gives the model's input window: a request
  estimated over the window less ``settings.WINDOW_MARGIN_PERCENT`` is split before it
  is sent. The estimate counts cl100k_base tokens, from the vocabulary litellm ships.
- **A refusal**: the provider raises ``litellm.ContextWindowExceededError``. It splits
  whatever the estimate said.

Splitting stops when more parts cannot help, because what is sent beside the document
(the prompt and the other documents' notes) has outgrown the window. The run then
raises ``litellm.ContextWindowExceededError`` with a note naming the document.
"""

from __future__ import annotations

import functools
import itertools
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any, NoReturn, TypeVar

import litellm
from litellm.exceptions import ContextWindowExceededError

from r3con import settings
from r3con.logging_setup import get_logger
from r3con.runs import TaskLogger
from r3con.runtime.llm import count_tokens, quiet_litellm

_log = get_logger("splitting")

T = TypeVar("T")

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
            "litellm's model map gives no input window for %s; a document is split "
            "only when the model refuses it",
            model,
        )
    return window


def _utf8_size(text: str) -> int:
    """``text``'s size in UTF-8 bytes, which no count of its tokens exceeds."""
    return len(text.encode("utf-8"))


def _kind(call: str, doc: int, k: int, n_parts: int) -> str:
    """The kind a call records for part ``k`` of ``n_parts`` of document ``doc``."""
    return f"{call}-d{doc}" if n_parts == 1 else f"{call}-d{doc}c{k}"


class Splits:
    """Where each document of one run is cut into parts, kept for the rest of the run.

    The window is ``model``'s ``max_input_tokens`` in litellm's model map, looked up
    once; ``line`` is that window less ``margin_percent``. Both are ``None`` when the
    map does not know the model. With a ``task_logger``, every split and stop is
    written to ``splits.json`` in the run folder as it happens.

    The documents of a stage are read in parallel: each document's cuts are touched
    only by its own worker, and the record and its file are written under a lock.
    """

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
        self._model = model
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

    def read_in_parts(
        self, doc: int, *, call: str, rest: str, send: Callable[[str, str], T]
    ) -> list[T]:
        """One result of ``send(part, kind)`` per part of ``documents[doc]``, in order,
        after splitting it as far as it must.

        ``call`` names the call (``"relevance-r2"``, ``"parse"``); a part's ``kind`` is
        ``{call}-d{doc}`` while the document is whole and ``{call}-d{doc}c{k}`` once it
        is split. ``rest`` is everything sent beside a part. ``send`` raises
        ``litellm.ContextWindowExceededError`` when the provider refuses a part; any
        other exception passes through untouched.

        With a known window: while the rest is under the line, halve until every part's
        estimate is under it, then send; a refusal halves again and starts over. With an
        unknown window: send, and halve on a refusal. Either way the parts accepted
        before a refusal are discarded and the level is read again.

        Raises:
            litellm.ContextWindowExceededError: with a note naming the document, when
                more parts cannot help: the rest alone is over the line; with an
                unknown window, a refused part is shorter than the rest; or a part of
                one character still does not fit.
        """
        count_rest = functools.cache(functools.partial(count_tokens, rest))
        while True:
            if self.line is not None:
                self._split_to_fit(doc, call, rest, count_rest, self.line)
            parts = self.parts(doc)
            results: list[T] = []
            for k, part in enumerate(parts):
                kind = _kind(call, doc, k, len(parts))
                try:
                    results.append(send(part, kind))
                except ContextWindowExceededError as error:
                    self._split_on_refusal(doc, kind, k, part, count_rest, error)
                    break
            else:
                self._record_parts(doc, call, len(parts))
                return results

    def _split_to_fit(
        self,
        doc: int,
        call: str,
        rest: str,
        count_rest: Callable[[], int],
        line: int,
    ) -> None:
        """Before sending, with a known window: stop if the rest alone is over the
        line; otherwise halve until every part's request is estimated under it."""
        rest_size = _utf8_size(rest)
        if rest_size > line and count_rest() > line:
            parts = self.parts(doc)
            kind = _kind(call, doc, 0, len(parts))
            estimate = count_rest() + count_tokens(parts[0])
            self._stop(
                doc,
                self._not_sent(kind, estimate),
                self._event(kind, "estimate", estimate, count_rest(), 0),
                error=None,
                clause=(
                    f"the prompt and notes sent with it are about {count_rest():,} "
                    f"tokens, over the {line:,}-token line by themselves"
                ),
            )
        while (over := self._first_over(doc, rest_size, count_rest, line)) is not None:
            k, estimate = over
            parts = self.parts(doc)
            kind = _kind(call, doc, k, len(parts))
            event = self._event(kind, "estimate", estimate, count_rest(), 0)
            if len(parts[k]) == 1:
                stop = self._not_sent(kind, estimate)
                self._stop(doc, stop, event, error=None, clause=_no_room(count_rest()))
            n_parts = self._split(doc, event, error=None)
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
        error: ContextWindowExceededError,
    ) -> None:
        """Halve after the provider refused part ``k``, or stop where more parts cannot
        help: a part of one character, or, with an unknown window, a part shorter than
        the rest."""
        part_tokens = count_tokens(part)
        event = self._event(
            kind, "refusal", count_rest() + part_tokens, count_rest(), k
        )
        if self.line is None and part_tokens < count_rest():
            self._stop(
                doc,
                error,
                event,
                error=str(error),
                clause=(
                    f"the part is about {part_tokens:,} tokens and the prompt and "
                    f"notes sent with it about {count_rest():,}"
                ),
            )
        if len(part) == 1:
            self._stop(
                doc, error, event, error=str(error), clause=_no_room(count_rest())
            )
        n_parts = self._split(doc, event, error=str(error))
        _log.warning(
            "%s: documents[%d] (Document %d) was refused as too long; reading it in %d "
            "parts",
            kind,
            doc,
            doc + 1,
            n_parts,
        )

    def _split(self, doc: int, event: dict[str, Any], *, error: str | None) -> int:
        """Halve every part of ``documents[doc]`` longer than one character, record the
        split with the provider's ``error`` message, if any, and return the number of
        parts."""
        starts = [0, *self._cuts[doc]]
        cuts = [
            start + halve(part)
            for start, part in zip(starts, self.parts(doc))
            if len(part) > 1
        ]
        self._cuts[doc] = sorted([*self._cuts[doc], *cuts])
        n_parts = len(self._cuts[doc]) + 1
        self._record_event(
            doc, {**event, "action": "split", "parts": n_parts, "error": error}
        )
        return n_parts

    def _stop(
        self,
        doc: int,
        stop: ContextWindowExceededError,
        event: dict[str, Any],
        *,
        error: str | None,
        clause: str,
    ) -> NoReturn:
        """Record the stop with the provider's ``error`` message, if any, add the note
        that says why more parts cannot help (``clause``), and raise ``stop``."""
        n_parts = len(self.parts(doc))
        self._record_event(
            doc, {**event, "action": "stop", "parts": n_parts, "error": error}
        )
        stop.add_note(
            f"r3con: reading documents[{doc}] (Document {doc + 1}) in more parts cannot "
            f"help in {event['call']}: {clause}. The relevant context has outgrown the "
            "model's window."
        )
        raise stop

    def _not_sent(self, kind: str, estimate: int) -> ContextWindowExceededError:
        """The error for a request r3con stops before sending it."""
        return ContextWindowExceededError(
            message=(
                f"r3con estimated {kind} at {estimate:,} tokens, over the "
                f"{self.line:,}-token line (the {self.max_input_tokens:,}-token input "
                f"window litellm's model map gives {self._model}, less "
                f"{self.margin_percent}%); it was not sent."
            ),
            model=self._model,
            llm_provider="r3con",
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

    def _record_event(self, doc: int, event: dict[str, Any]) -> None:
        with self._lock:
            entry = self._record["documents"].setdefault(
                str(doc), {"cuts": [], "parts": {}, "events": []}
            )
            entry["cuts"] = list(self._cuts[doc])
            entry["events"].append(event)
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


def _no_room(rest_tokens: int) -> str:
    return (
        "the part is one character, and the prompt and notes sent with it (about "
        f"{rest_tokens:,} tokens) leave it no room"
    )
