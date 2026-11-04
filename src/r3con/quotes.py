"""Finding a reply's quotes in the text its call read.

A quote is located by the first of four tiers that finds it. The first three find
it verbatim: as the model copied it, but for what normalizing leaves out and what an
ellipsis stands for:

- ``exact``: it is a substring of the text.
- ``normalized``: it is one once both are normalized alike (:func:`_normalize`), the
  quote without the spaces at its ends or the quotation marks around it.
- ``elided``: an ellipsis (``...``, ``…``, ``. . .``, or one in brackets, ``[...]``
  or ``(...)``) cuts it into fragments, and each fragment of at least
  :data:`MIN_ELIDED_FRAGMENT_CHARS` characters is found, in order, exactly or
  normalized; the shorter ones are ignored. The last fragment is placed as early as it
  can be, each one before it as late as it can be, and the span runs from the first
  fragment's start to the last one's end.

The fourth does not find it verbatim:

- ``fuzzy``: the model altered the text it quotes, a word changed or a table cell left
  out, say, and the quote, normalized, is only like the normalized text. The window of
  the text most like it (:func:`rapidfuzz.fuzz.partial_ratio_alignment`) is taken when
  it scores at least :data:`FUZZY_MIN_SCORE` of 100, and its ``score`` is kept. It
  locates the sentence the quote was taken from, for highlighting; it does not say
  the quote is there.

Normalizing reads a text one character at a time in NFKD, case-folded, with curly
quotes straight and dashes ``-``. Markup is ignored: an HTML tag or a ``|`` reads as a
space, and ``$``, ``*``, backticks and format characters, such as a soft hyphen, are
left out. Each run of whitespace is then one space. Every character it gives keeps the
position of the one it came from, so however a quote was found, its position is a span
of the original text. A quote no tier finds is kept with no position, as is one only
loosely like the text.
"""

from __future__ import annotations

import bisect
import functools
import re
import unicodedata
from collections.abc import Callable, Iterable
from typing import Literal, NamedTuple, TypedDict

from rapidfuzz import fuzz

MIN_ELIDED_FRAGMENT_CHARS = 10
FUZZY_MIN_SCORE = 90
_SCORE_DIGITS = 1

Match = Literal["exact", "normalized", "elided", "fuzzy"]
_Span = tuple[int, int]

_DOTS = r"(?:\.(?: ?\.){2,}|…)"
_ELLIPSIS = re.compile(rf"\[\s*{_DOTS}\s*\]|\(\s*{_DOTS}\s*\)|{_DOTS}")
# An HTML tag starts with a letter, so that "x < 5 and y > 3" holds none.
_UNIT = re.compile(r"(?P<tag></?[A-Za-z][^<>]*>)|.", re.DOTALL)
_UNIFIED = str.maketrans("‘’‚‛“”„‟‐‑‒–—−|", "''''\"\"\"\"------ ", "$*`")
_QUOTE_MARKS = "\"'“”‘’"
_FORMAT = "Cf"


class Quote(TypedDict):
    """A quote and where it is in its document: ``start`` and ``end`` as a slice, and
    the tier that found it, all ``None`` when no tier did; and for a ``fuzzy`` one
    only, its ``score``, to one decimal."""

    text: str
    start: int | None
    end: int | None
    match: Match | None
    score: float | None


class _Normal(NamedTuple):
    """A text normalized, and where in the original each of its characters came
    from."""

    text: str
    origins: list[int]


def locate_quotes(quotes: Iterable[str], text: str, offset: int = 0) -> list[Quote]:
    """Each of ``quotes`` that is not blank, in order, located in ``text`` by the first
    tier that finds it, its position shifted by ``offset``: where ``text`` starts in
    its document, for a part of one."""
    normal = functools.cache(functools.partial(_normalize, text))
    located: list[Quote] = []
    for quote in filter(str.strip, quotes):
        span, match, score = _locate(quote, text, normal) or (None, None, None)
        located.append(
            Quote(
                text=quote,
                start=None if span is None else span[0] + offset,
                end=None if span is None else span[1] + offset,
                match=match,
                score=score,
            )
        )
    return located


def _locate(
    quote: str, text: str, normal: Callable[[], _Normal]
) -> tuple[_Span, Match, float | None] | None:
    """Where ``quote`` is in ``text``, the first tier that finds it, and its score for
    a ``fuzzy`` one."""
    span = _exactly(quote, text)
    if span is not None:
        return span, "exact", None
    inner = _unquoted(quote)
    span = _normalized(inner, normal())
    if span is not None:
        return span, "normalized", None
    span = _elided(inner, text, normal())
    if span is not None:
        return span, "elided", None
    fuzzy = _fuzzy(inner, normal())
    return None if fuzzy is None else (fuzzy[0], "fuzzy", fuzzy[1])


def _elided(quote: str, text: str, normal: _Normal) -> _Span | None:
    """Where ``quote``'s fragments between its ellipses are in ``text``, in order,
    those too short to place ignored: the last at its first place after the others,
    and each one before it at its last place before the next."""
    pieces = _ELLIPSIS.split(quote)
    fragments = [
        f for f in map(str.strip, pieces) if len(f) >= MIN_ELIDED_FRAGMENT_CHARS
    ]
    if len(pieces) == 1 or not fragments:
        return None
    last: _Span = (0, 0)
    for fragment in fragments:
        found = _nearest(fragment, text, normal, after=last[1])
        if found is None:
            return None
        last = found
    start = last[0]
    for fragment in fragments[-2::-1]:
        found = _nearest(fragment, text, normal, before=start, last=True)
        # The forward pass placed this fragment before ``start``, so it is found.
        assert found is not None
        start = found[0]
    return start, last[1]


def _nearest(
    piece: str,
    text: str,
    normal: _Normal,
    *,
    after: int = 0,
    before: int | None = None,
    last: bool = False,
) -> _Span | None:
    """The first place ``piece`` is in ``text`` from ``after`` on, or with ``last``,
    the last place it ends by ``before``: exactly or normalized, whichever is
    nearer."""
    hits = [
        hit
        for hit in (
            _exactly(piece, text, after, before, last=last),
            _normalized(piece, normal, after, before, last=last),
        )
        if hit is not None
    ]
    if not hits:
        return None
    return max(hits) if last else min(hits)


def _exactly(
    piece: str,
    text: str,
    after: int = 0,
    before: int | None = None,
    *,
    last: bool = False,
) -> _Span | None:
    """Where ``piece`` is in ``text`` between ``after`` and ``before``: the first
    place, or with ``last``, the last."""
    find = text.rfind if last else text.find
    at = find(piece, after, len(text) if before is None else before)
    return None if at < 0 else (at, at + len(piece))


def _normalized(
    piece: str,
    normal: _Normal,
    after: int = 0,
    before: int | None = None,
    *,
    last: bool = False,
) -> _Span | None:
    """Where ``piece``, normalized and without the spaces at its ends, is in
    ``normal`` between ``after`` and ``before`` in the original text: the first place,
    or with ``last``, the last."""
    wanted = _normalize(piece).text.strip()
    origins = normal.origins
    low = bisect.bisect_left(origins, after)
    high = len(origins) if before is None else bisect.bisect_left(origins, before)
    find = normal.text.rfind if last else normal.text.find
    at = find(wanted, low, high)
    if not wanted or at < 0:
        return None
    return origins[at], origins[at + len(wanted) - 1] + 1


def _fuzzy(quote: str, normal: _Normal) -> tuple[_Span, float] | None:
    """Where the window of ``normal`` most like ``quote``, normalized, is in the
    original text, without the spaces at its ends, and its score, to one decimal: how
    like the quote that window is (:func:`rapidfuzz.fuzz.ratio`), when that is at
    least :data:`FUZZY_MIN_SCORE`. Rescoring the window keeps a quote longer than the
    text, which only holds all of it, from scoring 100."""
    wanted = _normalize(quote).text.strip()
    if not wanted:
        return None
    found = fuzz.partial_ratio_alignment(
        wanted, normal.text, score_cutoff=FUZZY_MIN_SCORE
    )
    if found is None:
        return None
    window = normal.text[found.dest_start : found.dest_end]
    start = found.dest_start + len(window) - len(window.lstrip())
    window = window.strip()
    score = fuzz.ratio(wanted, window, score_cutoff=FUZZY_MIN_SCORE)
    if not score:
        return None
    span = normal.origins[start], normal.origins[start + len(window) - 1] + 1
    return span, round(score, _SCORE_DIGITS)


def _unquoted(quote: str) -> str:
    """``quote`` without the spaces at its ends or the quotation marks around it."""
    inner = quote.strip()
    if len(inner) > 1 and inner[0] in _QUOTE_MARKS and inner[-1] in _QUOTE_MARKS:
        return inner[1:-1]
    return inner


def _normalize(text: str) -> _Normal:
    """``text`` normalized as the module says, each character it gives with the
    position of the one, or the HTML tag, it came from."""
    chars: list[str] = []
    origins: list[int] = []
    for unit in _UNIT.finditer(text):
        piece = " " if unit["tag"] else unicodedata.normalize("NFKD", unit[0])
        for char in piece.casefold().translate(_UNIFIED):
            spaced = char.isspace() and chars[-1:] == [" "]
            if spaced or unicodedata.category(char) == _FORMAT:
                continue
            chars.append(" " if char.isspace() else char)
            origins.append(unit.start())
    return _Normal("".join(chars), origins)
