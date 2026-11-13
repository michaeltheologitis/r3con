"""Bounded, order-preserving parallel map for per-document fan-out.

The concurrency *budget* (``R3CON_DOC_WORKERS``) lives in
:func:`r3con.settings.active_doc_workers`; callers pass the resolved number in as
``max_workers``, so a caller running several tasks concurrently keeps tasks × doc
fan-out under the served endpoint's concurrency cap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    func: Callable[[int, T], R],
    items: Sequence[T],
    *,
    max_workers: int,
    alone: int | None = None,
) -> list[R]:
    """Apply ``func(index, item)`` across ``items`` concurrently; return results in
    **input order**.

    - With ``max_workers <= 1`` or a single item, the calls run one after another in
      the calling thread, so an interrupt reaches the call directly.
    - Otherwise up to ``min(max_workers, len(items))`` calls run at once.
    - With ``alone``, an index, that item's call runs in the calling thread before any
      other starts, and the rest are mapped as above once it returns.
    - If calls raise, the lowest-indexed failure propagates once every call already
      started has finished, and calls not yet started are cancelled.

    ``func`` receives the item's index so a caller can tag its work (e.g. the
    log ``kind`` per document) without threading position through the result.
    """
    if alone is not None:
        first = func(alone, items[alone])
        others = [i for i in range(len(items)) if i != alone]
        rest = parallel_map(
            lambda k, item: func(others[k], item),
            [items[i] for i in others],
            max_workers=max_workers,
        )
        return [*rest[:alone], first, *rest[alone:]]
    if max_workers <= 1 or len(items) <= 1:
        return [func(i, item) for i, item in enumerate(items)]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as ex:
        return list(ex.map(func, range(len(items)), items))
