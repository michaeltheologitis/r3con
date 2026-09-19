"""Bounded, order-preserving parallel map for per-document fan-out.

The pipeline runs the same LLM call across *every document in a collection in
parallel* — once per document per round while surfacing relevance (stage 1),
and once per document while parsing (stage 2). Both want the same thing: run up
to ``max_workers`` calls concurrently, preserve input order in the results, and
surface the first exception. This is that one helper.

It is deliberately tiny and generic (no pipeline types) so both stages share one
code path. The concurrency *budget* (``R3CON_DOC_WORKERS``) lives in
:func:`r3con.settings.active_doc_workers`; callers pass the resolved number
in as ``max_workers`` so a caller running several tasks concurrently keeps
tasks × doc fan-out under the served endpoint's concurrency cap.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def parallel_map(
    func: Callable[[int, T], R], items: Sequence[T], *, max_workers: int
) -> list[R]:
    """Apply ``func(index, item)`` across ``items`` concurrently; return results in
    **input order**.

    - ``max_workers <= 1`` (or a single item) runs sequentially with no thread
      pool — the deterministic path used by tests and the degenerate one-doc case.
    - Otherwise up to ``min(max_workers, len(items))`` calls run at once.
    - The exception from the lowest-indexed failing call propagates out (every
      call is submitted up front, so the pool's context manager still runs and
      waits for all of them before the exception surfaces).

    ``func`` receives the item's index so a caller can tag its work (e.g. the
    log ``kind`` per document) without threading position through the result.
    """
    n = len(items)
    if n == 0:
        return []
    if max_workers <= 1 or n == 1:
        return [func(i, item) for i, item in enumerate(items)]

    results: list[R] = [None] * n  # type: ignore[list-item]
    with ThreadPoolExecutor(max_workers=min(max_workers, n)) as ex:
        futures = {ex.submit(func, i, item): i for i, item in enumerate(items)}
        for fut, idx in futures.items():
            results[idx] = fut.result()
    return results
