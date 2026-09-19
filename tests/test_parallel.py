"""Tests for ``r3con.parallel.parallel_map`` — the bounded, order-preserving
per-document fan-out helper.

Run with:  uv run python tests/test_parallel.py
"""

from __future__ import annotations

import time

from r3con.parallel import parallel_map


def test_empty_returns_empty() -> None:
    assert parallel_map(lambda i, x: x, [], max_workers=4) == []


def test_preserves_input_order_sequential() -> None:
    out = parallel_map(lambda i, x: x * 2, [1, 2, 3], max_workers=1)
    assert out == [2, 4, 6]


def test_passes_index() -> None:
    out = parallel_map(lambda i, x: (i, x), ["a", "b", "c"], max_workers=2)
    assert out == [(0, "a"), (1, "b"), (2, "c")]


def test_single_item_runs_without_pool() -> None:
    # max_workers > 1 but only one item → the sequential path; still correct.
    out = parallel_map(lambda i, x: x + 10, [5], max_workers=8)
    assert out == [15]


def test_order_preserved_under_out_of_order_completion() -> None:
    """Later items finish first (reverse sleep), yet results stay in input order."""
    def slow(i: int, x: int) -> int:
        time.sleep(0.01 * (3 - i))  # item 0 sleeps longest → completes last
        return x

    out = parallel_map(slow, [0, 1, 2], max_workers=3)
    assert out == [0, 1, 2]


def test_exception_propagates() -> None:
    def boom(i: int, x: int) -> int:
        if x == 2:
            raise RuntimeError("kaboom")
        return x

    try:
        parallel_map(boom, [1, 2, 3], max_workers=3)
    except RuntimeError as e:
        assert "kaboom" in str(e)
    else:
        raise AssertionError("expected RuntimeError to propagate")


if __name__ == "__main__":
    tests = [
        test_empty_returns_empty,
        test_preserves_input_order_sequential,
        test_passes_index,
        test_single_item_runs_without_pool,
        test_order_preserved_under_out_of_order_completion,
        test_exception_propagates,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
