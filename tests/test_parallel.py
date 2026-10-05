import time

import pytest

from r3con.parallel import parallel_map


@pytest.mark.parametrize(
    ("items", "max_workers", "results"),
    [
        ([], 4, []),
        ([1, 2, 3], 1, [(0, 1), (1, 2), (2, 3)]),
        (["a", "b", "c"], 2, [(0, "a"), (1, "b"), (2, "c")]),
        ([5], 8, [(0, 5)]),
    ],
)
def test_each_item_is_mapped_with_its_index_in_input_order(items, max_workers, results):
    assert parallel_map(lambda i, x: (i, x), items, max_workers=max_workers) == results


def test_results_keep_input_order_when_later_items_finish_first():
    def slow_first(i: int, x: int) -> int:
        time.sleep(0.01 * (3 - i))
        return x

    assert parallel_map(slow_first, [0, 1, 2], max_workers=3) == [0, 1, 2]


def test_an_items_exception_is_raised_from_the_map():
    def boom(i: int, x: int) -> int:
        if x == 2:
            raise RuntimeError("kaboom")
        return x

    with pytest.raises(RuntimeError, match="kaboom"):
        parallel_map(boom, [1, 2, 3], max_workers=3)
