"""Tests for `r3con.logging_setup` — the standard logger config.

Run with:  uv run python tests/test_logging_setup.py
"""

from __future__ import annotations

import io
import logging

from r3con.logging_setup import configure_logging, get_logger


def test_get_logger_is_r3con_namespaced() -> None:
    assert get_logger().name == "r3con"
    assert get_logger("relevance").name == "r3con.relevance"


def test_configure_is_idempotent_and_sets_level() -> None:
    logger = logging.getLogger("r3con")
    # Clean slate for the test.
    for h in list(logger.handlers):
        logger.removeHandler(h)
    configure_logging("INFO")
    configure_logging("INFO")  # second call must not add a second handler
    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO


def test_child_logs_reach_a_handler_at_info() -> None:
    """A child logger's INFO record reaches a handler on `grounded` (propagation)."""
    parent = get_logger()
    child = get_logger("relevance")
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    old = parent.level
    parent.addHandler(handler)
    parent.setLevel(logging.INFO)
    try:
        child.info("hello %d", 3)
    finally:
        parent.removeHandler(handler)
        parent.setLevel(old)
    assert "hello 3" in buf.getvalue()


if __name__ == "__main__":
    tests = [
        test_get_logger_is_r3con_namespaced,
        test_configure_is_idempotent_and_sets_level,
        test_child_logs_reach_a_handler_at_info,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
