import logging

import pytest

from r3con.logging_setup import configure_logging, get_logger

KEY = "not-a-real-key-q7RkT2mXv9LpA4sW"


def test_loggers_live_under_the_r3con_namespace():
    assert get_logger().name == "r3con"
    assert get_logger("relevance").name == "r3con.relevance"


def test_configuring_twice_attaches_one_handler_at_the_level_asked(monkeypatch):
    logger = logging.getLogger("r3con")
    monkeypatch.setattr(logger, "handlers", [])
    configure_logging("INFO")
    configure_logging("INFO")
    assert len(logger.handlers) == 1
    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_a_stages_records_reach_the_r3con_loggers_handlers(caplog):
    caplog.set_level(logging.INFO, logger="r3con")
    get_logger("relevance").info("hello %d", 3)
    assert [(r.name, r.getMessage()) for r in caplog.records] == [
        ("r3con.relevance", "hello 3")
    ]


@pytest.mark.parametrize(
    ("raw", "level"),
    [
        ("info", logging.INFO),
        ("Debug", logging.DEBUG),
        (" info ", logging.INFO),
        ("", logging.WARNING),
    ],
    ids=["lowercase", "mixed-case", "spaces", "empty"],
)
def test_the_log_level_variable_takes_any_case_and_spacing_and_empty_means_unset(
    monkeypatch, raw, level
):
    monkeypatch.setenv("R3CON_LOG_LEVEL", raw)
    configure_logging()
    assert logging.getLogger("r3con").level == level


def test_a_log_level_variable_that_is_not_a_level_is_refused_naming_it(monkeypatch):
    monkeypatch.setenv("R3CON_LOG_LEVEL", "loud")
    with pytest.raises(ValueError, match="^R3CON_LOG_LEVEL "):
        configure_logging()


def test_a_logged_message_and_its_traceback_have_keys_masked(caplog, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    caplog.set_level(logging.WARNING, logger="r3con")
    try:
        raise RuntimeError(f"echoed {KEY}")
    except RuntimeError as error:
        get_logger("llm").warning("sending again after %s", error, exc_info=True)
    assert "sending again after echoed REDACTED" in caplog.text
    assert "RuntimeError: echoed REDACTED" in caplog.text
    assert KEY not in caplog.text
