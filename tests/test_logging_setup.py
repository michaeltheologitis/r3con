import logging
import traceback

import pytest

from r3con.logging_setup import configure_logging, get_logger
from r3con.masking import register_key

KEY = "not-a-real-key-q7RkT2mXv9LpA4sW"


class Unprintable:
    """A log argument whose ``str`` raises, and whose ``repr`` raises too unless it is
    given one."""

    def __init__(self, shown: str | None = None) -> None:
        self.shown = shown

    def __str__(self) -> str:
        raise RuntimeError("no str")

    def __repr__(self) -> str:
        if self.shown is None:
            raise RuntimeError("no repr")
        return self.shown


class OwnTracebacks(logging.Handler):
    """Renders a record's ``exc_info`` itself, as rich, structlog or Sentry do, and
    the text a formatter would append only when there is none."""

    def __init__(self) -> None:
        super().__init__()
        self.shown: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.exc_info:
            self.shown.append("".join(traceback.format_exception(*record.exc_info)))
        else:
            self.shown.append(record.exc_text or "")


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


def test_a_handler_that_renders_tracebacks_itself_sees_the_chained_error_masked(
    monkeypatch,
):
    register_key(KEY)
    handler = OwnTracebacks()
    logger = get_logger("llm")
    monkeypatch.setattr(logger, "handlers", [handler])
    try:
        try:
            raise RuntimeError(f"inner {KEY}")
        except RuntimeError as inner:
            raise ValueError("outer") from inner
    except ValueError:
        logger.exception("gave up")
    [shown] = handler.shown
    assert "RuntimeError: inner REDACTED" in shown
    assert KEY not in shown


def test_a_logged_stack_is_masked(caplog):
    register_key(KEY)
    caplog.set_level(logging.WARNING, logger="r3con")
    get_logger("llm").warning("not-a-real-key-q7RkT2mXv9LpA4sW", stack_info=True)
    assert "Stack (most recent call last)" in caplog.text
    assert KEY not in caplog.text


@pytest.mark.parametrize(
    ("message", "argument", "logged"),
    [
        ("doc %d", "seven", "doc %d ('seven',)"),
        ("doc %s", Unprintable("Unprintable()"), "doc %s (Unprintable(),)"),
        ("doc %s", Unprintable(), "doc %s <unprintable args>"),
    ],
    ids=["wrong-type", "str-raises", "str-and-repr-raise"],
)
def test_a_log_call_whose_message_does_not_format_returns_and_logs_its_parts(
    caplog, message, argument, logged
):
    caplog.set_level(logging.WARNING, logger="r3con")
    get_logger("llm").warning(message, argument)
    assert caplog.messages == [logged]
