import logging

from r3con.logging_setup import configure_logging, get_logger


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
