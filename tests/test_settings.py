import re

import pytest

from r3con import settings
from r3con.settings import settings_snapshot


def test_the_snapshot_holds_each_cap_as_set_when_it_is_taken(monkeypatch):
    monkeypatch.setattr(settings, "SCHEMA_MAX_ATTEMPTS", 2)
    monkeypatch.setenv("R3CON_DOC_WORKERS", "3")
    snapshot = settings_snapshot()
    assert (snapshot["schema_max_attempts"], snapshot["doc_workers"]) == (2, 3)
    assert settings_snapshot(reasoning_max_turns=4)["reasoning_max_turns"] == 4
    assert list(snapshot) == [
        "doc_workers",
        "reasoning_max_turns",
        "schema_max_attempts",
        "parsing_max_attempts",
        "llm_num_retries",
        "llm_empty_content_retries",
        "reasoning_parse_max_toks",
        "window_margin_percent",
    ]


@pytest.mark.parametrize(
    ("name", "value", "floor"),
    [
        ("DOC_WORKERS", 0, 1),
        ("REASONING_MAX_TURNS", 0, 1),
        ("SCHEMA_MAX_ATTEMPTS", 0, 1),
        ("PARSING_MAX_ATTEMPTS", 0, 1),
        ("LLM_NUM_RETRIES", -1, 0),
        ("LLM_EMPTY_CONTENT_RETRIES", -1, 0),
        ("REASONING_PARSE_MAX_TOKS", -1, 0),
        ("WINDOW_MARGIN_PERCENT", -1, 0),
    ],
)
def test_a_cap_below_its_floor_is_refused(monkeypatch, name, value, floor):
    monkeypatch.setattr(settings, name, value)
    with pytest.raises(ValueError, match=f"^{name} must be >= {floor}, got {value}.$"):
        settings_snapshot()


def test_a_margin_above_its_ceiling_is_refused(monkeypatch):
    monkeypatch.setattr(settings, "WINDOW_MARGIN_PERCENT", 100)
    message = "WINDOW_MARGIN_PERCENT must be <= 99, got 100."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        settings_snapshot()


@pytest.mark.parametrize("value", [2.5, True, "3"], ids=["float", "bool", "str"])
def test_a_cap_that_is_not_an_integer_is_refused(monkeypatch, value):
    monkeypatch.setattr(settings, "SCHEMA_MAX_ATTEMPTS", value)
    message = f"SCHEMA_MAX_ATTEMPTS must be an integer >= 1, got {value!r}."
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        settings_snapshot()


def test_an_explicit_turn_cap_below_its_floor_is_refused():
    with pytest.raises(ValueError, match="REASONING_MAX_TURNS must be >= 1, got 0."):
        settings_snapshot(reasoning_max_turns=0)
