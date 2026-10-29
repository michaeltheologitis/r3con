import importlib
import sys

import pytest

from r3con import masking
from r3con.masking import _ERROR_CHARS, masked, one_line, register_key

# No shape litellm's own redactor knows: only r3con's masking can hide it.
KEY = "not-a-real-key-q7RkT2mXv9LpA4sW"
ECHOED = f"Incorrect API key provided: {KEY}"
MASKED = "Incorrect API key provided: REDACTED"
OPENAI_SHAPED = "Authorization: Bearer sk-proj-0123456789abcdefghij"


@pytest.mark.parametrize(
    ("text", "keys", "environment", "expected"),
    [
        (ECHOED, [KEY], {}, MASKED),
        (ECHOED, [], {"GROQ_API_KEY": KEY}, MASKED),
        (ECHOED, [], {"HF_TOKEN": KEY}, MASKED),
        (ECHOED, [], {"VERTEX_CLIENT_SECRET": KEY}, MASKED),
        (ECHOED, [], {"OPENAI_BASE_URL": KEY}, ECHOED),
        (
            "api_key EMPTY was refused",
            ["EMPTY"],
            {"HOSTED_VLLM_API_KEY": "EMPTY"},
            "api_key EMPTY was refused",
        ),
        (f"{KEY}-old, then {KEY}", [KEY, f"{KEY}-old"], {}, "REDACTED, then REDACTED"),
        (OPENAI_SHAPED, [], {}, "Authorization: REDACTED"),
    ],
    ids=[
        "given",
        "api-key-variable",
        "token-variable",
        "secret-variable",
        "other-variable-left-alone",
        "placeholder-left-alone",
        "longest-first",
        "litellms-patterns",
    ],
)
def test_masked_replaces_each_known_key_and_what_litellm_takes_for_one(
    monkeypatch, text, keys, environment, expected
):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    assert masked(text, *keys) == expected


def test_a_registered_key_is_masked_from_then_on_and_a_placeholder_never():
    register_key("EMPTY")
    assert masked(f"{ECHOED} EMPTY") == f"{ECHOED} EMPTY"
    register_key(KEY)
    assert masked(f"{ECHOED} EMPTY") == f"{MASKED} EMPTY"


@pytest.mark.parametrize(
    ("error", "line"),
    [
        (ValueError(), "ValueError"),
        (ValueError(" \n "), "ValueError"),
        (ValueError(f"\n{ECHOED}\nmore"), f"ValueError: {MASKED}"),
    ],
    ids=["no-message", "blank-message", "first-non-empty-line"],
)
def test_one_line_is_the_class_and_the_first_line_masked(error, line):
    assert one_line(error, KEY) == line


def test_one_line_masks_before_it_cuts_so_no_part_of_a_key_is_left():
    room = _ERROR_CHARS - len("ValueError: ") - len("REDACTED") - 2
    line = one_line(ValueError("x" * room + KEY), KEY)
    assert line == "ValueError: " + "x" * room + "REDACTED"


@pytest.fixture
def without_litellms_redactor():
    """r3con.masking imported as if litellm had moved its redactor."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "litellm.litellm_core_utils.secret_redaction", None)
        yield importlib.reload(masking)
    importlib.reload(masking)


def test_without_litellms_redactor_known_keys_are_still_masked(
    without_litellms_redactor,
):
    text = f"{OPENAI_SHAPED}; {ECHOED}"
    assert without_litellms_redactor.masked(text, KEY) == f"{OPENAI_SHAPED}; {MASKED}"
