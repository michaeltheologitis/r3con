import pytest
from pydantic import BaseModel, ValidationError

from r3con.masking import _ERROR_CHARS, masked, one_line, register_key

KEY = "not-a-real-key-q7RkT2mXv9LpA4sW"
ECHOED = f"Incorrect API key provided: {KEY}"
MASKED = "Incorrect API key provided: REDACTED"
# Each would be cut by a pattern without word boundaries, such as litellm's sk-[…]{13,}.
LOOKS_LIKE_A_KEY = "/home/u/risk-assessment-runs/logs/desk-reservations_2026_q3.csv"


class Parse(BaseModel):
    rows: list[int]
    name: str


def invalid(reply: str) -> ValidationError:
    try:
        Parse.model_validate_json(reply)
    except ValidationError as error:
        return error
    raise AssertionError(f"{reply!r} fits Parse")


@pytest.mark.parametrize(
    ("name", "value", "secret"),
    [
        ("OPENAI_API_KEY", KEY, True),
        ("HOSTED_VLLM_API_KEY", "vllm-key-123", True),
        ("WATSONX_APIKEY", KEY, True),
        ("AZURE_PASSWORD", KEY, True),
        ("AWS_BEARER_TOKEN_BEDROCK", KEY, True),
        ("GIGACHAT_CREDENTIALS", KEY, True),
        ("VERTEX_CLIENT_SECRET", KEY, True),
        ("GITHUB_TOKEN", "short-tok-12", False),
        ("SORT_KEY", "created_at_descending", False),
        ("S3_OBJECT_KEY", "documents-2026-q3-report", False),
        ("GOOGLE_APPLICATION_CREDENTIALS", "/home/u/service-account.json", False),
        ("HF_TOKEN_PATH", "~/.cache/huggingface/token", False),
        ("SLACK_TOKEN_LABEL", "the build server's token", False),
        ("OPENAI_BASE_URL", KEY, False),
    ],
    ids=[
        "api-key",
        "short-provider-key",
        "apikey",
        "password",
        "token",
        "credentials",
        "secret",
        "short-token",
        "bare-key",
        "bare-key-long",
        "credentials-path",
        "token-path",
        "whitespace",
        "not-a-credential",
    ],
)
def test_an_environment_value_is_masked_when_its_name_and_shape_say_secret(
    monkeypatch, name, value, secret
):
    monkeypatch.setenv(name, value)
    text = f"refused for {value}, see logs"
    assert masked(text) == ("refused for REDACTED, see logs" if secret else text)


def test_a_registered_key_is_masked_from_then_on_the_longest_first_but_a_placeholder():
    text = f"{KEY}-old, then {KEY}, sent as EMPTY to {LOOKS_LIKE_A_KEY}"
    assert masked(text) == text
    for key in ("EMPTY", KEY, f"{KEY}-old"):
        register_key(key)
    assert (
        masked(text) == f"REDACTED, then REDACTED, sent as EMPTY to {LOOKS_LIKE_A_KEY}"
    )


@pytest.mark.parametrize(
    ("error", "line"),
    [
        (ValueError(), "ValueError"),
        (ValueError(" \n "), "ValueError"),
        (
            ValueError("\nrefused\r\nTraceback (most recent call last):\n  File"),
            "ValueError: refused Traceback (most recent call last): File",
        ),
        (
            ValueError(f"the provider refused the request:\n\n{ECHOED}"),
            f"ValueError: the provider refused the request: {MASKED}",
        ),
        (
            invalid('{"rows": ["x"]}'),
            (
                "ValidationError: rows.0: Input should be a valid integer, unable to "
                "parse string as an integer; name: Field required"
            ),
        ),
        (
            invalid('{"rows": ['),
            "ValidationError: Invalid JSON: EOF while parsing a list at line 1 column 10",
        ),
    ],
    ids=[
        "no-message",
        "blank-message",
        "crlf-and-blank-lines",
        "detail-on-a-later-line",
        "each-validation-error",
        "invalid-json",
    ],
)
def test_one_line_is_the_class_and_the_whole_message_masked(error, line):
    register_key(KEY)
    assert one_line(error) == line


def test_one_line_masks_before_it_cuts_so_no_part_of_a_key_is_left():
    register_key(KEY)
    room = _ERROR_CHARS - len("ValueError: ") - len("REDACTED") - 2
    line = one_line(ValueError("x" * room + KEY))
    assert line == "ValueError: " + "x" * room + "REDACTED"
