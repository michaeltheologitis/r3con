import pwd
from datetime import UTC, date, datetime, time

import pytest
from pydantic import BaseModel, ValidationError

from r3con.masking import (
    _ERROR_CHARS,
    is_credential_name,
    masked,
    one_line,
    plain,
    register_key,
)

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
        ("OCI_KEY", KEY, True),
        ("SLACK_WEBHOOK_URL", f"https://hooks.slack.com/services/{KEY}", True),
        ("AWS_SECRET_ACCESS_KEY", f"/{KEY}", True),
        ("GITHUB_TOKEN", "short-tok-12", False),
        ("SORT_KEY", "created_at_descending", False),
        ("S3_OBJECT_KEY", "documents-2026-q3-report", False),
        ("SLACK_TOKEN_LABEL", "the build server's token", False),
        ("OPENAI_BASE_URL", KEY, False),
        ("GIT_AUTHOR_EMAIL", "michael@memos.example.com", False),
    ],
    ids=[
        "api-key",
        "short-provider-key",
        "apikey",
        "password",
        "token",
        "credentials",
        "secret",
        "oci-key",
        "webhook-url",
        "base64-key-starting-with-a-slash",
        "short-token",
        "bare-key",
        "bare-key-long",
        "whitespace",
        "not-a-credential",
        "author",
    ],
)
def test_an_environment_value_is_masked_when_its_name_and_shape_say_secret(
    monkeypatch, name, value, secret
):
    monkeypatch.setenv(name, value)
    text = f"refused for {value}, see logs"
    assert masked(text) == ("refused for REDACTED, see logs" if secret else text)


def test_a_credential_named_variable_holding_an_existing_folder_is_not_masked(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    text = f"{tmp_path / 'memo.txt'}: resolves outside {tmp_path}; skipping it"
    assert masked(text) == text


def test_a_credential_named_variable_holding_an_existing_file_is_not_masked(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("HOME", str(tmp_path))
    credentials = tmp_path / "service-account.json"
    token = tmp_path / ".cache" / "huggingface" / "token"
    token.parent.mkdir(parents=True)
    credentials.touch()
    token.touch()
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials))
    monkeypatch.setenv("HF_TOKEN_PATH", "~/.cache/huggingface/token")
    text = f"could not read {credentials} or ~/.cache/huggingface/token"
    assert masked(text) == text


def test_a_credential_starting_with_a_tilde_and_a_name_is_masked_with_no_user_lookup(
    monkeypatch,
):
    def lookup(name: str) -> None:
        pytest.fail(f"looked the user {name!r} up")

    monkeypatch.setattr(pwd, "getpwnam", lookup)
    monkeypatch.setenv("MEMO_READER_TOKEN", "~somename/memo-reader-2026")
    assert masked("refused ~somename/memo-reader-2026") == "refused REDACTED"


# Each name, and whether it says that what it holds is a credential.
CREDENTIAL_NAMES = {
    "api_key": True,
    "OPENAI_API_KEY": True,
    "WATSONX_APIKEY": True,
    "oci_key": True,
    "azure_ad_token": True,
    "langfuse_secret_key": True,
    "AZURE_PASSWORD": True,
    "vertex_credentials": True,
    "auth": True,
    "Authorization": True,
    "authentication": True,
    "x-auth-token": True,
    "auth_token": True,
    "basic_auth": True,
    "proxy_auth": True,
    "PROXY_AUTH": True,
    "basicAuth": True,
    "slack_webhook_url": True,
    "author": False,
    "GIT_AUTHOR_EMAIL": False,
    "authority": False,
    "AZURE_AUTHORITY_HOST": False,
    "prompt_cache_key": False,
    "cache_key": False,
    "langfuse_public_key": False,
}


@pytest.mark.parametrize(
    ("name", "credential"), CREDENTIAL_NAMES.items(), ids=CREDENTIAL_NAMES
)
def test_a_name_says_credential_by_its_words_and_auth_only_as_a_whole_word(
    name, credential
):
    assert is_credential_name(name) is credential


def test_a_registered_key_is_masked_from_then_on_the_longest_first_but_a_placeholder():
    text = f"{KEY}-old, then {KEY}, sent as EMPTY to {LOOKS_LIKE_A_KEY}"
    assert masked(text) == text
    for key in ("EMPTY", KEY, f"{KEY}-old"):
        register_key(key)
    assert (
        masked(text) == f"REDACTED, then REDACTED, sent as EMPTY to {LOOKS_LIKE_A_KEY}"
    )


@pytest.mark.parametrize(
    ("value", "text"),
    [
        (date(2024, 10, 21), "2024-10-21"),
        (datetime(2024, 10, 21, 9, 30, tzinfo=UTC), "2024-10-21T09:30:00+00:00"),
        (time(9, 30), "09:30:00"),
    ],
    ids=["date", "datetime", "time"],
)
def test_a_date_or_a_time_is_made_plain_as_its_iso_8601_text(value, text):
    assert plain(value) == text


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
