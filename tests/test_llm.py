import json
import logging
import random
import re
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from types import SimpleNamespace
from typing import Literal
from urllib.parse import quote

import litellm
import openai
import pytest
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info
from litellm.types.router import RetryPolicy
from pydantic import BaseModel, ConfigDict, Field, create_model

from r3con import settings
from r3con.masking import masked
from r3con.runs import StageRun, TaskLogger
from r3con.runtime.llm import (
    _RETRY_AFTER_JITTER_SECONDS,
    _SECONDS_DIGITS,
    Cancelled,
    _sleep,
    litellm_chat_completion,
    litellm_chat_completion_full,
    quiet_litellm,
    register_credentials,
    schema_as_sent,
)
from tests.fakes import DIGITS_KEY, GatewayKey

MODEL = "openai/gpt-6-luna"
VLLM = "hosted_vllm/fake"
OPENAI = "openai/fake"
BOTH_PATHS = pytest.mark.parametrize("model", [VLLM, OPENAI])
CHAT = "/v1/chat/completions"
VLLM_TOO_LONG = {
    "object": "error",
    "message": (
        "This model's maximum context length is 4000 tokens. However, you requested "
        "7956 tokens in the messages. Please reduce the length of the messages."
    ),
    "type": "BadRequestError",
    "param": None,
    "code": 400,
}
LOUD = (False, logging.INFO)
QUIET = (True, logging.CRITICAL)
ENTRY_GRACE_SECONDS = 0.2
THREAD_TIMEOUT_SECONDS = 5
SCRIPTED_ERROR = {"error": {"message": "scripted error", "type": "server_error"}}
UNKNOWN_PARAMETER = {
    "error": {"message": "unknown parameter: tone", "type": "invalid_request_error"}
}
# A new process's documents make its first model calls at once, as relevance does,
# and every line of pydantic's model rebuild hands the GIL to another thread: before
# pydantic 2.14, two threads building one of litellm's deferred models raced there.
# completion=litellm.completion has r3con send each call once: its own resend of a
# failed call would hide the race behind one retry.
FIRST_CALLS_AT_ONCE = """
import threading, time
import litellm, pydantic
from r3con.parallel import parallel_map
from r3con.runtime.llm import litellm_chat_completion_full

REBUILD = pydantic.BaseModel.model_rebuild.__func__.__code__
DOCUMENTS = 5
HAND_OVER_SECONDS = 0.001
rebuilt = threading.Event()


def hand_over_each_line(frame, event, arg):
    if event == "line":
        time.sleep(HAND_OVER_SECONDS)
    return hand_over_each_line


def trace_rebuilds(frame, event, arg):
    if frame.f_code is not REBUILD:
        return None
    rebuilt.set()
    return hand_over_each_line


threading.settrace(trace_rebuilds)
parallel_map(
    lambda i, _: litellm_chat_completion_full(
        model="hosted_vllm/fake", system_prompt="s", user_prompt=str(i),
        completion=litellm.completion, mock_response="ok",
    ),
    range(DOCUMENTS),
    max_workers=DOCUMENTS,
)
assert rebuilt.is_set(), "no thread ran BaseModel.model_rebuild, so no race was forced"
"""
FIRST_CALLS_TIMEOUT_SECONDS = 60
QUOTA_SPENT = {
    "error": {
        "message": "You exceeded your current quota, please check your plan.",
        "type": "insufficient_quota",
        "param": None,
        "code": "insufficient_quota",
    }
}
ANSWER = {
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "OK"},
            "finish_reason": "stop",
        }
    ]
}
OK = (200, ANSWER)
RATE_LIMITED = (429, SCRIPTED_ERROR)
TOO_LONG = (400, VLLM_TOO_LONG)
CALLER_RETRY_SETTINGS = {"num_retries", "max_retries", "retry_policy"}
LONGEST_FIRST_BACKOFF = 1
NOW = "Tue, 27 Oct 2026 12:00:00 GMT"
KEY = "sk-not-a-real-key-0123456789"
# No shape litellm's own redactor knows: only r3con's masking can hide it.
PLAIN_KEY = "not-a-real-key-q7RkT2mXv9LpA4sW"
SERVER = "https://llm.example.com/v1"
BAD_REQUEST = "BadRequestError: litellm.BadRequestError: "
CANCELLED = "Cancelled: cancel was set, so no further model call is sent"
LONG_RETRY_AFTER_SECONDS = 60
STOPPED_WITHIN_SECONDS = 5
SLOW_SEND_SECONDS = 0.05
COUNTS = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
DETAILED_USAGE = {
    **COUNTS,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 7},
}
# As OpenAI sends it: litellm's copy of cache_write_tokens must not be recorded.
OPENAI_USAGE = {
    **COUNTS,
    "prompt_tokens_details": {
        "cached_tokens": 0,
        "cache_write_tokens": 0,
        "audio_tokens": 0,
    },
    "completion_tokens_details": {
        "reasoning_tokens": 6,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    },
}
# A field litellm has no name for, as a newer server might send.
UNKNOWN_FIELD_USAGE = {**COUNTS, "queued_tokens": 3}
# Each of these litellm rewrites in its own usage: a count it fills with 0, the cache
# counts it copies under OpenAI's names.
NO_COMPLETION_COUNT_USAGE = {"prompt_tokens": 10, "total_tokens": 10}
DEEPSEEK_USAGE = {**COUNTS, "prompt_cache_hit_tokens": 4, "prompt_cache_miss_tokens": 6}
CACHE_CREATION_USAGE = {**COUNTS, "prompt_tokens_details": {"cache_creation_tokens": 5}}
# Zeros the server sent, which are kept, unlike the empty usage litellm starts with.
ZERO_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
RETRY_AFTER_SECONDS = 0.1
SERVED = "Qwen/Qwen3-8B-2504"
FINGERPRINT = "fp_3f1c2d"
THOUGHT = "The user wants OK."
CALLERS_METADATA = {"trace": "run-7"}
CREDENTIAL_NAME = "team-openai"


class Flat(BaseModel):
    name: str
    nickname: str | None = None


class Inner(BaseModel):
    kind: str
    note: str | None = None


class Nested(BaseModel):
    inner: Inner


class Listed(BaseModel):
    rows: list[Inner]


class Counts(BaseModel):
    by_site: dict[str, int]


class Number(BaseModel):
    x: int


@pytest.fixture
def loud_litellm(monkeypatch):
    """litellm free to print and its logger at INFO, until the test ends."""
    monkeypatch.setattr(litellm, "suppress_debug_info", LOUD[0])
    logger = logging.getLogger("LiteLLM")
    level = logger.level
    logger.setLevel(LOUD[1])
    yield
    logger.setLevel(level)


@pytest.fixture
def run(tmp_path) -> StageRun:
    return StageRun(stage="judge", task_logger=TaskLogger("run", root=tmp_path))


@pytest.fixture
def top_draws(monkeypatch):
    """Every random draw at the top of its range: the longest backoff and jitter."""
    monkeypatch.setattr(random, "uniform", lambda low, high: high)


def refused(message: str) -> litellm.BadRequestError:
    return litellm.BadRequestError(message=message, llm_provider="openai", model=MODEL)


def litellm_settings():
    return litellm.suppress_debug_info, logging.getLogger("LiteLLM").level


def ask(llm, **kwargs):
    return litellm_chat_completion(
        system_prompt="s", user_prompt="u", model=MODEL, completion=llm, **kwargs
    )


def ask_server(httpserver, model=VLLM, api_base=None, **kwargs):
    """One call through litellm itself, to the local server unless ``api_base`` names
    another, in vLLM's words unless ``model`` names another provider."""
    return litellm_chat_completion(
        system_prompt="s",
        user_prompt="u",
        model=model,
        api_base=api_base or httpserver.url_for("/v1"),
        api_key="sk-fake",
        **kwargs,
    )


def answers_in_turn(httpserver, *replies: tuple) -> None:
    """Script the local server to answer each ``(status, body)``, or ``(status, body,
    headers)``, in turn, and the last one to every request after it."""
    *first, (status, body, *headers) = replies
    for earlier_status, earlier_body, *earlier_headers in first:
        httpserver.expect_ordered_request(CHAT).respond_with_json(
            earlier_body, earlier_status, *earlier_headers
        )
    httpserver.expect_request(CHAT).respond_with_json(body, status, *headers)


def answering(message: dict | None = None, **reply) -> dict:
    """A server's reply of OK, its message with ``message``'s fields too, and with
    ``reply``'s fields."""
    [choice] = ANSWER["choices"]
    said = {**choice["message"], **(message or {})}
    return {**ANSWER, "choices": [{**choice, "message": said}], **reply}


def slowly(reply: str | BaseException):
    """A reply that takes ``SLOW_SEND_SECONDS`` to arrive: ``reply``, or ``reply``
    raised."""

    def answer(request: dict) -> str:
        time.sleep(SLOW_SEND_SECONDS)
        if isinstance(reply, BaseException):
            raise reply
        return reply

    return answer


def rate_limited(retry_after: int) -> tuple[int, dict, dict[str, str]]:
    return (429, SCRIPTED_ERROR, {"Retry-After": str(retry_after)})


def sent_schema(llm, schema: type[BaseModel]) -> dict:
    llm.replies("{}")
    litellm_chat_completion_full(
        system_prompt="s", user_prompt="u", model=MODEL, schema=schema, completion=llm
    )
    return llm.requests[0]["response_format"]["json_schema"]["schema"]


@pytest.mark.parametrize(
    ("schema", "path", "required"),
    [
        (Flat, [], ["name", "nickname"]),
        (Nested, [], ["inner"]),
        (Nested, ["$defs", "Inner"], ["kind", "note"]),
        (Listed, ["$defs", "Inner"], ["kind", "note"]),
    ],
)
def test_every_object_sent_in_strict_mode_is_closed_and_fully_required(
    llm, schema, path, required
):
    node = sent_schema(llm, schema)
    for key in path:
        node = node[key]
    assert node["required"] == required
    assert node["additionalProperties"] is False


@pytest.mark.parametrize(("n", "kept"), [(1_000, True), (1_001, False)])
def test_an_enum_past_openais_limit_is_sent_without_its_values(llm, n, kept):
    values = tuple(f"v{i}" for i in range(n))
    choice = create_model("Choice", pick=(Literal[values], ...))
    pick = sent_schema(llm, choice)["properties"]["pick"]
    assert pick.get("enum") == (list(values) if kept else None)
    assert pick["type"] == "string"


@pytest.mark.parametrize(
    ("sizes", "kept"),
    [
        ((700, 400), [False, True]),
        ((400, 700), [True, False]),
        ((500, 500), [True, True]),
        ((600, 300, 200), [False, True, True]),
    ],
)
def test_enums_over_openais_limit_together_are_dropped_largest_first(llm, sizes, kept):
    fields: dict = {
        f"pick{k}": (Literal[tuple(f"v{k}.{i}" for i in range(n))], ...)
        for k, n in enumerate(sizes)
    }
    picks = sent_schema(llm, create_model("Choices", **fields))["properties"]
    assert ["enum" in picks[f"pick{k}"] for k in range(len(sizes))] == kept


@pytest.mark.parametrize(
    ("pattern", "kept"),
    [
        (r"^\d{4}$", True),
        (r"^(?=\d)\w+$", False),
        (r"^(?!x)\w+$", False),
        (r"(?<=#)\d+", False),
        (r"(?<!#)\d+", False),
    ],
)
def test_a_pattern_with_a_lookaround_is_sent_without_it(llm, pattern, kept):
    coded = create_model(
        "Coded",
        __config__=ConfigDict(regex_engine="python-re"),
        code=(str, Field(pattern=pattern)),
    )
    code = sent_schema(llm, coded)["properties"]["code"]
    assert code.get("pattern") == (pattern if kept else None)
    assert code["type"] == "string"


def test_a_dict_field_is_sent_without_properties_left_open(llm):
    by_site = sent_schema(llm, Counts)["properties"]["by_site"]
    assert "required" not in by_site
    assert by_site["additionalProperties"] == {"type": "integer"}


def test_a_supplied_completion_is_called_once_per_request_without_retry_settings(llm):
    slow_down = litellm.RateLimitError(
        message="slow down", llm_provider="openai", model=MODEL
    )
    with pytest.raises(litellm.RateLimitError, match="slow down"):
        ask(llm.replies(slow_down, "ok"))
    assert len(llm.requests) == 1
    assert CALLER_RETRY_SETTINGS.isdisjoint(llm.requests[0])


def test_an_empty_reply_is_rerolled_with_the_same_request(llm):
    out = ask(llm.replies("", "\n", '{"x": 7}'), schema=Number)
    assert out == Number(x=7)
    first, *rerolls = llm.requests
    assert len(rerolls) == 2
    assert all(reroll["messages"] == first["messages"] for reroll in rerolls)
    assert not any("seed" in request for request in llm.requests)


def test_a_seed_the_caller_sends_is_resent_unchanged_on_a_reroll(llm):
    ask(llm.replies("", "", '{"x": 7}'), schema=Number, seed=5)
    assert [request["seed"] for request in llm.requests] == [5, 5, 5]


def test_empty_replies_raise_once_the_rerolls_run_out(llm):
    llm.replies("", "", "", "")
    with pytest.raises(ValueError, match=r"empty text after 4 attempt\(s\)\. For a"):
        ask(llm, max_empty_retries=3, schema=Number)
    assert len(llm.requests) == 4


def test_an_empty_plain_reply_is_returned_as_it_is(llm):
    assert ask(llm.replies("")) == ""
    assert len(llm.requests) == 1


def test_a_structured_reply_that_parses_is_not_rerolled(llm):
    assert ask(llm.replies('{"x": 1}'), schema=Number) == Number(x=1)
    assert len(llm.requests) == 1


def test_sampling_params_reach_the_request_untouched(llm):
    extra_body = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": False}}
    sampling = {"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5}
    llm.replies("ok")
    litellm_chat_completion_full(
        system_prompt="s",
        user_prompt="u",
        model="openai/gpt-4o",
        extra_body=extra_body,
        completion=llm,
        **sampling,
    )
    request = llm.requests[0]
    assert {key: request[key] for key in sampling} == sampling
    assert request["extra_body"] == extra_body


def test_what_litellm_writes_into_a_calls_metadata_never_reaches_the_next_call(
    llm, own_trace, logged
):
    metadata = {"trace": own_trace}
    for _ in range(2):
        ask(llm.replies("ok"), metadata=metadata)
        assert logged.acquire(timeout=THREAD_TIMEOUT_SECONDS)
    sent = [request["metadata"] for request in llm.requests]
    assert sent == [{"trace": own_trace}] * 2


def test_a_routers_credential_tag_stays_out_of_the_callers_tags(
    llm, monkeypatch, router, own_trace, logged
):
    monkeypatch.setattr(litellm, "completion", llm.replies("ok", "ok"))
    routing = router({"model": MODEL, "litellm_credential_name": CREDENTIAL_NAME})
    metadata = {"trace": own_trace, "tags": ["r3con"]}
    for _ in range(2):
        ask(routing.completion, metadata=metadata)
        assert logged.acquire(timeout=THREAD_TIMEOUT_SECONDS)
    assert metadata == {"trace": own_trace, "tags": ["r3con"]}
    sent = [request["metadata"]["tags"] for request in llm.requests]
    assert sent == [["r3con", f"Credential: {CREDENTIAL_NAME}"]] * 2


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize("name", ["metadata", "litellm_metadata"])
def test_litellms_timing_of_a_call_stays_out_of_the_callers_dict(httpserver, name):
    answers_in_turn(httpserver, OK)
    given = dict(CALLERS_METADATA)
    ask_server(httpserver, model=OPENAI, **{name: given})
    assert given == CALLERS_METADATA


def test_the_callers_completion_makes_the_request_and_is_not_sent_in_it(llm):
    assert ask(llm.replies("from my own connection")) == "from my own connection"
    assert [request["model"] for request in llm.requests] == [MODEL]
    assert "completion" not in llm.requests[0]


def test_without_a_completion_the_request_goes_to_litellm(llm, monkeypatch):
    monkeypatch.setattr(litellm, "completion", llm.replies("ok"))
    assert (
        litellm_chat_completion(system_prompt="s", user_prompt="u", model=MODEL) == "ok"
    )
    assert len(llm.requests) == 1


@pytest.mark.allow_hosts(["127.0.0.1"])
@BOTH_PATHS
@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504, 529])
def test_a_temporary_error_is_resent_as_many_times_as_settings_say(
    httpserver, monkeypatch, model, status
):
    monkeypatch.setattr(settings, "LLM_TEMPORARY_ERROR_RETRIES", 2)
    answers_in_turn(httpserver, (status, SCRIPTED_ERROR))
    with pytest.raises(openai.APIError):
        ask_server(httpserver, model=model)
    assert len(httpserver.log) == 3


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_refused_connection_is_resent(httpserver, monkeypatch, waits):
    monkeypatch.setattr(settings, "LLM_TEMPORARY_ERROR_RETRIES", 2)
    with socket.socket() as unused:
        unused.bind(("127.0.0.1", 0))
        refusing = f"http://127.0.0.1:{unused.getsockname()[1]}/v1"
        with pytest.raises(openai.APIError, match="Connection refused"):
            ask_server(httpserver, api_base=refusing)
    assert len(waits) == 2


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize(
    ("model", "status", "body"),
    [
        (VLLM, 400, UNKNOWN_PARAMETER),
        (VLLM, 401, {"error": "Unauthorized"}),
        (VLLM, 403, SCRIPTED_ERROR),
        (VLLM, 404, SCRIPTED_ERROR),
        (VLLM, 422, SCRIPTED_ERROR),
        (VLLM, *TOO_LONG),
        (VLLM, 429, QUOTA_SPENT),
        (OPENAI, 429, QUOTA_SPENT),
    ],
    ids=["400", "401", "403", "404", "422", "too-long", "quota", "openai-quota"],
)
def test_a_refusal_is_sent_once(httpserver, model, status, body):
    answers_in_turn(httpserver, (status, body))
    with pytest.raises(openai.APIError) as refused:
        ask_server(httpserver, model=model)
    assert refused.value.status_code == status
    assert len(httpserver.log) == 1


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_refusal_after_a_rate_limit_is_not_resent(httpserver):
    answers_in_turn(httpserver, RATE_LIMITED, TOO_LONG)
    with pytest.raises(litellm.ContextWindowExceededError, match="maximum context"):
        ask_server(httpserver)
    assert len(httpserver.log) == 2


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_each_backoff_doubles_up_to_the_wait_cap(
    httpserver, monkeypatch, waits, top_draws
):
    monkeypatch.setattr(settings, "LLM_TEMPORARY_ERROR_RETRIES", 3)
    monkeypatch.setattr(settings, "LLM_RETRY_MAX_WAIT_SECONDS", 2)
    answers_in_turn(httpserver, RATE_LIMITED)
    with pytest.raises(litellm.RateLimitError):
        ask_server(httpserver)
    assert waits == [1, 2, 2]


@pytest.mark.allow_hosts(["127.0.0.1"])
@BOTH_PATHS
@pytest.mark.parametrize(
    ("headers", "wait"),
    [
        (
            {"Retry-After-Ms": "1500", "Retry-After": "9"},
            1.5 + _RETRY_AFTER_JITTER_SECONDS,
        ),
        ({"Retry-After": "1.5"}, 1.5 + _RETRY_AFTER_JITTER_SECONDS),
        ({"Retry-After": "2"}, 2 + _RETRY_AFTER_JITTER_SECONDS),
        (
            {"Retry-After": "Tue, 27 Oct 2026 12:00:05 GMT"},
            5 + _RETRY_AFTER_JITTER_SECONDS,
        ),
        ({}, LONGEST_FIRST_BACKOFF),
        ({"Retry-After": "soon"}, LONGEST_FIRST_BACKOFF),
    ],
    ids=["ms-first", "fractional", "integer", "http-date", "absent", "garbage"],
)
def test_the_wait_is_read_from_the_retry_after_headers_else_backed_off(
    httpserver, monkeypatch, waits, top_draws, model, headers, wait
):
    monkeypatch.setattr(time, "time", lambda: parsedate_to_datetime(NOW).timestamp())
    answers_in_turn(httpserver, (429, SCRIPTED_ERROR, headers), OK)
    assert ask_server(httpserver, model=model) == "OK"
    assert waits == [wait]


def test_retry_after_headers_are_read_whatever_their_case(
    llm, monkeypatch, waits, top_draws
):
    slow_down = litellm.RateLimitError(
        message="slow down", llm_provider="openai", model=MODEL
    )
    slow_down.litellm_response_headers = {"RETRY-AFTER": "2"}
    monkeypatch.setattr(litellm, "completion", llm.replies(slow_down, "ok"))
    assert (
        litellm_chat_completion(system_prompt="s", user_prompt="u", model=MODEL) == "ok"
    )
    assert waits == [2 + _RETRY_AFTER_JITTER_SECONDS]


@pytest.mark.allow_hosts(["127.0.0.1"])
@BOTH_PATHS
@pytest.mark.parametrize("retry_after", [0, 30, 90])
def test_a_retry_after_is_waited_out_with_jitter_even_past_the_backoff_cap(
    httpserver, monkeypatch, waits, top_draws, model, retry_after
):
    monkeypatch.setattr(settings, "LLM_RETRY_MAX_WAIT_SECONDS", 60)
    answers_in_turn(
        httpserver, rate_limited(retry_after), rate_limited(retry_after), OK
    )
    assert ask_server(httpserver, model=model) == "OK"
    wait = retry_after + _RETRY_AFTER_JITTER_SECONDS
    assert waits == [wait, wait]


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_retry_after_longer_than_the_time_left_gives_up_at_once(
    httpserver, monkeypatch
):
    monkeypatch.setattr(settings, "LLM_RETRY_MAX_TOTAL_SECONDS", 300)
    answers_in_turn(httpserver, rate_limited(301), OK)
    with pytest.raises(litellm.RateLimitError):
        ask_server(httpserver)
    assert len(httpserver.log) == 1


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize(
    ("reply", "waited"),
    [(RATE_LIMITED, [1, 2, 2]), (rate_limited(3), [4])],
    ids=["backoff-cut-to-the-time-left", "retry-after-past-the-time-left"],
)
def test_no_resend_starts_after_the_total_time_cap(
    httpserver, monkeypatch, waits, top_draws, reply, waited
):
    monkeypatch.setattr(settings, "LLM_RETRY_MAX_TOTAL_SECONDS", 5)
    monkeypatch.setattr(time, "monotonic", lambda: sum(waits))
    answers_in_turn(httpserver, reply)
    with pytest.raises(litellm.RateLimitError):
        ask_server(httpserver)
    assert waits == waited


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_each_resend_is_logged_as_a_warning_naming_the_error_and_the_wait(
    httpserver, caplog
):
    caplog.set_level(logging.WARNING, logger="r3con")
    answers_in_turn(httpserver, RATE_LIMITED, RATE_LIMITED, OK)
    ask_server(httpserver)
    resends = [record for record in caplog.records if record.name == "r3con.llm"]
    assert [record.levelno for record in resends] == [logging.WARNING] * 2
    resend = re.compile(r"in \S+ seconds as it raised RateLimitError")
    assert all(resend.search(record.getMessage()) for record in resends)


@pytest.mark.allow_hosts(["127.0.0.1"])
@BOTH_PATHS
def test_a_resend_warning_masks_the_key_the_error_echoes(httpserver, caplog, model):
    caplog.set_level(logging.WARNING, logger="r3con")
    echoed = {"error": {"message": f"slow down, {PLAIN_KEY}", "type": "server_error"}}
    answers_in_turn(httpserver, (429, echoed), OK)
    litellm_chat_completion(
        system_prompt="s",
        user_prompt="u",
        model=model,
        api_base=httpserver.url_for("/v1"),
        api_key=PLAIN_KEY,
    )
    [resend] = caplog.records
    assert "as it raised RateLimitError" in resend.getMessage()
    assert "slow down, REDACTED" in caplog.text
    assert PLAIN_KEY not in caplog.text


@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize(
    ("own_setting", "requests"),
    [
        ({"num_retries": 1}, 2),
        ({"max_retries": 0}, 1),
        ({"retry_policy": RetryPolicy(RateLimitErrorRetries=2)}, 3),
    ],
    ids=["num_retries", "max_retries", "retry_policy"],
)
def test_a_callers_own_retry_setting_replaces_r3cons_resends(
    httpserver, own_setting, requests
):
    answers_in_turn(httpserver, RATE_LIMITED)
    with pytest.raises(litellm.RateLimitError):
        ask_server(httpserver, **own_setting)
    assert len(httpserver.log) == requests


def test_a_recorded_call_carries_litellms_price_for_it(llm, priced_model, run):
    ask(llm.replies("ok"), run=run)

    [step] = run.steps
    assert step.usd == (
        step.tokens["prompt"] * priced_model["input_cost_per_token"]
        + step.tokens["completion"] * priced_model["output_cost_per_token"]
    )


def test_a_call_litellm_cannot_price_is_recorded_without_a_cost(llm, run):
    litellm_chat_completion(
        system_prompt="s",
        user_prompt="u",
        model=VLLM,
        completion=llm.replies("ok"),
        run=run,
    )

    assert [step.usd for step in run.steps] == [None]


def slowed_down(retry_after: str) -> litellm.RateLimitError:
    error = litellm.RateLimitError(
        message="slow down", llm_provider="openai", model=MODEL
    )
    error.litellm_response_headers = {"retry-after": retry_after}
    return error


@pytest.mark.parametrize(
    ("rate_limits", "sends", "resend_wait_seconds"),
    [(0, 1, 0), (2, 3, 4.01)],
    ids=["first-time", "after-two-rate-limits-to-hundredths"],
)
def test_a_recorded_call_says_how_often_it_was_sent_and_how_long_it_waited(
    llm, monkeypatch, run, top_draws, rate_limits, sends, resend_wait_seconds
):
    errors = [slowed_down(retry_after="1.004") for _ in range(rate_limits)]
    monkeypatch.setattr(litellm, "completion", llm.replies(*errors, "ok"))

    litellm_chat_completion(system_prompt="s", user_prompt="u", model=MODEL, run=run)

    [step] = run.steps
    assert (step.sends, step.resend_wait_seconds) == (sends, resend_wait_seconds)
    assert len(llm.requests) == sends


def test_a_call_r3con_did_not_resend_itself_records_no_sends_or_wait(llm, run):
    ask(llm.replies("ok"), run=run)

    [step] = run.steps
    assert (step.sends, step.resend_wait_seconds) == (None, None)


@pytest.mark.parametrize(
    ("errors", "sends", "resend_wait_seconds", "recorded"),
    [
        (
            [slowed_down(retry_after="1.004") for _ in range(3)],
            3,
            4.01,
            "RateLimitError: litellm.RateLimitError: slow down",
        ),
        (
            [refused("unknown parameter: tone")],
            1,
            0,
            f"{BAD_REQUEST}unknown parameter: tone",
        ),
    ],
    ids=["gives-up-after-two-resends", "refused-at-once"],
)
def test_a_call_that_raises_is_recorded_and_its_error_raised_as_it_was(
    llm, monkeypatch, run, top_draws, errors, sends, resend_wait_seconds, recorded
):
    monkeypatch.setattr(settings, "LLM_TEMPORARY_ERROR_RETRIES", 2)
    monkeypatch.setattr(litellm, "completion", llm.replies(*errors))

    with pytest.raises(type(errors[-1])) as raised:
        litellm_chat_completion(
            system_prompt="s", user_prompt="u", model=MODEL, run=run
        )

    assert raised.value is errors[-1]
    [step] = run.steps
    assert (step.sends, step.resend_wait_seconds) == (sends, resend_wait_seconds)
    assert (step.error, step.response, step.tokens) == (recorded, {}, None)
    assert len(llm.requests) == sends


def test_a_supplied_completion_that_raises_is_recorded_without_sends_or_wait(llm, run):
    with pytest.raises(litellm.RateLimitError):
        ask(llm.replies(slowed_down(retry_after="1")), run=run)

    [step] = run.steps
    assert (step.sends, step.resend_wait_seconds) == (None, None)


@pytest.mark.allow_hosts(["127.0.0.1"])
@BOTH_PATHS
@pytest.mark.parametrize(
    ("reply", "recorded"),
    [
        (answering(usage=COUNTS), COUNTS),
        (answering(usage={**COUNTS, "prompt_tokens_details": None}), COUNTS),
        (answering(usage=DETAILED_USAGE), DETAILED_USAGE),
        (answering(usage=OPENAI_USAGE), OPENAI_USAGE),
        (answering(usage=UNKNOWN_FIELD_USAGE), UNKNOWN_FIELD_USAGE),
        (answering(usage=NO_COMPLETION_COUNT_USAGE), NO_COMPLETION_COUNT_USAGE),
        (answering(usage=DEEPSEEK_USAGE), DEEPSEEK_USAGE),
        (answering(usage=CACHE_CREATION_USAGE), CACHE_CREATION_USAGE),
        (answering(usage=ZERO_USAGE), ZERO_USAGE),
        (ANSWER, None),
    ],
    ids=[
        "counts-only",
        "null-details",
        "details",
        "openais-cache-write",
        "unknown-field",
        "a-count-missing",
        "deepseeks-names",
        "cache-creation",
        "zero-counts",
        "no-usage",
    ],
)
def test_a_recorded_call_keeps_each_usage_field_the_server_sent_and_no_other(
    httpserver, run, model, reply, recorded
):
    answers_in_turn(httpserver, (200, reply))
    ask_server(httpserver, model=model, run=run)

    [step] = run.steps
    assert step.usage == recorded


@pytest.mark.allow_hosts(["127.0.0.1"])
@BOTH_PATHS
@pytest.mark.parametrize(
    ("reply", "recorded"),
    [
        (
            answering(model=SERVED, system_fingerprint=FINGERPRINT),
            (SERVED, FINGERPRINT),
        ),
        (answering(model=SERVED), (SERVED, None)),
        (ANSWER, (None, None)),
    ],
    ids=["fingerprinted", "no-fingerprint", "neither"],
)
def test_a_recorded_call_names_the_model_and_fingerprint_the_server_answered_with(
    httpserver, run, model, reply, recorded
):
    answers_in_turn(httpserver, (200, reply))
    ask_server(httpserver, model=model, run=run)

    [step] = run.steps
    assert (step.served_model, step.fingerprint) == recorded


@pytest.mark.allow_hosts(["127.0.0.1"])
@BOTH_PATHS
@pytest.mark.parametrize(
    ("message", "recorded"),
    [
        ({"reasoning_content": THOUGHT}, THOUGHT),
        ({"reasoning": THOUGHT}, THOUGHT),
        ({"content": f"<think>{THOUGHT}</think>OK"}, THOUGHT),
        ({}, None),
    ],
    ids=["reasoning_content", "reasoning", "think-block", "none"],
)
def test_a_recorded_call_keeps_the_reasoning_text_the_server_sent(
    httpserver, run, model, message, recorded
):
    answers_in_turn(httpserver, (200, answering(message)))
    ask_server(httpserver, model=model, run=run)

    [step] = run.steps
    assert step.response.get("reasoning") == recorded


@pytest.mark.allow_hosts(["127.0.0.1"])
@BOTH_PATHS
def test_a_resent_call_records_the_body_of_the_send_that_answered(
    httpserver, run, model
):
    answers_in_turn(
        httpserver, RATE_LIMITED, (200, answering(model=SERVED, usage=COUNTS))
    )
    ask_server(httpserver, model=model, run=run)

    [step] = run.steps
    assert (step.sends, step.usage, step.served_model) == (2, COUNTS, SERVED)


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_routers_call_records_the_body_its_deployment_answered_with(
    httpserver, run, router
):
    api_base = httpserver.url_for("/v1")
    routing = router({"model": VLLM, "api_base": api_base, "api_key": "sk-fake"})
    answers_in_turn(httpserver, (200, answering(model=SERVED, usage=COUNTS)))
    litellm_chat_completion(
        system_prompt="s",
        user_prompt="u",
        model=VLLM,
        completion=routing.completion,
        run=run,
    )

    [step] = run.steps
    assert (step.usage, step.served_model) == (COUNTS, SERVED)


@pytest.mark.allow_hosts(["127.0.0.1"])
@BOTH_PATHS
def test_the_hook_that_reads_the_reply_sends_nothing_to_the_server(
    httpserver, run, model
):
    answers_in_turn(httpserver, OK)
    ask_server(httpserver, model=model, run=run)

    [(request, _)] = httpserver.log
    assert set(request.get_json()) == {"model", "messages"}


@pytest.mark.allow_hosts(["127.0.0.1"])
@BOTH_PATHS
def test_a_callers_own_logger_fn_is_still_called_and_the_body_still_recorded(
    httpserver, run, model
):
    events = []
    answers_in_turn(httpserver, (200, answering(usage=COUNTS)))
    ask_server(
        httpserver,
        model=model,
        run=run,
        logger_fn=lambda details: events.append(details["log_event_type"]),
    )

    assert "post_api_call" in events
    assert run.steps[0].usage == COUNTS


def test_a_call_r3con_does_not_record_goes_out_without_the_hook(llm):
    ask(llm.replies("ok"))

    assert "logger_fn" not in llm.requests[0]


@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_gemini_reply_records_its_usage_metadata_and_model_version(httpserver, run):
    usage = {"promptTokenCount": 11, "candidatesTokenCount": 2, "totalTokenCount": 13}
    httpserver.expect_request(re.compile(".*:generateContent")).respond_with_json(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "OK"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": usage,
            "modelVersion": "gemini-served-1",
        }
    )
    litellm_chat_completion(
        system_prompt="s",
        user_prompt="u",
        model="gemini/gemini-2.5-flash",
        api_base=httpserver.url_for(""),
        api_key="fake-gemini-key",
        run=run,
    )

    [step] = run.steps
    assert (step.usage, step.served_model) == (usage, "gemini-served-1")


def test_a_supplied_completion_that_hands_over_no_body_records_no_usage_model_or_fingerprint(
    run,
):
    reply = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
        usage={"prompt_tokens": 3},
    )
    assert ask(lambda **request: reply, run=run) == "ok"

    [step] = run.steps
    assert (step.error, step.usage, step.served_model, step.fingerprint) == (None,) * 4


@pytest.mark.parametrize("rate_limits", [0, 1], ids=["first-time", "after-a-resend"])
def test_a_recorded_call_says_how_long_it_took_resends_and_waits_included(
    llm, monkeypatch, run, rate_limits
):
    # r3con's own wait, which the autouse `waits` fixture replaces with a record.
    monkeypatch.setattr("r3con.runtime.llm._sleep", _sleep)
    monkeypatch.setattr(random, "uniform", lambda low, high: low)
    error = slowed_down(retry_after=str(RETRY_AFTER_SECONDS))
    errors = [slowly(error) for _ in range(rate_limits)]
    monkeypatch.setattr(litellm, "completion", llm.replies(*errors, slowly("ok")))

    litellm_chat_completion(system_prompt="s", user_prompt="u", model=MODEL, run=run)

    [step] = run.steps
    waited = rate_limits * RETRY_AFTER_SECONDS
    assert step.resend_wait_seconds == pytest.approx(waited, abs=0.01)
    assert (rate_limits + 1) * SLOW_SEND_SECONDS + waited <= step.seconds
    assert step.seconds < STOPPED_WITHIN_SECONDS
    assert step.seconds == round(step.seconds, _SECONDS_DIGITS)


def test_a_call_that_raises_records_how_long_it_took_and_nothing_of_a_reply(llm, run):
    with pytest.raises(litellm.BadRequestError):
        ask(llm.replies(slowly(refused("no"))), run=run)

    [step] = run.steps
    assert SLOW_SEND_SECONDS <= step.seconds
    assert (step.usage, step.served_model, step.fingerprint) == (None, None, None)


@pytest.mark.parametrize("supplied", [False, True], ids=["litellm", "supplied"])
def test_once_cancel_is_set_a_call_is_neither_sent_nor_recorded(
    llm, monkeypatch, run, supplied
):
    monkeypatch.setattr(litellm, "completion", llm)
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(Cancelled, match="no further model call"):
        litellm_chat_completion(
            system_prompt="s",
            user_prompt="u",
            model=MODEL,
            completion=llm if supplied else None,
            run=run,
            cancel=cancel,
        )
    assert llm.requests == []
    assert run.steps == []


@pytest.mark.parametrize(
    "cancel", [None, threading.Event()], ids=["without-cancel", "with-cancel"]
)
def test_a_wait_to_resend_is_the_one_sleep_whether_or_not_cancel_is_given(
    llm, monkeypatch, waits, top_draws, cancel
):
    monkeypatch.setattr(
        litellm, "completion", llm.replies(slowed_down(retry_after="2"), "ok")
    )
    reply = litellm_chat_completion(
        system_prompt="s", user_prompt="u", model=MODEL, cancel=cancel
    )
    assert reply == "ok"
    assert waits == [2 + _RETRY_AFTER_JITTER_SECONDS]


def test_a_wait_to_resend_ends_at_once_when_cancel_is_set_and_nothing_is_resent(
    llm, monkeypatch, run
):
    # r3con's own wait, which the autouse `waits` fixture replaces with a record.
    monkeypatch.setattr("r3con.runtime.llm._sleep", _sleep)
    cancel = threading.Event()

    def slowed_down_and_cancelled(request: dict) -> str:
        cancel.set()
        raise slowed_down(retry_after=str(LONG_RETRY_AFTER_SECONDS))

    monkeypatch.setattr(
        litellm, "completion", llm.replies(slowed_down_and_cancelled, "ok")
    )
    started = time.monotonic()
    with pytest.raises(Cancelled):
        litellm_chat_completion(
            system_prompt="s", user_prompt="u", model=MODEL, run=run, cancel=cancel
        )
    assert time.monotonic() - started < STOPPED_WITHIN_SECONDS
    assert len(llm.requests) == 1
    [step] = run.steps
    assert (step.sends, step.error) == (1, CANCELLED)


def test_a_failed_call_costs_nothing_so_the_stages_cost_stays_a_number(
    llm, priced_model, run
):
    llm.replies("ok", refused("no"))

    ask(llm, run=run)
    with pytest.raises(litellm.BadRequestError):
        ask(llm, run=run)

    answered, failed = run.steps
    assert failed.usd == 0.0
    assert run.compute_totals()["usd"] == answered.usd > 0


@pytest.mark.parametrize(
    ("error", "api_key", "recorded"),
    [
        (
            refused("refused\nTraceback (most recent call last):"),
            KEY,
            f"{BAD_REQUEST}refused Traceback (most recent call last):",
        ),
        (
            refused(f"bad key Bearer {KEY}"),
            KEY,
            f"{BAD_REQUEST}bad key Bearer REDACTED",
        ),
        (
            refused("maximum context length is 4000 tokens"),
            "x",
            f"{BAD_REQUEST}maximum context length is 4000 tokens",
        ),
    ],
    ids=["one-line", "key-masked", "placeholder-key-left-alone"],
)
def test_a_recorded_error_is_one_line_without_the_key(
    llm, run, error, api_key, recorded
):
    with pytest.raises(type(error)):
        ask(llm.replies(error), run=run, api_key=api_key)

    [step] = run.steps
    assert step.error == recorded


def test_litellm_reads_its_bundled_price_map():
    assert len(litellm.model_cost) > 0
    source = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["is_env_forced"] is True


def test_nested_quiet_blocks_leave_litellm_as_they_found_it(loud_litellm):
    with quiet_litellm(), quiet_litellm():
        assert litellm_settings() == QUIET
    assert litellm_settings() == LOUD


def test_a_quiet_block_in_another_thread_leaves_litellm_as_it_found_it(loud_litellm):
    starting, entered, main_left = (threading.Event() for _ in range(3))
    seen_inside = []

    def quiet_until_main_leaves():
        starting.set()
        with quiet_litellm():
            seen_inside.append(litellm_settings())
            entered.set()
            main_left.wait(timeout=THREAD_TIMEOUT_SECONDS)

    other = threading.Thread(target=quiet_until_main_leaves)
    with quiet_litellm():
        other.start()
        assert starting.wait(timeout=THREAD_TIMEOUT_SECONDS)
        entered.wait(timeout=ENTRY_GRACE_SECONDS)
    main_left.set()
    other.join(timeout=THREAD_TIMEOUT_SECONDS)

    assert not other.is_alive()
    assert seen_inside == [QUIET]
    assert litellm_settings() == LOUD


def test_every_credential_a_request_sends_is_registered_for_masking(
    monkeypatch, router
):
    monkeypatch.setenv("MY_GATEWAY", "gateway-from-the-environment")
    routing = router(
        {"model": MODEL, "api_key": "router-key-0123456789"},
        {"model": OPENAI, "api_key": "os.environ/MY_GATEWAY"},
        {"model": VLLM, "extra_headers": {"x-tenant-auth": "tenant-0123456789"}},
    )
    params = {
        "api_key": "call-key-0123456789",
        "extra_headers": {
            "x-gateway-auth": "header-0123456789",
            "Authorization": "Bearer bearer-token-0123456789",
        },
        "azure_ad_token": "ad-token-0123456789",
        "user": "end-user-0123456789",
        "prompt_cache_key": "summaries-cache-2026",
        "max_tokens": 4096,
    }
    sent = (
        "router-key-0123456789 gateway-from-the-environment tenant-0123456789 "
        "call-key-0123456789 header-0123456789 bearer-token-0123456789 "
        "ad-token-0123456789"
    )
    register_credentials(params, routing.completion)
    assert masked(f"{sent} end-user-0123456789 summaries-cache-2026 4096") == (
        "REDACTED " * 7 + "end-user-0123456789 summaries-cache-2026 4096"
    )


class Credential(BaseModel):
    auth_token: str = Field(description="The token the memo says the gateway takes")


class CredentialAudit(BaseModel):
    credentials: list[Credential]


def test_a_structured_requests_documents_and_schema_are_data_not_credentials(llm):
    document = "The memo's gateway is https://reader:memo-password-2026@gw.example.com"
    llm.replies({"credentials": []})
    litellm_chat_completion(
        system_prompt="List the credentials the memo names.",
        user_prompt=document,
        model=MODEL,
        schema=CredentialAudit,
        completion=llm,
    )
    data = f"{document} {json.dumps(schema_as_sent(CredentialAudit))}"
    assert masked(data) == data


@dataclass
class TenantKey:
    api_key: str


# Each request's params, and the credential among them a provider may echo.
SENT_CREDENTIALS = {
    "url-user-name": (
        {"api_base": f"https://{PLAIN_KEY}@llm.example.com/v1"},
        PLAIN_KEY,
    ),
    "url-code": ({"api_base": f"{SERVER}?code={PLAIN_KEY}"}, PLAIN_KEY),
    "url-sig": ({"api_base": f"{SERVER}?sv=2026-01-01&sig={PLAIN_KEY}"}, PLAIN_KEY),
    "url-signature": ({"api_base": f"{SERVER}?signature={PLAIN_KEY}"}, PLAIN_KEY),
    "in-a-pydantic-model": ({"extra_body": GatewayKey(api_key=PLAIN_KEY)}, PLAIN_KEY),
    "in-a-dataclass": ({"extra_body": TenantKey(api_key=PLAIN_KEY)}, PLAIN_KEY),
    "in-a-set": ({"extra_body": {"auth_tokens": {PLAIN_KEY}}}, PLAIN_KEY),
    "an-integer": ({"api_key": DIGITS_KEY}, str(DIGITS_KEY)),
    "header-named-like-request-data": ({"headers": {"metadata": PLAIN_KEY}}, PLAIN_KEY),
    "callback-key-in-metadata": (
        {"metadata": {"langfuse_secret_key": PLAIN_KEY}},
        PLAIN_KEY,
    ),
    "callback-key-in-litellm-metadata": (
        {"litellm_metadata": {"langsmith_api_key": PLAIN_KEY}},
        PLAIN_KEY,
    ),
    "key-in-extra-body-metadata": (
        {"extra_body": {"metadata": {"api_key": PLAIN_KEY}}},
        PLAIN_KEY,
    ),
    "mcp-tool-header": (
        {
            "tools": [
                {"type": "mcp", "headers": {"Authorization": f"Bearer {PLAIN_KEY}"}}
            ]
        },
        PLAIN_KEY,
    ),
    "header-of-a-tool-in-extra-body": (
        {"extra_body": {"tools": [{"headers": {"X-Key": PLAIN_KEY}}]}},
        PLAIN_KEY,
    ),
    "base64-key-starting-with-a-slash": (
        {"aws_secret_access_key": f"/{PLAIN_KEY}"},
        f"/{PLAIN_KEY}",
    ),
}


@pytest.mark.parametrize(
    ("params", "credential"), SENT_CREDENTIALS.values(), ids=SENT_CREDENTIALS
)
def test_a_credential_is_registered_whatever_its_shape_or_place_in_the_params(
    params, credential
):
    register_credentials(params)
    assert masked(f"refused {credential}") == "refused REDACTED"


def test_a_one_shot_iterator_among_the_params_reaches_the_completion_whole(llm):
    llm.replies("ok")
    litellm_chat_completion(
        system_prompt="s",
        user_prompt="u",
        model=VLLM,
        completion=llm,
        stop=iter(["END", "STOP"]),
    )
    [request] = llm.requests
    assert list(request["stop"]) == ["END", "STOP"]


def test_bytes_that_are_not_text_among_the_params_reach_the_completion(llm):
    llm.replies("ok")
    ask(llm, extra_body={"blob": b"\xff"})
    [request] = llm.requests
    assert request["extra_body"] == {"blob": b"\xff"}


def test_a_url_key_is_masked_as_written_and_percent_decoded():
    key = "not-a-real/key+q7RkT2mXv9LpA4sW="
    url = f"{SERVER}?key={quote(key, safe='')}"
    register_credentials({"api_base": url})
    assert masked(f"{url} {key}") == f"{SERVER}?key=REDACTED REDACTED"


def test_first_model_calls_made_at_once_in_a_new_process_all_get_replies():
    try:
        first_calls = subprocess.run(
            [sys.executable, "-c", FIRST_CALLS_AT_ONCE],
            capture_output=True,
            text=True,
            check=False,
            timeout=FIRST_CALLS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as hung:
        pytest.fail(f"the first calls hung; their stderr:\n{hung.stderr}")
    assert first_calls.returncode == 0, first_calls.stderr
