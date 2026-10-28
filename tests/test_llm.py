import logging
import random
import re
import socket
import threading
import time
from email.utils import parsedate_to_datetime
from typing import Literal

import litellm
import openai
import pytest
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info
from litellm.types.router import RetryPolicy
from pydantic import BaseModel, create_model

from r3con import settings
from r3con.runs import StageRun, TaskLogger
from r3con.runtime.llm import (
    _RETRY_AFTER_JITTER_SECONDS,
    litellm_chat_completion,
    litellm_chat_completion_full,
    quiet_litellm,
)

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
def top_draws(monkeypatch):
    """Every random draw at the top of its range: the longest backoff and jitter."""
    monkeypatch.setattr(random, "uniform", lambda low, high: high)


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


def test_an_empty_structured_reply_is_rerolled_with_the_same_request(llm):
    out = ask(llm.replies("", "", '{"x": 7}'), schema=Number)
    assert out == Number(x=7)
    first, *rerolls = llm.requests
    assert len(rerolls) == 2
    assert all(reroll["messages"] == first["messages"] for reroll in rerolls)
    assert not any("seed" in request for request in llm.requests)


def test_a_seed_the_caller_sends_is_resent_unchanged_on_a_reroll(llm):
    ask(llm.replies("", "", '{"x": 7}'), schema=Number, seed=5)
    assert [request["seed"] for request in llm.requests] == [5, 5, 5]


@pytest.mark.parametrize(
    "asking", [{"schema": Number}, {"reroll_empty": True}], ids=["structured", "plain"]
)
def test_empty_replies_raise_once_the_rerolls_run_out(llm, asking):
    llm.replies("", "", "", "")
    with pytest.raises(ValueError, match=r"empty text after 4 attempt\(s\)\. For a"):
        ask(llm, max_empty_retries=3, **asking)
    assert len(llm.requests) == 4


def test_an_empty_plain_reply_is_returned_as_it_is(llm):
    assert ask(llm.replies("")) == ""
    assert len(llm.requests) == 1


def test_an_empty_plain_reply_is_rerolled_when_asked(llm):
    assert ask(llm.replies("", "", "ok"), reroll_empty=True) == "ok"
    first, *rerolls = llm.requests
    assert len(rerolls) == 2
    assert all(reroll["messages"] == first["messages"] for reroll in rerolls)


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


def test_a_recorded_call_carries_litellms_price_for_it(llm, monkeypatch, tmp_path):
    usd_in, usd_out = 1.0, 10.0
    entry = {
        "input_cost_per_token": usd_in,
        "output_cost_per_token": usd_out,
        "litellm_provider": "openai",
        "mode": "chat",
    }
    monkeypatch.setitem(litellm.model_cost, "openai/priced", entry)
    run = StageRun(stage="judge", task_logger=TaskLogger("run", root=tmp_path))

    litellm_chat_completion(
        system_prompt="s",
        user_prompt="u",
        model="openai/priced",
        completion=llm.replies("ok"),
        run=run,
    )

    [step] = run.steps
    assert (
        step.usd == step.tokens["prompt"] * usd_in + step.tokens["completion"] * usd_out
    )


def test_a_call_litellm_cannot_price_is_recorded_without_a_cost(llm, tmp_path):
    run = StageRun(stage="judge", task_logger=TaskLogger("run", root=tmp_path))

    litellm_chat_completion(
        system_prompt="s",
        user_prompt="u",
        model=VLLM,
        completion=llm.replies("ok"),
        run=run,
    )

    assert [step.usd for step in run.steps] == [None]


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
