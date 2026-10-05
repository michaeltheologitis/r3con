import litellm
import pytest
from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info
from pydantic import BaseModel

from r3con.runtime.llm import litellm_chat_completion, litellm_chat_completion_full
from r3con.settings import settings

MODEL = "openai/gpt-5.6-luna"


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


def ask(llm, **kwargs):
    return litellm_chat_completion(
        system_prompt="s", user_prompt="u", model=MODEL, completion=llm, **kwargs
    )


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


def test_a_dict_field_is_sent_without_properties_left_open(llm):
    by_site = sent_schema(llm, Counts)["properties"]["by_site"]
    assert "required" not in by_site
    assert by_site["additionalProperties"] == {"type": "integer"}


def test_the_retry_count_sent_is_read_from_settings_when_the_call_runs(
    llm, monkeypatch
):
    monkeypatch.setattr(settings, "LLM_NUM_RETRIES", 3)
    ask(llm.replies("ok"))
    assert llm.requests[0]["num_retries"] == 3


def test_a_callers_retry_count_wins(llm):
    ask(llm.replies("ok"), num_retries=2)
    assert llm.requests[0]["num_retries"] == 2


def test_an_empty_structured_reply_is_rerolled_with_the_next_seed(llm):
    out = ask(llm.replies("", "", '{"x": 7}'), schema=Number, seed=0)
    assert out == Number(x=7)
    assert [request["seed"] for request in llm.requests] == [0, 1, 2]


def test_empty_structured_replies_raise_once_the_rerolls_run_out(llm):
    llm.replies("", "", "", "")
    with pytest.raises(ValueError, match="empty text after 4 attempt"):
        ask(llm, schema=Number, seed=0, max_empty_retries=3)
    assert len(llm.requests) == 4


def test_an_empty_plain_reply_is_returned_as_it_is(llm):
    assert ask(llm.replies("")) == ""
    assert len(llm.requests) == 1


def test_a_structured_reply_that_parses_is_not_rerolled(llm):
    assert ask(llm.replies('{"x": 1}'), schema=Number, seed=5) == Number(x=1)
    assert [request["seed"] for request in llm.requests] == [5]


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
def test_a_transient_server_error_is_retried(httpserver):
    error = {"error": {"message": "scripted 500", "type": "server_error"}}
    message = {"role": "assistant", "content": "OK"}
    for status, body in [
        (500, error),
        (500, error),
        (200, {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}]}),
    ]:
        httpserver.expect_ordered_request("/v1/chat/completions").respond_with_json(
            body, status=status
        )
    reply = litellm_chat_completion(
        system_prompt="s",
        user_prompt="u",
        model="hosted_vllm/fake",
        api_base=httpserver.url_for("/v1"),
        api_key="sk-fake",
    )
    assert reply == "OK"
    assert len(httpserver.log) == 3


def test_litellm_reads_its_bundled_price_map():
    assert len(litellm.model_cost) > 0
    source = get_model_cost_map_source_info()
    assert source["source"] == "local"
    assert source["is_env_forced"] is True
