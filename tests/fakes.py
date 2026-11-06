import copy
import json
import re
import threading
from collections import deque
from collections.abc import Callable, Mapping
from typing import Any, Literal

import litellm
from litellm.exceptions import ContextWindowExceededError

_completion = litellm.completion

Stage = Literal["relevance", "schema", "parsing", "reasoning"]
Request = dict[str, Any]
Content = str | dict[str, Any]
Reply = Content | BaseException | Callable[[Request], Content]
USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
FINGERPRINT = "fp_fake"
SCHEMA_REPLY = """Thought: one row per memo, with its lines.
<schema>
class Line(BaseModel):
    text: str


class Memo(BaseModel):
    title: str
    lines: list[Line]


class Parse(BaseModel):
    memos: list[Memo]
</schema>"""
ANSWERING_SCHEMA = """Thought: one row per person.
<schema>
class Row(BaseModel):
    who: str


class Parse(BaseModel):
    rows: list[Row]
</schema>"""
# The line, and the blank one after it, that heads a parse request's user message.
PARSE_HEADING = re.compile(r"\ADocument ID: (\w+)\n\n")
GATE_TIMEOUT_SECONDS = 10
UNSCRIPTED_SHOWN_CHARS = 200


class UnscriptedRequest(AssertionError):
    """A request reached a fake that had no reply for it."""


def _schema_name(request: Request) -> str | None:
    """The name of the schema ``request`` asks its reply to fit, if any."""
    return request.get("response_format", {}).get("json_schema", {}).get("name")


def stage_of(request: Request) -> Stage:
    """The pipeline stage that built ``request``, read from the shape r3con gives it."""
    if "response_format" in request:
        schema = request["response_format"]["json_schema"]["name"]
        by_schema: dict[str, Stage] = {
            "RelevanceSummary": "relevance",
            "ReasoningAnswer": "reasoning",
        }
        return by_schema.get(schema, "parsing")
    messages = request["messages"]
    if len(messages) > 2:
        return "reasoning"
    return "schema" if "Output:" in messages[1]["content"] else "reasoning"


class FakeLLM:
    """A scripted ``litellm.completion`` that answers through litellm's own mock.

    ``replies(...)`` queues replies for whichever requests come next; ``answers(...)``
    sets each stage's reply for when the queue is empty (one reply for every request
    of that stage, or a list consumed in order). A reply is the response's content (a
    dict is sent as its JSON), an exception to raise, or a callable from the request to
    the content. A relevance request gets a text reply as its summary, quoting and
    citing nothing; a parsing request for ``{parse, provenance_ids}`` gets a dict
    reply without ``parse`` as its parse, citing nothing; a direct reasoning request
    gets a text reply as it is.
    ``refuses_over(chars)`` makes it a model with a window: a longer request is
    refused as too long instead of answered; ``refuses_over(tokens=...)`` counts the
    request as a provider does, in tokens.
    Each request sets ``called`` as it arrives, its reply already chosen, then waits
    at ``gate``, open until a test clears it, for at most ``GATE_TIMEOUT_SECONDS``
    before it is answered.
    """

    def __init__(self) -> None:
        self.requests: list[Request] = []
        self.peak_in_flight = 0
        self.called = threading.Event()
        self.gate = threading.Event()
        self.gate.set()
        self._queue: deque[Reply] = deque()
        self._by_stage: dict[str, Reply | deque[Reply]] = {}
        self._limit: int | None = None
        self._token_limit: int | None = None
        self._in_flight = 0
        self._lock = threading.Lock()

    def replies(self, *replies: Reply) -> "FakeLLM":
        with self._lock:
            self._queue.extend(replies)
        return self

    def answers(self, **by_stage: Reply | list[Reply]) -> "FakeLLM":
        with self._lock:
            for stage, reply in by_stage.items():
                self._by_stage[stage] = (
                    deque(reply) if isinstance(reply, list) else reply
                )
        return self

    def refuses_over(
        self, chars: int | None = None, *, tokens: int | None = None
    ) -> "FakeLLM":
        """From now on, a request whose messages total more than ``chars`` characters,
        or more than ``tokens`` tokens with its response format (``request_tokens``), is
        recorded and refused with litellm's ``ContextWindowExceededError``, without
        using up a scripted reply."""
        with self._lock:
            self._limit, self._token_limit = chars, tokens
        return self

    def requests_for(self, stage: Stage) -> list[Request]:
        return [request for request in self.requests if stage_of(request) == stage]

    def __call__(self, **request: Any) -> Any:
        snapshot = copy.deepcopy(request)
        with self._lock:
            self.requests.append(snapshot)
            self.called.set()
            size = sum(len(message["content"]) for message in request["messages"])
            if self._limit is not None and size > self._limit:
                reply = too_long(self._limit, size, request["model"])
            elif (
                self._token_limit is not None
                and (count := request_tokens(request)) > self._token_limit
            ):
                reply = too_long(self._token_limit, count, request["model"])
            else:
                reply = self._next_reply(snapshot)
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            self.gate.wait(GATE_TIMEOUT_SECONDS)
            if isinstance(reply, BaseException):
                raise reply
            content = reply(copy.deepcopy(snapshot)) if callable(reply) else reply
            if (
                isinstance(content, str)
                and _schema_name(snapshot) == "RelevanceSummary"
            ):
                content = _summary_reply(content, snapshot)
            if (
                isinstance(content, dict)
                and "parse" not in content
                and _schema_name(snapshot) == "ParseReply"
            ):
                content = {"parse": content, "provenance_ids": []}
            if isinstance(content, dict):
                content = json.dumps(content)
            return mock_completion(request, content)
        finally:
            with self._lock:
                self._in_flight -= 1

    def _next_reply(self, request: Request) -> Reply:
        if self._queue:
            return self._queue.popleft()
        stage = stage_of(request)
        answer = self._by_stage.get(stage)
        if isinstance(answer, deque):
            answer = answer.popleft() if answer else None
        if answer is None:
            raise unscripted(request)
        return answer


def unscripted(request: Request) -> UnscriptedRequest:
    """The error for a ``request`` that a fake has no reply for."""
    last = request["messages"][-1]["content"][:UNSCRIPTED_SHOWN_CHARS]
    return UnscriptedRequest(
        f"no reply scripted for this {stage_of(request)} request: {last!r}"
    )


def mock_completion(request: Request, content: str) -> Any:
    """litellm's response to ``request`` when its own mock replies ``content``, the
    reply's body handed to the request's ``logger_fn``, as litellm hands a server's:
    its model the model asked for, without the provider."""
    # The dict form, because litellm treats mock_response="" as absent.
    choice = {"index": 0, "finish_reason": "stop", "message": {"content": content}}
    body = {
        "model": request["model"].partition("/")[2] or request["model"],
        "system_fingerprint": FINGERPRINT,
        "choices": [choice],
        "usage": USAGE,
    }
    response = _completion(**request, mock_response=body)
    # litellm's mock skips the call a real send makes with the body it got.
    logger_fn = request.get("logger_fn")
    if logger_fn is not None:
        logger_fn(
            {"log_event_type": "post_api_call", "original_response": json.dumps(body)}
        )
    return response


def _summary_reply(summary: str, request: Request) -> dict[str, Any]:
    """A relevance reply of ``summary``, quoting and citing nothing, in the fields
    ``request``'s schema has: no ``provenance_ids`` in round 1."""
    fields = request["response_format"]["json_schema"]["schema"]["properties"]
    reply = {"summary": summary, "quotes": [], "provenance_ids": []}
    return {field: reply[field] for field in fields}


def is_later_round(request: Request) -> bool:
    """Whether a relevance ``request`` is a later round's, whose reply cites the
    summaries it shows."""
    schema = request["response_format"]["json_schema"]["schema"]
    return "provenance_ids" in schema["properties"]


def citable_lists(request: Request) -> dict[str, Any]:
    """The JSON schema of each list of ids a direct reasoning reply may cite."""
    schema = request["response_format"]["json_schema"]["schema"]
    return schema["$defs"]["ProvenanceIds"]["properties"]


def cited_documents(provenance: Mapping[str, Any]) -> set[str]:
    """The ids of the documents whose snippets or records an answer's ``provenance``
    cites."""
    views = (provenance["summary_ids"], provenance["parse_ids"])
    return {id_.partition(".")[0] for ids in views for id_ in ids}


def request_tokens(request: Request) -> int:
    """A request's cl100k tokens as a provider counts them: its messages' contents and
    its response format's schema as JSON."""
    texts = [message["content"] for message in request["messages"]]
    if "response_format" in request:
        texts.append(json.dumps(request["response_format"]["json_schema"]["schema"]))
    return sum(len(litellm.encode(text=text)) for text in texts)


def too_long(limit: int, size: int, model: str) -> ContextWindowExceededError:
    """The error litellm raises for a provider that refuses a request as too long."""
    return ContextWindowExceededError(
        message=(
            f"This model's maximum context length is {limit} tokens. However, you "
            f"requested {size} tokens in the messages."
        ),
        model=model,
        llm_provider="hosted_vllm",
    )
