import os

# Before litellm is imported: otherwise it fetches its price map from GitHub, and
# collection runs outside pytest-socket's reach.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import copy
import json
import logging
import threading
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, Literal

import litellm
import pytest

# Its import-time IPv6 probe opens a socket: here, before sockets are blocked.
import urllib3  # noqa: F401
import yaml

litellm.suppress_debug_info = True
_completion = litellm.completion

Stage = Literal["relevance", "schema", "parsing", "reasoning"]
Request = dict[str, Any]
Reply = str | BaseException | Callable[[Request], str]

R3CON_VARIABLES = (
    "R3CON_LOGS_DIR",
    "R3CON_DOC_WORKERS",
    "R3CON_PROMPTS_DIR",
    "R3CON_CONFIGS_DIR",
    "R3CON_LOG_LEVEL",
)
USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
ANSWERING_SCHEMA = """Thought: one row per person.
<schema>
class Row(BaseModel):
    who: str


class Parse(BaseModel):
    rows: list[Row]
</schema>"""
ANSWERING_CODE = (
    '<code>\nfinal_answer("; ".join(r["who"] for r in parse["rows"]))\n</code>'
)


class UnscriptedRequest(AssertionError):
    """A request reached a fake that had no reply for it."""


def stage_of(request: Request) -> Stage:
    """The pipeline stage that built ``request``, read from the shape r3con gives it."""
    if "response_format" in request:
        return "parsing"
    messages = request["messages"]
    if len(messages) > 2:
        return "reasoning"
    user = messages[1]["content"]
    if user.startswith("Input:\n<task>"):
        return "schema" if "Output:" in user else "reasoning"
    return "relevance"


class FakeLLM:
    """A scripted ``litellm.completion`` that answers through litellm's own mock.

    ``replies(...)`` queues replies for whichever requests come next; ``answers(...)``
    sets each stage's reply for when the queue is empty (one reply for every request
    of that stage, or a list consumed in order). A reply is the response's content, an
    exception to raise, or a callable from the request to the content.
    """

    def __init__(self) -> None:
        self.requests: list[Request] = []
        self.peak_in_flight = 0
        self._queue: deque[Reply] = deque()
        self._by_stage: dict[str, Reply | deque[Reply]] = {}
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

    def requests_for(self, stage: Stage) -> list[Request]:
        return [request for request in self.requests if stage_of(request) == stage]

    def __call__(self, **request: Any) -> litellm.ModelResponse:
        snapshot = copy.deepcopy(request)
        with self._lock:
            self.requests.append(snapshot)
            reply = self._next_reply(snapshot)
            self._in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self._in_flight)
        try:
            if isinstance(reply, BaseException):
                raise reply
            content = reply(copy.deepcopy(snapshot)) if callable(reply) else reply
            # The dict form, because litellm treats mock_response="" as absent.
            choice = {
                "index": 0,
                "finish_reason": "stop",
                "message": {"content": content},
            }
            return _completion(
                **request, mock_response={"choices": [choice], "usage": USAGE}
            )
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
            last = request["messages"][-1]["content"][:200]
            raise UnscriptedRequest(
                f"no reply scripted for this {stage} request: {last!r}"
            )
        return answer


def _first_line(request: Request) -> str:
    return request["messages"][1]["content"].splitlines()[0]


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def answering_llm(llm: FakeLLM) -> FakeLLM:
    """``llm``, answering every stage so that any run completes: each document's
    first line is its note and its one parsed row, and the answer joins the rows."""
    return llm.answers(
        relevance=lambda request: f"Notes on {_first_line(request)}.",
        schema=ANSWERING_SCHEMA,
        parsing=lambda request: json.dumps({"rows": [{"who": _first_line(request)}]}),
        reasoning=ANSWERING_CODE,
    )


@pytest.fixture
def window(monkeypatch: pytest.MonkeyPatch) -> Callable[[int], str]:
    """Maps a model with an input window of ``tokens`` in litellm's model map and
    returns its name. litellm caches its lookup per name, so the name carries the
    window: one name always means one window, across tests."""

    def register(tokens: int) -> str:
        model = f"hosted_vllm/window-{tokens}"
        entry = {"max_input_tokens": tokens, "litellm_provider": "hosted_vllm"}
        monkeypatch.setitem(litellm.model_cost, model, {**entry, "mode": "chat"})
        return model

    return register


@pytest.fixture
def configs(tmp_path):
    """Writes ``name -> fields`` into the working directory's config overlay."""
    root = tmp_path / "configs"
    root.mkdir()

    def write(**by_name: dict) -> None:
        for name, fields in by_name.items():
            (root / f"{name}.yaml").write_text(yaml.safe_dump(fields))

    return write


@pytest.fixture(autouse=True)
def _isolated(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Iterator[None]:
    for name in R3CON_VARIABLES:
        # Set, then delete, so the undo also removes a value the code under test writes.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.chdir(tmp_path)
    if not any(request.node.get_closest_marker(m) for m in ("live", "allow_hosts")):
        monkeypatch.setattr(litellm, "completion", FakeLLM())
    logger = logging.getLogger("r3con")
    level, propagate, handlers = logger.level, logger.propagate, list(logger.handlers)
    yield
    logger.setLevel(level)
    logger.propagate = propagate
    logger.handlers[:] = handlers


@pytest.fixture(scope="session")
def httpserver_listen_address() -> tuple[str, int]:
    """Serve pytest-httpserver on 127.0.0.1, the host the tests' ``allow_hosts`` names.

    Its default, ``localhost``, resolves to ::1 first on GitHub's runners, and
    pytest-socket refuses that connection.
    """
    return ("127.0.0.1", 0)
