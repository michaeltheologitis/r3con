import os

# Before litellm is imported: otherwise it fetches its price map from GitHub, and
# collection runs outside pytest-socket's reach.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import copy
import json
import logging
import re
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
MEMOS = Path(__file__).resolve().parents[1] / "examples" / "memos"
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
    ``refuses_over(chars)`` makes it a model with a window: a longer request is
    refused as too long instead of answered; ``refuses_over(tokens=...)`` counts the
    request as a provider does, in tokens. ``refused`` holds every request refused as
    too long.
    """

    def __init__(self) -> None:
        self.requests: list[Request] = []
        self.refused: list[Request] = []
        self.peak_in_flight = 0
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

    def __call__(self, **request: Any) -> litellm.ModelResponse:
        snapshot = copy.deepcopy(request)
        with self._lock:
            self.requests.append(snapshot)
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
            if isinstance(reply, litellm.ContextWindowExceededError):
                self.refused.append(snapshot)
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


def request_tokens(request: Request) -> int:
    """A request's cl100k tokens as a provider counts them: its messages' contents and
    its response format's schema as JSON."""
    texts = [message["content"] for message in request["messages"]]
    if "response_format" in request:
        texts.append(json.dumps(request["response_format"]["json_schema"]["schema"]))
    return sum(len(litellm.encode(text=text)) for text in texts)


def too_long(limit: int, size: int, model: str) -> litellm.ContextWindowExceededError:
    """The error litellm raises for a provider that refuses a request as too long."""
    return litellm.ContextWindowExceededError(
        message=(
            f"This model's maximum context length is {limit} tokens. However, you "
            f"requested {size} tokens in the messages."
        ),
        model=model,
        llm_provider="hosted_vllm",
    )


def grow_registry(chars: int, at: float = 0.6) -> str:
    """The contractor registry memo grown to about ``chars`` characters by filler
    paragraphs, with its heading first and its code lines and footer at fraction
    ``at``. The filler names no contractor and counts nothing."""
    heading, body = (MEMOS / "04_contractor_registry.txt").read_text().split("\n\n", 1)
    room = chars - len(heading) - len(body)
    before = _filler(0, round(room * at))
    after = _filler(len(before), room - _size(before))
    return _PARAGRAPH.join([heading, *before, body.strip(), *after])


def _filler(start: int, chars: int) -> list[str]:
    """Paragraphs of procurement and insurance boilerplate, each tagged uniquely from
    ``start`` on, about ``chars`` characters with their breaks."""
    paragraphs: list[str] = []
    while _size(paragraphs) < chars:
        i = start + len(paragraphs)
        tag = "".join(chr(ord("a") + int(digit)) for digit in str(i))
        paragraphs.append(_BOILERPLATE[i % len(_BOILERPLATE)].format(tag=tag))
    return paragraphs


def _size(paragraphs: list[str]) -> int:
    return sum(len(paragraph) + len(_PARAGRAPH) for paragraph in paragraphs)


_PARAGRAPH = "\n\n"
_BOILERPLATE = (
    (
        "Procurement note, {tag}. Purchase orders above the delegated limit need a second "
        "signature from the regional office before a contractor is engaged. Quotations are "
        "kept on file for the period the finance policy sets, and a contractor that "
        "declines to quote is recorded as such rather than left out."
    ),
    (
        "Insurance note, {tag}. Each approved contractor holds public liability and "
        "employer's liability cover at the levels the framework agreement sets. "
        "Certificates are renewed every year and checked by the compliance team; a lapsed "
        "certificate suspends new work orders until a current one arrives."
    ),
    (
        "Onboarding note, {tag}. A new contractor completes the site induction, the "
        "permit-to-work briefing and the lone-working assessment before a first visit. "
        "Induction records are held by the site office and are not reproduced in this "
        "extract."
    ),
    (
        "Payment note, {tag}. Invoices are matched against the work order and the "
        "completion sheet signed on site. A disputed line is held, not rejected, and the "
        "contractor is told in writing which line is held and why."
    ),
    (
        "Review note, {tag}. The facilities board reviews this registry every quarter. "
        "Changes to scope, rates or contact details take effect from the first day of the "
        "following month and are sent to every site manager."
    ),
    (
        "Records note, {tag}. This extract leaves out rates, bank details and named "
        "contacts, which are held in the procurement system. Requests for the full record "
        "go to the facilities administrator."
    ),
)


_SITES = (
    "Ashby",
    "Brook",
    "Calder",
    "Dunmore",
    "Elsworth",
    "Fairlie",
    "Glenholm",
    "Harrow",
    "Ingle",
    "Jarrow",
    "Kelso",
    "Linton",
)


def quiet_site_memos(n: int) -> list[str]:
    """``n`` site memos of about 230 characters, each under its own site name, that log
    no equipment incident, under CT-204 (even k) or CT-311 (odd k): added to the five
    memos, they cannot change the answer."""
    return [
        f"MEMO — {_SITES[k % 12]} {k:03d} site, Q3 operations review\n\n"
        "Equipment incidents logged this quarter: 0.\nNo faults were reported.\n"
        "Planned maintenance at this site is carried out under contractor code "
        f"{'CT-204' if k % 2 == 0 else 'CT-311'}.\n"
        f"Site manager: {_SITES[(k * 7) % 12]} Hale."
        for k in range(n)
    ]


_WORD_BUDGET = re.compile(r"Keep it under (\d+) words")


def keeping_to_budget(
    reply: str | Callable[[Request], str],
) -> Callable[[Request], str]:
    """A relevance reply from a model that obeys a word budget: when the system prompt
    asks for fewer than W words, the reply's first W words."""

    def obeying(request: Request) -> str:
        note = reply(request) if callable(reply) else reply
        budget = _WORD_BUDGET.search(request["messages"][0]["content"])
        return note if budget is None else " ".join(note.split()[: int(budget[1])])

    return obeying


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
def grown_registry() -> Callable[..., str]:
    return grow_registry


@pytest.fixture
def quiet_sites() -> Callable[[int], list[str]]:
    return quiet_site_memos


@pytest.fixture
def within_budget() -> Callable[..., Callable[[Request], str]]:
    return keeping_to_budget


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
