import copy
import json
import os
import re
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import litellm
import toga
from keyring.backend import KeyringBackend
from keyring.compat import properties
from keyring.errors import PasswordDeleteError
from litellm.exceptions import AuthenticationError, ContextWindowExceededError
from pydantic import BaseModel

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
TWO_LINE_MEMOS = {
    "q3/halloran.txt": "Halloran memo\nThe Halloran memo logs two sites.",
    "q3/merrow.txt": "Merrow memo\nThe Merrow memo logs two sites.",
    "calder.pdf": "Calder memo\nThe Calder memo logs two sites.",
}
"""Three memos of two lines each, by name."""
SEARCH = "https://memos.search.windows.net"
# A key of unquoted digits, which YAML reads as an integer.
DIGITS_KEY = 123456789012345678
# Where a process a test starts finds FileKeyring's file.
KEYRING_FILE = "R3CON_TEST_KEYRING_FILE"
# Below keyring's null store's, -1, and its failing one's, 0.
BELOW_ANY_PICKED = -2


class GatewayKey(BaseModel):
    """A caller's own credential object, passed among a run's params."""

    api_key: str


def on_your_data(parameters: dict) -> dict:
    """Params with an Azure OpenAI "On Your Data" search source of ``parameters``."""
    source = {"endpoint": SEARCH, "index_name": "memos", **parameters}
    return {
        "extra_body": {"data_sources": [{"type": "azure_search", "parameters": source}]}
    }


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


ANSWERING_CODE = (
    '<code>\nfinal_answer("; ".join(r["who"] for r in parse["rows"]), '
    'provenance_ids={"summary_ids": [], '
    '"parse_ids": [r["document"] for r in parse["rows"]]})\n</code>'
)


def first_line(request: Request) -> str:
    """The first line of the document a request sends, below a parse request's
    ``Document ID`` line."""
    return PARSE_HEADING.sub("", request["messages"][1]["content"]).splitlines()[0]


def joined_rows(request: Request) -> Content:
    """A reasoning reply that joins the parse's rows and cites their documents for
    their records: on the coding path, code that does; on the direct path, the rows of
    the parse its prompt shows, joined, and their documents' ids."""
    if "response_format" not in request:
        return ANSWERING_CODE
    system = request["messages"][0]["content"]
    shown = system.split("## The parsed data (`parse`)", 1)[1].split("\n## Rules")[0]
    rows = json.loads(shown[shown.index("{") : shown.rindex("}") + 1])["rows"]
    return {
        "answer": "; ".join(row["who"] for row in rows),
        "provenance_ids": {
            "summary_ids": [],
            "parse_ids": [row["document"] for row in rows],
        },
    }


ANSWERING: dict[str, Content | Callable[[Request], Content]] = {
    "relevance": lambda request: f"Notes on {first_line(request)}.",
    "schema": ANSWERING_SCHEMA,
    "parsing": lambda request: {"rows": [{"who": first_line(request)}]},
    "reasoning": joined_rows,
}
"""What ``answering_llm`` answers each stage with, so that any run completes: each
document's first line is its note and its one parsed row, and the answer joins the
rows, on either reasoning path."""


def failing_once(
    when: Callable[[Request], bool], otherwise: Content | Callable[[Request], Content]
) -> Reply:
    """A reply that refuses the first request ``when`` holds for, as a provider whose
    key is refused does, and answers every other with ``otherwise``."""
    refused: list[Request] = []

    def reply(request: Request) -> Content:
        if not refused and when(request):
            refused.append(request)
            raise bad_key()
        return otherwise(request) if callable(otherwise) else otherwise

    return reply


def reading(line: str, *, later_round: bool = False) -> Callable[[Request], bool]:
    """Whether a request reads the document whose first line is ``line``, in a later
    relevance round only with ``later_round``."""
    return lambda request: (
        first_line(request) == line and (is_later_round(request) or not later_round)
    )


def predating_resume(run_dir: Path) -> None:
    """``run_dir`` as a run folder from before resume leaves it: no ``documents.json``."""
    (run_dir / "documents.json").unlink()


def reading_moved_prompts(run_dir: Path) -> None:
    """A ``./prompts`` overlay of the parsing prompt the run read, which shadows it."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    stage = "structuring/parsing"
    read = Path(manifest["sources"]["prompts"][stage])
    overlay = Path("prompts") / stage / read.name
    overlay.parent.mkdir(parents=True)
    overlay.write_text(read.read_text())


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


def without_trail(provenance: Mapping[str, Any]) -> dict[str, Any]:
    """An answer's ``provenance`` without its ``trail``, which ``test_provenance.py``
    pins."""
    return {key: value for key, value in provenance.items() if key != "trail"}


def quote(text: str, start: int | None, end: int | None, match: str | None) -> dict:
    """A snippet's quote of ``text``, found from ``start`` to ``end`` by ``match``."""
    return {"text": text, "start": start, "end": end, "match": match, "score": None}


def memo_line(document_ids: Sequence[str], i: int, k: int) -> dict[str, Any]:
    """Line ``k`` of the ``i``-th of ``TWO_LINE_MEMOS``, as a quote behind an answer
    over them, ``document_ids`` giving each memo's id."""
    name, memo = list(TWO_LINE_MEMOS.items())[i]
    text = memo.splitlines()[k]
    start = memo.index(text)
    return {
        "document": document_ids[i],
        "name": name,
        "start": start,
        "end": start + len(text),
        "text": text,
    }


def quoting_a_line_a_round(
    shown_ids: Callable[[Request], list[str]],
) -> Callable[[Request], Content]:
    """A relevance reply quoting the document's first line in round 1 and its second
    later, then citing the first id shown."""

    def reply(request: Request) -> Content:
        first, second = request["messages"][1]["content"].splitlines()
        if not is_later_round(request):
            return {"summary": f"Notes on {first}.", "quotes": [first]}
        cites = shown_ids(request)[:1]
        return {
            "summary": f"Notes on {first}.",
            "quotes": [second],
            "provenance_ids": cites,
        }

    return reply


def quoting_its_first_line_and_citing_all_shown(
    shown_ids: Callable[[Request], list[str]],
) -> Callable[[Request], Content]:
    """A relevance reply quoting the first line of the text it is sent, and, in a
    later round, citing every summary shown."""

    def reply(request: Request) -> Content:
        first = request["messages"][1]["content"].splitlines()[0]
        quoting = {"summary": f"Notes on {first}.", "quotes": [first]}
        if not is_later_round(request):
            return quoting
        return {**quoting, "provenance_ids": shown_ids(request)}

    return reply


def citing(
    answer: str, summaries: slice = slice(0), parses: slice = slice(0)
) -> Callable[[Request], Content]:
    """A direct reasoning reply of ``answer`` citing the ``summaries`` and the
    ``parses`` of the ids its schema lists, none by default."""

    def reply(request: Request) -> Content:
        lists = citable_lists(request)

        def listed(view: str) -> list[str]:
            return lists[view]["items"].get("enum", [])

        return {
            "answer": answer,
            "provenance_ids": {
                "summary_ids": listed("summary_ids")[summaries],
                "parse_ids": listed("parse_ids")[parses],
            },
        }

    return reply


def in_json(name: str, change: Callable[[Any], None]) -> Callable[[Path], None]:
    """Rewrites the JSON file ``name`` of a run folder after ``change``."""

    def rewrite(run_dir: Path) -> None:
        data = json.loads((run_dir / name).read_text())
        change(data)
        (run_dir / name).write_text(json.dumps(data))

    return rewrite


def a_parse_citation_short(result: dict) -> None:
    """The parse's citations with the last document's left out."""
    result["provenance"] = result["provenance"][:-1]


def snippets_without_quotes(result: dict) -> None:
    """As r3context 0.6.4 wrote them."""
    for round_ in result["rounds"]:
        for snippet in round_["snippets"]:
            del snippet["quotes"]


def bad_key() -> AuthenticationError:
    return AuthenticationError(
        message="bad key", llm_provider="openai", model="gpt-6-luna"
    )


class FakeKeyring(KeyringBackend):
    """A credential store in memory, for keyring's ``set_keyring``: it answers as
    keyring's backends do, and holds each password by its service and account in
    ``passwords``. With no ``priority``, keyring never picks it by itself."""

    def __init__(self) -> None:
        super().__init__()
        self.passwords: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.passwords.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.passwords[service, username] = password

    def delete_password(self, service: str, username: str) -> None:
        if self.passwords.pop((service, username), None) is None:
            raise PasswordDeleteError(f"No {username} in {service}")


class FileKeyring(KeyringBackend):
    """A credential store in the JSON file the ``KEYRING_FILE`` variable names, as
    ``{service: {account: password}}``, for a process a test starts through keyring's
    own ``PYTHON_KEYRING_BACKEND``, so the test can read what it saved. It reads and
    saves; keyring never picks it by itself."""

    # The way keyring's own backends give theirs.
    @properties.classproperty
    def priority(cls) -> float:
        return BELOW_ANY_PICKED

    def _stored(self) -> dict[str, dict[str, str]]:
        path = Path(os.environ[KEYRING_FILE])
        return json.loads(path.read_text()) if path.exists() else {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._stored().get(service, {}).get(username)

    def set_password(self, service: str, username: str, password: str) -> None:
        stored = self._stored()
        stored.setdefault(service, {})[username] = password
        Path(os.environ[KEYRING_FILE]).write_text(json.dumps(stored))


class FakeCommand:
    """Toga's command, which keeps what it is made with: ``action`` is its handler."""

    ABOUT = "about"

    def __init__(
        self,
        action: Callable[..., None] | None,
        text: str,
        *,
        group: toga.Group,
        section: int = 0,
        order: int = 0,
        id: str | None = None,
    ) -> None:
        self.action = action
        self.text = text
        self.group, self.section, self.order = group, section, order
        self.id = id or text


class FakeCommands(dict[str, FakeCommand]):
    """Toga's set of an app's commands, by id, which its menus show. Toga 0.5.7 adds
    its About command before the app's startup, which on the Mac is the app menu's
    first, in section -1."""

    def __init__(self) -> None:
        about = FakeCommand(
            None, "About", group=toga.Group.APP, section=-1, id=FakeCommand.ABOUT
        )
        super().__init__({about.id: about})

    def add(self, *commands: FakeCommand) -> None:
        self.update((command.id, command) for command in commands)


class FakeUpdaterController:
    """Sparkle's SPUStandardUpdaterController. Its init answers nil unless it
    ``initializes``; its updater can check for updates once started, unless it
    ``fails`` to start; and it counts the checks it is asked for."""

    def __init__(self, *, initializes: bool, fails: bool) -> None:
        self.initializes, self.fails = initializes, fails
        self.updater: SimpleNamespace | None = None
        self.checks = 0

    def initWithStartingUpdater(
        self, start: bool, *, updaterDelegate: object, userDriverDelegate: object
    ) -> "FakeUpdaterController | None":
        self.updater = SimpleNamespace(canCheckForUpdates=start and not self.fails)
        return self if self.initializes else None

    def checkForUpdates(self, sender: object) -> None:
        self.checks += 1


class FakeObjC:
    """rubicon.objc's ``ObjCClass``, as the Mac app's Objective-C runtime answers it.
    The main bundle's private Frameworks folder is ``frameworks``. NSBundle finds a
    bundle at a path only where there is a folder. Sparkle's loads unless ``loads`` is
    False, which makes its updater controller found unless ``has_controller`` is False.
    Each controller made is in ``controllers``; one answers its init with nil if
    ``initializes`` is False, and its updater fails to start if ``starts`` is False."""

    def __init__(self, frameworks: Path) -> None:
        self.frameworks = frameworks
        self.loads = self.has_controller = self.initializes = self.starts = True
        self.controllers: list[FakeUpdaterController] = []
        main_bundle = SimpleNamespace(privateFrameworksPath=str(frameworks))
        self.classes: dict[str, SimpleNamespace] = {
            "NSBundle": SimpleNamespace(
                mainBundle=main_bundle, bundleWithPath=self._bundle_with_path
            )
        }

    def ObjCClass(self, name: str) -> SimpleNamespace:
        try:
            return self.classes[name]
        except KeyError:
            # As rubicon raises for a class the runtime does not have.
            raise NameError(f"ObjC Class '{name}' couldn't be found.") from None

    def _bundle_with_path(self, path: str) -> SimpleNamespace | None:
        return SimpleNamespace(load=self._load) if Path(path).is_dir() else None

    def _load(self) -> bool:
        if self.loads and self.has_controller:
            controller = SimpleNamespace(alloc=self._alloc)
            self.classes["SPUStandardUpdaterController"] = controller
        return self.loads

    def _alloc(self) -> FakeUpdaterController:
        controller = FakeUpdaterController(
            initializes=self.initializes, fails=not self.starts
        )
        self.controllers.append(controller)
        return controller
