import contextlib
import json
import logging
import re
import socket
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast, get_args

import httpx2
import keyring
import litellm
import pytest
import yaml
from keyring.backend import KeyringBackend
from keyring.backends import fail
from keyring.errors import KeyringLocked, PasswordDeleteError, PasswordSetError
from litellm.exceptions import RateLimitError
from starlette.testclient import TestClient
from starlette.types import ASGIApp

from r3con import r3con
from r3con.app.server import (
    APP_KEY_SERVICE,
    DEFAULT_RELEVANCE_ROUNDS,
    DIRECT_ONLY,
    ENCRYPTED_ZIP,
    KEYLESS_SERVER_KEY,
    MAC_APP_KEY_SERVICE,
    MAX_RELEVANCE_ROUNDS,
    MAX_ZIP_BYTES,
    MAX_ZIP_ENTRIES,
    NO_SAVE_DIALOG,
    NOT_ONE_INDEX_EACH,
    REPRESENTATION_FILES,
    TOO_MANY_BYTES,
    TOO_MANY_ENTRIES,
    UNREADABLE_IMPORT,
    UNREADABLE_RUN,
    UNREADABLE_ZIP,
    UNWRITABLE_IMPORT,
    UNWRITABLE_SAVE,
    ZIP_ENCRYPTED_FLAG,
    ChooseSavePath,
    ServerState,
    _server,
    create_app,
    listen,
    serve,
)
from r3con.masking import MASK
from r3con.r3con import read_documents
from r3con.settings import CONFIGS_DIR
from tests.documents import (
    CERTIFICATE_ENCRYPTION,
    CORRUPT_DOCX,
    write_docx,
    write_pdf,
    write_zip,
)
from tests.fakes import (
    GATE_TIMEOUT_SECONDS,
    SCHEMA_REPLY,
    FakeKeyring,
    FakeLLM,
    a_parse_citation_short,
    in_json,
)

litellm.suppress_debug_info = True

PORT = 8765
TOKEN = "the-launch-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BASE_URL = f"http://127.0.0.1:{PORT}"
MEMOS = Path(__file__).resolve().parents[2] / "examples" / "memos"
APP_DIR = Path(__file__).resolve().parents[2] / "src" / "r3con" / "app"
LONG_RETRY_AFTER_SECONDS = 60
# Far below GATE_TIMEOUT_SECONDS, which ends a held call by itself.
STILL_WAITING_SECONDS = 0.5
FORCED_EXIT_SECONDS = 5
WORKER_PREFIX = "r3con-app-"
QUESTION = "Which contractor's sites logged the most equipment incidents in Q3?"
MODEL = "openai/gpt-6-luna"
# A bare OpenAI name that litellm's model map places from its floor, 1.103, on.
BARE_OPENAI_MODEL = "gpt-5"
# litellm's way to a model an OpenAI-compatible server serves: openai/ and its name.
SERVED_OPENAI_MODEL = "openai/google/gemma-4-12B-it"
OPENAI_KEY = "sk-stored-openai-key-1234"
OTHER_KEY = "other-endpoint-key-5678"
ENVIRONMENT_KEY = "key-from-the-environment-9012"
ENDPOINT = "http://127.0.0.1:9/v1"
SECOND_ENDPOINT = "http://127.0.0.1:11/v1"
SECOND_KEY = "second-server-key-3456"
# A known provider's URL, by which litellm places a bare name with that provider.
GROQ_ENDPOINT = "https://api.groq.com/openai/v1"
CUSTOM_HEADERS = "Authorization: Bearer custom-header-key-7890\nX-Gateway-Key: gate"
# A header's value a config file's params may send, as an OpenAI-compatible gateway takes.
GATE_KEY = "gateway-header-key-2468"
STORED_OPENAI = {"provider": "openai", "key": OPENAI_KEY}
OTHER = {"provider": "other", "endpoint": ENDPOINT, "key": OTHER_KEY}
KEYLESS_OTHER = {"provider": "other", "endpoint": ENDPOINT}
SECOND_SERVER = {"provider": "other", "endpoint": SECOND_ENDPOINT, "key": SECOND_KEY}
ASK = {"question": "Which site logged the fewest?", "model": MODEL}
PDF_TEXT = "Incidents this quarter: 7."
CORRUPT_PDF = b"%PDF-1.4 truncated garbage"
REPORT = [
    "Incidents by site:",
    [["Site", "Incidents"], ["Halloran", "7"], ["Merrow", "3"]],
    "Merrow's count is provisional.",
]
DOCUMENT_TEXT = "/api/documents/text"
UNEXTRACTED = "name: must end in one of .pdf, .docx, .html, .htm"
DOCUMENT_ZIP = "/api/documents/zip"
ZIP_NAME = "memos.zip"
MEMO = "A memo.\n"
LONG_MEMO = {"memo.txt": "A memo, long enough to compress. " * 20}
# A zip entry's local header, before its name.
LOCAL_HEADER_BYTES = 30
# Where memo.txt's compressed bytes start: past its local header and its name.
LONG_MEMO_DATA_AT = LOCAL_HEADER_BYTES + len("memo.txt")
CENTRAL_DIRECTORY_ENTRY = b"PK\x01\x02"
LOCAL_ENTRY = b"PK\x03\x04"
# Where an entry's flags are in each of its two headers, and its claimed unzipped
# size in the central directory's.
CENTRAL_FLAGS_AT = 8
LOCAL_FLAGS_AT = 6
CENTRAL_SIZE_AT = 24
RUN_EVENTS = [
    "run_started",
    "relevance_started",
    "relevance_finished",
    "schema_started",
    "schema_finished",
    "parsing_started",
    "parsing_finished",
    "reasoning_started",
    "reasoning_finished",
    "run_finished",
]
ASK_EVENTS = ["ask_started", "reasoning_started", "reasoning_finished", "ask_finished"]
TOO_FEW_ROUNDS = "relevance_rounds: Input should be greater than or equal to 1"
TOO_MANY_ROUNDS = (
    f"relevance_rounds: Input should be less than or equal to {MAX_RELEVANCE_ROUNDS}"
)
NOT_A_ROUND_COUNT = "relevance_rounds: Input should be a valid integer"
ANSWER = {
    "answer": "Northgate",
    "provenance_ids": {"summary_ids": [], "parse_ids": []},
}
SAVES = "/api/saves"
IMPORTS = "/api/imports"
# A server a crafted file's params would send an imported run's asks to, with its key.
ELSEWHERE = "http://127.0.0.1:12/v1"
MAX_SAVE_BYTES = 1_000
SAVED_NAME = "which-site-logged-the-most-incidents.html"
CHOSEN_NAME = "chosen.html"
SAVED_TAB = b"<!doctype html><title>a saved tab</title>"
REMEMBERED_OPENAI = {"source": "here", "hint": OPENAI_KEY[-4:], "remembered": True}
UNTIL_CLOSED_OPENAI = {**REMEMBERED_OPENAI, "remembered": False}


class BrokenKeyring(KeyringBackend):
    """A credential store that fails every call, its error quoting what it was given,
    as a store's own might."""

    def get_password(self, service: str, username: str) -> str | None:
        raise KeyringLocked(f"Can't read {service}/{username}")

    def set_password(self, service: str, username: str, password: str) -> None:
        raise PasswordSetError(f"Can't store {password} as {service}/{username}")

    def delete_password(self, service: str, username: str) -> None:
        raise PasswordDeleteError(f"Can't delete {service}/{username}")


KEPT_UNTIL_CLOSED = "The openai key is kept only until the app closes"
MAY_LOAD_AGAIN = "The openai key may load again next launch"


def store_failed(what: str, error: str) -> str:
    """What the log says of a failed call to the credential store."""
    return f"{what}: the credential store failed ({error})."


# A store that fails, and keyring's own for a system where it finds none, each with
# what the log says as a launch reads it, and as a key is saved, then removed.
LAUNCH_FAILURES = [
    pytest.param(
        BrokenKeyring,
        [
            store_failed(
                f"The {provider} key saved earlier isn't loaded", "KeyringLocked"
            )
            for provider in ("openai", "anthropic", "gemini")
        ],
        id="failing",
    ),
    pytest.param(
        fail.Keyring,
        [store_failed("No key saved earlier is loaded", "NoKeyringError")],
        id="none",
    ),
]
SAVE_FAILURES = [
    pytest.param(
        BrokenKeyring,
        [
            store_failed(KEPT_UNTIL_CLOSED, "PasswordSetError"),
            store_failed(MAY_LOAD_AGAIN, "KeyringLocked"),
        ],
        id="failing",
    ),
    pytest.param(
        fail.Keyring,
        [store_failed(KEPT_UNTIL_CLOSED, "NoKeyringError")],
        id="none",
    ),
]


class UnwritableKeyring(FakeKeyring):
    """A store that reads and deletes, but fails to save."""

    def set_password(self, service: str, username: str, password: str) -> None:
        raise PasswordSetError("Can't store it")


class UndeletableKeyring(FakeKeyring):
    """A store that saves and reads, but fails to delete, quoting what it holds."""

    def delete_password(self, service: str, username: str) -> None:
        held = self.passwords[service, username]
        raise PasswordDeleteError(f"Can't delete {held}")


class LockedItemKeyring(FakeKeyring):
    """A store that fails to read its OpenAI key, as one whose dialog was denied."""

    def get_password(self, service: str, username: str) -> str | None:
        if username == "openai":
            raise KeyringLocked("Can't get password from keychain")
        return super().get_password(service, username)


@pytest.fixture
def llm() -> FakeLLM:
    """A model that answers every stage, so that a run or an ask over it finishes."""
    return FakeLLM().answers(
        relevance="A site memo.",
        schema=SCHEMA_REPLY,
        parsing={"memos": [{"title": "Northgate", "lines": []}]},
        reasoning=ANSWER,
    )


@pytest.fixture
def held_llm(llm: FakeLLM) -> FakeLLM:
    """``llm``, holding every request until its gate opens."""
    llm.gate.clear()
    return llm


@pytest.fixture
def one_call_at_a_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """r3con reads one document at a time, so a held model holds the only call."""
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def client_for(
    model: FakeLLM,
    logs_dir: Path,
    choose_save_path: ChooseSavePath | None = None,
    **options: Any,
) -> TestClient:
    state = ServerState(
        logs_dir=logs_dir, completion=model, choose_save_path=choose_save_path
    )
    app = create_app(token=TOKEN, port=PORT, state=state, **options)
    return TestClient(app, base_url=BASE_URL, headers=AUTH)


@pytest.fixture
def client(llm: FakeLLM, logs_dir: Path) -> Iterator[TestClient]:
    with client_for(llm, logs_dir) as client:
        yield client


def memos() -> list[str]:
    return [path.read_text() for path in sorted(MEMOS.iterdir())]


def run_body(model: str = MODEL, endpoint: str | None = None) -> dict[str, Any]:
    body = {"question": QUESTION, "model": model, "documents": memos()}
    return body if endpoint is None else {**body, "endpoint": endpoint}


def events_in(stream: str) -> list[dict[str, Any]]:
    """The events of a ``text/event-stream`` body, in order."""
    return [
        json.loads(line.removeprefix("data: "))
        for line in stream.splitlines()
        if line.startswith("data: ")
    ]


def start_run(
    client: TestClient, model: str = MODEL, endpoint: str | None = None
) -> str:
    response = client.post("/api/runs", json=run_body(model, endpoint))
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def events_of(client: TestClient, path: str) -> list[dict[str, Any]]:
    response = client.get(path)
    assert response.headers["content-type"].startswith("text/event-stream")
    return events_in(response.text)


def run_events(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    return events_of(client, f"/api/runs/{run_id}/events")


def ask_events(
    client: TestClient, run_id: str, ask: dict[str, Any] = ASK
) -> list[dict[str, Any]]:
    response = client.post(f"/api/runs/{run_id}/asks", json=ask)
    assert response.status_code == 202, response.text
    return events_of(client, f"/api/asks/{response.json()['ask_id']}/events")


def finished_run(
    client: TestClient, model: str = MODEL, endpoint: str | None = None
) -> str:
    run_id = start_run(client, model, endpoint)
    last = run_events(client, run_id)[-1]
    assert last["type"] == "run_finished", last
    return run_id


def text_of(client: TestClient, path: Path) -> httpx2.Response:
    """The server's answer for the text of the file at ``path``, sent as it is."""
    return client.post(
        DOCUMENT_TEXT, params={"name": path.name}, content=path.read_bytes()
    )


def write_page(path: Path) -> None:
    """A web page at ``path``: a heading, and a script its text leaves out."""
    path.write_text(f"<script>var site;</script><h1>{REPORT[0]}</h1>", encoding="utf-8")


def zipped(tmp_path: Path, entries: Mapping[str, str | bytes]) -> Path:
    path = tmp_path / ZIP_NAME
    write_zip(path, entries)
    return path


def unzip(client: TestClient, body: bytes) -> httpx2.Response:
    """The server's answer for the zip ``body``, sent as it is."""
    return client.post(DOCUMENT_ZIP, params={"name": ZIP_NAME}, content=body)


def unzipped_names(response: httpx2.Response) -> list[str]:
    return [file["name"] for file in response.json()["files"]]


def transport(llm: FakeLLM) -> list[dict[str, str]]:
    """The key and endpoint each request was sent with."""
    return [
        {name: call[name] for name in ("api_key", "api_base") if name in call}
        for call in llm.requests
    ]


def worker_of(job_id: str) -> threading.Thread:
    """The thread the server started for the job ``job_id``, while it runs."""
    return next(t for t in threading.enumerate() if t.name == WORKER_PREFIX + job_id)


def wait_for_worker(job_id: str) -> None:
    """Return once the thread the server started for the job ``job_id`` has ended, and
    so has logged its failure, if any: the job's stream ends before that. A thread no
    longer listed has ended."""
    for thread in threading.enumerate():
        if thread.name == WORKER_PREFIX + job_id:
            thread.join(GATE_TIMEOUT_SECONDS)
            assert not thread.is_alive(), f"{thread.name} still running"


def rate_limited(**request: Any) -> Any:
    """A ``litellm.completion`` that answers every request with a 429 asking for a
    wait longer than any test."""
    error = RateLimitError(message="slow down", llm_provider="openai", model=MODEL)
    # Where litellm keeps a provider's headers, on the errors it raises.
    vars(error)["litellm_response_headers"] = {
        "retry-after": str(LONG_RETRY_AFTER_SECONDS)
    }
    raise error


def http_client(url: str) -> httpx2.Client:
    """A client for the server that ``url``, with its token, opens."""
    base_url, token = url.split("/#token=")
    return httpx2.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=GATE_TIMEOUT_SECONDS,
        trust_env=False,
    )


@contextlib.contextmanager
def launched(
    logs_dir: Path,
    model: FakeLLM | None = None,
    key_service: str = APP_KEY_SERVICE,
) -> Iterator[httpx2.Client]:
    """A client for a server started as ``r3con app`` starts one, or with
    ``key_service``, as the Mac app does, stopped after."""
    url, stop = serve(logs_dir=logs_dir, completion=model, key_service=key_service)
    try:
        with http_client(url) as http:
            yield http
    finally:
        stop()


def shut_down_with_a_stream_open(
    url: str, model: FakeLLM, shut_down: Callable[[], None]
) -> tuple[str, dict[str, Any]]:
    """Over a real connection: start a run and read its stream's first event; once
    ``model`` holds its first call, call ``shut_down``, and read on until the server's
    shutdown ends the stream, whole, though the run is still going. Returns the run's
    id and that first event."""
    with http_client(url) as http:
        run_id = http.post("/api/runs", json=run_body()).json()["run_id"]
        with http.stream("GET", f"/api/runs/{run_id}/events") as response:
            lines = response.iter_lines()
            first = next(line for line in lines if line.startswith("data: "))
            assert model.called.wait(GATE_TIMEOUT_SECONDS)
            shut_down()
            # Read to the end of its body: uvicorn cutting it short raises here.
            for _ in lines:
                pass
    return run_id, json.loads(first.removeprefix("data: "))


def first_event_then_the_rest(
    url: str, gate: threading.Event
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Over a real connection: start a run, read its stream's first event while the
    model is held at ``gate``, then open the gate and read the rest."""
    with http_client(url) as http:
        assert http.get("/").status_code == 200
        run_id = http.post("/api/runs", json=run_body()).json()["run_id"]
        with http.stream("GET", f"/api/runs/{run_id}/events") as response:
            lines = response.iter_lines()
            first = next(line for line in lines if line.startswith("data: "))
            gate.set()
            rest = events_in("\n".join(lines))
    return json.loads(first.removeprefix("data: ")), rest


def test_the_page_is_served_without_the_token(client: TestClient):
    del client.headers["Authorization"]
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<html" in response.text
    assert response.headers["content-security-policy"] == "frame-ancestors 'none'"


@pytest.mark.parametrize(
    ("name", "content_type"),
    [("app.css", "text/css"), ("app.js", "text/javascript")],
)
def test_the_pages_stylesheet_and_script_are_served_without_the_token(
    client: TestClient, name: str, content_type: str
):
    del client.headers["Authorization"]
    response = client.get(f"/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"] == f"{content_type}; charset=utf-8"
    assert response.content == (APP_DIR / name).read_bytes()


@pytest.mark.parametrize("name", ["app.css", "app.js"])
def test_the_page_links_its_stylesheet_and_script(client: TestClient, name: str):
    assert f'"{name}"' in client.get("/").text


@pytest.mark.parametrize("path", ["/", "/app.css", "/app.js"])
def test_the_page_and_its_files_are_revalidated_on_every_load(
    client: TestClient, path: str
):
    assert client.get(path).headers["cache-control"] == "no-cache"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/keys"),
        ("POST", f"{DOCUMENT_TEXT}?name=memo.pdf"),
        ("POST", f"{DOCUMENT_ZIP}?name={ZIP_NAME}"),
        ("POST", f"{SAVES}?name={SAVED_NAME}"),
        ("POST", IMPORTS),
    ],
)
@pytest.mark.parametrize(
    "authorization",
    [None, "Bearer wrong-token", "the-launch-token", "Basic the-launch-token"],
)
def test_an_api_call_without_this_launchs_token_is_refused(
    client: TestClient, method: str, path: str, authorization: str | None
):
    del client.headers["Authorization"]
    headers = {} if authorization is None else {"Authorization": authorization}
    response = client.request(method, path, headers=headers, content=CORRUPT_PDF)
    assert response.status_code == 401
    assert "token" in response.json()["error"]


def test_an_api_call_with_this_launchs_token_is_answered(client: TestClient):
    assert client.get("/api/keys").status_code == 200


@pytest.mark.parametrize("path", ["/", "/app.css", "/app.js", "/api/keys"])
@pytest.mark.parametrize(
    "host", ["evil.example", f"evil.example:{PORT}", f"127.0.0.1:{PORT + 1}"]
)
def test_a_request_that_names_another_host_is_refused(
    client: TestClient, path: str, host: str
):
    assert client.get(path, headers={"Host": host}).status_code == 403


@pytest.mark.parametrize(
    ("method", "path"),
    [
        *(("GET", path) for path in ["/", "/app.css", "/app.js", "/api/keys"]),
        ("POST", f"{DOCUMENT_TEXT}?name=memo.docx"),
        ("POST", f"{DOCUMENT_ZIP}?name={ZIP_NAME}"),
    ],
)
@pytest.mark.parametrize(
    "origin", ["http://evil.example", "null", f"http://127.0.0.1:{PORT + 1}"]
)
def test_a_request_from_another_origin_is_refused(
    client: TestClient, method: str, path: str, origin: str
):
    response = client.request(method, path, headers={"Origin": origin})
    assert response.status_code == 403


@pytest.mark.parametrize("host", [f"127.0.0.1:{PORT}", f"localhost:{PORT}"])
def test_a_request_from_this_servers_own_page_is_answered(
    client: TestClient, host: str
):
    headers = {"Host": host, "Origin": f"http://{host}"}
    assert client.get("/api/keys", headers=headers).status_code == 200


def test_keys_say_where_each_comes_from_and_only_a_stored_ones_last_four(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GEMINI_API_KEY", ENVIRONMENT_KEY)
    monkeypatch.setenv("OPENAI_API_KEY", ENVIRONMENT_KEY)
    responses = [
        client.post("/api/keys", json=STORED_OPENAI),
        client.post("/api/keys", json=OTHER),
        client.get("/api/keys"),
    ]
    assert responses[-1].json() == {
        "openai": REMEMBERED_OPENAI,
        "anthropic": None,
        "gemini": {"source": "environment", "variable": "GEMINI_API_KEY"},
        "other": [{"endpoint": ENDPOINT, "key": OTHER_KEY[-4:]}],
    }
    responses += [
        client.request("DELETE", "/api/keys", json={"provider": "openai"}),
        client.request("DELETE", "/api/keys", json=KEYLESS_OTHER),
    ]
    assert responses[-1].json() == {
        "openai": {"source": "environment", "variable": "OPENAI_API_KEY"},
        "anthropic": None,
        "gemini": {"source": "environment", "variable": "GEMINI_API_KEY"},
        "other": [],
    }
    for response in responses:
        assert response.status_code == 200
        for key in (OPENAI_KEY, OTHER_KEY, ENVIRONMENT_KEY):
            assert key[:-4] not in response.text


@pytest.mark.parametrize("key", ["EMPTY", "short-key-1"])
def test_a_key_too_short_to_spare_its_last_four_has_no_hint(
    client: TestClient, key: str
):
    client.post("/api/keys", json={"provider": "anthropic", "key": key})
    status = client.post("/api/keys", json={**OTHER, "key": key}).json()
    assert status["anthropic"] == {"source": "here", "hint": "", "remembered": True}
    assert status["other"] == [{"endpoint": ENDPOINT, "key": ""}]
    assert key[-4:] not in json.dumps(status)


def test_servers_are_stored_by_url_in_the_order_added_and_forgotten_one_at_a_time(
    client: TestClient,
):
    client.post("/api/keys", json=OTHER)
    client.post("/api/keys", json=SECOND_SERVER)
    status = client.post("/api/keys", json=KEYLESS_OTHER).json()
    assert status["other"] == [
        {"endpoint": ENDPOINT, "key": None},
        {"endpoint": SECOND_ENDPOINT, "key": SECOND_KEY[-4:]},
    ]
    status = client.request("DELETE", "/api/keys", json=KEYLESS_OTHER).json()
    assert status["other"] == [{"endpoint": SECOND_ENDPOINT, "key": SECOND_KEY[-4:]}]


@pytest.mark.parametrize("slashes", ["/", "//"])
def test_a_server_url_with_trailing_slashes_is_the_same_server(
    client: TestClient, llm: FakeLLM, slashes: str
):
    client.post("/api/keys", json={**OTHER, "endpoint": f"{ENDPOINT}{slashes}"})
    status = client.post("/api/keys", json=KEYLESS_OTHER).json()
    assert status["other"] == [{"endpoint": ENDPOINT, "key": None}]
    finished_run(client, "hosted_vllm/qwen", f"{ENDPOINT}{slashes}")
    assert {call["api_base"] for call in llm.requests} == {ENDPOINT}
    forget = {"provider": "other", "endpoint": f"{ENDPOINT}{slashes}"}
    assert client.request("DELETE", "/api/keys", json=forget).json()["other"] == []


@pytest.mark.parametrize(
    ("stored", "model_name", "sent"),
    [
        (STORED_OPENAI, MODEL, {"api_key": OPENAI_KEY}),
        (STORED_OPENAI, BARE_OPENAI_MODEL, {"api_key": OPENAI_KEY}),
        (STORED_OPENAI, SERVED_OPENAI_MODEL, {"api_key": OPENAI_KEY}),
        (STORED_OPENAI, "anthropic/claude-sonnet-5-5", {}),
        (OTHER, "hosted_vllm/qwen", {}),
        (OTHER, BARE_OPENAI_MODEL, {}),
        (OTHER, "claude-sonnet-5-5", {}),
        (OTHER, "gemini-2.5-pro", {}),
        (OTHER, "azure/x", {}),
    ],
)
def test_a_model_on_no_server_gets_only_its_providers_stored_key(
    client: TestClient,
    llm: FakeLLM,
    stored: dict[str, str],
    model_name: str,
    sent: dict[str, str],
):
    client.post("/api/keys", json=stored)
    finished_run(client, model_name)
    assert transport(llm) == [sent] * len(llm.requests)


@pytest.mark.parametrize(
    ("model_name", "variable"),
    [
        ("hosted_vllm/qwen", "HOSTED_VLLM_API_KEY"),
        ("lm_studio/qwen", "LM_STUDIO_API_KEY"),
        ("llamafile/qwen", "LLAMAFILE_API_KEY"),
        ("ollama/llama3", "OLLAMA_API_KEY"),
        ("ollama_chat/llama3", "OLLAMA_API_KEY"),
        ("custom_openai/qwen", "OPENAI_API_KEY"),
    ],
)
def test_a_keyless_server_is_sent_a_placeholder_never_a_key_from_the_environment(
    client: TestClient,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    variable: str,
):
    monkeypatch.setenv(variable, ENVIRONMENT_KEY)
    client.post("/api/keys", json=KEYLESS_OTHER)
    run_events(client, start_run(client, model_name, ENDPOINT))
    placeholder = KEYLESS_SERVER_KEY.get_secret_value()
    assert llm.requests
    assert transport(llm) == [{"api_key": placeholder, "api_base": ENDPOINT}] * len(
        llm.requests
    )


@pytest.mark.parametrize(
    ("model_name", "endpoint", "sent"),
    [
        (
            "hosted_vllm/qwen",
            ENDPOINT,
            {"api_key": KEYLESS_SERVER_KEY.get_secret_value(), "api_base": ENDPOINT},
        ),
        (
            "hosted_vllm/qwen",
            SECOND_ENDPOINT,
            {"api_key": SECOND_KEY, "api_base": SECOND_ENDPOINT},
        ),
        (
            "ollama_chat/llama3",
            SECOND_ENDPOINT,
            {"api_key": SECOND_KEY, "api_base": SECOND_ENDPOINT},
        ),
        (
            SERVED_OPENAI_MODEL,
            ENDPOINT,
            {"api_key": KEYLESS_SERVER_KEY.get_secret_value(), "api_base": ENDPOINT},
        ),
        (
            SERVED_OPENAI_MODEL,
            SECOND_ENDPOINT,
            {"api_key": SECOND_KEY, "api_base": SECOND_ENDPOINT},
        ),
        (
            MODEL,
            ENDPOINT,
            {"api_key": KEYLESS_SERVER_KEY.get_secret_value(), "api_base": ENDPOINT},
        ),
        (
            BARE_OPENAI_MODEL,
            SECOND_ENDPOINT,
            {"api_key": SECOND_KEY, "api_base": SECOND_ENDPOINT},
        ),
    ],
)
def test_a_model_on_a_server_reaches_it_as_typed_with_its_key_in_a_run_and_an_ask(
    client: TestClient,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    endpoint: str,
    sent: dict[str, str],
):
    monkeypatch.setenv("OPENAI_API_KEY", ENVIRONMENT_KEY)
    client.post("/api/keys", json=STORED_OPENAI)
    client.post("/api/keys", json=KEYLESS_OTHER)
    client.post("/api/keys", json=SECOND_SERVER)
    run_id = finished_run(client, model_name, endpoint)
    ask = {**ASK, "model": model_name, "endpoint": endpoint}
    assert ask_events(client, run_id, ask)[-1]["type"] == "ask_finished"
    assert {call["model"] for call in llm.requests} == {model_name}
    assert transport(llm) == [sent] * len(llm.requests)


@pytest.mark.parametrize(
    ("model_name", "endpoint", "wrong"),
    [
        ("hosted_vllm/qwen", "http://127.0.0.1:10/v1", "isn't in settings"),
        (
            "anthropic/claude-sonnet-5-5",
            ENDPOINT,
            "anthropic/claude-sonnet-5-5 isn't a self-hosted or openai/ model",
        ),
        ("gemini/gemini-2.5-pro", ENDPOINT, "isn't a self-hosted or openai/ model"),
        (BARE_OPENAI_MODEL, GROQ_ENDPOINT, "isn't a self-hosted or openai/ model"),
        (
            SERVED_OPENAI_MODEL,
            ENDPOINT,
            "would send the server the headers in OPENAI_CUSTOM_HEADERS",
        ),
        ("lm_studio/qwen", ENDPOINT, "OPENAI_CUSTOM_HEADERS"),
        ("github_copilot/x", ENDPOINT, "isn't a self-hosted or openai/ model"),
        ("chatgpt/x", ENDPOINT, "isn't a self-hosted or openai/ model"),
    ],
)
def test_an_unknown_server_or_a_model_that_would_send_it_other_credentials_is_refused(
    client: TestClient,
    llm: FakeLLM,
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
    endpoint: str,
    wrong: str,
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", ENVIRONMENT_KEY)
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", CUSTOM_HEADERS)
    client.post("/api/keys", json=STORED_OPENAI)
    client.post("/api/keys", json=KEYLESS_OTHER)
    client.post("/api/keys", json={"provider": "other", "endpoint": GROQ_ENDPOINT})
    run_id = finished_run(client)
    llm.requests.clear()
    responses = [
        client.post("/api/runs", json=run_body(model_name, endpoint)),
        client.post(
            f"/api/runs/{run_id}/asks",
            json={**ASK, "model": model_name, "endpoint": endpoint},
        ),
    ]
    for response in responses:
        assert response.status_code == 422
        assert wrong in response.json()["error"]
    assert llm.requests == []


def test_a_hosted_vllm_model_on_a_server_runs_while_openai_custom_headers_is_set(
    client: TestClient, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", CUSTOM_HEADERS)
    client.post("/api/keys", json=KEYLESS_OTHER)
    finished_run(client, "hosted_vllm/qwen", ENDPOINT)
    placeholder = KEYLESS_SERVER_KEY.get_secret_value()
    assert transport(llm) == [{"api_key": placeholder, "api_base": ENDPOINT}] * len(
        llm.requests
    )


def test_a_name_litellm_cannot_place_is_sent_to_its_server_which_litellm_then_refuses(
    client: TestClient, llm: FakeLLM
):
    client.post("/api/keys", json=OTHER)
    last = run_events(client, start_run(client, "qwen3-32b", ENDPOINT))[-1]
    assert llm.requests
    assert transport(llm) == [{"api_key": OTHER_KEY, "api_base": ENDPOINT}] * len(
        llm.requests
    )
    assert last["type"] == "run_failed"
    assert "LLM Provider NOT provided" in last["error"]


def test_a_pasted_key_and_endpoint_are_used_and_masked_without_their_whitespace(
    client: TestClient, llm: FakeLLM
):
    client.post("/api/keys", json={"provider": "openai", "key": f"{OPENAI_KEY}\n"})
    client.post("/api/keys", json={**OTHER, "endpoint": f" {ENDPOINT}\n"})
    assert client.get("/api/keys").json()["other"][0]["endpoint"] == ENDPOINT
    llm.answers(reasoning=RuntimeError(f"Incorrect API key provided: {OPENAI_KEY}."))
    last = run_events(client, start_run(client))[-1]
    assert {call["api_key"] for call in llm.requests} == {OPENAI_KEY}
    assert last["error"] == "RuntimeError: Incorrect API key provided: REDACTED."
    llm.answers(reasoning=ANSWER)
    llm.requests.clear()
    finished_run(client, "hosted_vllm/qwen", f"{ENDPOINT} ")
    assert {call["api_base"] for call in llm.requests} == {ENDPOINT}


def test_a_placeholder_key_too_short_to_be_a_secret_is_not_masked(
    client: TestClient, llm: FakeLLM
):
    client.post("/api/keys", json={**OTHER, "key": "EMPTY"})
    llm.answers(relevance=RuntimeError("The model sent an EMPTY reply."))
    last = run_events(client, start_run(client, "hosted_vllm/qwen", ENDPOINT))[-1]
    assert last["error"] == "RuntimeError: The model sent an EMPTY reply."


def test_with_a_keyless_server_a_bare_openai_model_on_no_server_never_reaches_it(
    client: TestClient, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", ENVIRONMENT_KEY)
    client.post("/api/keys", json=KEYLESS_OTHER)
    finished_run(client, BARE_OPENAI_MODEL)
    assert llm.requests
    assert all("api_base" not in call for call in llm.requests)


def test_a_stored_key_is_in_no_response_event_or_file(
    client: TestClient, logs_dir: Path
):
    responses = [client.post("/api/keys", json=STORED_OPENAI)]
    run_id = start_run(client)
    responses.append(client.get(f"/api/runs/{run_id}/events"))
    responses.append(client.post(f"/api/runs/{run_id}/asks", json=ASK))
    responses.append(client.get(f"/api/asks/{responses[-1].json()['ask_id']}/events"))
    responses.append(client.get("/api/keys"))
    assert all(OPENAI_KEY not in response.text for response in responses)
    files = [path for path in logs_dir.rglob("*") if path.is_file()]
    assert files
    assert all(OPENAI_KEY not in path.read_text() for path in files)


@pytest.mark.parametrize("held", ["stored", "environment"])
def test_a_key_a_provider_quotes_in_its_error_is_masked_in_the_event_and_the_log(
    client: TestClient,
    llm: FakeLLM,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
    held: str,
):
    if held == "stored":
        client.post("/api/keys", json=STORED_OPENAI)
        key = OPENAI_KEY
    else:
        monkeypatch.setenv("OPENAI_API_KEY", ENVIRONMENT_KEY)
        key = ENVIRONMENT_KEY
    llm.answers(relevance=RuntimeError(f"Incorrect API key provided: {key}."))
    with caplog.at_level(logging.ERROR, logger="r3con.app"):
        run_id = start_run(client)
        last = run_events(client, run_id)[-1]
        wait_for_worker(run_id)
    assert last["type"] == "run_failed"
    assert last["error"] == "RuntimeError: Incorrect API key provided: REDACTED."
    logged = [r.getMessage() for r in caplog.records if r.name == "r3con.app"]
    assert logged == [f"run_failed: {last['error']}"]


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_saved_key_is_remembered_by_the_next_launch(
    logs_dir: Path, credential_store: FakeKeyring
):
    with launched(logs_dir) as http:
        pasted = {**STORED_OPENAI, "key": f" {OPENAI_KEY}\n"}
        saved = http.post("/api/keys", json=pasted).json()["openai"]
    assert credential_store.passwords == {(APP_KEY_SERVICE, "openai"): OPENAI_KEY}
    with launched(logs_dir) as http:
        loaded = http.get("/api/keys").json()["openai"]
    assert saved == loaded == REMEMBERED_OPENAI


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize(
    ("saved_by", "loaded_by"),
    [(APP_KEY_SERVICE, MAC_APP_KEY_SERVICE), (MAC_APP_KEY_SERVICE, APP_KEY_SERVICE)],
    ids=["r3con app's, by the Mac app", "the Mac app's, by r3con app"],
)
def test_a_key_saved_by_one_program_is_not_loaded_by_the_other(
    logs_dir: Path, credential_store: FakeKeyring, saved_by: str, loaded_by: str
):
    with launched(logs_dir, key_service=saved_by) as http:
        http.post("/api/keys", json=STORED_OPENAI)
    assert credential_store.passwords == {(saved_by, "openai"): OPENAI_KEY}
    with launched(logs_dir, key_service=loaded_by) as http:
        assert http.get("/api/keys").json()["openai"] is None
    with launched(logs_dir, key_service=saved_by) as http:
        assert http.get("/api/keys").json()["openai"] == REMEMBERED_OPENAI


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_launch_reads_a_key_from_the_store_as_a_post_would_store_it(
    logs_dir: Path, credential_store: FakeKeyring
):
    credential_store.passwords[APP_KEY_SERVICE, "openai"] = f" {OPENAI_KEY}\n"
    credential_store.passwords[APP_KEY_SERVICE, "gemini"] = " \n"
    with launched(logs_dir) as http:
        status = http.get("/api/keys").json()
    assert (status["openai"], status["gemini"]) == (REMEMBERED_OPENAI, None)


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_key_from_the_store_is_sent_and_masked_like_any_other(
    logs_dir: Path,
    credential_store: FakeKeyring,
    llm: FakeLLM,
    caplog: pytest.LogCaptureFixture,
):
    credential_store.passwords[APP_KEY_SERVICE, "openai"] = OPENAI_KEY
    llm.answers(relevance=RuntimeError(f"Incorrect API key provided: {OPENAI_KEY}."))
    with caplog.at_level(logging.INFO, logger="r3con"), launched(logs_dir, llm) as http:
        run_id = http.post("/api/runs", json=run_body()).json()["run_id"]
        events = http.get(f"/api/runs/{run_id}/events")
        wait_for_worker(run_id)
    assert {call["api_key"] for call in llm.requests} == {OPENAI_KEY}
    last = events_in(events.text)[-1]
    assert last["error"] == "RuntimeError: Incorrect API key provided: REDACTED."
    assert OPENAI_KEY not in events.text
    assert OPENAI_KEY not in caplog.text
    files = [path for path in logs_dir.rglob("*") if path.is_file()]
    assert files
    assert all(OPENAI_KEY not in path.read_text() for path in files)


def test_forgetting_a_key_removes_it_from_the_store(
    client: TestClient, credential_store: FakeKeyring
):
    client.post("/api/keys", json=STORED_OPENAI)
    client.post("/api/keys", json={"provider": "gemini", "key": SECOND_KEY})
    status = client.request("DELETE", "/api/keys", json={"provider": "openai"}).json()
    assert status["openai"] is None
    assert credential_store.passwords == {(APP_KEY_SERVICE, "gemini"): SECOND_KEY}


def test_a_key_the_store_fails_to_delete_is_still_forgotten_here_and_the_log_says_so(
    client: TestClient, caplog: pytest.LogCaptureFixture
):
    keyring.set_keyring(UndeletableKeyring())
    client.post("/api/keys", json=STORED_OPENAI)
    with caplog.at_level(logging.INFO, logger="r3con"):
        forget = {"provider": "openai"}
        response = client.request("DELETE", "/api/keys", json=forget)
    assert (response.status_code, response.json()["openai"]) == (200, None)
    said = store_failed(MAY_LOAD_AGAIN, "PasswordDeleteError")
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.app"] == [said]
    assert OPENAI_KEY not in caplog.text


def test_removing_a_key_a_failed_save_left_in_the_store_removes_it_there(
    client: TestClient,
):
    store = UnwritableKeyring()
    store.passwords[APP_KEY_SERVICE, "openai"] = SECOND_KEY
    keyring.set_keyring(store)
    saved = client.post("/api/keys", json=STORED_OPENAI).json()["openai"]
    assert saved == UNTIL_CLOSED_OPENAI
    client.request("DELETE", "/api/keys", json={"provider": "openai"})
    assert store.passwords == {}


@pytest.mark.parametrize(("store", "said"), SAVE_FAILURES)
def test_without_a_working_store_a_key_lasts_until_the_app_closes_and_is_never_logged(
    client: TestClient,
    llm: FakeLLM,
    caplog: pytest.LogCaptureFixture,
    store: Callable[[], KeyringBackend],
    said: list[str],
):
    keyring.set_keyring(store())
    with caplog.at_level(logging.INFO, logger="r3con"):
        responses = [client.post("/api/keys", json=STORED_OPENAI)]
        finished_run(client)
        responses.append(client.get("/api/keys"))
        responses.append(
            client.request("DELETE", "/api/keys", json={"provider": "openai"})
        )
    assert [response.status_code for response in responses] == [200] * 3
    kept = [response.json()["openai"] for response in responses]
    assert kept == [UNTIL_CLOSED_OPENAI, UNTIL_CLOSED_OPENAI, None]
    assert {call["api_key"] for call in llm.requests} == {OPENAI_KEY}
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.app"] == said
    assert OPENAI_KEY not in caplog.text
    assert all(OPENAI_KEY not in response.text for response in responses)


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize(("store", "said"), LAUNCH_FAILURES)
def test_a_launch_without_a_working_store_starts_with_no_saved_key_and_says_why(
    logs_dir: Path,
    caplog: pytest.LogCaptureFixture,
    store: Callable[[], KeyringBackend],
    said: list[str],
):
    keyring.set_keyring(store())
    with caplog.at_level(logging.INFO, logger="r3con"), launched(logs_dir) as http:
        status = http.get("/api/keys").json()
    assert status == {"openai": None, "anthropic": None, "gemini": None, "other": []}
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.app"] == said


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_key_the_store_cannot_read_leaves_the_others_loaded(logs_dir: Path):
    store = LockedItemKeyring()
    store.passwords[APP_KEY_SERVICE, "openai"] = OPENAI_KEY
    store.passwords[APP_KEY_SERVICE, "gemini"] = SECOND_KEY
    keyring.set_keyring(store)
    with launched(logs_dir) as http:
        status = http.get("/api/keys").json()
    assert status["openai"] is None
    assert status["gemini"] == {
        "source": "here",
        "hint": SECOND_KEY[-4:],
        "remembered": True,
    }


def test_a_run_streams_its_events_in_order_and_replays_them_to_a_late_reader(
    client: TestClient, logs_dir: Path
):
    run_id = start_run(client)
    events = run_events(client, run_id)
    types = [event["type"] for event in events]
    assert list(dict.fromkeys(types)) == RUN_EVENTS
    assert types[-1] == "run_finished"
    assert run_events(client, run_id) == events
    assert events[0]["run_dir"] == events[-1]["run_dir"]
    assert Path(events[-1]["run_dir"]).parent == logs_dir


MEMO_NAMES = sorted(path.name for path in MEMOS.iterdir())


@pytest.mark.parametrize(
    ("names", "named"),
    [
        (None, ["0", "1", "2", "3", "4"]),
        (MEMO_NAMES, MEMO_NAMES),
        (
            ["notes.txt", "notes.txt", "notes.txt (2)", "a/notes.txt", "notes.txt"],
            [
                "notes.txt",
                "notes.txt (3)",
                "notes.txt (2)",
                "a/notes.txt",
                "notes.txt (4)",
            ],
        ),
    ],
    ids=["none", "given", "given-twice"],
)
def test_a_run_names_its_documents_by_the_names_sent_each_once(
    client: TestClient, names: list[str] | None, named: list[str]
):
    body = run_body() if names is None else {**run_body(), "names": names}
    response = client.post("/api/runs", json=body)
    assert response.status_code == 202, response.text
    events = run_events(client, response.json()["run_id"])
    assert events[0]["document_names"] == named
    assert events[-1]["type"] == "run_finished"


@pytest.mark.parametrize(
    ("sent", "rounds"),
    [
        ({}, DEFAULT_RELEVANCE_ROUNDS),
        ({"relevance_rounds": 1}, 1),
        ({"relevance_rounds": MAX_RELEVANCE_ROUNDS}, MAX_RELEVANCE_ROUNDS),
    ],
    ids=["default", "one", "most"],
)
def test_a_run_builds_its_representation_over_the_relevance_rounds_sent(
    client: TestClient, sent: dict[str, int], rounds: int
):
    response = client.post("/api/runs", json={**run_body(), **sent})
    assert response.status_code == 202, response.text
    events = run_events(client, response.json()["run_id"])
    manifest = json.loads((Path(events[0]["run_dir"]) / "manifest.json").read_text())
    finished = {
        (e["round"], e["doc_id"]) for e in events if e["type"] == "relevance_finished"
    }
    every = {(k, d) for k in range(1, rounds + 1) for d in events[0]["document_ids"]}
    assert events[0]["rounds"] == manifest["config"]["relevance_rounds"] == rounds
    assert finished == every
    assert events[-1]["type"] == "run_finished"


def test_an_ask_keeps_its_runs_rounds_and_takes_none_of_its_own(client: TestClient):
    response = client.post("/api/runs", json={**run_body(), "relevance_rounds": 3})
    run_id = response.json()["run_id"]
    assert run_events(client, run_id)[-1]["type"] == "run_finished"
    events = ask_events(client, run_id)
    manifest = json.loads((Path(events[-1]["ask_dir"]) / "manifest.json").read_text())
    refused = client.post(
        f"/api/runs/{run_id}/asks", json={**ASK, "relevance_rounds": 2}
    )
    assert manifest["config"]["relevance_rounds"] == 3
    assert refused.status_code == 422
    assert "relevance_rounds: Extra inputs are not permitted" in refused.json()["error"]


def test_an_ask_over_a_finished_run_streams_its_own_events(client: TestClient):
    run_id = finished_run(client)
    run_dir = Path(run_events(client, run_id)[0]["run_dir"])
    events = ask_events(client, run_id)
    assert [event["type"] for event in events] == ASK_EVENTS
    assert events[0]["question"] == ASK["question"]
    assert Path(events[-1]["ask_dir"]).parent == run_dir / "asks"


def test_an_ask_over_a_run_still_running_is_refused(held_llm: FakeLLM, logs_dir: Path):
    with client_for(held_llm, logs_dir) as client:
        run_id = start_run(client)
        assert held_llm.called.wait(GATE_TIMEOUT_SECONDS)
        response = client.post(f"/api/runs/{run_id}/asks", json=ASK)
        held_llm.gate.set()
        assert run_events(client, run_id)[-1]["type"] == "run_finished"
    assert response.status_code == 409
    assert "still running" in response.json()["error"]


def test_an_ask_over_a_run_that_failed_in_reasoning_is_answered(
    client: TestClient, llm: FakeLLM
):
    llm.answers(reasoning=RuntimeError("the model is down"))
    run_id = start_run(client)
    last = run_events(client, run_id)[-1]
    assert (last["type"], last["stage"]) == ("run_failed", "reasoning")
    llm.answers(reasoning=ANSWER)
    assert [event["type"] for event in ask_events(client, run_id)] == ASK_EVENTS


def test_an_ask_over_a_run_that_failed_before_parsing_fails_saying_why(
    client: TestClient, llm: FakeLLM
):
    llm.answers(relevance=RuntimeError("the model is down"))
    run_id = start_run(client)
    assert run_events(client, run_id)[-1]["stage"] == "relevance"
    events = ask_events(client, run_id)
    assert [(event["type"], event["stage"]) for event in events] == [
        ("ask_failed", None)
    ]
    assert "never finished parsing" in events[0]["error"]


def test_an_ask_that_would_send_a_masked_param_fails_naming_it_and_sends_nothing(
    client: TestClient, llm: FakeLLM
):
    run_id = finished_run(client)
    manifest_path = Path(run_events(client, run_id)[0]["run_dir"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["params"] = {"extra_headers": {"Authorization": "REDACTED"}}
    manifest_path.write_text(json.dumps(manifest))
    sent = len(llm.requests)
    events = ask_events(client, run_id)
    assert [(event["type"], event["stage"]) for event in events] == [
        ("ask_failed", None)
    ]
    assert events[0]["error"] == (
        "ValueError: REDACTED, the mask a run's manifest saves in place of a "
        "credential, would be sent: pass the real value in `params`, which replaces "
        "all of the run's params. Masked params: `extra_headers.Authorization`."
    )
    assert llm.requests[sent:] == []


def test_an_ask_over_a_run_folder_it_cannot_read_fails_saying_why_and_sends_nothing(
    client: TestClient, llm: FakeLLM
):
    run_id = finished_run(client)
    run_dir = Path(run_events(client, run_id)[0]["run_dir"])
    in_json("structuring/parsing/result.json", a_parse_citation_short)(run_dir)
    sent = len(llm.requests)
    events = ask_events(client, run_id)
    assert [(event["type"], event["stage"]) for event in events] == [
        ("ask_failed", None)
    ]
    assert events[0]["error"] == (
        f"ValueError: can't read {run_dir} as a run folder: ValueError: zip() "
        "argument 2 is shorter than argument 1"
    )
    assert llm.requests[sent:] == []
    assert not (run_dir / "asks").exists()


def representation_of(client: TestClient, run_id: str) -> dict[str, Any]:
    response = client.get(f"/api/runs/{run_id}/representation")
    assert response.status_code == 200, response.text
    return response.json()


def test_a_runs_representation_is_all_an_ask_reads_of_its_folder(
    client: TestClient, llm: FakeLLM, tmp_path: Path
):
    files = representation_of(client, finished_run(client))
    copy = tmp_path / "copy"
    for path, content in files.items():
        (copy / path).parent.mkdir(parents=True, exist_ok=True)
        (copy / path).write_text(json.dumps(content), encoding="utf-8")
    answer = r3con.ask(copy, ASK["question"], completion=llm)
    assert list(files) == list(REPRESENTATION_FILES)
    assert answer.answer == ANSWER["answer"]


def test_a_runs_representation_leaves_out_the_manifests_paths_on_this_computer(
    client: TestClient,
):
    run_id = finished_run(client)
    run_dir = Path(run_events(client, run_id)[0]["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    local = manifest.pop("sources")
    assert local["prompts"]
    assert representation_of(client, run_id)["manifest.json"] == manifest


def test_a_runs_representation_names_no_server_header_or_masked_value(
    client: TestClient, tmp_path: Path
):
    """A run on a stored server, under a config file where the server runs whose
    params, as r3con's own config params may, carry another server and a header."""
    config = yaml.safe_load((CONFIGS_DIR / "default.yaml").read_text())
    params = {"api_base": SECOND_ENDPOINT, "extra_headers": {"X-Gateway-Key": GATE_KEY}}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "default.yaml").write_text(
        yaml.safe_dump({**config, "params": params})
    )
    client.post("/api/keys", json=OTHER)
    run_id = finished_run(client, SERVED_OPENAI_MODEL, ENDPOINT)
    run_dir = Path(run_events(client, run_id)[0]["run_dir"])
    files = representation_of(client, run_id)
    saved = json.dumps(files)
    written = json.loads((run_dir / "manifest.json").read_text())["config"]["params"]
    assert written["api_base"] == SECOND_ENDPOINT
    assert files["manifest.json"]["config"]["params"] == {}
    assert [
        held
        for held in (ENDPOINT, SECOND_ENDPOINT, OTHER_KEY, GATE_KEY, MASK)
        if held in saved
    ] == []


def imported(client: TestClient, files: dict[str, Any]) -> str:
    response = client.post(IMPORTS, json=files)
    assert response.status_code == 201, response.text
    return response.json()["run_id"]


def run_folders(logs_dir: Path) -> list[str]:
    return sorted(folder.name for folder in logs_dir.iterdir())


def test_a_saved_runs_representation_imports_as_a_new_run_asked_like_any_other(
    client: TestClient, logs_dir: Path
):
    saved = representation_of(client, finished_run(client))
    before = run_folders(logs_dir)
    run_id = imported(client, saved)
    run_dir = Path(run_events(client, run_id)[0]["run_dir"])
    written = sorted(str(path.relative_to(run_dir)) for path in run_dir.rglob("*.*"))
    events = ask_events(client, run_id)
    assert [event["type"] for event in events] == ASK_EVENTS
    assert run_dir.parent == logs_dir and run_dir.name not in before
    assert written == sorted(REPRESENTATION_FILES)
    resaved = representation_of(client, run_id)
    assert {path: resaved[path] for path in REPRESENTATION_FILES[1:]} == {
        path: saved[path] for path in REPRESENTATION_FILES[1:]
    }


def test_an_imported_run_keeps_no_param_override_or_other_field_of_its_manifest(
    client: TestClient, llm: FakeLLM
):
    saved = representation_of(client, finished_run(client))
    config = saved["manifest.json"]["config"]
    saved["manifest.json"] = {
        **saved["manifest.json"],
        "config": {**config, "params": {"api_base": ELSEWHERE}, "overrides": {"x": 1}},
        "ask_of": "another-run",
        "sources": {"config": "/somewhere/default.yaml"},
    }
    client.post("/api/keys", json=STORED_OPENAI)
    run_id = imported(client, saved)
    run_dir = Path(run_events(client, run_id)[0]["run_dir"])
    manifest = json.loads((run_dir / "manifest.json").read_text())
    sent = len(llm.requests)
    assert ask_events(client, run_id)[-1]["type"] == "ask_finished"
    assert sorted(manifest) == [
        "config",
        "litellm_model",
        "r3con_version",
        "run_folder",
        "task",
    ]
    assert sorted(manifest["config"]) == [
        "model",
        "name",
        "prompts",
        "reasoning",
        "relevance_rounds",
    ]
    assert transport(llm)[sent:] == [{"api_key": OPENAI_KEY}]


def renamed_first_id(files: dict[str, Any]) -> None:
    for name in ("ids.json", "names.json"):
        files[name]["renamed"] = files[name].pop(next(iter(files[name])))


@pytest.mark.parametrize(
    ("damage", "logged"),
    [
        (lambda f: f.update({"../escape.json": {}}), "Extra inputs"),
        (lambda f: f.pop("ids.json"), "ids.json: Field required"),
        (lambda f: f.update({"ids.json": []}), "Input should be an object"),
        (
            lambda f: f["manifest.json"]["config"]["prompts"].update(
                {"reasoning/direct": "../../../escape"}
            ),
            "String should match pattern",
        ),
        (
            lambda f: f["manifest.json"]["config"].update({"name": "../x"}),
            "String should match pattern",
        ),
        (lambda f: f["manifest.json"]["config"].update({"seed": 7}), "Extra inputs"),
        (
            lambda f: f["manifest.json"]["config"].update({"reasoning": "coding"}),
            DIRECT_ONLY,
        ),
        (
            lambda f: f["ids.json"].update({next(iter(f["ids.json"])): -1}),
            NOT_ONE_INDEX_EACH,
        ),
        (
            lambda f: f["ids.json"].update(dict.fromkeys(list(f["ids.json"])[:2], 0)),
            NOT_ONE_INDEX_EACH,
        ),
        (lambda f: f["names.json"].update({"extra": "x.txt"}), NOT_ONE_INDEX_EACH),
        (lambda f: f["relevance/result.json"].pop("rounds"), "refused (ValueError)"),
        (
            lambda f: f["manifest.json"]["config"]["prompts"].update(
                {"reasoning/direct": "v999"}
            ),
            "refused (FileNotFoundError)",
        ),
        (renamed_first_id, "refused (ValueError)"),
    ],
    ids=[
        "path",
        "missing",
        "not-an-object",
        "prompt-path",
        "config-path",
        "config-field",
        "coding",
        "negative-index",
        "repeated-index",
        "names-differ",
        "no-rounds",
        "no-such-prompt",
        "unknown-snippet-id",
    ],
)
def test_an_import_not_a_run_r3con_can_ask_is_refused_plainly_and_writes_nothing(
    client: TestClient,
    logs_dir: Path,
    caplog: pytest.LogCaptureFixture,
    damage: Callable[[dict[str, Any]], object],
    logged: str,
):
    saved = representation_of(client, finished_run(client))
    before = run_folders(logs_dir)
    damage(saved)
    with caplog.at_level(logging.WARNING, logger="r3con.app"):
        response = client.post(IMPORTS, json=saved)
    said = [r.getMessage() for r in caplog.records if r.name == "r3con.app"]
    assert (response.status_code, response.json()) == (
        422,
        {"error": UNREADABLE_IMPORT},
    )
    assert len(said) == 1 and logged in said[0]
    assert run_folders(logs_dir) == before


def test_a_refused_imports_log_line_keeps_a_key_from_the_file_on_one_line(
    client: TestClient, caplog: pytest.LogCaptureFixture
):
    saved = representation_of(client, finished_run(client))
    saved["ids.json"] = {"x\nFAKE warning line": "no"}
    with caplog.at_level(logging.WARNING, logger="r3con.app"):
        client.post(IMPORTS, json=saved)
    said = [r.getMessage() for r in caplog.records if r.name == "r3con.app"]
    assert len(said) == 1 and "\n" not in said[0]
    assert "x\\nFAKE warning line" in said[0]


def test_an_import_whose_folder_cannot_be_written_is_refused_and_logged(
    client: TestClient, llm: FakeLLM, tmp_path: Path, caplog: pytest.LogCaptureFixture
):
    saved = representation_of(client, finished_run(client))
    taken = tmp_path / "taken"
    taken.write_text("a file where the logs dir would be")
    with (
        client_for(llm, taken) as elsewhere,
        caplog.at_level(logging.WARNING, logger="r3con.app"),
    ):
        response = elsewhere.post(IMPORTS, json=saved)
    logged = [r.getMessage() for r in caplog.records if r.name == "r3con.app"]
    assert (response.status_code, response.json()) == (
        500,
        {"error": UNWRITABLE_IMPORT},
    )
    assert len(logged) == 1 and "an imported run isn't written" in logged[0]
    assert taken.read_text() == "a file where the logs dir would be"


def test_an_import_over_the_body_limit_is_refused_and_writes_nothing(
    llm: FakeLLM, logs_dir: Path
):
    with client_for(llm, logs_dir) as client:
        saved = representation_of(client, finished_run(client))
    before = run_folders(logs_dir)
    body = json.dumps(saved)
    with client_for(llm, logs_dir, max_body_bytes=len(body) - 1) as client:
        response = client.post(IMPORTS, content=body)
    assert response.status_code == 413
    assert run_folders(logs_dir) == before


def test_the_representation_of_a_run_still_running_is_refused(
    held_llm: FakeLLM, logs_dir: Path
):
    with client_for(held_llm, logs_dir) as client:
        run_id = start_run(client)
        assert held_llm.called.wait(GATE_TIMEOUT_SECONDS)
        response = client.get(f"/api/runs/{run_id}/representation")
        held_llm.gate.set()
        assert run_events(client, run_id)[-1]["type"] == "run_finished"
    assert response.status_code == 409
    assert "still running" in response.json()["error"]


@pytest.mark.parametrize("path", REPRESENTATION_FILES)
@pytest.mark.parametrize(
    ("damage", "error"),
    [
        (
            lambda file: file.write_text("{not json", encoding="utf-8"),
            "JSONDecodeError",
        ),
        (Path.unlink, "FileNotFoundError"),
    ],
    ids=["not json", "missing"],
)
def test_the_representation_of_a_run_folder_with_a_file_it_cannot_read_is_refused(
    client: TestClient,
    caplog: pytest.LogCaptureFixture,
    path: str,
    damage: Callable[[Path], object],
    error: str,
):
    run_id = finished_run(client)
    run_dir = Path(run_events(client, run_id)[0]["run_dir"])
    damage(run_dir / path)
    with caplog.at_level(logging.WARNING, logger="r3con.app"):
        response = client.get(f"/api/runs/{run_id}/representation")
    logged = [r.getMessage() for r in caplog.records if r.name == "r3con.app"]
    assert response.status_code == 409
    assert response.json() == {"error": UNREADABLE_RUN}
    assert logged == [f"{run_dir}: can't read it to save ({error})"]


class FakeSaveDialog:
    """A Save dialog, at the screen: it answers each name it is shown with
    ``chooses``, as the user would, ``None`` for a cancel."""

    def __init__(self, chooses: Path | None) -> None:
        self.chooses = chooses
        self.shown: list[str] = []

    def __call__(self, name: str) -> Path | None:
        self.shown.append(name)
        return self.chooses


def save(client: TestClient, name: str = SAVED_NAME, tab: bytes = SAVED_TAB):
    return client.post(SAVES, params={"name": name}, content=tab)


def test_a_tab_is_saved_where_the_save_dialog_chose_named_as_the_page_suggested(
    llm: FakeLLM, logs_dir: Path, tmp_path: Path
):
    dialog = FakeSaveDialog(tmp_path / CHOSEN_NAME)
    with client_for(llm, logs_dir, dialog) as client:
        response = save(client)
    assert response.status_code == 204
    assert dialog.shown == [SAVED_NAME]
    assert (tmp_path / CHOSEN_NAME).read_bytes() == SAVED_TAB


def test_a_save_cancelled_in_the_dialog_is_answered_and_writes_nothing(
    llm: FakeLLM, logs_dir: Path, tmp_path: Path
):
    dialog = FakeSaveDialog(None)
    with client_for(llm, logs_dir, dialog) as client:
        response = save(client)
    assert (response.status_code, dialog.shown) == (204, [SAVED_NAME])
    # The working directory, where a write to the bare suggested name would land.
    assert list(tmp_path.iterdir()) == []


def test_a_save_over_an_earlier_one_replaces_it(
    llm: FakeLLM, logs_dir: Path, tmp_path: Path
):
    earlier = tmp_path / CHOSEN_NAME
    earlier.write_text("an earlier saved tab")
    with client_for(llm, logs_dir, FakeSaveDialog(earlier)) as client:
        response = save(client)
    assert response.status_code == 204
    assert [path.name for path in tmp_path.iterdir()] == [CHOSEN_NAME]
    assert earlier.read_bytes() == SAVED_TAB


@pytest.mark.parametrize(
    "name",
    [
        "../tab.html",
        "/tmp/tab.html",
        "saved/tab.html",
        "saved\\tab.html",
        "tab.htm",
        "",
    ],
)
def test_a_save_not_named_as_one_html_file_is_refused_before_any_dialog(
    llm: FakeLLM, logs_dir: Path, tmp_path: Path, name: str
):
    dialog = FakeSaveDialog(tmp_path / CHOSEN_NAME)
    with client_for(llm, logs_dir, dialog) as client:
        response = save(client, name)
    assert response.status_code == 422
    assert "name: String should match pattern" in response.json()["error"]
    assert dialog.shown == []


def test_a_save_over_the_body_limit_is_refused_before_any_dialog(
    llm: FakeLLM, logs_dir: Path, tmp_path: Path
):
    dialog = FakeSaveDialog(tmp_path / CHOSEN_NAME)
    with client_for(llm, logs_dir, dialog, max_body_bytes=MAX_SAVE_BYTES) as client:
        response = save(client, tab=b"x" * (MAX_SAVE_BYTES + 1))
    assert response.status_code == 413
    assert dialog.shown == []


# "folder" stands for a file the save would replace, and must leave as it was.
@pytest.mark.parametrize(
    ("chosen", "error"),
    [(f"gone/{CHOSEN_NAME}", "FileNotFoundError"), ("folder", "IsADirectoryError")],
)
def test_a_save_where_the_file_cannot_be_written_is_refused_and_logged(
    llm: FakeLLM,
    logs_dir: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    chosen: str,
    error: str,
):
    (tmp_path / "folder").mkdir()
    (tmp_path / "folder" / "kept.txt").write_text("kept")
    with (
        client_for(llm, logs_dir, FakeSaveDialog(tmp_path / chosen)) as client,
        caplog.at_level(logging.WARNING, logger="r3con.app"),
    ):
        response = save(client)
    logged = [r.getMessage() for r in caplog.records if r.name == "r3con.app"]
    assert response.status_code == 500
    assert response.json() == {"error": UNWRITABLE_SAVE}
    assert logged == [f"{tmp_path / chosen}: a saved tab isn't written ({error})"]
    assert [path.name for path in tmp_path.rglob("*")] == ["folder", "kept.txt"]


def test_a_server_with_no_save_dialog_refuses_a_save_for_its_page_to_download(
    client: TestClient,
):
    response = save(client)
    assert (response.status_code, response.json()) == (404, {"error": NO_SAVE_DIALOG})


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.parametrize(
    ("dialog", "flag"), [(None, ""), (FakeSaveDialog(None), "&save=dialog")]
)
def test_only_a_server_with_a_save_dialog_opens_its_page_saying_so(
    logs_dir: Path, dialog: FakeSaveDialog | None, flag: str
):
    listener = listen(0)
    try:
        _, url = _server(listener, logs_dir, None, choose_save_path=dialog)
    finally:
        listener.close()
    assert re.fullmatch(rf"http://127\.0\.0\.1:\d+/#token=[\w-]+{flag}", url)


def test_a_run_r3con_refuses_before_it_starts_ends_its_stream_and_cannot_be_asked(
    client: TestClient, llm: FakeLLM, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("r3con.settings.REASONING_MAX_TURNS", 0)
    run_id = start_run(client)
    events = run_events(client, run_id)
    assert [(event["type"], event["stage"]) for event in events] == [
        ("run_failed", None)
    ]
    assert events[0]["error"] == (
        "ValueError: REASONING_MAX_TURNS must be >= 1, got 0."
    )
    assert llm.requests == []
    response = client.post(f"/api/runs/{run_id}/asks", json=ASK)
    assert response.status_code == 409
    assert "never started" in response.json()["error"]


@pytest.mark.parametrize(
    ("change", "wrong"),
    [
        ({"question": ""}, "question: must not be blank"),
        ({"question": " \n"}, "question: must not be blank"),
        ({"model": None}, "model:"),
        ({"documents": []}, "documents: List should have at least 1 item"),
        ({"documents": ["a memo", "  "]}, "documents.1: must not be blank"),
        ({"documents": [{"text": "a memo"}]}, "documents.0: Input should be a"),
        ({"documents": "a memo"}, "documents:"),
        ({"names": ["a.txt"]}, "names: must give one name per document"),
        ({"names": ["a.txt", " ", "c", "d", "e"]}, "names.1: must not be blank"),
        ({"temperature": 0}, "temperature: Extra inputs are not permitted"),
        ({"endpoint": "file:///etc"}, "endpoint: String should match pattern"),
        ({"endpoint": "http://me:pw@127.0.0.1:9/v1"}, "endpoint: must not hold a user"),
        ({"endpoint": "http:///"}, "endpoint: must name a host"),
        ({"relevance_rounds": 0}, TOO_FEW_ROUNDS),
        ({"relevance_rounds": MAX_RELEVANCE_ROUNDS + 1}, TOO_MANY_ROUNDS),
        ({"relevance_rounds": "3"}, NOT_A_ROUND_COUNT),
        ({"relevance_rounds": True}, NOT_A_ROUND_COUNT),
        ({"relevance_rounds": 2.5}, NOT_A_ROUND_COUNT),
    ],
)
def test_a_run_body_that_is_not_valid_is_refused_with_what_is_wrong(
    client: TestClient, llm: FakeLLM, change: dict[str, Any], wrong: str
):
    response = client.post("/api/runs", json={**run_body(), **change})
    assert response.status_code == 422
    assert wrong in response.json()["error"]
    assert llm.requests == []


def test_a_body_that_is_not_json_is_refused(client: TestClient):
    response = client.post("/api/runs", content=b"{not json")
    assert response.status_code == 422
    assert response.json()["error"].startswith("body: Invalid JSON")


@pytest.mark.parametrize(
    "path",
    ["/api/runs", f"{DOCUMENT_TEXT}?name=memo.pdf", f"{DOCUMENT_ZIP}?name={ZIP_NAME}"],
)
# Streamed, the body has no Content-Length, so only reading it can find it too long.
@pytest.mark.parametrize("streamed", [False, True], ids=["sized", "streamed"])
def test_a_body_over_the_limit_is_refused(
    llm: FakeLLM, logs_dir: Path, path: str, streamed: bool
):
    body = {"question": QUESTION, "model": MODEL, "documents": ["x" * 1_001]}
    content = json.dumps(body).encode()
    with client_for(llm, logs_dir, max_body_bytes=1_000) as client:
        response = client.post(path, content=iter([content]) if streamed else content)
    assert response.status_code == 413
    assert llm.requests == []


@pytest.mark.parametrize(
    ("text", "to_unicode", "extracted"),
    [
        (PDF_TEXT, None, PDF_TEXT),
        # pypdf reads a ToUnicode CMap with surrogatepass: half a pair stays half.
        ("Memo A here", {"A": "D800"}, "Memo \ufffd here"),
        ("Memo A here", {"A": "D83DDE00"}, "Memo \U0001f600 here"),
        # A pair mapped from two codes comes as two code units.
        ("Memo AB here", {"A": "D83D", "B": "DE00"}, "Memo \U0001f600 here"),
    ],
    ids=["plain", "half-a-pair", "pair", "pair-from-two-codes"],
)
def test_a_pdfs_text_is_extracted_as_read_documents_extracts_it(
    client: TestClient,
    tmp_path: Path,
    text: str,
    to_unicode: dict[str, str] | None,
    extracted: str,
):
    pdf = tmp_path / "memo.pdf"
    write_pdf(pdf, text, to_unicode=to_unicode)
    response = text_of(client, pdf)
    assert response.status_code == 200
    assert response.json() == {"text": extracted, "reason": None}
    assert read_documents(pdf) == {"memo.pdf": extracted}


# A Word file or a web page is read by its name's suffix, in any case, even with nothing
# before it.
@pytest.mark.parametrize(
    ("name", "write"),
    [
        ("report.docx", lambda path: write_docx(path, REPORT)),
        ("REPORT.DOCX", lambda path: write_docx(path, REPORT)),
        (".docx", lambda path: write_docx(path, REPORT)),
        ("page.html", write_page),
        ("PAGE.HTM", write_page),
    ],
)
def test_a_word_files_or_web_pages_text_is_extracted_as_read_documents_extracts_it(
    client: TestClient, tmp_path: Path, name: str, write: Callable[[Path], None]
):
    path = tmp_path / name
    write(path)
    response = text_of(client, path)
    [text] = read_documents(path).values()
    assert response.status_code == 200
    assert response.json() == {"text": text, "reason": None}


def test_the_page_sends_the_server_each_suffix_r3con_extracts():
    script = (APP_DIR / "app.js").read_text(encoding="utf-8")
    listed = re.search(r"const EXTRACTED_SUFFIXES = \[(.*?)\];", script)
    assert listed is not None
    assert re.findall(r"'([^']+)'", listed[1]) == list(r3con._EXTRACTORS)


def test_the_page_has_a_line_for_each_reason_a_pdf_has_no_text():
    script = (APP_DIR / "app.js").read_text(encoding="utf-8")
    table = re.search(r"const PDF_NO_TEXT = \{(.*?)\};", script, re.DOTALL)
    assert table is not None
    lines = re.findall(r"^\s*(\w+):", table[1], re.MULTILINE)
    assert lines == list(get_args(r3con._PdfNoText))


@pytest.mark.parametrize(
    ("name", "write", "logged", "reason"),
    [
        (
            "memo.pdf",
            write_pdf,
            (
                "memo.pdf: no extractable text — likely a scan, which needs OCR first; "
                "skipping it"
            ),
            "no_text",
        ),
        (
            "memo.pdf",
            lambda path: write_pdf(path, PDF_TEXT, password="secret"),
            "memo.pdf: password-protected (FileNotDecryptedError); skipping it",
            "password",
        ),
        (
            "memo.pdf",
            lambda path: write_pdf(path, PDF_TEXT, trailer=CERTIFICATE_ENCRYPTION),
            "memo.pdf: password-protected (NotImplementedError); skipping it",
            "password",
        ),
        (
            "memo.pdf",
            lambda path: path.write_bytes(CORRUPT_PDF),
            "memo.pdf: not readable as a PDF (PdfStreamError); skipping it",
            "unreadable",
        ),
        (
            "memo.docx",
            lambda path: path.write_bytes(CORRUPT_DOCX),
            "memo.docx: not readable as a Word file (BadZipFile); skipping it",
            None,
        ),
        (
            "memo.docx",
            write_docx,
            "memo.docx: no extractable text; skipping it",
            None,
        ),
        (
            "memo.html",
            lambda path: path.write_bytes(b"<script>var site;</script>"),
            "memo.html: no extractable text; skipping it",
            None,
        ),
    ],
    ids=[
        "scanned-pdf",
        "password-pdf",
        "certificate-pdf",
        "corrupt-pdf",
        "corrupt-docx",
        "empty-docx",
        "script-only-html",
    ],
)
def test_a_file_with_no_text_or_unreadable_has_empty_text_is_logged_and_a_pdf_says_why(
    client: TestClient,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    name: str,
    write: Callable[[Path], None],
    logged: str,
    reason: str | None,
):
    file = tmp_path / name
    write(file)
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        response = text_of(client, file)
    assert response.status_code == 200
    assert response.json() == {"text": "", "reason": reason}
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        logged
    ]


@pytest.mark.parametrize(
    ("params", "wrong"),
    [
        ({}, "name: Field required"),
        ({"name": " "}, "name: must not be blank"),
        ({"name": "memo.pdf", "pages": "1"}, "pages: Extra inputs are not permitted"),
        ({"name": "memo.pdf", "content": "x"}, "content: Extra inputs"),
        ({"name": "memo.txt"}, UNEXTRACTED),
        ({"name": "memo.doc"}, UNEXTRACTED),
        ({"name": "memo"}, UNEXTRACTED),
        ({"name": "memo.xml"}, UNEXTRACTED),
    ],
)
def test_a_document_without_a_valid_name_is_refused_with_what_is_wrong(
    client: TestClient, params: dict[str, str], wrong: str
):
    response = client.post(DOCUMENT_TEXT, params=params, content=CORRUPT_PDF)
    assert response.status_code == 422
    assert wrong in response.json()["error"]


def test_a_zips_documents_come_back_by_path_as_a_folder_walk_reads_them(
    client: TestClient, tmp_path: Path
):
    folder = tmp_path / "memos"
    (folder / "reports").mkdir(parents=True)
    (folder / "01_memo.txt").write_bytes(b"Windows line ends\r\nand a bad byte: \xff\n")
    (folder / "notes.md").write_text("# Notes\n")
    write_docx(folder / "reports" / "02_report.docx", REPORT)
    write_pdf(folder / "reports" / "03_memo.pdf", PDF_TEXT)
    files = sorted(path for path in folder.rglob("*") if path.is_file())
    names = [path.relative_to(folder).as_posix() for path in files]
    for hidden in [".notes.md", ".git/notes.txt", "__MACOSX/01_memo.txt"]:
        (folder / hidden).parent.mkdir(exist_ok=True)
        (folder / hidden).write_text("Left out of a folder walk, and of a zip.\n")
    every = (path for path in folder.rglob("*") if path.is_file())
    archive = zipped(
        tmp_path,
        {path.relative_to(folder).as_posix(): path.read_bytes() for path in every},
    )
    response = unzip(client, archive.read_bytes())
    assert response.status_code == 200
    unzipped = response.json()
    assert [file["name"] for file in unzipped["files"]] == names
    by_name = {file["name"]: file["text"] for file in unzipped["files"]}
    assert by_name == read_documents(folder)
    assert [file["size"] for file in unzipped["files"]] == [
        path.stat().st_size for path in files
    ]
    assert unzipped["skipped"] == 0


def test_a_zips_documents_come_in_the_order_read_documents_walks_a_folder(
    client: TestClient, tmp_path: Path
):
    folder = tmp_path / "memos"
    for path in ["a-b/x.txt", "a/b.txt", "a.txt"]:
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_text(f"{path}\n")
    archive = zipped(
        tmp_path, {path: f"{path}\n" for path in ["a-b/x.txt", "a/b.txt", "a.txt"]}
    )
    response = unzip(client, archive.read_bytes())
    assert unzipped_names(response) == ["a/b.txt", "a-b/x.txt", "a.txt"]
    files = response.json()["files"]
    assert [(f["name"], f["text"]) for f in files] == list(
        read_documents(folder).items()
    )


def test_a_zip_of_a_hidden_folder_reads_its_documents_as_a_walk_of_that_folder_does(
    client: TestClient, tmp_path: Path
):
    entries = {
        ".notes/a.txt": "A note.\n",
        ".notes/.git/b.txt": "A hidden folder's note.\n",
        "__MACOSX/.notes/._a.txt": "x",
    }
    for path, text in entries.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    response = unzip(client, zipped(tmp_path, entries).read_bytes())
    assert unzipped_names(response) == ["a.txt"]
    texts = {file["name"]: file["text"] for file in response.json()["files"]}
    assert texts == read_documents(tmp_path / ".notes")


@pytest.mark.parametrize(
    ("paths", "names"),
    [
        (["memos/a.txt", "memos/q3/b.txt"], ["a.txt", "q3/b.txt"]),
        (
            ["memos/a.txt", "memos/q3/b.txt", "__MACOSX/memos/._a.txt"],
            ["a.txt", "q3/b.txt"],
        ),
        (["a/notes.txt", "b/notes.txt"], ["a/notes.txt", "b/notes.txt"]),
        (["memos/a.txt", "b.txt"], ["b.txt", "memos/a.txt"]),
    ],
    ids=["one-folder", "one-folder-from-finder", "two-folders", "a-file-at-the-top"],
)
def test_a_zips_documents_are_named_below_a_top_folder_they_all_share(
    client: TestClient, tmp_path: Path, paths: list[str], names: list[str]
):
    archive = zipped(tmp_path, {path: MEMO for path in paths})
    assert unzipped_names(unzip(client, archive.read_bytes())) == names


def test_hidden_macos_and_lock_entries_are_left_out_uncounted_and_others_counted(
    client: TestClient, tmp_path: Path
):
    entries = {
        "memos/": "",
        "memos/01_memo.txt": MEMO,
        "memos/.DS_Store": "x",
        "memos/.git/notes.txt": "A hidden folder's note.\n",
        "__MACOSX/memos/._01_memo.txt": "x",
        "__MACOSX/memos/02_memo.txt": "macOS's metadata, named like a memo.\n",
        "memos/~$report.docx": "x",
        "memos/notes.py": "print('not a document')\n",
        "memos/photo.jpg": "x",
    }
    response = unzip(client, zipped(tmp_path, entries).read_bytes())
    assert response.status_code == 200
    memo = {"name": "01_memo.txt", "size": len(MEMO), "text": MEMO, "reason": None}
    assert response.json() == {"files": [memo], "skipped": 2}


def test_a_zip_in_the_zip_is_skipped_unopened(client: TestClient, tmp_path: Path):
    inner = tmp_path / "inner.zip"
    write_zip(inner, {"inner_memo.txt": "A memo in a zip in the zip.\n"})
    entries = {"outer_memo.txt": MEMO, "inner.zip": inner.read_bytes()}
    response = unzip(client, zipped(tmp_path, entries).read_bytes())
    memo = {"name": "outer_memo.txt", "size": len(MEMO), "text": MEMO, "reason": None}
    assert response.json() == {"files": [memo], "skipped": 1}


def test_a_zipped_document_with_no_text_or_unreadable_comes_back_empty_saying_why(
    client: TestClient, tmp_path: Path
):
    scan, locked = tmp_path / "scan.pdf", tmp_path / "locked.pdf"
    write_pdf(scan)
    write_pdf(locked, PDF_TEXT, password="secret")
    entries = {
        "scan.pdf": scan.read_bytes(),
        "locked.pdf": locked.read_bytes(),
        "blank.txt": "",
        "bad.pdf": CORRUPT_PDF,
    }
    response = unzip(client, zipped(tmp_path, entries).read_bytes())
    empty = [
        ("bad.pdf", len(CORRUPT_PDF), "unreadable"),
        ("blank.txt", 0, None),
        ("locked.pdf", locked.stat().st_size, "password"),
        ("scan.pdf", scan.stat().st_size, "no_text"),
    ]
    assert response.json() == {
        "files": [
            {"name": name, "size": size, "text": "", "reason": reason}
            for name, size, reason in empty
        ],
        "skipped": 0,
    }


def _flip_a_compressed_byte(body: bytes) -> bytes:
    flipped = body[LONG_MEMO_DATA_AT] ^ 0xFF
    return body[:LONG_MEMO_DATA_AT] + bytes([flipped]) + body[LONG_MEMO_DATA_AT + 1 :]


@pytest.mark.parametrize(
    "damage",
    [
        lambda body: b"",
        lambda body: CORRUPT_DOCX,
        lambda body: body[: len(body) // 2],
        _flip_a_compressed_byte,
    ],
    ids=["empty", "garbage", "cut-short", "bad-data"],
)
def test_a_corrupt_zip_is_refused_saying_so_and_logged(
    client: TestClient,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    damage: Callable[[bytes], bytes],
):
    body = damage(zipped(tmp_path, LONG_MEMO).read_bytes())
    with caplog.at_level(logging.WARNING, logger="r3con.app"):
        response = unzip(client, body)
    assert response.status_code == 422
    assert response.json() == {"error": UNREADABLE_ZIP}
    [logged] = [r.getMessage() for r in caplog.records if r.name == "r3con.app"]
    assert logged.startswith(f"{ZIP_NAME}: not a readable zip (")


def test_a_zip_whose_entry_holds_more_than_it_claims_is_refused_as_corrupt(
    client: TestClient, tmp_path: Path
):
    body = bytearray(zipped(tmp_path, LONG_MEMO).read_bytes())
    size_at = body.index(CENTRAL_DIRECTORY_ENTRY) + CENTRAL_SIZE_AT
    body[size_at : size_at + 4] = (1).to_bytes(4, "little")
    response = unzip(client, bytes(body))
    assert response.status_code == 422
    assert response.json() == {"error": UNREADABLE_ZIP}


def test_an_encrypted_zip_is_refused_saying_so(client: TestClient, tmp_path: Path):
    body = bytearray(zipped(tmp_path, LONG_MEMO).read_bytes())
    body[body.index(LOCAL_ENTRY) + LOCAL_FLAGS_AT] |= ZIP_ENCRYPTED_FLAG
    body[body.index(CENTRAL_DIRECTORY_ENTRY) + CENTRAL_FLAGS_AT] |= ZIP_ENCRYPTED_FLAG
    response = unzip(client, bytes(body))
    assert response.status_code == 422
    assert response.json() == {"error": ENCRYPTED_ZIP}


@pytest.mark.parametrize(
    ("entries", "status"), [(MAX_ZIP_ENTRIES, 200), (MAX_ZIP_ENTRIES + 1, 422)]
)
def test_a_zip_of_more_entries_than_the_cap_is_refused(
    client: TestClient, tmp_path: Path, entries: int, status: int
):
    memos = {f"{i:04}.txt": MEMO for i in range(entries)}
    response = unzip(client, zipped(tmp_path, memos).read_bytes())
    assert response.status_code == status
    if status == 422:
        assert response.json() == {"error": TOO_MANY_ENTRIES}


def test_a_zip_whose_documents_would_unzip_past_the_cap_is_refused(
    client: TestClient, tmp_path: Path
):
    entries = {"a.txt": MEMO, "b.txt": b"a" * MAX_ZIP_BYTES}
    response = unzip(client, zipped(tmp_path, entries).read_bytes())
    assert response.status_code == 422
    assert response.json() == {"error": TOO_MANY_BYTES}


@pytest.mark.parametrize(
    ("params", "wrong"),
    [
        ({}, "name: Field required"),
        ({"name": " "}, "name: must not be blank"),
        ({"name": ZIP_NAME, "pages": "1"}, "pages: Extra inputs are not permitted"),
    ],
)
def test_a_zip_without_a_valid_name_is_refused_with_what_is_wrong(
    client: TestClient, tmp_path: Path, params: dict[str, str], wrong: str
):
    body = zipped(tmp_path, {"memo.txt": MEMO}).read_bytes()
    response = client.post(DOCUMENT_ZIP, params=params, content=body)
    assert response.status_code == 422
    assert wrong in response.json()["error"]


def test_the_page_walks_a_folder_for_the_suffixes_read_documents_walks_for():
    script = (APP_DIR / "app.js").read_text(encoding="utf-8")
    typed = re.search(r"const SUFFIX_TYPES = \{(.*?)\};", script, re.DOTALL)
    assert typed is not None
    walked = re.search(r"const FOLDER_SUFFIXES = (.*?);", script)
    assert walked is not None
    assert walked[1] == "Object.keys(SUFFIX_TYPES)"
    listed = set(re.findall(r"'([^']+)': '", typed[1]))
    assert listed == r3con.TEXT_SUFFIXES


@pytest.mark.parametrize(
    ("body", "wrong"),
    [
        ({"provider": "mistral", "key": OPENAI_KEY}, "does not match any of the"),
        ({"provider": "openai", "key": " "}, "openai.key: must not be blank"),
        ({"provider": "openai"}, "openai.key: Field required"),
        ({**STORED_OPENAI, "x": 1}, "openai.x: Extra inputs"),
        ({"provider": "other", "key": OTHER_KEY}, "other.endpoint: Field required"),
        ({"provider": "other", "endpoint": "file:///etc"}, "other.endpoint"),
        (
            {**OTHER, "endpoint": f"http://me:{OTHER_KEY}@127.0.0.1:9/v1"},
            "other.endpoint: must not hold a user or password",
        ),
        ({**OTHER, "endpoint": "http:///"}, "other.endpoint: must name a host"),
        ({**OTHER, "endpoint": "https:////v1"}, "other.endpoint: must name a host"),
    ],
)
def test_a_key_body_that_is_not_valid_is_refused_without_echoing_the_key(
    client: TestClient, body: dict[str, Any], wrong: str
):
    response = client.post("/api/keys", json=body)
    assert response.status_code == 422
    assert wrong in response.json()["error"]
    assert OPENAI_KEY not in response.text
    assert OTHER_KEY not in response.text


def test_forgetting_a_server_takes_its_url(client: TestClient):
    client.post("/api/keys", json=OTHER)
    response = client.request("DELETE", "/api/keys", json={"provider": "other"})
    assert response.status_code == 422
    assert "other.endpoint: Field required" in response.json()["error"]
    assert client.get("/api/keys").json()["other"] == [
        {"endpoint": ENDPOINT, "key": OTHER_KEY[-4:]}
    ]


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/runs/nothing/events"),
        ("POST", "/api/runs/nothing/asks"),
        ("GET", "/api/runs/nothing/representation"),
        ("POST", "/api/runs/nothing/cancel"),
        ("GET", "/api/asks/nothing/events"),
        ("POST", "/api/asks/nothing/cancel"),
    ],
)
def test_a_run_or_ask_this_server_did_not_start_is_not_found(
    client: TestClient, method: str, path: str
):
    response = client.request(method, path, json=ASK)
    assert response.status_code == 404
    assert "nothing" in response.json()["error"]


@pytest.mark.usefixtures("one_call_at_a_time")
def test_a_cancelled_run_sends_no_further_call_and_its_stream_ends_with_run_failed(
    held_llm: FakeLLM, logs_dir: Path, caplog: pytest.LogCaptureFixture
):
    with (
        caplog.at_level(logging.INFO, logger="r3con.app"),
        client_for(held_llm, logs_dir) as client,
    ):
        run_id = start_run(client)
        assert held_llm.called.wait(GATE_TIMEOUT_SECONDS)
        worker = worker_of(run_id)
        response = client.post(f"/api/runs/{run_id}/cancel")
        held_llm.gate.set()
        last = run_events(client, run_id)[-1]
        worker.join(GATE_TIMEOUT_SECONDS)
    assert response.status_code == 202
    assert len(held_llm.requests) == 1
    assert (last["type"], last["stage"]) == ("run_failed", "relevance")
    assert last["error"].startswith("Cancelled: ")
    logged = [
        (r.levelno, r.getMessage()) for r in caplog.records if r.name == "r3con.app"
    ]
    assert logged == [(logging.INFO, f"run_failed: {last['error']}")]


def test_a_cancelled_ask_sends_no_further_call_and_its_stream_ends_with_ask_failed(
    llm: FakeLLM, logs_dir: Path
):
    with client_for(llm, logs_dir) as client:
        run_id = finished_run(client)
        llm.gate.clear()
        llm.called.clear()
        # A reply that does not fit, which the ask would send again.
        llm.answers(reasoning="not an answer")
        sent = len(llm.requests)
        ask_id = client.post(f"/api/runs/{run_id}/asks", json=ASK).json()["ask_id"]
        assert llm.called.wait(GATE_TIMEOUT_SECONDS)
        response = client.post(f"/api/asks/{ask_id}/cancel")
        llm.gate.set()
        last = events_of(client, f"/api/asks/{ask_id}/events")[-1]
    assert response.status_code == 202
    assert len(llm.requests) == sent + 1
    assert (last["type"], last["stage"]) == ("ask_failed", "reasoning")
    assert last["error"].startswith("Cancelled: ")


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_serve_streams_a_runs_events_as_they_happen_and_stops(
    held_llm: FakeLLM, logs_dir: Path
):
    url, stop = serve(logs_dir=logs_dir, completion=held_llm)
    try:
        first, rest = first_event_then_the_rest(url, held_llm.gate)
    finally:
        held_llm.gate.set()
        stop()
    assert first["type"] == "run_started"
    assert rest[-1]["type"] == "run_finished"
    with pytest.raises(httpx2.ConnectError):
        httpx2.get(url, trust_env=False)


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.usefixtures("one_call_at_a_time")
def test_stop_waits_for_a_run_on_a_held_call_and_returns_once_it_has_ended(
    held_llm: FakeLLM, logs_dir: Path
):
    url, stop = serve(logs_dir=logs_dir, completion=held_llm)
    stopping = threading.Thread(target=stop)
    run_id, started = shut_down_with_a_stream_open(url, held_llm, stopping.start)
    worker = worker_of(run_id)
    stopping.join(STILL_WAITING_SECONDS)
    assert stopping.is_alive()
    held_llm.gate.set()
    stopping.join(GATE_TIMEOUT_SECONDS)
    assert not stopping.is_alive()
    assert not worker.is_alive()
    assert len(held_llm.requests) == 1
    error = Path(started["run_dir"]) / "relevance" / "error.txt"
    assert "Cancelled" in error.read_text()


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
@pytest.mark.usefixtures("one_call_at_a_time")
def test_a_forced_exit_ends_the_wait_at_once_and_the_run_left_going_ends_cleanly(
    held_llm: FakeLLM,
    logs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    unhandled: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(threading, "excepthook", unhandled.append)
    listener = listen(0)
    server, url = _server(listener, logs_dir, held_llm)
    serving = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
    serving.start()

    def ctrl_c() -> None:
        server.should_exit = True

    run_id, _ = shut_down_with_a_stream_open(url, held_llm, ctrl_c)
    worker = worker_of(run_id)
    server.force_exit = True  # as a second Ctrl-C sets it
    serving.join(FORCED_EXIT_SECONDS)
    assert not serving.is_alive()
    assert worker.is_alive()
    held_llm.gate.set()
    worker.join(GATE_TIMEOUT_SECONDS)
    assert not worker.is_alive()
    assert unhandled == []
    assert not [r for r in caplog.records if "on_event raised" in r.getMessage()]


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_run_posted_once_the_server_is_stopping_is_refused_and_starts_nothing(
    llm: FakeLLM, logs_dir: Path
):
    listener = listen(0)
    server, url = _server(listener, logs_dir, llm)
    serving = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
    serving.start()
    server.should_exit = True
    serving.join(GATE_TIMEOUT_SECONDS)
    # A request the server was still handling as its shutdown began.
    base_url, token = url.split("/#token=")
    client = TestClient(
        cast(ASGIApp, server.config.app),
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"},
    )
    response = client.post("/api/runs", json=run_body())
    assert response.status_code == 503
    assert response.json() == {"error": "The server is stopping."}
    assert (server.app_state.jobs, server.app_state.workers) == (
        {"run": {}, "ask": {}},
        [],
    )
    assert not logs_dir.exists()
    assert llm.requests == []


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_connection_the_server_accepts_sends_with_tcp_nodelay(logs_dir: Path):
    listener = listen(0)
    server, url = _server(listener, logs_dir, None)
    serving = threading.Thread(target=server.run, kwargs={"sockets": [listener]})
    serving.start()
    try:
        with http_client(url) as http:
            assert http.get("/api/keys").status_code == 200
            (connection,) = server.server_state.connections
            accepted = connection.transport.get_extra_info("socket")
            nodelay = accepted.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
    finally:
        server.should_exit = True
        serving.join(GATE_TIMEOUT_SECONDS)
    assert not serving.is_alive()
    assert nodelay


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_a_shutdown_with_a_stream_open_cancels_its_run_before_closing_the_stream(
    logs_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "completion", rate_limited)
    url, stop = serve(logs_dir=logs_dir)
    with http_client(url) as http:
        run_id = http.post("/api/runs", json=run_body()).json()["run_id"]
        worker = worker_of(run_id)
        with http.stream("GET", f"/api/runs/{run_id}/events") as response:
            lines = response.iter_lines()
            first = next(line for line in lines if line.startswith("data: "))
            stop()
            rest = events_in("\n".join(lines))
    assert not worker.is_alive()
    last = rest[-1]
    assert (last["type"], last["stage"]) == ("run_failed", "relevance")
    assert last["error"].startswith("Cancelled: ")
    run_dir = Path(json.loads(first.removeprefix("data: "))["run_dir"])
    assert "Cancelled" in (run_dir / "relevance" / "error.txt").read_text()
