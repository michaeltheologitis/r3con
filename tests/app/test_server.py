import contextlib
import json
import logging
import re
import socket
import threading
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import httpx2
import litellm
import pytest
from litellm.exceptions import RateLimitError
from starlette.testclient import TestClient
from starlette.types import ASGIApp

from r3con import r3con
from r3con.app.server import (
    ENCRYPTED_ZIP,
    MAX_ZIP_BYTES,
    MAX_ZIP_ENTRIES,
    TOO_MANY_BYTES,
    TOO_MANY_ENTRIES,
    UNREADABLE_ZIP,
    ZIP_ENCRYPTED_FLAG,
    ServerState,
    _server,
    create_app,
    listen,
    serve,
)
from r3con.r3con import read_documents
from tests.documents import CORRUPT_DOCX, write_docx, write_pdf, write_zip
from tests.fakes import GATE_TIMEOUT_SECONDS, FakeLLM

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
QUESTION = "Which contractor's sites logged the most equipment incidents in Q3?"
MODEL = "openai/gpt-6-luna"
# A bare OpenAI name that litellm's model map places from its floor, 1.103, on.
BARE_OPENAI_MODEL = "gpt-5"
OPENAI_KEY = "sk-stored-openai-key-1234"
OTHER_KEY = "other-endpoint-key-5678"
ENVIRONMENT_KEY = "key-from-the-environment-9012"
ENDPOINT = "http://127.0.0.1:9/v1"
STORED_OPENAI = {"provider": "openai", "key": OPENAI_KEY}
OTHER = {"provider": "other", "endpoint": ENDPOINT, "key": OTHER_KEY}
KEYLESS_OTHER = {"provider": "other", "endpoint": ENDPOINT}
ASK = {"question": "Which site logged the fewest?", "model": MODEL}
PDF_TEXT = "Incidents this quarter: 7."
CORRUPT_PDF = b"%PDF-1.4 truncated garbage"
REPORT = [
    "Incidents by site:",
    [["Site", "Incidents"], ["Halloran", "7"], ["Merrow", "3"]],
    "Merrow's count is provisional.",
]
DOCUMENT_TEXT = "/api/documents/text"
UNEXTRACTED = "name: must end in one of .pdf, .docx"
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
SCHEMA_REPLY = """Thought: one row per site.
<schema>
class Row(BaseModel):
    site: str


class Parse(BaseModel):
    rows: list[Row]
</schema>"""
ANSWER = {
    "answer": "Northgate",
    "provenance_ids": {"summary_ids": [], "parse_ids": []},
}


@pytest.fixture
def llm() -> FakeLLM:
    """A model that answers every stage, so that a run or an ask over it finishes."""
    return FakeLLM().answers(
        relevance="A site memo.",
        schema=SCHEMA_REPLY,
        parsing={"rows": [{"site": "Northgate"}]},
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


def client_for(model: FakeLLM, logs_dir: Path, **options: Any) -> TestClient:
    state = ServerState(logs_dir=logs_dir, completion=model)
    app = create_app(token=TOKEN, port=PORT, state=state, **options)
    return TestClient(app, base_url=BASE_URL, headers=AUTH)


@pytest.fixture
def client(llm: FakeLLM, logs_dir: Path) -> Iterator[TestClient]:
    with client_for(llm, logs_dir) as client:
        yield client


def memos() -> list[str]:
    return [path.read_text() for path in sorted(MEMOS.iterdir())]


def run_body(model: str = MODEL) -> dict[str, Any]:
    return {"question": QUESTION, "model": model, "documents": memos()}


def events_in(stream: str) -> list[dict[str, Any]]:
    """The events of a ``text/event-stream`` body, in order."""
    return [
        json.loads(line.removeprefix("data: "))
        for line in stream.splitlines()
        if line.startswith("data: ")
    ]


def start_run(client: TestClient, model: str = MODEL) -> str:
    response = client.post("/api/runs", json=run_body(model))
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def events_of(client: TestClient, path: str) -> list[dict[str, Any]]:
    response = client.get(path)
    assert response.headers["content-type"].startswith("text/event-stream")
    return events_in(response.text)


def run_events(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    return events_of(client, f"/api/runs/{run_id}/events")


def ask_events(client: TestClient, run_id: str) -> list[dict[str, Any]]:
    response = client.post(f"/api/runs/{run_id}/asks", json=ASK)
    assert response.status_code == 202, response.text
    return events_of(client, f"/api/asks/{response.json()['ask_id']}/events")


def finished_run(client: TestClient, model: str = MODEL) -> str:
    run_id = start_run(client, model)
    last = run_events(client, run_id)[-1]
    assert last["type"] == "run_finished", last
    return run_id


def text_of(client: TestClient, path: Path) -> httpx2.Response:
    """The server's answer for the text of the file at ``path``, sent as it is."""
    return client.post(
        DOCUMENT_TEXT, params={"name": path.name}, content=path.read_bytes()
    )


def zipped(tmp_path: Path, entries: Mapping[str, str | bytes]) -> Path:
    path = tmp_path / ZIP_NAME
    write_zip(path, entries)
    return path


def unzip(client: TestClient, body: bytes, **params: str) -> httpx2.Response:
    """The server's answer for the zip ``body``, sent as it is."""
    return client.post(DOCUMENT_ZIP, params={"name": ZIP_NAME, **params}, content=body)


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
    return next(t for t in threading.enumerate() if t.name == f"r3con-app-{job_id}")


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


def shut_down_with_a_stream_open(
    url: str, model: FakeLLM, shut_down: Callable[[], None]
) -> tuple[str, dict[str, Any]]:
    """Over a real connection: start a run and read its stream's first event; once
    ``model`` holds its first call, call ``shut_down``, and read on until the server's
    shutdown closes the stream, after its grace. Returns the run's id and that first
    event."""
    with http_client(url) as http:
        run_id = http.post("/api/runs", json=run_body()).json()["run_id"]
        with http.stream("GET", f"/api/runs/{run_id}/events") as response:
            lines = response.iter_lines()
            first = next(line for line in lines if line.startswith("data: "))
            assert model.called.wait(GATE_TIMEOUT_SECONDS)
            shut_down()
            # Cut short, so without the end of its body.
            with contextlib.suppress(httpx2.RemoteProtocolError):
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


def test_keys_say_which_providers_have_one_and_never_what_it_is(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("GEMINI_API_KEY", ENVIRONMENT_KEY)
    responses = [
        client.post("/api/keys", json=STORED_OPENAI),
        client.post("/api/keys", json=OTHER),
        client.get("/api/keys"),
    ]
    assert responses[-1].json() == {
        "openai": True,
        "anthropic": False,
        "gemini": True,
        "other": {"endpoint": ENDPOINT, "key": True},
    }
    responses += [
        client.request("DELETE", "/api/keys", json={"provider": "openai"}),
        client.request("DELETE", "/api/keys", json={"provider": "other"}),
    ]
    assert responses[-1].json() == {
        "openai": False,
        "anthropic": False,
        "gemini": True,
        "other": None,
    }
    for response in responses:
        assert response.status_code == 200
        for key in (OPENAI_KEY, OTHER_KEY, ENVIRONMENT_KEY):
            assert key not in response.text


@pytest.mark.parametrize(
    ("stored", "model_name", "sent"),
    [
        (STORED_OPENAI, MODEL, {"api_key": OPENAI_KEY}),
        (STORED_OPENAI, BARE_OPENAI_MODEL, {"api_key": OPENAI_KEY}),
        (STORED_OPENAI, "anthropic/claude-sonnet-5-5", {}),
        (OTHER, "hosted_vllm/qwen", {"api_key": OTHER_KEY, "api_base": ENDPOINT}),
        (KEYLESS_OTHER, "hosted_vllm/qwen", {"api_base": ENDPOINT}),
        (OTHER, BARE_OPENAI_MODEL, {}),
        (OTHER, "claude-sonnet-5-5", {}),
        (OTHER, "gemini-2.5-pro", {}),
        (OTHER, "azure/x", {}),
        (OTHER, "ollama_chat/llama3", {"api_key": OTHER_KEY, "api_base": ENDPOINT}),
    ],
)
def test_a_stored_key_reaches_every_call_for_its_providers_model(
    client: TestClient,
    llm: FakeLLM,
    stored: dict[str, str],
    model_name: str,
    sent: dict[str, str],
):
    client.post("/api/keys", json=stored)
    finished_run(client, model_name)
    assert transport(llm) == [sent] * len(llm.requests)


def test_a_name_litellm_cannot_place_is_sent_to_other_which_litellm_then_refuses(
    client: TestClient, llm: FakeLLM
):
    client.post("/api/keys", json=OTHER)
    last = run_events(client, start_run(client, "qwen3-32b"))[-1]
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
    assert client.get("/api/keys").json()["other"]["endpoint"] == ENDPOINT
    llm.answers(reasoning=RuntimeError(f"Incorrect API key provided: {OPENAI_KEY}."))
    last = run_events(client, start_run(client))[-1]
    assert {call["api_key"] for call in llm.requests} == {OPENAI_KEY}
    assert last["error"] == "RuntimeError: Incorrect API key provided: REDACTED."
    llm.answers(reasoning=ANSWER)
    llm.requests.clear()
    finished_run(client, "hosted_vllm/qwen")
    assert {call["api_base"] for call in llm.requests} == {ENDPOINT}


def test_a_placeholder_key_too_short_to_be_a_secret_is_not_masked(
    client: TestClient, llm: FakeLLM
):
    client.post("/api/keys", json={**OTHER, "key": "EMPTY"})
    llm.answers(relevance=RuntimeError("The model sent an EMPTY reply."))
    last = run_events(client, start_run(client, "hosted_vllm/qwen"))[-1]
    assert last["error"] == "RuntimeError: The model sent an EMPTY reply."


def test_with_a_keyless_other_a_bare_openai_model_never_reaches_its_endpoint(
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
        last = run_events(client, start_run(client))[-1]
    assert last["type"] == "run_failed"
    assert last["error"] == "RuntimeError: Incorrect API key provided: REDACTED."
    logged = [r.getMessage() for r in caplog.records if r.name == "r3con.app"]
    assert logged == [f"run_failed: {last['error']}"]


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
        ({"temperature": 0}, "temperature: Extra inputs are not permitted"),
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
    assert response.json() == {"text": extracted}
    assert read_documents(pdf) == [extracted]


# A Word file is read by its name's suffix, in any case, even with nothing before it.
@pytest.mark.parametrize("name", ["report.docx", "REPORT.DOCX", ".docx"])
def test_a_word_files_text_is_extracted_as_read_documents_extracts_it(
    client: TestClient, tmp_path: Path, name: str
):
    path = tmp_path / name
    write_docx(path, REPORT)
    response = text_of(client, path)
    [text] = read_documents(path)
    assert response.status_code == 200
    assert response.json() == {"text": text}


def test_the_page_sends_the_server_each_suffix_r3con_extracts():
    script = (APP_DIR / "app.js").read_text(encoding="utf-8")
    listed = re.search(r"const EXTRACTED_SUFFIXES = \[(.*?)\];", script)
    assert listed is not None
    assert re.findall(r"'([^']+)'", listed[1]) == list(r3con._EXTRACTORS)


@pytest.mark.parametrize(
    ("name", "write", "logged"),
    [
        (
            "memo.pdf",
            write_pdf,
            (
                "memo.pdf: no extractable text — a scanned PDF needs OCR first; "
                "skipping it"
            ),
        ),
        (
            "memo.pdf",
            lambda path: path.write_bytes(CORRUPT_PDF),
            "memo.pdf: not readable as a PDF (PdfStreamError); skipping it",
        ),
        (
            "memo.docx",
            lambda path: path.write_bytes(CORRUPT_DOCX),
            "memo.docx: not readable as a Word file (BadZipFile); skipping it",
        ),
        (
            "memo.docx",
            write_docx,
            "memo.docx: no extractable text; skipping it",
        ),
    ],
    ids=["scanned-pdf", "corrupt-pdf", "corrupt-docx", "empty-docx"],
)
def test_a_file_with_no_text_layer_or_that_cannot_be_read_has_empty_text_and_is_logged(
    client: TestClient,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    name: str,
    write: Callable[[Path], None],
    logged: str,
):
    file = tmp_path / name
    write(file)
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        response = text_of(client, file)
    assert response.status_code == 200
    assert response.json() == {"text": ""}
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
        ({"name": "memo.html"}, UNEXTRACTED),
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
    archive = zipped(tmp_path, {n: f.read_bytes() for n, f in zip(names, files)})
    response = unzip(client, archive.read_bytes())
    assert response.status_code == 200
    unzipped = response.json()
    assert [file["name"] for file in unzipped["files"]] == names
    assert [file["text"] for file in unzipped["files"]] == read_documents(folder)
    assert [file["size"] for file in unzipped["files"]] == [
        path.stat().st_size for path in files
    ]
    assert unzipped["skipped"] == unzipped["unread"] == 0


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
    assert [f["text"] for f in response.json()["files"]] == read_documents(folder)


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
    memo = {"name": "01_memo.txt", "size": len(MEMO), "text": MEMO}
    assert response.json() == {"files": [memo], "skipped": 2, "unread": 0}


def test_a_zip_in_the_zip_is_skipped_unopened(client: TestClient, tmp_path: Path):
    inner = tmp_path / "inner.zip"
    write_zip(inner, {"inner_memo.txt": "A memo in a zip in the zip.\n"})
    entries = {"outer_memo.txt": MEMO, "inner.zip": inner.read_bytes()}
    response = unzip(client, zipped(tmp_path, entries).read_bytes())
    memo = {"name": "outer_memo.txt", "size": len(MEMO), "text": MEMO}
    assert response.json() == {"files": [memo], "skipped": 1, "unread": 0}


def test_a_zip_read_to_a_limit_reads_its_first_documents_and_counts_the_rest(
    client: TestClient, tmp_path: Path
):
    memos = {f"{i}.txt": f"Memo {i}.\n" for i in (3, 1, 2)}
    response = unzip(client, zipped(tmp_path, memos).read_bytes(), limit="2")
    assert unzipped_names(response) == ["1.txt", "2.txt"]
    assert response.json()["unread"] == 1


def test_a_zip_read_to_a_limit_counts_only_documents_with_text(
    client: TestClient, tmp_path: Path
):
    memos = {
        "1.txt": "",
        "2.txt": " \n",
        **{f"{i}.txt": f"Memo {i}.\n" for i in range(3, 7)},
    }
    response = unzip(client, zipped(tmp_path, memos).read_bytes(), limit="3")
    assert unzipped_names(response) == ["1.txt", "2.txt", "3.txt", "4.txt", "5.txt"]
    assert response.json()["unread"] == 1


def test_a_zipped_document_with_no_text_or_unreadable_comes_back_empty_to_refuse(
    client: TestClient, tmp_path: Path
):
    scan = tmp_path / "scan.pdf"
    write_pdf(scan)
    entries = {"scan.pdf": scan.read_bytes(), "blank.txt": "", "bad.pdf": CORRUPT_PDF}
    response = unzip(client, zipped(tmp_path, entries).read_bytes())
    assert response.json() == {
        "files": [
            {"name": "bad.pdf", "size": len(CORRUPT_PDF), "text": ""},
            {"name": "blank.txt", "size": 0, "text": ""},
            {"name": "scan.pdf", "size": scan.stat().st_size, "text": ""},
        ],
        "skipped": 0,
        "unread": 0,
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


# A document past the limit is not read, so it counts for nothing against the cap.
@pytest.mark.parametrize(("limit", "status"), [(None, 422), ("1", 200)])
def test_a_zip_whose_documents_read_would_unzip_past_the_cap_is_refused(
    client: TestClient, tmp_path: Path, limit: str | None, status: int
):
    entries = {"a.txt": MEMO, "b.txt": b"a" * MAX_ZIP_BYTES}
    body = zipped(tmp_path, entries).read_bytes()
    response = unzip(client, body, **({} if limit is None else {"limit": limit}))
    assert response.status_code == status
    if status == 422:
        assert response.json() == {"error": TOO_MANY_BYTES}


@pytest.mark.parametrize(
    ("params", "wrong"),
    [
        ({}, "name: Field required"),
        ({"name": " "}, "name: must not be blank"),
        ({"name": ZIP_NAME, "pages": "1"}, "pages: Extra inputs are not permitted"),
        ({"name": ZIP_NAME, "limit": "0"}, "limit: Input should be greater than"),
        ({"name": ZIP_NAME, "limit": "all"}, "limit: Input should be a valid integer"),
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
    plain = re.search(r"const PLAIN_TEXT_SUFFIXES = \[(.*?)\];", script)
    assert plain is not None
    walked = re.search(r"const FOLDER_SUFFIXES = (.*?);", script)
    assert walked is not None
    assert walked[1] == "[...PLAIN_TEXT_SUFFIXES, ...EXTRACTED_SUFFIXES]"
    listed = set(re.findall(r"'([^']+)'", plain[1]))
    assert listed == r3con.TEXT_SUFFIXES - set(r3con._EXTRACTORS)


@pytest.mark.parametrize(
    ("body", "wrong"),
    [
        ({"provider": "mistral", "key": OPENAI_KEY}, "does not match any of the"),
        ({"provider": "openai", "key": " "}, "openai.key: must not be blank"),
        ({"provider": "openai"}, "openai.key: Field required"),
        ({**STORED_OPENAI, "x": 1}, "openai.x: Extra inputs"),
        ({"provider": "other", "key": OTHER_KEY}, "other.endpoint: Field required"),
        ({"provider": "other", "endpoint": "file:///etc"}, "other.endpoint"),
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


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/api/runs/nothing/events"),
        ("POST", "/api/runs/nothing/asks"),
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
