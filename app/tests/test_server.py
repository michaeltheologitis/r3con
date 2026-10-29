import json
import logging
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx2
import litellm
import pytest
from r3con_app.server import create_app, serve
from starlette.testclient import TestClient

litellm.suppress_debug_info = True

PORT = 8765
TOKEN = "the-launch-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BASE_URL = f"http://127.0.0.1:{PORT}"
MEMOS = Path(__file__).resolve().parents[2] / "examples" / "memos"
GATE_TIMEOUT_SECONDS = 10
QUESTION = "Which contractor's sites logged the most equipment incidents in Q3?"
MODEL = "openai/gpt-6-luna"
OPENAI_KEY = "sk-stored-openai-key-1234"
OTHER_KEY = "other-endpoint-key-5678"
ENVIRONMENT_KEY = "key-from-the-environment-9012"
ENDPOINT = "http://127.0.0.1:9/v1"
STORED_OPENAI = {"provider": "openai", "key": OPENAI_KEY}
OTHER = {"provider": "other", "endpoint": ENDPOINT, "key": OTHER_KEY}
KEYLESS_OTHER = {"provider": "other", "endpoint": ENDPOINT}
ASK = {"question": "Which site logged the fewest?", "model": MODEL}
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
USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
SCHEMA_REPLY = """Thought: one row per site.
<schema>
class Row(BaseModel):
    site: str


class Parse(BaseModel):
    rows: list[Row]
</schema>"""
# Each reply by the name of the structure its request asks for; "schema" asks for none.
REPLIES = {
    "schema": SCHEMA_REPLY,
    "RelevanceSummary": json.dumps({"summary": "A site memo.", "provenance_ids": []}),
    "ParseReply": json.dumps(
        {"parse": {"rows": [{"site": "Northgate"}]}, "provenance_ids": []}
    ),
    "ReasoningAnswer": json.dumps(
        {
            "answer": "Northgate",
            "provenance_ids": {"summary_ids": [], "parse_ids": []},
        }
    ),
}


def opened() -> threading.Event:
    gate = threading.Event()
    gate.set()
    return gate


@dataclass
class FakeModel:
    """A ``completion=`` that answers r3con's requests as a model would, through
    litellm's own mock, and keeps them. Each request waits at ``gate``; a request for
    the reply named ``fail_on`` raises ``error`` instead of being answered."""

    gate: threading.Event = field(default_factory=opened)
    fail_on: str | None = None
    error: Exception = field(default_factory=lambda: RuntimeError("the model is down"))
    requests: list[dict[str, Any]] = field(default_factory=list)
    called: threading.Event = field(default_factory=threading.Event)

    def __call__(self, **request: Any) -> Any:
        self.requests.append(request)
        self.called.set()
        self.gate.wait(GATE_TIMEOUT_SECONDS)
        reply = request.get("response_format", {}).get("json_schema", {})
        name = reply.get("name", "schema")
        if name == self.fail_on:
            raise self.error
        choice = {
            "index": 0,
            "finish_reason": "stop",
            "message": {"content": REPLIES[name]},
        }
        return litellm.completion(
            **request, mock_response={"choices": [choice], "usage": USAGE}
        )


@pytest.fixture
def llm() -> FakeModel:
    return FakeModel()


@pytest.fixture
def held_llm() -> FakeModel:
    """A fake model that holds every request until its gate opens."""
    model = FakeModel()
    model.gate.clear()
    return model


@pytest.fixture
def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / "runs"


def client_for(model: FakeModel, logs_dir: Path, **options: Any) -> TestClient:
    app = create_app(
        token=TOKEN, port=PORT, logs_dir=logs_dir, completion=model, **options
    )
    return TestClient(app, base_url=BASE_URL, headers=AUTH)


@pytest.fixture
def client(llm: FakeModel, logs_dir: Path) -> Iterator[TestClient]:
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


def transport(llm: FakeModel) -> list[dict[str, str]]:
    """The key and endpoint each request was sent with."""
    return [
        {name: call[name] for name in ("api_key", "api_base") if name in call}
        for call in llm.requests
    ]


def http_client(url: str) -> httpx2.Client:
    """A client for the server that ``url``, with its token, opens."""
    base_url, token = url.split("/#token=")
    return httpx2.Client(
        base_url=base_url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=GATE_TIMEOUT_SECONDS,
        trust_env=False,
    )


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
    "authorization",
    [None, "Bearer wrong-token", "the-launch-token", "Basic the-launch-token"],
)
def test_an_api_call_without_this_launchs_token_is_refused(
    client: TestClient, authorization: str | None
):
    del client.headers["Authorization"]
    headers = {} if authorization is None else {"Authorization": authorization}
    response = client.get("/api/keys", headers=headers)
    assert response.status_code == 401
    assert "token" in response.json()["error"]


def test_an_api_call_with_this_launchs_token_is_answered(client: TestClient):
    assert client.get("/api/keys").status_code == 200


@pytest.mark.parametrize("path", ["/", "/api/keys"])
@pytest.mark.parametrize(
    "host", ["evil.example", f"evil.example:{PORT}", f"127.0.0.1:{PORT + 1}"]
)
def test_a_request_that_names_another_host_is_refused(
    client: TestClient, path: str, host: str
):
    assert client.get(path, headers={"Host": host}).status_code == 403


@pytest.mark.parametrize(
    "origin", ["http://evil.example", "null", f"http://127.0.0.1:{PORT + 1}"]
)
def test_a_request_from_another_origin_is_refused(client: TestClient, origin: str):
    response = client.post("/api/runs", headers={"Origin": origin}, json={})
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
        (STORED_OPENAI, "gpt-6-luna", {"api_key": OPENAI_KEY}),
        (STORED_OPENAI, "anthropic/claude-sonnet-5-5", {}),
        (OTHER, "hosted_vllm/qwen", {"api_key": OTHER_KEY, "api_base": ENDPOINT}),
        (KEYLESS_OTHER, "hosted_vllm/qwen", {"api_base": ENDPOINT}),
        (OTHER, "gpt-6-luna", {}),
        (OTHER, "claude-sonnet-5-5", {}),
        (OTHER, "gemini-2.5-pro", {}),
        (OTHER, "azure/x", {}),
        (OTHER, "ollama_chat/llama3", {"api_key": OTHER_KEY, "api_base": ENDPOINT}),
    ],
)
def test_a_stored_key_reaches_every_call_for_its_providers_model(
    client: TestClient,
    llm: FakeModel,
    stored: dict[str, str],
    model_name: str,
    sent: dict[str, str],
):
    client.post("/api/keys", json=stored)
    finished_run(client, model_name)
    assert transport(llm) == [sent] * len(llm.requests)


def test_a_name_litellm_cannot_place_is_sent_to_other_which_litellm_then_refuses(
    client: TestClient, llm: FakeModel
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
    client: TestClient, llm: FakeModel
):
    client.post("/api/keys", json={"provider": "openai", "key": f"{OPENAI_KEY}\n"})
    client.post("/api/keys", json={**OTHER, "endpoint": f" {ENDPOINT}\n"})
    assert client.get("/api/keys").json()["other"]["endpoint"] == ENDPOINT
    llm.fail_on = "ReasoningAnswer"
    llm.error = RuntimeError(f"Incorrect API key provided: {OPENAI_KEY}.")
    last = run_events(client, start_run(client))[-1]
    assert {call["api_key"] for call in llm.requests} == {OPENAI_KEY}
    assert last["error"] == "RuntimeError: Incorrect API key provided: [key]."
    llm.fail_on = None
    llm.requests.clear()
    finished_run(client, "hosted_vllm/qwen")
    assert {call["api_base"] for call in llm.requests} == {ENDPOINT}


def test_a_placeholder_key_too_short_to_be_a_secret_is_not_masked(
    client: TestClient, llm: FakeModel
):
    client.post("/api/keys", json={**OTHER, "key": "EMPTY"})
    llm.fail_on = "RelevanceSummary"
    llm.error = RuntimeError("The model sent an EMPTY reply.")
    last = run_events(client, start_run(client, "hosted_vllm/qwen"))[-1]
    assert last["error"] == "RuntimeError: The model sent an EMPTY reply."


def test_with_a_keyless_other_a_bare_openai_model_never_reaches_its_endpoint(
    client: TestClient, llm: FakeModel, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("OPENAI_API_KEY", ENVIRONMENT_KEY)
    client.post("/api/keys", json=KEYLESS_OTHER)
    finished_run(client, "gpt-6-luna")
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


def test_a_key_a_provider_quotes_in_its_error_is_masked_in_the_event_and_the_log(
    client: TestClient, llm: FakeModel, caplog: pytest.LogCaptureFixture
):
    client.post("/api/keys", json=STORED_OPENAI)
    llm.fail_on = "RelevanceSummary"
    llm.error = RuntimeError(f"Incorrect API key provided: {OPENAI_KEY}.")
    with caplog.at_level(logging.ERROR, logger="r3con.app"):
        last = run_events(client, start_run(client))[-1]
    assert last["type"] == "run_failed"
    assert last["error"] == "RuntimeError: Incorrect API key provided: [key]."
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


def test_an_ask_over_a_run_still_running_is_refused(
    held_llm: FakeModel, logs_dir: Path
):
    with client_for(held_llm, logs_dir) as client:
        run_id = start_run(client)
        assert held_llm.called.wait(GATE_TIMEOUT_SECONDS)
        response = client.post(f"/api/runs/{run_id}/asks", json=ASK)
        held_llm.gate.set()
        assert run_events(client, run_id)[-1]["type"] == "run_finished"
    assert response.status_code == 409
    assert "still running" in response.json()["error"]


def test_an_ask_over_a_run_that_failed_in_reasoning_is_answered(
    client: TestClient, llm: FakeModel
):
    llm.fail_on = "ReasoningAnswer"
    run_id = start_run(client)
    last = run_events(client, run_id)[-1]
    assert (last["type"], last["stage"]) == ("run_failed", "reasoning")
    llm.fail_on = None
    assert [event["type"] for event in ask_events(client, run_id)] == ASK_EVENTS


def test_an_ask_over_a_run_that_failed_before_parsing_fails_saying_why(
    client: TestClient, llm: FakeModel
):
    llm.fail_on = "RelevanceSummary"
    run_id = start_run(client)
    assert run_events(client, run_id)[-1]["stage"] == "relevance"
    events = ask_events(client, run_id)
    assert [(event["type"], event["stage"]) for event in events] == [
        ("ask_failed", None)
    ]
    assert "never finished parsing" in events[0]["error"]


def test_a_run_r3con_refuses_before_it_starts_ends_its_stream_and_cannot_be_asked(
    client: TestClient, llm: FakeModel, monkeypatch: pytest.MonkeyPatch
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
    client: TestClient, llm: FakeModel, change: dict[str, Any], wrong: str
):
    response = client.post("/api/runs", json={**run_body(), **change})
    assert response.status_code == 422
    assert wrong in response.json()["error"]
    assert llm.requests == []


def test_a_body_that_is_not_json_is_refused(client: TestClient):
    response = client.post("/api/runs", content=b"{not json")
    assert response.status_code == 422
    assert response.json()["error"].startswith("body: Invalid JSON")


def test_a_body_over_the_limit_is_refused(llm: FakeModel, logs_dir: Path):
    body = {"question": QUESTION, "model": MODEL, "documents": ["x" * 1_001]}
    with client_for(llm, logs_dir, max_body_bytes=1_000) as client:
        response = client.post("/api/runs", json=body)
    assert response.status_code == 413
    assert llm.requests == []


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
        ("GET", "/api/asks/nothing/events"),
    ],
)
def test_a_run_or_ask_this_server_did_not_start_is_not_found(
    client: TestClient, method: str, path: str
):
    response = client.request(method, path, json=ASK)
    assert response.status_code == 404
    assert "nothing" in response.json()["error"]


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_serve_streams_a_runs_events_as_they_happen_and_stops(
    held_llm: FakeModel, logs_dir: Path
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
def test_a_run_the_server_stops_under_ends_without_an_unhandled_error(
    held_llm: FakeModel,
    logs_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    unhandled: list[threading.ExceptHookArgs] = []
    monkeypatch.setattr(threading, "excepthook", unhandled.append)
    held_llm.fail_on = "RelevanceSummary"
    url, stop = serve(logs_dir=logs_dir, completion=held_llm)
    with http_client(url) as http:
        assert http.post("/api/runs", json=run_body()).status_code == 202
    assert held_llm.called.wait(GATE_TIMEOUT_SECONDS)
    stop()
    held_llm.gate.set()
    workers = [t for t in threading.enumerate() if t.name.startswith("r3con-app-")]
    assert workers
    for worker in workers:
        worker.join(GATE_TIMEOUT_SECONDS)
    assert not any(worker.is_alive() for worker in workers)
    assert unhandled == []
    assert not [r for r in caplog.records if "on_event raised" in r.getMessage()]
