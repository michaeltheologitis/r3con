import json
import threading
from pathlib import Path
from typing import Any

import httpx2
import pytest
from conftest import GATE_TIMEOUT_SECONDS, PORT, FakeModel, events_in, memos
from r3con_app import server
from r3con_app.server import serve
from starlette.testclient import TestClient

QUESTION = "Which contractor's sites logged the most equipment incidents in Q3?"
MODEL = "openai/gpt-6-luna"
OPENAI_KEY = "sk-stored-openai-key-1234"
OTHER_KEY = "other-endpoint-key-5678"
ENVIRONMENT_KEY = "gemini-key-from-the-environment"
ENDPOINT = "http://127.0.0.1:9/v1"
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
ASK = {"question": "Which site logged the fewest?", "model": MODEL}


def start_run(client: TestClient, model: str = MODEL) -> str:
    body = {"question": QUESTION, "model": model, "documents": memos()}
    response = client.post("/api/runs", json=body)
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def events_of(client: TestClient, path: str) -> list[dict[str, Any]]:
    response = client.get(path)
    assert response.headers["content-type"].startswith("text/event-stream")
    return events_in(response.text)


def finished_run(client: TestClient, model: str = MODEL) -> str:
    run_id = start_run(client, model)
    last = events_of(client, f"/api/runs/{run_id}/events")[-1]
    assert last["type"] == "run_finished", last
    return run_id


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
    other = {"provider": "other", "endpoint": ENDPOINT, "key": OTHER_KEY}
    responses = [
        client.post("/api/keys", json={"provider": "openai", "key": OPENAI_KEY}),
        client.post("/api/keys", json=other),
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
        ({"provider": "openai", "key": OPENAI_KEY}, MODEL, {"api_key": OPENAI_KEY}),
        ({"provider": "openai", "key": OPENAI_KEY}, "anthropic/claude-sonnet-5-5", {}),
        (
            {"provider": "other", "endpoint": ENDPOINT, "key": OTHER_KEY},
            "hosted_vllm/qwen",
            {"api_key": OTHER_KEY, "api_base": ENDPOINT},
        ),
        (
            {"provider": "other", "endpoint": ENDPOINT},
            "hosted_vllm/qwen",
            {"api_base": ENDPOINT},
        ),
        ({"provider": "other", "endpoint": ENDPOINT, "key": OTHER_KEY}, MODEL, {}),
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
    transport = [
        {name: call[name] for name in ("api_key", "api_base") if name in call}
        for call in llm.requests
    ]
    assert transport == [sent] * len(llm.requests)


def test_a_stored_key_is_in_no_response_event_or_file(
    client: TestClient, data_dir: Path
):
    responses = [
        client.post("/api/keys", json={"provider": "openai", "key": OPENAI_KEY})
    ]
    run_id = start_run(client)
    responses.append(client.get(f"/api/runs/{run_id}/events"))
    responses.append(client.post(f"/api/runs/{run_id}/asks", json=ASK))
    responses.append(client.get(f"/api/asks/{responses[-1].json()['ask_id']}/events"))
    responses.append(client.get("/api/keys"))
    assert all(OPENAI_KEY not in response.text for response in responses)
    files = [path for path in data_dir.rglob("*") if path.is_file()]
    assert files
    assert all(OPENAI_KEY not in path.read_text() for path in files)


def test_a_run_streams_its_events_in_order_and_replays_them_to_a_late_reader(
    client: TestClient, data_dir: Path
):
    run_id = start_run(client)
    events = events_of(client, f"/api/runs/{run_id}/events")
    types = [event["type"] for event in events]
    assert list(dict.fromkeys(types)) == RUN_EVENTS
    assert types[-1] == "run_finished"
    assert events_of(client, f"/api/runs/{run_id}/events") == events
    assert Path(events[-1]["run_dir"]).parent == data_dir / "runs"


def test_an_ask_over_a_finished_run_streams_its_own_events(client: TestClient):
    run_id = finished_run(client)
    run_dir = Path(events_of(client, f"/api/runs/{run_id}/events")[-1]["run_dir"])
    response = client.post(f"/api/runs/{run_id}/asks", json=ASK)
    assert response.status_code == 202, response.text
    events = events_of(client, f"/api/asks/{response.json()['ask_id']}/events")
    assert [event["type"] for event in events] == ASK_EVENTS
    assert events[0]["question"] == ASK["question"]
    assert Path(events[-1]["ask_dir"]).parent == run_dir / "asks"


def test_a_run_whose_model_fails_ends_its_stream_and_cannot_be_asked(
    client: TestClient, llm: FakeModel
):
    llm.error = RuntimeError("the model is down")
    run_id = start_run(client)
    last = events_of(client, f"/api/runs/{run_id}/events")[-1]
    assert last["type"] == "run_failed"
    assert last["stage"] == "relevance"
    assert "the model is down" in last["error"]
    response = client.post(f"/api/runs/{run_id}/asks", json=ASK)
    assert response.status_code == 409
    assert "finished run" in response.json()["error"]


def test_a_run_r3con_refuses_before_it_starts_still_ends_its_stream(
    client: TestClient, llm: FakeModel, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("r3con.settings.REASONING_MAX_TURNS", 0)
    run_id = start_run(client)
    events = events_of(client, f"/api/runs/{run_id}/events")
    assert [(event["type"], event["stage"]) for event in events] == [
        ("run_failed", None)
    ]
    assert "REASONING_MAX_TURNS must be >= 1" in events[0]["error"]
    assert llm.requests == []


@pytest.mark.parametrize(
    ("change", "wrong"),
    [
        ({"question": ""}, "question: must not be blank"),
        ({"question": " \n"}, "question: must not be blank"),
        ({"model": None}, "model:"),
        ({"documents": []}, "documents: List should have at least 1 item"),
        ({"documents": [{"name": "a.txt", "text": "  "}]}, "documents.0.text"),
        ({"documents": [{"text": "a memo"}]}, "documents.0.name: Field required"),
        ({"documents": "a memo"}, "documents:"),
        ({"temperature": 0}, "temperature: Extra inputs are not permitted"),
    ],
)
def test_a_run_body_that_is_not_valid_is_refused_with_what_is_wrong(
    client: TestClient, llm: FakeModel, change: dict[str, Any], wrong: str
):
    body = {"question": QUESTION, "model": MODEL, "documents": memos(), **change}
    response = client.post("/api/runs", json=body)
    assert response.status_code == 422
    assert wrong in response.json()["error"]
    assert llm.requests == []


def test_a_body_that_is_not_json_is_refused(client: TestClient):
    response = client.post("/api/runs", content=b"{not json")
    assert response.status_code == 422
    assert response.json()["error"].startswith("body: Invalid JSON")


def test_a_body_over_the_limit_is_refused(
    client: TestClient, llm: FakeModel, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(server, "MAX_BODY_BYTES", 1_000)
    documents = [{"name": "long.txt", "text": "x" * 1_001}]
    body = {"question": QUESTION, "model": MODEL, "documents": documents}
    response = client.post("/api/runs", json=body)
    assert response.status_code == 413
    assert "1,000 bytes" in response.json()["error"]
    assert llm.requests == []


@pytest.mark.parametrize(
    ("body", "wrong"),
    [
        ({"provider": "mistral", "key": OPENAI_KEY}, "does not match any of the"),
        ({"provider": "openai", "key": " "}, "openai.key: must not be blank"),
        ({"provider": "openai"}, "openai.key: Field required"),
        ({"provider": "openai", "key": OPENAI_KEY, "x": 1}, "openai.x: Extra inputs"),
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


def first_event_then_the_rest(
    base_url: str, token: str, gate: threading.Event
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Over a real connection: start a run, read its stream's first event while the
    model is held at ``gate``, then open the gate and read the rest."""
    headers = {"Authorization": f"Bearer {token}"}
    timeout = GATE_TIMEOUT_SECONDS
    with httpx2.Client(
        base_url=base_url, headers=headers, timeout=timeout, trust_env=False
    ) as http:
        assert http.get("/").status_code == 200
        body = {"question": QUESTION, "model": MODEL, "documents": memos()}
        run_id = http.post("/api/runs", json=body).json()["run_id"]
        with http.stream("GET", f"/api/runs/{run_id}/events") as response:
            lines = response.iter_lines()
            first = next(line for line in lines if line.startswith("data: "))
            gate.set()
            return json.loads(first.removeprefix("data: ")), events_in("\n".join(lines))


@pytest.mark.enable_socket
@pytest.mark.allow_hosts(["127.0.0.1"])
def test_serve_streams_a_runs_events_as_they_happen_and_stops(data_dir: Path):
    gate = threading.Event()
    url, stop = serve(data_dir=data_dir, completion=FakeModel(gate=gate))
    base_url, token = url.split("/#token=")
    try:
        first, rest = first_event_then_the_rest(base_url, token, gate)
    finally:
        gate.set()
        stop()
    assert first["type"] == "run_started"
    assert rest[-1]["type"] == "run_finished"
    with pytest.raises(httpx2.ConnectError):
        httpx2.get(base_url, trust_env=False)
