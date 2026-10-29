import os

# Before litellm is imported: otherwise it fetches its price map from GitHub.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import json
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
import pytest
from r3con_app.server import KEY_VARIABLES, create_app
from starlette.testclient import TestClient

litellm.suppress_debug_info = True

PORT = 8765
TOKEN = "the-launch-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
BASE_URL = f"http://127.0.0.1:{PORT}"
MEMOS = Path(__file__).resolve().parents[2] / "examples" / "memos"
GATE_TIMEOUT_SECONDS = 10
USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
SCHEMA_REPLY = """Thought: one row per site.
<schema>
class Row(BaseModel):
    site: str


class Parse(BaseModel):
    rows: list[Row]
</schema>"""
STRUCTURED_REPLIES = {
    "RelevanceSummary": {"summary": "A site memo.", "provenance_ids": []},
    "ParseReply": {"parse": {"rows": [{"site": "Northgate"}]}, "provenance_ids": []},
    "ReasoningAnswer": {
        "answer": "Northgate",
        "provenance_ids": {"summary_ids": [], "parse_ids": []},
    },
}


@dataclass
class FakeModel:
    """A ``completion=`` that answers r3con's requests as a model would, through
    litellm's own mock, and keeps them. It raises ``error`` instead, when set, and
    waits for ``gate`` first, when given."""

    gate: threading.Event | None = None
    error: Exception | None = None
    requests: list[dict[str, Any]] = field(default_factory=list)

    def __call__(self, **request: Any) -> Any:
        self.requests.append(request)
        if self.gate is not None:
            self.gate.wait(GATE_TIMEOUT_SECONDS)
        if self.error is not None:
            raise self.error
        response_format = request.get("response_format", {})
        schema = response_format.get("json_schema", {}).get("name")
        content = (
            SCHEMA_REPLY if schema is None else json.dumps(STRUCTURED_REPLIES[schema])
        )
        choice = {"index": 0, "finish_reason": "stop", "message": {"content": content}}
        return litellm.completion(
            **request, mock_response={"choices": [choice], "usage": USAGE}
        )


def memos() -> list[dict[str, str]]:
    return [{"name": p.name, "text": p.read_text()} for p in sorted(MEMOS.iterdir())]


def events_in(stream: str) -> list[dict[str, Any]]:
    """The events of a ``text/event-stream`` body, in order."""
    return [
        json.loads(line.removeprefix("data: "))
        for line in stream.splitlines()
        if line.startswith("data: ")
    ]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for variable in KEY_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def llm() -> FakeModel:
    return FakeModel()


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    return tmp_path / "data"


@pytest.fixture
def client(llm: FakeModel, data_dir: Path) -> Iterator[TestClient]:
    app = create_app(token=TOKEN, port=PORT, data_dir=data_dir, completion=llm)
    with TestClient(app, base_url=BASE_URL, headers=AUTH) as client:
        yield client
