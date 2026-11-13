import errno
import importlib.metadata
import json
import threading
from types import MappingProxyType

import litellm
import pytest
import yaml
from pydantic import BaseModel, Field

from r3con import settings
from r3con.config import PROMPT_STAGES, RunConfig
from r3con.masking import masked, register_key
from r3con.pipeline import run_pipeline
from r3con.runs import (
    StageRun,
    TaskLogger,
    call_totals,
    fingerprints,
    new_run_folder,
    normalize_model_name,
    reopen_for_resume,
    run_lock,
    write_manifest,
)
from r3con.runtime.llm import register_credentials
from r3con.settings import settings_snapshot
from tests.fakes import DIGITS_KEY, GatewayKey, on_your_data

MODEL = "openai/gpt-6-luna"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
ERROR_LINE = "RateLimitError: slow down"
TOKENS = {"prompt": 10, "completion": 2, "total": 12}
USAGE = {
    "prompt_tokens": 10,
    "completion_tokens": 2,
    "total_tokens": 12,
    "completion_tokens_details": {"reasoning_tokens": 1},
}
THOUGHT = "Count the memos first."
KEY = "not-a-real-key-q7RkT2mXv9LpA4sW"
PEM = f"-----BEGIN PRIVATE KEY-----\n{KEY}\n-----END PRIVATE KEY-----\n"
SAP_SERVICE_KEY = json.dumps({"clientid": "sb-memo-reader", "clientsecret": KEY})
SERVER = "https://llm.example.com/v1"
# A JSON Schema whose fields' names look like a credential's.
SCHEMA = {
    "type": "object",
    "properties": {
        "author": {"type": "string", "description": "The memo's signed author"},
        "auth_token": {"type": "string", "description": "The session's sign-in token"},
        "token_count": {
            "type": "integer",
            "description": "How many tokens the memo spends",
        },
    },
}
# Texts and the SHA-256 of their UTF-8 bytes, as sha256sum gives it.
FINGERPRINTED = {
    "Halloran memo": "df64b9ac9984a7c355d77857db4f84ea0f8e3e8efdee95e2993a52347c554ea5",
    "Merrow memo · café": "872e3e95b4ce367bc4dae4bd12ffe0dbfb5afd15217e1ba728c1ae1fa906a379",
    # Half a surrogate pair, passed through as its three bytes, ED A0 80.
    "Calder memo, half a pair: \ud800": (
        "387608192a392fe5a3d3253a6c9cc8d2f1b41f1d84e50d10162c56e729a9ad52"
    ),
}
TOOLS = [{"type": "function", "function": {"name": "sign_in", "parameters": SCHEMA}}]
# A field of a schema, which the rows below name like a credential.
FIELD = {"type": "string", "description": "The session's sign-in token"}
DEPENDENT_FIELDS = ["auth_token_issuer"]
SEARCH_KEY = {"type": "api_key", "key": KEY}
MCP_TOOL = {
    "type": "mcp",
    "server_label": "memos",
    "server_url": "https://mcp.example.com/sse",
    "headers": {"Authorization": f"Bearer {KEY}"},
}


# Each run's params, and its manifest's record of them.
RECORDED_PARAMS = {
    "nested-in-extra-body": (
        {"extra_body": {"api_key": KEY, "top_k": 20}},
        {"extra_body": {"api_key": "REDACTED", "top_k": 20}},
    ),
    "headers-whatever-their-name": (
        {"headers": {"x-functions-key": KEY}},
        {"headers": {"x-functions-key": "REDACTED"}},
    ),
    "dict-valued-credential": (
        {"vertex_credentials": {"private_key": PEM, "client_id": "108374628401"}},
        {"vertex_credentials": {"private_key": "REDACTED", "client_id": "REDACTED"}},
    ),
    "oci-key": ({"oci_key": PEM}, {"oci_key": "REDACTED"}),
    "url-password": (
        {"api_base": f"https://reader:{KEY}@llm.example.com/v1"},
        {"api_base": "https://reader:REDACTED@llm.example.com/v1"},
    ),
    "url-key": (
        {"api_base": f"{SERVER}?key={KEY}"},
        {"api_base": f"{SERVER}?key=REDACTED"},
    ),
    "url-user-name": (
        {"api_base": f"https://{KEY}@llm.example.com/v1"},
        {"api_base": "https://REDACTED@llm.example.com/v1"},
    ),
    "url-code-beside-a-public-api-version": (
        {"api_base": f"{SERVER}?api-version=2024-10-21&code={KEY}"},
        {"api_base": f"{SERVER}?api-version=2024-10-21&code=REDACTED"},
    ),
    "an-integer-key": ({"api_key": DIGITS_KEY}, {"api_key": "REDACTED"}),
    "in-a-pydantic-model": (
        {"extra_body": GatewayKey(api_key=KEY)},
        {"extra_body": {"api_key": "REDACTED"}},
    ),
    "schemas-beside-a-key": (
        {"tools": TOOLS, "extra_body": {"api_key": KEY, "guided_json": SCHEMA}},
        {"tools": TOOLS, "extra_body": {"api_key": "REDACTED", "guided_json": SCHEMA}},
    ),
    "azure-on-your-data-search-key": (
        on_your_data({"authentication": SEARCH_KEY}),
        on_your_data({"authentication": {**SEARCH_KEY, "key": "REDACTED"}}),
    ),
    "azure-on-your-data-embedding-key": (
        on_your_data({"embedding_dependency": {"authentication": SEARCH_KEY}}),
        on_your_data(
            {
                "embedding_dependency": {
                    "authentication": {**SEARCH_KEY, "key": "REDACTED"}
                }
            }
        ),
    ),
}
# Credentials litellm reads from within what a request sends, and the run's record.
WITHIN_REQUEST_DATA = {
    "callback-key-in-metadata": (
        {"metadata": {"langfuse_secret_key": KEY}},
        {"metadata": {"langfuse_secret_key": "REDACTED"}},
    ),
    "callback-key-in-litellm-metadata": (
        {"litellm_metadata": {"langsmith_api_key": KEY}},
        {"litellm_metadata": {"langsmith_api_key": "REDACTED"}},
    ),
    "key-in-extra-body-metadata": (
        {"extra_body": {"metadata": {"api_key": KEY}}},
        {"extra_body": {"metadata": {"api_key": "REDACTED"}}},
    ),
    "mcp-tool-header": (
        {"tools": [MCP_TOOL]},
        {"tools": [{**MCP_TOOL, "headers": {"Authorization": "REDACTED"}}]},
    ),
    "header-of-a-tool-in-extra-body": (
        {"extra_body": {"tools": [{"headers": {"X-Key": KEY}}]}},
        {"extra_body": {"tools": [{"headers": {"X-Key": "REDACTED"}}]}},
    ),
}
# A run's environment and params, with KEY where litellm finds a secret within one.
SECRETS_WITHIN = {
    "sap-service-keys-client-secret": ({"AICORE_SERVICE_KEY": SAP_SERVICE_KEY}, {}),
    "sap-private-keys-line": ({"AICORE_KEY_STR": PEM}, {}),
    "first-hosts-password": (
        {},
        {"api_base": f"postgres://u:{KEY}@db1.example.com,v:pw-0123456789@db2/memos"},
    ),
    "second-hosts-password": (
        {},
        {"api_base": f"postgres://u:pw-0123456789@db1.example.com,v:{KEY}@db2/memos"},
    ),
}
# Params with no credential, though names in them look like one's.
CREDENTIAL_FREE_PARAMS = {
    "metadata-author": {"metadata": {"author": "Halloran memo desk"}},
    "tool-schema": {"tools": TOOLS},
    "structured-outputs-schema": {
        "extra_body": {"structured_outputs": {"json": SCHEMA}}
    },
    "response-format-schema": {
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "memo", "schema": SCHEMA},
        }
    },
    "openrouter-headers": {
        "extra_headers": {
            "HTTP-Referer": "https://memos.example.com",
            "X-Title": "Halloran memo desk",
        }
    },
    "definitions-schema": {
        "extra_body": {"guided_json": {"definitions": {"auth_token": FIELD}}}
    },
    "pattern-properties-schema": {
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "memo",
                "schema": {
                    "type": "object",
                    "patternProperties": {"^token_.*$": FIELD},
                },
            },
        }
    },
    "dependent-schemas-schema": {
        "extra_body": {
            "guided_json": {
                "dependentSchemas": {"auth_token": {"required": DEPENDENT_FIELDS}}
            }
        }
    },
    "dependent-required-schema": {
        "extra_body": {
            "guided_json": {"dependentRequired": {"auth_token": DEPENDENT_FIELDS}}
        }
    },
    "anthropic-headers": {
        "extra_headers": {
            "anthropic-beta": "context-1m-2025-08-07",
            "anthropic-version": "2023-06-01",
        }
    },
    "http-headers": {
        "extra_headers": {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "memo-desk/2026.11",
        }
    },
    "google-quota-project-header": {
        "extra_headers": {"x-goog-user-project": "memo-reader-2026"}
    },
    "azure-ml-deployment-header": {
        "extra_headers": {"azureml-model-deployment": "memo-reader-blue"}
    },
    "openai-headers": {
        "extra_headers": {
            "OpenAI-Organization": "org-memo-desk-2026",
            "OpenAI-Project": "proj_memoReader2026",
            "OpenAI-Beta": "assistants=v2",
        }
    },
    "api-version-in-any-case": {"api_base": f"{SERVER}?API-Version=2024-10-21"},
    "watsonx-version": {"api_base": f"{SERVER}?version=2024-05-31"},
    "prompt-cache-key": {"prompt_cache_key": "summaries-cache-2026"},
}


class Sample(BaseModel):
    name: str
    count: int = Field(description="some count")


def reply(content: str = "ok", finish: str = "stop") -> dict:
    return {"role": "assistant", "content": content, "finish_reason": finish}


def config(**fields) -> RunConfig:
    fields = {"model": MODEL, "prompts": dict.fromkeys(PROMPT_STAGES, "v1"), **fields}
    return RunConfig(**fields)


@pytest.fixture
def logger(tmp_path) -> TaskLogger:
    return TaskLogger("run", root=tmp_path)


def flushed_transcript(run: StageRun) -> dict:
    return yaml.safe_load(run.flush().read_text(encoding="utf-8"))


def test_a_task_logger_writes_into_one_flat_folder(tmp_path):
    logger = TaskLogger("gpt__r3__abc123", task_id="t1", root=tmp_path)
    assert logger.dir == tmp_path / "gpt__r3__abc123"
    assert logger.dir.is_dir()
    assert (logger.folder, logger.task_id) == ("gpt__r3__abc123", "t1")
    assert TaskLogger("solo", root=tmp_path).task_id == "solo"


def test_a_new_run_folder_is_a_unique_utc_timestamp_and_a_hex_suffix():
    first, second = new_run_folder(), new_run_folder()
    assert first != second
    stamp, _, suffix = first.partition("_")
    assert len(stamp) == len("20260613T142233Z")
    assert stamp.endswith("Z")
    assert len(suffix) == 8
    assert set(suffix) <= set("0123456789abcdef")


def test_text_and_json_are_written_under_their_names_and_overwritten(logger):
    logger.write_text("answer", "first")
    logger.write_text("answer", "final\n")
    logger.write_json("score", {"score": 1, "gold": "yes"})
    assert (logger.dir / "answer.txt").read_text() == "final\n"
    assert json.loads((logger.dir / "score.json").read_text()) == {
        "score": 1,
        "gold": "yes",
    }


def test_pydantic_values_are_written_as_data_and_classes_as_their_schema(logger):
    logger.write_json("parse", {"attempt": 1, "result": Sample(name="x", count=3)})
    logger.write_json("schema", Sample)
    parse = json.loads((logger.dir / "parse.json").read_text())
    assert parse == {"attempt": 1, "result": {"name": "x", "count": 3}}
    assert "properties" in json.loads((logger.dir / "schema.json").read_text())


def test_a_value_json_cannot_hold_is_written_as_its_repr(logger):
    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque>"

    logger.write_json("oddity", {"obj": Opaque()})
    assert json.loads((logger.dir / "oddity.json").read_text()) == {"obj": "<Opaque>"}


def test_non_ascii_is_written_as_text_not_escapes(logger):
    path = logger.write_json("r", {"label": "应付账款"})
    raw = path.read_text(encoding="utf-8")
    assert "应付账款" in raw
    assert "\\u" not in raw
    assert json.loads(raw) == {"label": "应付账款"}


def test_a_nested_name_creates_its_folders(logger):
    path = logger.write_json("reasoning/result", {"answer": "x"})
    assert path == logger.dir / "reasoning" / "result.json"
    path = logger.write_yaml("structuring/schema/transcript", {"messages": []})
    assert path == logger.dir / "structuring" / "schema" / "transcript.yaml"


def test_a_stage_run_flushes_its_calls_and_its_transcript_into_its_folder(logger):
    run = StageRun(stage="structuring/schema", task_logger=logger, model=MODEL)
    assert run.dir == logger.dir / "structuring" / "schema"
    run.add_step(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        response=reply("a"),
        tokens={"prompt": 5, "completion": 1, "total": 6},
    )
    transcript = flushed_transcript(run)
    assert transcript["stage"] == "structuring/schema"
    assert transcript["model"] == "gpt-6-luna"
    assert "seed" not in transcript
    assert [m["content"] for m in transcript["messages"]] == ["s", "u", "a"]
    assert (run.dir / "calls.json").is_file()


def test_each_call_records_the_document_its_kind_names(logger):
    run = StageRun(stage="structuring/parsing", task_logger=logger)
    run.add_step(kind="relevance-r1-d2", messages=[], response=reply("a"))
    run.add_step(kind="parse-d0", messages=[], response=reply("b", finish="length"))
    run.add_step(kind="llm_call", messages=[], response=reply("c"))
    assert run.flush(write_transcript=False) == run.dir / "calls.json"
    calls = json.loads((run.dir / "calls.json").read_text())
    assert [(c["kind"], c["doc"], c["chunk"]) for c in calls] == [
        ("relevance-r1-d2", 2, None),
        ("parse-d0", 0, None),
        ("llm_call", None, None),
    ]
    assert [c["output"] for c in calls] == ["a", "b", "c"]
    assert calls[1]["finish_reason"] == "length"
    assert not (run.dir / "transcript.yaml").exists()


def test_calls_keep_non_ascii_as_text(logger):
    run = StageRun(stage="structuring/parsing", task_logger=logger)
    run.add_step(
        messages=[{"role": "user", "content": "文档"}], response=reply("应付账款")
    )
    run.flush(write_transcript=False)
    raw = (run.dir / "calls.json").read_text(encoding="utf-8")
    assert "应付账款" in raw
    assert "文档" in raw
    assert "\\u" not in raw


@pytest.mark.parametrize(
    ("calls", "totals"),
    [
        (
            [
                (TOKENS, 0.25, None),
                ({"prompt": 25, "completion": 6, "total": 31}, 0.5, None),
            ],
            {
                "n_steps": 2,
                "tokens": {"prompt": 35, "completion": 8, "total": 43},
                "usd": 0.75,
            },
        ),
        (
            [(TOKENS, 0.25, None), (None, None, None)],
            {
                "n_steps": 2,
                "tokens": TOKENS,
                "usd": None,
            },
        ),
        (
            [(TOKENS, 0.25, None), (None, 0.0, ERROR_LINE)],
            {"n_steps": 1, "tokens": TOKENS, "usd": 0.25},
        ),
        ([], {"n_steps": 0, "tokens": None, "usd": 0.0}),
    ],
    ids=["priced", "one-unpriced", "one-failed", "none"],
)
def test_a_stage_totals_its_steps_tokens_and_cost(logger, calls, totals):
    run = StageRun(stage="reasoning", task_logger=logger)
    for tokens, usd, error in calls:
        run.add_step(messages=[], response=reply(), tokens=tokens, usd=usd, error=error)
    assert run.compute_totals() == totals
    assert type(run.compute_totals()["usd"]) is type(totals["usd"])


def test_each_call_records_what_it_cost(logger):
    run = StageRun(stage="relevance", task_logger=logger)
    run.add_step(messages=[], response=reply(), usd=0.25)
    run.add_step(messages=[], response=reply())
    run.flush(write_transcript=False)
    calls = json.loads((run.dir / "calls.json").read_text())
    assert [call["usd"] for call in calls] == [0.25, None]


def test_each_call_records_how_often_it_was_sent_and_how_long_it_waited(logger):
    run = StageRun(stage="relevance", task_logger=logger)
    run.add_step(messages=[], response=reply(), sends=3, resend_wait_seconds=2.5)
    run.add_step(messages=[], response=reply())
    run.flush(write_transcript=False)
    calls = json.loads((run.dir / "calls.json").read_text())
    recorded = [(call["sends"], call["resend_wait_seconds"]) for call in calls]
    assert recorded == [(3, 2.5), (None, None)]


def test_each_call_records_its_reply_and_how_long_it_took(logger):
    run = StageRun(stage="relevance", task_logger=logger)
    run.add_step(
        messages=[],
        response={**reply("a"), "reasoning": THOUGHT},
        tokens=TOKENS,
        usage=USAGE,
        served_model="gpt-6-luna",
        fingerprint="fp_3f1c2d",
        seconds=1.25,
    )
    run.flush(write_transcript=False)

    [call] = json.loads((run.dir / "calls.json").read_text())
    assert call["usage"] == USAGE
    assert (call["served_model"], call["fingerprint"]) == ("gpt-6-luna", "fp_3f1c2d")
    assert (call["reasoning"], call["output"], call["seconds"]) == (THOUGHT, "a", 1.25)


def test_a_resumed_stage_run_writes_its_earlier_calls_back_first_and_counts_them(
    logger,
):
    earlier = StageRun(stage="reasoning", task_logger=logger, model=MODEL)
    sent = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    earlier.add_step(
        kind="turn-1",
        messages=sent,
        response={**reply("A"), "reasoning": THOUGHT},
        tokens=TOKENS,
        usage=USAGE,
        usd=0.5,
        seconds=3.25,
    )
    earlier.add_step(
        kind="turn-2", messages=sent, response={}, usd=0.0, error=ERROR_LINE
    )
    earlier.flush()
    calls = json.loads((earlier.dir / "calls.json").read_text())

    resumed = StageRun(stage="reasoning", task_logger=logger, model=MODEL, resumed=True)
    assert resumed.add_step(kind="turn-2", messages=sent, response=reply("B")).step == 3
    resumed.flush()
    rewritten = json.loads((earlier.dir / "calls.json").read_text())
    assert rewritten[:2] == calls
    assert [call["output"] for call in rewritten[2:]] == ["B"]
    steps = [*earlier.steps, *resumed.steps]
    assert resumed.compute_totals() == call_totals(map(vars, steps))
    assert flushed_transcript(resumed)["messages"][-1]["content"] == "B"
    assert (
        StageRun(stage="reasoning", task_logger=logger).compute_totals()["n_steps"] == 0
    )


def test_a_resumed_stage_run_starts_afresh_from_a_calls_json_it_cannot_read(
    logger, caplog
):
    folder = logger.dir / "relevance"
    folder.mkdir()
    (folder / "calls.json").write_text('[{"step": 1, "kind": "relev')
    assert StageRun(stage="relevance", task_logger=logger, resumed=True).steps == []
    assert "calls.json can't be read" in caplog.text


def test_a_json_file_is_written_whole_or_left_as_it_was(logger, monkeypatch):
    logger.write_json("relevance/result", {"rounds": [1]})

    def killed(source, target):
        raise KeyboardInterrupt

    monkeypatch.setattr("r3con.runs.os.replace", killed)
    with pytest.raises(KeyboardInterrupt):
        logger.write_json("relevance/result", {"rounds": [1, 2]})
    folder = logger.dir / "relevance"
    assert json.loads((folder / "result.json").read_text()) == {"rounds": [1]}
    assert [path.name for path in folder.iterdir()] == ["result.json"]


@pytest.mark.parametrize(
    "error",
    [NotImplementedError("flock"), OSError(errno.ENOSYS, "Function not implemented")],
    ids=["filelock-without-a-fallback", "the-file-system"],
)
def test_a_folder_on_a_file_system_without_locks_is_written_unlocked_with_a_warning(
    logger, monkeypatch, caplog, error
):
    class Unlockable:
        def __init__(self, path, timeout):
            pass

        def acquire(self):
            raise error

    monkeypatch.setattr("r3con.runs.FileLock", Unlockable)
    with run_lock(logger.dir):
        logger.write_json("manifest", {})
    assert "can't be locked" in caplog.text
    assert (logger.dir / "manifest.json").is_file()


def test_a_run_folder_is_held_by_one_run_or_resume_at_a_time(logger):
    held = pytest.raises(RuntimeError, match="a run or resume is going in it;")
    with run_lock(logger.dir), held, run_lock(logger.dir):
        pass
    with run_lock(logger.dir):
        assert (logger.dir / "run.lock").is_file()


def test_reopening_a_folder_for_a_resume_records_it_and_ends_a_line_cut_short(logger):
    write_manifest(logger, task="Who?", config=config(), n_docs=1)
    (logger.dir / "events.jsonl").write_text('{"type": "run_started"}\n{"type": "rel')
    reopen_for_resume(logger, settings_snapshot())
    reopen_for_resume(logger, {**settings_snapshot(), "doc_workers": 2})
    manifest = json.loads((logger.dir / "manifest.json").read_text())
    version = importlib.metadata.version("r3context")
    assert [
        (resume["r3con_version"], resume["settings"]["doc_workers"])
        for resume in manifest["resumes"]
    ] == [(version, settings_snapshot()["doc_workers"]), (version, 2)]
    assert (logger.dir / "events.jsonl").read_text().endswith('{"type": "rel\n')


def test_a_transcripts_last_reply_shows_its_reasoning(logger):
    run = StageRun(stage="reasoning", task_logger=logger)
    run.add_step(
        messages=[{"role": "user", "content": "u"}],
        response={**reply("a"), "reasoning": THOUGHT},
    )
    assert flushed_transcript(run)["messages"][-1]["reasoning"] == THOUGHT


def test_each_call_records_its_error_and_one_that_raised_no_output(logger):
    run = StageRun(stage="relevance", task_logger=logger)
    run.add_step(messages=[], response=reply("a"))
    run.add_step(messages=[], response={}, usd=0.0, error=ERROR_LINE)
    run.flush(write_transcript=False)
    calls = json.loads((run.dir / "calls.json").read_text())
    recorded = [
        (call["error"], call["finish_reason"], call["output"]) for call in calls
    ]
    assert recorded == [(None, "stop", "a"), (ERROR_LINE, None, None)]


def test_a_transcript_ending_in_a_call_that_raised_shows_what_it_sent(logger):
    run = StageRun(stage="reasoning", task_logger=logger)
    first = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}]
    run.add_step(kind="turn-1", messages=first, response=reply("A1"))
    second = [*first, {"role": "assistant", "content": "A1"}]
    second.append({"role": "user", "content": "<observation>obs</observation>"})
    run.add_step(kind="turn-2", messages=second, response={}, usd=0.0, error=ERROR_LINE)
    contents = [m["content"] for m in flushed_transcript(run)["messages"]]
    assert contents == ["S", "U1", "A1", "<observation>obs</observation>"]


def test_a_multi_turn_transcript_is_the_last_turns_whole_thread(logger):
    run = StageRun(stage="reasoning", task_logger=logger)
    first = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}]
    run.add_step(kind="turn-1", messages=first, response=reply("A1"))
    second = [*first, {"role": "assistant", "content": "A1"}]
    second.append({"role": "user", "content": "<observation>obs</observation>"})
    run.add_step(kind="turn-2", messages=second, response=reply("A2 final"))
    contents = [m["content"] for m in flushed_transcript(run)["messages"]]
    assert contents == ["S", "U1", "A1", "<observation>obs</observation>", "A2 final"]


def test_a_retried_stages_transcript_is_its_final_attempt(logger):
    run = StageRun(stage="structuring/schema", task_logger=logger)
    system = {"role": "system", "content": "S"}
    run.add_step(
        messages=[system, {"role": "user", "content": "first"}], response=reply("a1")
    )
    run.add_step(
        messages=[system, {"role": "user", "content": "retry"}], response=reply("a2")
    )
    contents = [m["content"] for m in flushed_transcript(run)["messages"]]
    assert contents == ["S", "retry", "a2"]
    calls = json.loads((run.dir / "calls.json").read_text())
    assert [c["output"] for c in calls] == ["a1", "a2"]


def test_a_stage_with_no_steps_flushes_an_empty_transcript(logger):
    run = StageRun(stage="structuring/schema", task_logger=logger)
    assert flushed_transcript(run)["messages"] == []


def test_multiline_text_is_written_as_a_readable_block(logger):
    run = StageRun(stage="reasoning", task_logger=logger)
    run.add_step(messages=[], response=reply("1. Thorir  \n2. Gudrun  \n3. Amleth"))
    path = run.flush()
    raw = path.read_text(encoding="utf-8")
    assert "content: |" in raw
    assert "\\n" not in raw
    assert (
        yaml.safe_load(raw)["messages"][0]["content"]
        == "1. Thorir\n2. Gudrun\n3. Amleth"
    )


def test_steps_added_from_many_threads_are_all_kept_and_numbered_once(logger):
    run = StageRun(stage="relevance", task_logger=logger)

    def add(i: int) -> None:
        run.add_step(kind=f"relevance-r1-d{i}", messages=[], response=reply())

    threads = [threading.Thread(target=add, args=(i,)) for i in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(step.step for step in run.steps) == list(range(1, 51))


@pytest.mark.parametrize(
    ("model", "name"),
    [
        ("openai/gpt-6-luna", "gpt-6-luna"),
        ("hosted_vllm/Qwen/Qwen3-8B", "Qwen3-8B"),
        ("gpt-6-luna", "gpt-6-luna"),
        (None, None),
    ],
)
def test_a_model_is_recorded_without_its_provider_prefix(model, name):
    assert normalize_model_name(model) == name


def test_the_manifest_records_the_identity_and_the_installed_version(logger):
    run_config = config(model="hosted_vllm/Some/Model-X", relevance_rounds=3)
    write_manifest(
        logger, task="what happened?", config=run_config, n_docs=3, context_chars=1234
    )
    manifest = json.loads((logger.dir / "manifest.json").read_text())
    assert manifest["r3con_version"] == importlib.metadata.version("r3context")
    assert manifest["litellm_version"] == importlib.metadata.version("litellm")
    assert manifest["task"] == "what happened?"
    assert manifest["n_docs"] == 3
    assert manifest["context_chars"] == 1234
    assert manifest["run_folder"] == "run"
    assert manifest["config"]["model"] == "Model-X"
    recorded = manifest["config"]
    assert recorded["relevance_rounds"] == 3
    assert "seed" not in recorded
    assert recorded["prompts"] == dict.fromkeys(PROMPT_STAGES, "v1")
    assert manifest["settings"]["reasoning_max_turns"] == settings.REASONING_MAX_TURNS
    assert manifest["created"].startswith("20")


def test_the_manifest_keeps_the_model_string_litellm_is_called_with(logger):
    write_manifest(logger, task="q", config=config(model=QWEN), n_docs=1)
    manifest = json.loads((logger.dir / "manifest.json").read_text())
    assert manifest["litellm_model"] == QWEN
    assert manifest["config"]["model"] == "Qwen3.5-35B-A3B"


@pytest.mark.parametrize(
    "key",
    ["not-a-real-key-q7RkT2mXv9LpA4sW", 'not-a-"real"-key\\q7RkT2mXv9'],
    ids=["plain", "json-escapes-it"],
)
def test_the_manifest_masks_a_registered_key_among_the_runs_params(logger, key):
    register_key(key)
    params = {"extra_headers": {"x-gateway-auth": key}, "temperature": 0.3}
    write_manifest(logger, task="q", config=config(params=params), n_docs=1)
    recorded = json.loads((logger.dir / "manifest.json").read_text())["config"]
    assert recorded["params"] == {
        "extra_headers": {"x-gateway-auth": "REDACTED"},
        "temperature": 0.3,
    }


def refused_with(llm, logger, params, error: Exception) -> None:
    """Run with ``params`` until relevance's first request raises ``error``."""
    with pytest.raises(type(error)):
        run_pipeline(
            task="q",
            documents=["d"],
            config=config(params=params),
            task_logger=logger,
            completion=llm.answers(relevance=error),
        )


@pytest.mark.parametrize(
    ("params", "recorded"), RECORDED_PARAMS.values(), ids=RECORDED_PARAMS
)
def test_a_run_records_each_credential_among_its_params_masked(
    llm, logger, params, recorded
):
    refused = litellm.AuthenticationError("refused", llm_provider="openai", model=MODEL)
    refused_with(llm, logger, params, refused)
    manifest = json.loads((logger.dir / "manifest.json").read_text())
    assert manifest["config"]["params"] == recorded


@pytest.mark.parametrize(
    ("params", "recorded"), WITHIN_REQUEST_DATA.values(), ids=WITHIN_REQUEST_DATA
)
def test_a_credential_within_the_requests_data_is_registered_and_recorded_masked(
    llm, logger, params, recorded
):
    echoed = litellm.AuthenticationError(
        f"refused {KEY}", llm_provider="openai", model=MODEL
    )
    refused_with(llm, logger, params, echoed)
    written = [path.read_text() for path in logger.dir.rglob("*") if path.is_file()]
    assert not [text for text in written if KEY in text]
    manifest = json.loads((logger.dir / "manifest.json").read_text())
    assert manifest["config"]["params"] == recorded


@pytest.mark.parametrize(
    "params", CREDENTIAL_FREE_PARAMS.values(), ids=CREDENTIAL_FREE_PARAMS
)
def test_params_with_no_credential_are_not_registered_and_recorded_as_they_are(
    llm, logger, params
):
    refused = litellm.AuthenticationError("refused", llm_provider="openai", model=MODEL)
    refused_with(llm, logger, params, refused)
    assert masked(json.dumps(params)) == json.dumps(params)
    manifest = json.loads((logger.dir / "manifest.json").read_text())
    assert manifest["config"]["params"] == params


def test_the_manifest_records_a_mapping_among_the_params_as_a_dict_masked(logger):
    params = {"extra_headers": MappingProxyType({"Authorization": f"Bearer {KEY}"})}
    register_credentials(params)
    write_manifest(logger, task="q", config=config(params=params), n_docs=1)
    recorded = json.loads((logger.dir / "manifest.json").read_text())["config"]
    assert recorded["params"] == {"extra_headers": {"Authorization": "REDACTED"}}


def test_a_provider_error_that_echoes_the_servers_password_is_written_masked(
    llm, logger, events_in
):
    server = f"https://reader:{KEY}@llm.example.com/v1"
    down = litellm.APIConnectionError(
        f"can't reach {server}", llm_provider="openai", model=MODEL
    )
    with pytest.raises(litellm.APIConnectionError):
        run_pipeline(
            task="q",
            documents=["d"],
            config=config(),
            task_logger=logger,
            api_base=server,
            completion=llm.answers(relevance=down),
        )
    written = [path.read_text() for path in logger.dir.rglob("*") if path.is_file()]
    assert not [text for text in written if KEY in text]
    [failed] = [e for e in events_in(logger.dir) if e["type"] == "run_failed"]
    assert "can't reach https://reader:REDACTED@llm.example.com/v1" in failed["error"]


@pytest.mark.parametrize(
    ("environment", "params"), SECRETS_WITHIN.values(), ids=SECRETS_WITHIN
)
def test_a_secret_within_a_credential_is_masked_in_every_file_when_an_error_echoes_it(
    llm, logger, monkeypatch, environment, params
):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    echoed = litellm.AuthenticationError(
        f"refused {KEY}", llm_provider="openai", model=MODEL
    )
    refused_with(llm, logger, params, echoed)
    written = [path.read_text() for path in logger.dir.rglob("*") if path.is_file()]
    assert not [text for text in written if KEY in text]
    assert "refused REDACTED" in (logger.dir / "relevance" / "error.txt").read_text()


def test_the_manifest_records_the_settings_it_is_given(logger):
    caps = {**settings_snapshot(), "reasoning_max_turns": 3}
    write_manifest(logger, task="q", config=config(), n_docs=1, settings=caps)
    assert json.loads((logger.dir / "manifest.json").read_text())["settings"] == caps
    write_manifest(logger, task="q", config=config(), n_docs=1)
    recorded = json.loads((logger.dir / "manifest.json").read_text())["settings"]
    assert recorded == settings_snapshot()


def test_a_fingerprint_is_the_sha256_of_a_texts_utf8_bytes_by_its_id_in_order():
    ids = ["c3", "a1", "b2"]
    fingerprinted = fingerprints(ids, list(FINGERPRINTED))
    assert list(fingerprinted.items()) == list(zip(ids, FINGERPRINTED.values()))


def test_the_manifest_is_written_before_the_first_stage_fails(llm, logger):
    down = litellm.APIConnectionError(
        "provider down", llm_provider="openai", model=MODEL
    )
    with pytest.raises(litellm.APIConnectionError, match="provider down"):
        run_pipeline(
            task="q",
            documents=["d"],
            config=config(),
            task_logger=logger,
            completion=llm.answers(relevance=down),
        )
    assert json.loads((logger.dir / "manifest.json").read_text())["task"] == "q"


def test_the_manifest_names_the_file_each_prompt_and_the_config_came_from(
    logger, tmp_path, monkeypatch
):
    overlaid = tmp_path / "overlay" / "relevance" / "later-rounds" / "v1.yaml"
    overlaid.parent.mkdir(parents=True)
    overlaid.write_text("instructions: |-\n  OVERLAID\n")
    monkeypatch.setenv("R3CON_PROMPTS_DIR", str(tmp_path / "overlay"))
    write_manifest(logger, task="q", config=config(), n_docs=1)
    sources = json.loads((logger.dir / "manifest.json").read_text())["sources"]
    assert sources["config"] == str(settings.CONFIGS_DIR / "default.yaml")
    assert sources["prompts"] == {
        stage: str(
            overlaid
            if stage == "relevance/later-rounds"
            else settings.PROMPTS_DIR / stage / "v1.yaml"
        )
        for stage in PROMPT_STAGES
    }
