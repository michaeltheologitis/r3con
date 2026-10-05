import importlib.metadata
import json
import threading

import litellm
import pytest
import yaml
from pydantic import BaseModel, Field

from r3con.config import PROMPT_STAGES, RunConfig
from r3con.pipeline import run_pipeline
from r3con.runs import (
    StageRun,
    TaskLogger,
    new_run_folder,
    normalize_model_name,
    write_manifest,
)
from r3con.settings import settings

MODEL = "openai/gpt-6-luna"


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
    run = StageRun(stage="structuring/schema", task_logger=logger, model=MODEL, seed=0)
    assert run.dir == logger.dir / "structuring" / "schema"
    run.add_step(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
        response=reply("a"),
        tokens={"prompt": 5, "completion": 1, "total": 6},
    )
    transcript = flushed_transcript(run)
    assert transcript["stage"] == "structuring/schema"
    assert transcript["model"] == "gpt-6-luna"
    assert transcript["seed"] == 0
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
    ("tokens", "totals"),
    [
        (
            [
                {"prompt": 10, "completion": 2, "total": 12},
                {"prompt": 25, "completion": 6, "total": 31},
            ],
            {"n_steps": 2, "tokens": {"prompt": 35, "completion": 8, "total": 43}},
        ),
        ([None], {"n_steps": 1, "tokens": None}),
        ([], {"n_steps": 0, "tokens": None}),
    ],
)
def test_a_stage_totals_its_steps_and_their_tokens(logger, tokens, totals):
    run = StageRun(stage="reasoning", task_logger=logger)
    for used in tokens:
        run.add_step(messages=[], response=reply(), tokens=used)
    assert run.compute_totals() == totals


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
    run_config = config(model="hosted_vllm/Some/Model-X", seed=7, relevance_rounds=3)
    write_manifest(
        logger, task="what happened?", config=run_config, n_docs=3, context_chars=1234
    )
    manifest = json.loads((logger.dir / "manifest.json").read_text())
    assert manifest["r3con_version"] == importlib.metadata.version("r3context")
    assert manifest["task"] == "what happened?"
    assert manifest["n_docs"] == 3
    assert manifest["context_chars"] == 1234
    assert manifest["run_folder"] == "run"
    assert manifest["config"]["model"] == "Model-X"
    recorded = manifest["config"]
    assert (recorded["seed"], recorded["relevance_rounds"]) == (7, 3)
    assert recorded["prompts"] == dict.fromkeys(PROMPT_STAGES, "v1")
    assert manifest["settings"]["reasoning_max_turns"] == settings.REASONING_MAX_TURNS
    assert manifest["created"].startswith("20")


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
    overlay = tmp_path / "overlay"
    (overlay / "relevance").mkdir(parents=True)
    (overlay / "relevance" / "v1.yaml").write_text("instructions: |-\n  OVERLAID\n")
    monkeypatch.setenv("R3CON_PROMPTS_DIR", str(overlay))
    write_manifest(logger, task="q", config=config(), n_docs=1)
    sources = json.loads((logger.dir / "manifest.json").read_text())["sources"]
    assert sources["config"] == str(settings.CONFIGS_DIR / "default.yaml")
    assert sources["prompts"] == {
        stage: str(
            overlay / "relevance" / "v1.yaml"
            if stage == "relevance"
            else settings.PROMPTS_DIR / stage / "v1.yaml"
        )
        for stage in PROMPT_STAGES
    }
