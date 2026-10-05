import json
import re
from collections import Counter

import litellm
import pytest
import yaml

from r3con import settings
from r3con.config import PROMPT_STAGES, RunConfig
from r3con.pipeline import run_pipeline
from r3con.runs import TaskLogger
from r3con.stages.structuring.parsing import SchemaError

MODEL = "openai/gpt-6-luna"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
DOCS = ["Halloran memo", "Merrow memo"]
LATER_STAGES = ("schema", "parsing", "reasoning")
NEVER_COMMITS = "<code>\nprint(1)\n</code>"
STAGE_FOLDERS = {
    "relevance": "relevance",
    "schema": "structuring/schema",
    "parsing": "structuring/parsing",
    "reasoning": "reasoning",
}
CONFIG = {
    "name": "test",
    "model": MODEL,
    "relevance_rounds": 2,
    "prompts": dict.fromkeys(PROMPT_STAGES, "v1"),
}


def answer(
    llm, task_logger=None, *, documents=DOCS, max_reasoning_turns=None, **fields
):
    """A whole run over ``documents`` on CONFIG with ``fields`` changed, ``llm``
    answering."""
    return run_pipeline(
        task="Who?",
        documents=documents,
        config=RunConfig(**{**CONFIG, **fields}),
        max_reasoning_turns=max_reasoning_turns,
        task_logger=task_logger,
        completion=llm,
    )


def provider_down() -> litellm.APIConnectionError:
    return litellm.APIConnectionError(
        "provider down", llm_provider="openai", model=MODEL
    )


def read_json(logger, name: str):
    return json.loads((logger.dir / f"{name}.json").read_text())


def notes_by_round():
    reads = Counter()

    def note(request) -> str:
        document = request["messages"][1]["content"]
        reads[document] += 1
        return f"Round {reads[document]} notes on {document}."

    return note


def system_prompts(llm, stage) -> list[str]:
    return [request["messages"][0]["content"] for request in llm.requests_for(stage)]


def test_a_run_answers_and_leaves_every_stages_artifacts(answering_llm, tmp_path):
    logger = TaskLogger("run", root=tmp_path)
    result = answer(answering_llm, logger)
    assert str(result) == result.answer == "Halloran memo; Merrow memo"
    assert result.relevant_context == [
        "Notes on Halloran memo.",
        "Notes on Merrow memo.",
    ]
    assert result.structured_context == {
        "rows": [
            {"document": 1, "who": "Halloran memo"},
            {"document": 2, "who": "Merrow memo"},
        ]
    }
    assert "class Parse(BaseModel)" in result.schema_code
    assert result.source_docs == {"rows": [0, 1]}
    assert result.run_dir == logger.dir
    written = {str(p.relative_to(logger.dir)) for p in logger.dir.rglob("*.*")}
    assert written == {
        "manifest.json",
        "relevance/calls.json",
        "relevance/result.json",
        "structuring/schema/calls.json",
        "structuring/schema/result.json",
        "structuring/schema/transcript.yaml",
        "structuring/parsing/calls.json",
        "structuring/parsing/result.json",
        "reasoning/calls.json",
        "reasoning/result.json",
        "reasoning/transcript.yaml",
    }
    relevance = json.loads((logger.dir / "relevance/result.json").read_text())
    assert (relevance["n_rounds"], relevance["n_docs"]) == (2, 2)
    assert [r["round"] for r in relevance["rounds"]] == [1, 2]
    assert relevance["totals"] == {
        "n_steps": 4,
        "tokens": {"prompt": 40, "completion": 80, "total": 120},
    }


def test_every_later_stage_reads_the_final_rounds_notes_only(answering_llm):
    answer(answering_llm.answers(relevance=notes_by_round()))
    for stage in LATER_STAGES:
        for prompt in system_prompts(answering_llm, stage):
            assert "Round 2 notes on Halloran memo." in prompt
            assert "Round 2 notes on Merrow memo." in prompt
            assert "Round 1 notes" not in prompt


def test_the_reasoning_prompt_names_each_records_document(answering_llm):
    answer(answering_llm)
    [prompt] = system_prompts(answering_llm, "reasoning")
    assert '"document": 1,\n      "who": "Halloran memo"' in prompt
    assert '"document": 2,\n      "who": "Merrow memo"' in prompt


@pytest.mark.parametrize("rounds", [1, 3])
def test_each_document_is_read_once_per_configured_round(answering_llm, rounds):
    answer(answering_llm, relevance_rounds=rounds)
    assert len(answering_llm.requests_for("relevance")) == rounds * len(DOCS)


def test_the_configs_params_are_in_every_request(answering_llm):
    params = {"temperature": 0.7, "extra_body": {"top_k": 20}}
    answer(answering_llm, model="openai/gpt-4o", params=params)
    for request in answering_llm.requests:
        assert {key: request[key] for key in params} == params


def test_no_request_carries_a_seed_unless_params_sets_one(answering_llm):
    answer(answering_llm)
    assert not any("seed" in request for request in answering_llm.requests)
    answering_llm.requests.clear()
    answer(answering_llm, model="openai/gpt-4o", params={"seed": 7})
    assert [request["seed"] for request in answering_llm.requests] == [7] * 8


def test_each_stage_renders_the_prompt_version_its_config_pins(answering_llm, tmp_path):
    versions = dict(zip(PROMPT_STAGES, ["vA", "vB", "vC", "vD"], strict=True))
    for stage, version in versions.items():
        prompt = yaml.safe_load((settings.PROMPTS_DIR / stage / "v1.yaml").read_text())
        prompt["instructions"] += f"\nMARKER-{version}"
        path = tmp_path / "prompts" / stage / f"{version}.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(prompt))
    answer(answering_llm, prompts=versions)
    stages = dict(zip(PROMPT_STAGES, ["relevance", *LATER_STAGES], strict=True))
    for stage, version in versions.items():
        for prompt in system_prompts(answering_llm, stages[stage]):
            assert set(re.findall(r"MARKER-v\w", prompt)) == {f"MARKER-{version}"}


@pytest.mark.parametrize(
    ("prompts", "error"),
    [
        ({**CONFIG["prompts"], "reasoning": "v9"}, FileNotFoundError),
        ({"relevance": "v1"}, ValueError),
    ],
    ids=["missing-file", "missing-stage"],
)
def test_a_config_that_cannot_render_its_prompts_is_refused_before_any_request(
    answering_llm, tmp_path, prompts, error
):
    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(error):
        answer(answering_llm, logger, prompts=prompts)
    assert answering_llm.requests == []
    assert not (logger.dir / "manifest.json").exists()


@pytest.mark.parametrize("stage", list(STAGE_FOLDERS))
def test_a_failing_stage_leaves_its_calls_and_traceback_and_names_the_run_folder(
    answering_llm, tmp_path, stage
):
    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(litellm.APIConnectionError, match="provider down") as failure:
        answer(answering_llm.answers(**{stage: provider_down()}), logger)
    assert failure.value.__notes__ == [f"r3con: partial artifacts in {logger.dir}"]
    folder = logger.dir / STAGE_FOLDERS[stage]
    error = (folder / "error.txt").read_text()
    assert "APIConnectionError: litellm.APIConnectionError: provider down" in error
    assert (folder / "calls.json").is_file()
    assert not (folder / "result.json").exists()
    earlier = list(STAGE_FOLDERS.values())[: list(STAGE_FOLDERS).index(stage)]
    assert all((logger.dir / name / "result.json").is_file() for name in earlier)


def test_a_stage_that_fails_midway_keeps_the_calls_made_before_it(
    answering_llm, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    down = provider_down()
    with pytest.raises(litellm.APIConnectionError):
        answer(answering_llm.answers(relevance=["one", "two", down, down]), logger)
    calls = read_json(logger, "relevance/calls")
    assert sorted(call["kind"] for call in calls) == [
        "relevance-r1-d0",
        "relevance-r1-d1",
    ]


def test_an_interrupted_stage_is_recorded_like_a_failure(answering_llm, tmp_path):
    logger = TaskLogger("run", root=tmp_path)
    answering_llm.answers(relevance=KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        answer(answering_llm, logger, documents=DOCS[:1])
    assert "KeyboardInterrupt" in (logger.dir / "relevance" / "error.txt").read_text()


def test_without_a_logger_a_failure_carries_no_note(answering_llm, tmp_path):
    with pytest.raises(litellm.APIConnectionError) as failure:
        answer(answering_llm.answers(schema=provider_down()))
    assert not hasattr(failure.value, "__notes__")
    assert list(tmp_path.iterdir()) == []


def test_without_a_logger_nothing_is_written(answering_llm, tmp_path):
    result = answer(answering_llm)
    assert result.answer == "Halloran memo; Merrow memo"
    assert result.run_dir is None
    assert list(tmp_path.iterdir()) == []


def test_every_request_of_a_run_goes_through_the_callers_completion(
    answering_llm, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    answer(answering_llm, logger, model=QWEN)
    assert len(answering_llm.requests) == 8
    assert {r["model"] for r in answering_llm.requests} == {QWEN}
    assert "completion" not in (logger.dir / "manifest.json").read_text()


@pytest.mark.parametrize(
    ("setting", "explicit"), [(2, None), (5, 2)], ids=["in-settings", "explicit"]
)
def test_a_turn_cap_bounds_the_run_and_is_recorded(
    answering_llm, tmp_path, monkeypatch, setting, explicit
):
    monkeypatch.setattr(settings, "REASONING_MAX_TURNS", setting)
    logger = TaskLogger("run", root=tmp_path)
    answering_llm.answers(reasoning=NEVER_COMMITS)
    answer(answering_llm, logger, max_reasoning_turns=explicit)
    assert len(answering_llm.requests_for("reasoning")) == 3
    assert read_json(logger, "reasoning/result")["n_turns"] == 2
    assert read_json(logger, "manifest")["settings"]["reasoning_max_turns"] == 2


@pytest.mark.parametrize(
    ("cap", "stage", "reply"),
    [
        ("SCHEMA_MAX_ATTEMPTS", "schema", "class Foo(BaseModel):\n    x: int"),
        ("PARSING_MAX_ATTEMPTS", "parsing", '{"rows": [{"who": 7}]}'),
    ],
)
def test_an_attempt_cap_set_in_settings_bounds_its_stage(
    answering_llm, monkeypatch, cap, stage, reply
):
    monkeypatch.setattr(settings, cap, 2)
    with pytest.raises(SchemaError, match="exhausted 2 attempts"):
        answer(answering_llm.answers(**{stage: reply}), documents=DOCS[:1])
    assert len(answering_llm.requests_for(stage)) == 2


def test_the_manifest_records_the_margin(answering_llm, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "WINDOW_MARGIN_PERCENT", 20)
    logger = TaskLogger("run", root=tmp_path)
    answer(answering_llm, logger)
    assert read_json(logger, "manifest")["settings"]["window_margin_percent"] == 20


def test_a_cap_below_its_floor_sends_and_writes_nothing(llm, tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "REASONING_MAX_TURNS", 0)
    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(ValueError, match="REASONING_MAX_TURNS must be >= 1, got 0."):
        answer(llm, logger)
    assert llm.requests == []
    assert not (logger.dir / "manifest.json").exists()
