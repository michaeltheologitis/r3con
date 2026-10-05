import json
import re
from collections import Counter

import litellm
import pytest
import yaml

from r3con.config import PROMPT_STAGES, RunConfig
from r3con.pipeline import run_pipeline
from r3con.runs import TaskLogger
from r3con.settings import settings

MODEL = "openai/gpt-6-luna"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
DOCS = ["Halloran memo", "Merrow memo"]
LATER_STAGES = ("schema", "parsing", "reasoning")
CONFIG = {
    "name": "test",
    "model": MODEL,
    "seed": 0,
    "relevance_rounds": 2,
    "prompts": dict.fromkeys(PROMPT_STAGES, "v1"),
}


def answer(llm, task_logger=None, **fields):
    """A whole run over DOCS on CONFIG with ``fields`` changed, ``llm`` answering."""
    return run_pipeline(
        task="Who?",
        documents=DOCS,
        config=RunConfig(**{**CONFIG, **fields}),
        task_logger=task_logger,
        completion=llm,
    )


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


def test_the_configs_params_and_seed_are_in_every_request(answering_llm):
    params = {"temperature": 0.7, "extra_body": {"top_k": 20}}
    answer(answering_llm, model="openai/gpt-4o", seed=7, params=params)
    for request in answering_llm.requests:
        assert request["temperature"] == 0.7
        assert request["extra_body"] == {"top_k": 20}
        assert request["seed"] == 7


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


def test_a_reasoning_failure_is_raised_and_its_traceback_left_on_disk(
    answering_llm, tmp_path
):
    too_long = litellm.ContextWindowExceededError(
        "ctx too long", model=MODEL, llm_provider="openai"
    )
    answering_llm.answers(reasoning=too_long)
    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(litellm.ContextWindowExceededError, match="ctx too long"):
        answer(answering_llm, logger)
    assert (
        "ContextWindowExceededError" in (logger.dir / "reasoning/error.txt").read_text()
    )
    assert not (logger.dir / "reasoning/result.json").exists()
    assert (logger.dir / "manifest.json").is_file()
    assert (logger.dir / "structuring/parsing/result.json").is_file()


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
