import json
import re
from collections import Counter
from pathlib import Path

import litellm
import pytest
import yaml

from r3con import read_documents, settings
from r3con.config import PROMPT_STAGES, RunConfig
from r3con.pipeline import run_pipeline
from r3con.runs import TaskLogger
from r3con.stages.structuring.parsing import SchemaError

MODEL = "openai/gpt-6-luna"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
MEMOS = Path(__file__).resolve().parents[1] / "examples" / "memos"
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
V2 = {**CONFIG["prompts"], "reasoning": "v2"}
READ_IN_PARTS = "Some documents were too long to read whole and were read in parts"
REGISTRY_IN_4_PARTS = ["1", "2", "3", "4.1", "4.2", "4.3", "4.4", "5"]
ROUTINE = "The site logged its routine checks and found nothing out of the ordinary. "


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


def memos_with(registry: str) -> list[str]:
    """The five memos, with the contractor registry (documents[3]) replaced."""
    documents = read_documents(MEMOS)
    documents[3] = registry
    return documents


def summary_headings(llm) -> list[str]:
    """The headings of the reasoning prompt's summaries, without its examples'."""
    [prompt, *_] = system_prompts(llm, "reasoning")
    section = prompt.split("## Document summaries", 1)[1].split("## The parsed", 1)[0]
    return re.findall(r"^### Document ([\d.]+)$", section, re.MULTILINE)


def chars(request) -> int:
    return sum(len(message["content"]) for message in request["messages"])


def tokens(request) -> int:
    return sum(len(litellm.encode(text=m["content"])) for m in request["messages"])


def users(llm) -> list[str]:
    return [request["messages"][1]["content"] for request in llm.requests]


def long_notes_for(registry: str, size: int):
    """A relevance reply: ``size`` characters for a part of ``registry``, and the
    first line for any other document."""
    note = "A long note on one part of the contractor registry. " * (size // 50 + 1)

    def reply(request) -> str:
        document = request["messages"][1]["content"]
        return note[:size] if document in registry else f"Notes on {document[:20]}."

    return reply


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


def test_a_refused_document_is_read_in_parts_and_keeps_its_number(
    answering_llm, grown_registry, tmp_path
):
    registry = grown_registry(40_000)
    logger = TaskLogger("run", root=tmp_path)
    llm = answering_llm.refuses_over(16_000)
    result = answer(llm, logger, documents=memos_with(registry), model=QWEN, prompts=V2)
    assert len(result.relevant_context) == 5
    assert summary_headings(llm) == REGISTRY_IN_4_PARTS
    assert READ_IN_PARTS in system_prompts(llm, "reasoning")[0]
    assert result.source_docs == {"rows": [0, 1, 2, 3, 3, 3, 3, 4]}
    stamps = [row["document"] for row in result.structured_context["rows"]]
    assert stamps == [1, 2, 3, 4, 4, 4, 4, 5]
    part_notes = read_json(logger, "relevance/result")["rounds"][0]["snippets"][3]
    for request in llm.requests_for("relevance"):
        part = request["messages"][1]["content"]
        if part in registry and part != registry:
            system = request["messages"][0]["content"]
            assert not any(note in system for note in part_notes)
    over = [request for request in llm.requests if chars(request) > 16_000]
    assert len(over) == len({json.dumps(r["messages"]) for r in over}) == 2
    record = read_json(logger, "splits")["documents"]["3"]
    assert len(record["cuts"]) == 3
    assert [(e["cause"], e["action"]) for e in record["events"]] == [
        ("refusal", "split"),
        ("refusal", "split"),
    ]
    assert record["parts"] == {"relevance-r1": 4, "relevance-r2": 4, "parse": 4}


def test_a_document_over_the_line_is_split_before_sending(
    answering_llm, grown_registry, window, tmp_path
):
    logger = TaskLogger("run", root=tmp_path)
    documents = memos_with(grown_registry(40_000))
    answer(answering_llm, logger, documents=documents, model=window(5_000), prompts=V2)
    assert summary_headings(answering_llm) == REGISTRY_IN_4_PARTS
    assert max(tokens(request) for request in answering_llm.requests) <= 4_250
    events = read_json(logger, "splits")["documents"]["3"]["events"]
    assert {event["cause"] for event in events} == {"estimate"}


def test_an_unsplit_run_sends_what_a_v1_reasoning_run_sends(
    answering_llm, window, tmp_path, monkeypatch
):
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")
    documents, model = read_documents(MEMOS), window(1_000_000)
    sent = {}
    for version in ("v1", "v2"):
        answering_llm.requests.clear()
        logger = TaskLogger(version, root=tmp_path)
        prompts = {**CONFIG["prompts"], "reasoning": version}
        answer(answering_llm, logger, documents=documents, model=model, prompts=prompts)
        sent[version] = [
            (r["messages"], r.get("response_format")) for r in answering_llm.requests
        ]
        assert not (logger.dir / "splits.json").exists()
    assert len(sent["v2"]) == 17
    assert sent["v2"] == sent["v1"]


@pytest.mark.parametrize("measured", [False, True], ids=["by-refusal", "by-estimate"])
def test_a_later_stage_splits_further_without_renumbering(
    answering_llm, grown_registry, window, tmp_path, measured
):
    registry = grown_registry(40_000)
    llm = answering_llm.answers(relevance=long_notes_for(registry, 2_000))
    if not measured:
        llm.refuses_over(25_000)
    logger = TaskLogger("run", root=tmp_path)
    model = window(6_000) if measured else QWEN
    result = answer(
        llm, logger, documents=memos_with(registry), model=model, prompts=V2
    )
    record = read_json(logger, "splits")["documents"]["3"]
    assert record["parts"] == {"relevance-r1": 2, "relevance-r2": 2, "parse": 4}
    assert summary_headings(llm) == ["1", "2", "3", "4.1", "4.2", "5"]
    kinds = [call["kind"] for call in read_json(logger, "structuring/parsing/calls")]
    assert sorted(kind for kind in kinds if "-d3" in kind) == [
        f"parse-d3c{k}" for k in range(4)
    ]
    assert len(llm.requests_for("relevance")) == 12 if measured else 13
    assert result.source_docs == {"rows": [0, 1, 2, 3, 3, 3, 3, 4]}


@pytest.mark.parametrize(
    ("measured", "clause"),
    [
        (False, "the part is about "),
        (True, "over the 5,100-token line by themselves"),
    ],
    ids=["by-refusal", "by-estimate"],
)
def test_splitting_stops_at_the_relevant_contexts_line(
    answering_llm, window, tmp_path, monkeypatch, measured, clause
):
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")
    reports = [f"Report {i:02d}. " + ROUTINE * 5 for i in range(40)]
    llm = answering_llm.answers(relevance=(ROUTINE * 14)[:1_000])
    if not measured:
        llm.refuses_over(20_000)
    logger = TaskLogger("run", root=tmp_path)
    model = window(6_000) if measured else QWEN
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        answer(llm, logger, documents=reports, model=model, prompts=V2)
    note, folder = stop.value.__notes__
    assert note.startswith(
        "r3con: reading documents[0] (Document 1) in more parts cannot help in "
        "relevance-r2-d0: "
    )
    assert clause in note
    assert note.endswith("The relevant context has outgrown the model's window.")
    assert folder == f"r3con: partial artifacts in {logger.dir}"
    assert "TASK-" not in note
    assert [user for user in users(llm) if user in reports[0]] == [reports[0]] * (
        1 if measured else 2
    )
    assert len(llm.requests) == (40 if measured else 41)
    [(doc, record)] = read_json(logger, "splits")["documents"].items()
    assert (doc, record["cuts"]) == ("0", [])
    assert [event["action"] for event in record["events"]] == ["stop"]


def test_a_stopped_run_keeps_its_calls_and_splits(answering_llm, tmp_path):
    # Relevance and the schema fit under the limit; a parse call, which carries every
    # note and a whole document, does not, and its document is shorter than its rest.
    documents = [f"{name} memo. " + ROUTINE * 66 for name in ("Halloran", "Merrow")]
    llm = answering_llm.answers(relevance=(ROUTINE * 41)[:3_000]).refuses_over(13_500)
    logger = TaskLogger("run", root=tmp_path)
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        answer(llm, logger, documents=documents, model=QWEN, prompts=V2)
    assert "in more parts cannot help in parse-d0" in stop.value.__notes__[0]
    written = {str(p.relative_to(logger.dir)) for p in logger.dir.rglob("*.*")}
    assert {
        "relevance/result.json",
        "structuring/schema/result.json",
        "structuring/parsing/calls.json",
        "structuring/parsing/error.txt",
        "splits.json",
    } <= written
    assert "structuring/parsing/result.json" not in written
    events = read_json(logger, "splits")["documents"]["0"]["events"]
    assert [(e["call"], e["action"]) for e in events] == [("parse-d0", "stop")]
