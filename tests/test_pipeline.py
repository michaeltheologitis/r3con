import functools
import json
import re
from collections import Counter
from pathlib import Path

import litellm
import pytest
import yaml

from r3con import read_documents, settings
from r3con.config import PROMPT_STAGES, RunConfig, load_config
from r3con.notes import NotesTooLong
from r3con.pipeline import run_pipeline
from r3con.runs import TaskLogger
from r3con.stages.relevance import join_parts
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
SAMPLE = "only a SAMPLE"
LINE_8192 = 6_963


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


def memo_note(request) -> str:
    """A relevance reply of about 200 characters, 33 words, that names the document by
    its first line, as a note on one of the memos does."""
    first = request["messages"][1]["content"].splitlines()[0]
    note = (
        f"{first}: an operations document; it reports what each site logged and the "
        "contractor code its maintenance runs under, which the registry maps to a name. "
    ) * 4
    return note[:200].rsplit(" ", 1)[0] + "."


def whole_document_row(request) -> str:
    """A parse reply whose one row is the whole document, so the parse fills the
    reasoning prompt."""
    return json.dumps({"rows": [{"who": request["messages"][1]["content"]}]})


def memos_run(llm, documents, model, tmp_path):
    """A run of the default config over ``documents`` on ``model``, ``llm`` answering."""
    logger = TaskLogger("run", root=tmp_path)
    result = run_pipeline(
        task="Who?",
        documents=documents,
        config=load_config("default", model=model),
        task_logger=logger,
        completion=llm,
    )
    return logger, result


def first_turns(llm) -> list[dict]:
    return [r for r in llm.requests_for("reasoning") if len(r["messages"]) == 2]


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
    assert len(llm.requests_for("relevance")) == (12 if measured else 13)
    assert result.source_docs == {"rows": [0, 1, 2, 3, 3, 3, 3, 4]}


@pytest.mark.parametrize("measured", [False, True], ids=["by-refusal", "by-estimate"])
def test_notes_that_do_not_fit_stop_the_run_where_the_relevance_prompt_cannot_shorten_them(
    answering_llm, window, tmp_path, monkeypatch, measured
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
    assert note == (
        "r3con: relevance-r2-d0 carries the notes of 39 documents (about 7,410 "
        "tokens), the bigger part of a request that does not fit, and the relevance "
        "prompt this run pins cannot ask for shorter notes (the shipped v2 can). The "
        "relevant context has outgrown the model's window."
    )
    assert folder == f"r3con: partial artifacts in {logger.dir}"
    assert "TASK-" not in note
    assert [user for user in users(llm) if user in reports[0]] == [reports[0]] * (
        1 if measured else 2
    )
    assert len(llm.requests) == (40 if measured else 41)
    assert not (logger.dir / "splits.json").exists()
    events = read_json(logger, "notes")["events"]
    assert [(e["call"], e["action"]) for e in events] == [("relevance-r2-d0", "stop")]


def test_a_stopped_run_keeps_its_calls_and_splits(answering_llm, tmp_path):
    # Relevance and the schema fit under the limit; a parse call, which carries every
    # note and a whole document, does not, and its document is shorter than its rest.
    documents = [f"{name} memo. " + ROUTINE * 66 for name in ("Halloran", "Merrow")]
    llm = answering_llm.answers(relevance=(ROUTINE * 14)[:1_000]).refuses_over(10_000)
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


def test_reasoning_shows_samples_when_the_whole_parse_would_not_fit(
    answering_llm, quiet_sites, within_budget, window, tmp_path
):
    documents = read_documents(MEMOS) + quiet_sites(55)
    llm = answering_llm.answers(
        relevance=within_budget(memo_note), parsing=whole_document_row
    )
    _, result = memos_run(llm, documents, window(8_192), tmp_path)
    assert result.answer == "; ".join(documents)
    assert len(llm.requests) == 182
    assert max(tokens(request) for request in llm.requests) <= LINE_8192
    [turn] = first_turns(llm)
    assert SAMPLE in turn["messages"][0]["content"]


def test_a_refused_first_turn_goes_again_once_with_samples(
    answering_llm, quiet_sites, within_budget, tmp_path
):
    documents = read_documents(MEMOS) + quiet_sites(55)
    llm = answering_llm.answers(
        relevance=within_budget(memo_note), parsing=whole_document_row
    ).refuses_over(tokens=8_192)
    _, result = memos_run(llm, documents, QWEN, tmp_path)
    assert result.answer == "; ".join(documents)
    assert len(llm.requests) == 183
    whole, samples = first_turns(llm)
    assert SAMPLE not in whole["messages"][0]["content"]
    assert tokens(whole) > 8_192
    assert SAMPLE in samples["messages"][0]["content"]


def test_shortening_stops_where_ten_word_notes_cannot_fit(
    answering_llm, quiet_sites, within_budget, window, tmp_path
):
    documents = read_documents(MEMOS) + quiet_sites(395)
    llm = answering_llm.answers(relevance=within_budget(memo_note))
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        memos_run(llm, documents, window(8_192), tmp_path)
    assert type(stop.value) is litellm.ContextWindowExceededError
    assert "r3con estimated relevance-r2-d" in str(stop.value)
    assert stop.value.__notes__[0] == (
        "r3con: even at 10 words each, the notes of 400 documents would take about "
        "6,733 tokens, over the 2,599 that reasoning leaves them, so reading them "
        "shorter cannot help. The relevant context has outgrown the model's window."
    )
    assert len(llm.requests) == 400
    assert all(
        "summaries of the other documents" not in s
        for s in system_prompts(llm, "relevance")
    )
    assert all("Keep it under" not in s for s in system_prompts(llm, "relevance"))
    events = json.loads((tmp_path / "run" / "notes.json").read_text())["events"]
    assert [(e["action"], e["round"], e["words"]) for e in events] == [
        ("stop", 1, None)
    ]


def test_an_ordinary_run_sends_what_a_v1_relevance_run_sends(
    answering_llm, window, tmp_path, monkeypatch
):
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")
    documents, model = read_documents(MEMOS), window(1_000_000)
    sent = {}
    for version in ("v1", "v2"):
        answering_llm.requests.clear()
        logger = TaskLogger(version, root=tmp_path)
        prompts = {**V2, "relevance": version}
        answer(answering_llm, logger, documents=documents, model=model, prompts=prompts)
        sent[version] = [
            (r["messages"], r.get("response_format")) for r in answering_llm.requests
        ]
        assert not (logger.dir / "notes.json").exists()
    assert len(sent["v2"]) == 17
    assert sent["v2"] == sent["v1"]


def first_lines(documents: list[str]) -> str:
    """The answer ``answering_llm`` gives over ``documents``."""
    return "; ".join(document.splitlines()[0] for document in documents)


def test_notes_that_do_not_fit_are_read_again_shorter_by_estimate(
    answering_llm, quiet_sites, within_budget, window, tmp_path
):
    documents = read_documents(MEMOS) + quiet_sites(115)
    llm = answering_llm.answers(relevance=within_budget(memo_note))
    logger, result = memos_run(llm, documents, window(8_192), tmp_path)
    assert result.answer == first_lines(documents)
    assert max(tokens(request) for request in llm.requests) <= LINE_8192
    assert not (logger.dir / "splits.json").exists()
    assert len(result.relevant_context) == 120
    assert summary_headings(llm) == [str(n) for n in range(1, 121)]
    events = read_json(logger, "notes")["events"]
    assert [
        (e["call"], e["cause"], e["action"], e["round"], e["words"]) for e in events
    ] == [("reasoning", "estimate", "read again", 2, 16)]
    assert len(llm.requests) == 482


def test_notes_that_do_not_fit_are_read_again_shorter_on_a_refusal(
    answering_llm, quiet_sites, within_budget, tmp_path
):
    documents = read_documents(MEMOS) + quiet_sites(115)
    llm = answering_llm.answers(relevance=within_budget(memo_note))
    logger, result = memos_run(
        llm.refuses_over(tokens=8_192), documents, QWEN, tmp_path
    )
    assert result.answer == first_lines(documents)
    events = read_json(logger, "notes")["events"]
    assert events and {event["cause"] for event in events} == {"refusal"}
    assert {event["action"] for event in events} == {"read again"}
    assert all(event["words"] >= 10 for event in events)
    refused = [json.dumps(request["messages"]) for request in llm.refused]
    assert len(refused) == len(set(refused))
    assert len(llm.requests) == 485


READ_AGAIN = {
    "by-refusal": (
        20_000,
        [
            ("relevance-r2-d0", 1, 88),
            ("relevance-r2-w88-d0", 1, 44),
            ("reasoning", 2, 22),
        ],
        247,
    ),
    "by-refusal-of-the-schema": (
        17_000,
        [("relevance-r2-d0", 1, 88), ("relevance-r2-w88-d0", 1, 44), ("schema", 2, 22)],
        246,
    ),
    "by-estimate": (None, [("relevance-r2-d0", 1, 44)], 162),
}


@pytest.mark.parametrize(
    ("refused_over", "read_again", "n_requests"), READ_AGAIN.values(), ids=READ_AGAIN
)
def test_notes_that_do_not_fit_are_read_again_shorter(
    answering_llm,
    within_budget,
    window,
    tmp_path,
    monkeypatch,
    refused_over,
    read_again,
    n_requests,
):
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")
    reports = [f"Report {i:02d}. " + ROUTINE * 5 for i in range(40)]
    llm = answering_llm.answers(relevance=within_budget((ROUTINE * 14)[:1_000]))
    if refused_over:
        llm.refuses_over(refused_over)
    logger = TaskLogger("run", root=tmp_path)
    model = QWEN if refused_over else window(6_000)
    prompts = {**V2, "relevance": "v2"}
    result = answer(llm, logger, documents=reports, model=model, prompts=prompts)
    assert len(result.relevant_context) == 40
    events = read_json(logger, "notes")["events"]
    assert [(e["call"], e["round"], e["words"]) for e in events] == read_again
    assert {e["cause"] for e in events} == {"refusal" if refused_over else "estimate"}
    if not refused_over:
        assert max(tokens(request) for request in llm.requests) <= 5_100
        assert events[0]["sized_for"] == "reasoning"
    assert len(llm.requests) == n_requests


def refused_parse_run(llm, within_budget, tmp_path, monkeypatch, *, relevance=None):
    """Ten reports on an unknown window whose every request fits under 18,500
    characters but the last report's parse call, whose notes (ten of 1,000
    characters, 177 words each) are bigger than its 5,000 characters."""
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")
    reports = [f"Report {i:02d}. " + ROUTINE for i in range(9)]
    reports.append("Report 09. " + (ROUTINE * 67)[:5_000])
    note = within_budget((ROUTINE * 14)[:1_000])
    llm.answers(relevance=note if relevance is None else relevance(note))
    logger = TaskLogger("run", root=tmp_path)
    prompts = {**V2, "relevance": "v2"}
    run = functools.partial(
        answer, llm.refuses_over(18_500), logger, documents=reports, model=QWEN
    )
    return logger, functools.partial(run, prompts=prompts)


def test_a_stage_sent_again_keeps_its_discarded_calls(
    answering_llm, within_budget, tmp_path, monkeypatch
):
    logger, run = refused_parse_run(answering_llm, within_budget, tmp_path, monkeypatch)
    run()
    [event] = read_json(logger, "notes")["events"]
    assert (event["call"], event["cause"], event["discarded"]) == (
        "parse-d9",
        "refusal",
        9,
    )
    assert (event["round"], event["words"]) == (2, 88)
    parses = [call["kind"] for call in read_json(logger, "structuring/parsing/calls")]
    assert parses == [f"parse-d{i}" for i in range(9)] + [
        f"parse-d{i}" for i in range(10)
    ]
    reads = [call["kind"] for call in read_json(logger, "relevance/calls")]
    assert reads[20:] == [f"relevance-r2-w88-d{i}" for i in range(10)]
    rounds = read_json(logger, "relevance/result")["rounds"]
    assert [r.get("max_words") for r in rounds] == [None, 88]


def test_the_relevant_context_is_the_notes_reasoning_saw(
    answering_llm, within_budget, tmp_path, monkeypatch
):
    logger, run = refused_parse_run(answering_llm, within_budget, tmp_path, monkeypatch)
    result = run()
    reasoning_prompt = system_prompts(answering_llm, "reasoning")[-1]
    last_round = read_json(logger, "relevance/result")["rounds"][-1]["snippets"]
    assert result.relevant_context == [join_parts(s) for s in last_round]
    assert all(len(note.split()) == 88 for note in result.relevant_context)
    assert all(f"\n{note}\n" in reasoning_prompt for note in result.relevant_context)


def test_a_failure_while_reading_again_is_recorded_by_both_stages_named_once(
    answering_llm, within_budget, tmp_path, monkeypatch
):
    def down_when_read_again(note):
        return [note] * 20 + [provider_down()]

    logger, run = refused_parse_run(
        answering_llm,
        within_budget,
        tmp_path,
        monkeypatch,
        relevance=down_when_read_again,
    )
    with pytest.raises(litellm.APIConnectionError) as failure:
        run()
    assert failure.value.__notes__ == [f"r3con: partial artifacts in {logger.dir}"]
    for stage in ("relevance", "structuring/parsing"):
        assert "provider down" in (logger.dir / stage / "error.txt").read_text()
    assert (logger.dir / "relevance" / "result.json").is_file()


NINE_WORDS = "Report logged its routine checks and found nothing unusual."


@pytest.mark.parametrize(
    ("measured", "relevance", "reply", "note"),
    [
        (True, "v2", (ROUTINE * 14)[:1_000], "r3con: even at 10 words each"),
        (False, "v2", NINE_WORDS, "r3con: relevance-r2-d0 was refused as too long"),
        (True, "v1", (ROUTINE * 14)[:1_000], "r3con: relevance-r2-d0 carries"),
    ],
    ids=["floor-by-estimate", "floor-by-refusal", "v1"],
)
def test_no_stop_leaves_a_run_as_notes_too_long(
    answering_llm, window, tmp_path, measured, relevance, reply, note
):
    reports = [f"Report {i:02d}. " + ROUTINE * 5 for i in range(40)]
    llm = answering_llm.answers(relevance=reply)
    if not measured:
        llm.refuses_over(tokens=1_000)
    logger = TaskLogger("run", root=tmp_path)
    model = window(3_500) if measured else QWEN
    prompts = {**V2, "relevance": relevance}
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        answer(llm, logger, documents=reports, model=model, prompts=prompts)
    assert type(stop.value) is litellm.ContextWindowExceededError
    assert not isinstance(stop.value, NotesTooLong)
    first, folder = stop.value.__notes__
    assert first.startswith(note)
    assert first.endswith("The relevant context has outgrown the model's window.")
    assert folder == f"r3con: partial artifacts in {logger.dir}"
    assert [e["action"] for e in read_json(logger, "notes")["events"]] == ["stop"]
