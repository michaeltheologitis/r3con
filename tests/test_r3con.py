import errno
import importlib.metadata
import itertools
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import litellm
import pytest
from filelock import FileLock
from litellm.llms.github_copilot.authenticator import (
    Authenticator as CopilotAuthenticator,
)

import r3con
from r3con import Answer, RunConfig
from r3con import r3con as namespace
from r3con.config import PROMPT_STAGES, REASONING_PATHS, load_config
from r3con.documents import read_documents
from r3con.r3con import ask, resume, run, trace
from tests.documents import write_pdf, write_tree
from tests.fakes import (
    ANSWERING,
    ANSWERING_SCHEMA,
    FINGERPRINT,
    TWO_LINE_MEMOS,
    USAGE,
    a_parse_citation_short,
    bad_key,
    cited_documents,
    citing,
    failing_once,
    first_line,
    in_json,
    is_later_round,
    predating_resume,
    quoting_a_line_a_round,
    quoting_its_first_line_and_citing_all_shown,
    reading,
    reading_moved_prompts,
    silent_when,
    snippets_without_quotes,
    stage_of,
    without_trail,
)

PINS_A_MISSING_FILE = {**dict.fromkeys(PROMPT_STAGES, "v1"), "reasoning/coding": "v9"}
DOCS = ["Halloran memo", "Merrow memo", "Calder memo"]
NAMES = ["memos/halloran.txt", "memos/merrow.txt", "calder.txt"]
NAMED_DOCS = dict(zip(NAMES, DOCS, strict=True))
JOINED = "Halloran memo; Merrow memo; Calder memo"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
GPT_4O = "openai/gpt-4o"
COPILOT = "github_copilot/gpt-x"
# No shape litellm's own redactor knows: only r3con's masking can hide it.
KEY = "not-a-real-key-q7RkT2mXv9LpA4sW"
NAMES_A_KEY = "The setup memo's example config reads api_key=abcdefgh12345."
ECHOING = {
    "message": f"Incorrect API key provided: {KEY}",
    "llm_provider": "openai",
    "model": "gpt-6-luna",
}
REFUSED_LINE = (
    "AuthenticationError: litellm.AuthenticationError: Incorrect API key provided: "
    "REDACTED"
)
RUN_PARAMS = {"temperature": 0.3, "extra_body": {"top_k": 20}}
GATEWAY_HEADER = {"extra_headers": {"Authorization": f"Bearer {KEY}"}}
# An Azure api_version as YAML reads one written unquoted: a date.
DATED_PARAMS = {"api_version": date(2024, 10, 21)}
SAVED_DATED_PARAMS = {"api_version": "2024-10-21"}
MASKED_HEADER_REFUSAL = (
    "REDACTED, the mask a run's manifest saves in place of a credential, would be "
    "sent: pass the real value in `params`, which replaces all of the run's params. "
    "Masked params: `extra_headers.Authorization`."
)
CANCELLED = "Cancelled: cancel was set, so no further model call is sent"
BARRIER_TIMEOUT_SECONDS = 10
# How long the user takes to enter a device sign-in's code, long enough for every
# document's first call to start meanwhile.
USER_ENTERS_CODE_SECONDS = 0.2
LOGGED_TIMEOUT_SECONDS = 5
ASK_FILES = {
    "manifest.json",
    "events.jsonl",
    "reasoning/calls.json",
    "reasoning/result.json",
    "reasoning/transcript.yaml",
}


def finished_run(llm, tmp_path: Path, documents=DOCS, **options) -> Answer:
    """A run of "Who?" over ``documents``, ``llm`` answering, its folder under
    ``tmp_path / "logs"``."""
    logs = tmp_path / "logs"
    return run("Who?", documents, completion=llm, logs_dir=logs, **options)


def reasoning_calls(folder: Path) -> list[dict[str, Any]]:
    return json.loads((folder / "reasoning" / "calls.json").read_text())


def what_each_call_records(calls: list[dict[str, Any]]) -> set[tuple]:
    """Each call's kind, what its reply's body said and how long it took, as
    ``(kind, served_model, fingerprint, usage, seconds is a number)``."""
    return {
        (
            call["kind"],
            call["served_model"],
            call["fingerprint"],
            json.dumps(call["usage"]),
            isinstance(call["seconds"], float),
        )
        for call in calls
    }


def files_holding(text: str, folder: Path) -> list[str]:
    """The files under ``folder`` that hold ``text``, by path."""
    files = [path for path in folder.rglob("*") if path.is_file()]
    return [str(path.relative_to(folder)) for path in files if text in path.read_text()]


def the_runs_own_files(run_dir: Path) -> dict[str, bytes]:
    """Every file in ``run_dir`` but those under ``asks/``, by path."""
    return {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file() and path.relative_to(run_dir).parts[0] != "asks"
    }


def snippets_without_provenance(result: dict) -> None:
    for round_ in result["rounds"]:
        round_["snippets"] = [
            {"id": s["id"], "summary": s["summary"]} for s in round_["snippets"]
        ]


def snippets_as_plain_text(result: dict) -> None:
    """As r3context 0.4.0 wrote them."""
    for round_ in result["rounds"]:
        round_["snippets"] = [s["summary"] for s in round_["snippets"]]


def without_provenance(result: dict) -> None:
    del result["provenance"]


def with_a_seed(manifest: dict) -> None:
    """As r3con before 0.2.0 recorded its config."""
    manifest["config"]["seed"] = 7


def with_one_reasoning_prompt(manifest: dict) -> None:
    """As r3context 0.4.0 pinned it."""
    prompts = manifest["config"]["prompts"]
    del prompts["reasoning/direct"], prompts["reasoning/coding"]
    prompts["reasoning"] = "v2"


def quoting_only_in_round_one(shown_ids):
    """A relevance reply quoting the whole document in round 1, and later quoting
    nothing and citing the first id shown."""

    def reply(request):
        if not is_later_round(request):
            document = request["messages"][1]["content"]
            return {"summary": f"Notes on {document}.", "quotes": [document]}
        cites = shown_ids(request)[:1]
        return {"summary": "Later notes.", "quotes": [], "provenance_ids": cites}

    return reply


@pytest.fixture
def quoted_run(answering_llm, shown_ids, tmp_path) -> Answer:
    """A finished run over ``NAMED_DOCS`` whose snippets quote their document in round
    1 only and whose answer cites the first summary, ``answering_llm`` answering so
    from then on."""
    answering_llm.answers(
        relevance=quoting_only_in_round_one(shown_ids),
        reasoning=citing("Halloran memo", slice(1)),
    )
    return finished_run(answering_llm, tmp_path, NAMED_DOCS)


@pytest.mark.parametrize(
    ("documents", "error", "message"),
    [
        ("./docs", TypeError, r"not a single string or path.*\[text\].*read_documents"),
        ("just some document text", TypeError, "not a single string or path"),
        (Path("./docs"), TypeError, "not a single string or path"),
        (
            42,
            TypeError,
            "a sequence of strings, or a mapping of names to them — got int",
        ),
        ({"a.txt": "fine", "b.txt": 42}, TypeError, "item 'b.txt' is int"),
        ({1: "a memo"}, TypeError, "names must be strings — 1 is int"),
        (["fine", 42], TypeError, "item 1 is int"),
        (["real", "   ", "also real"], ValueError, "item 1 is blank"),
        (["  ", ""], ValueError, "item 0 is blank"),
        ([], ValueError, "is empty"),
    ],
)
def test_documents_that_are_not_texts_are_refused_before_any_request(
    llm, documents, error, message
):
    with pytest.raises(error, match=message):
        run("Who?", documents, completion=llm)
    assert llm.requests == []
    assert not Path("logs").exists()


def test_a_blank_document_given_by_name_is_refused_by_its_name_alone(llm):
    with pytest.raises(ValueError) as refused:
        run("Who?", {"a.txt": "a memo", "b.txt": " "}, completion=llm)
    assert str(refused.value) == (
        "`documents` item 'b.txt' is blank: every document must have text. Leave it "
        "out, or read the files with r3con.read_documents(...), which skips empty ones."
    )
    assert llm.requests == []


@pytest.mark.parametrize("wrap", [list, tuple, iter])
def test_each_document_text_is_read_as_given_and_reported_in_order(answering_llm, wrap):
    documents = ["one", "two", "three"]
    result = run("Who?", wrap(documents), completion=answering_llm)
    assert result.relevant_context == [f"Notes on {d}." for d in documents]
    assert result.answer == "one; two; three"


def test_a_document_that_names_a_file_is_never_read_off_disk(answering_llm, tmp_path):
    (tmp_path / "Q3 was strong.").write_text("SOMETHING ELSE", encoding="utf-8")
    result = run("Who?", ["Q3 was strong."], completion=answering_llm)
    assert result.answer == "Q3 was strong."
    reads = answering_llm.requests_for("relevance")
    assert {request["messages"][1]["content"] for request in reads} == {
        "Q3 was strong."
    }


def test_a_prebuilt_config_refuses_overrides_it_would_swallow(llm):
    config = RunConfig(model="openai/m", prompts=dict.fromkeys(PROMPT_STAGES, "v1"))
    with pytest.raises(ValueError, match=r"already a RunConfig.*\['model'\]"):
        run("q", ["a document"], config=config, model="openai/other", completion=llm)
    assert llm.requests == []


@pytest.mark.parametrize(
    "config",
    [
        "default",
        RunConfig(model="openai/m", prompts=dict.fromkeys(PROMPT_STAGES, "v1")),
    ],
    ids=["by-name", "run-config"],
)
def test_a_cap_below_its_floor_is_refused_before_a_run_folder_exists(
    llm, tmp_path, monkeypatch, config
):
    monkeypatch.setattr(r3con.settings, "REASONING_MAX_TURNS", 0)
    with pytest.raises(ValueError, match="REASONING_MAX_TURNS must be >= 1, got 0."):
        run("Who?", ["memo"], config=config, completion=llm, logs_dir=tmp_path / "logs")
    assert not (tmp_path / "logs").exists()
    assert llm.requests == []


@pytest.mark.parametrize(
    ("config", "error", "message"),
    [
        ("exp", FileNotFoundError, "'reasoning/coding' version 'v9'"),
        (
            RunConfig(model="openai/m", prompts=PINS_A_MISSING_FILE),
            FileNotFoundError,
            "'reasoning/coding' version 'v9'",
        ),
        (
            RunConfig(model="openai/m", prompts={"reasoning/direct": "v1"}),
            ValueError,
            "missing prompt versions",
        ),
    ],
    ids=["by-name", "run-config", "run-config-missing-a-stage"],
)
def test_a_config_missing_a_prompt_is_refused_before_a_run_folder_exists(
    llm, configs, tmp_path, config, error, message
):
    configs(exp={"model": "openai/m", "prompts": PINS_A_MISSING_FILE})
    with pytest.raises(error, match=message):
        run("Who?", ["memo"], config=config, completion=llm, logs_dir=tmp_path / "logs")
    assert not (tmp_path / "logs").exists()
    assert llm.requests == []


@pytest.mark.parametrize(
    "model", ["anthropic/claude-sonnet-5-5", "gemini/gemini-3.8-flash"]
)
def test_a_run_on_a_provider_that_takes_no_seed_completes(answering_llm, model):
    result = run("Who?", ["Halloran memo"], model=model, completion=answering_llm)
    assert result.answer == "Halloran memo"
    assert {request["model"] for request in answering_llm.requests} == {model}


@pytest.mark.parametrize("reasoning", REASONING_PATHS)
def test_the_reasoning_path_is_chosen_per_run_and_recorded(
    answering_llm, tmp_path, reasoning
):
    result = run(
        "Who?",
        ["Halloran memo"],
        reasoning=reasoning,
        completion=answering_llm,
        logs_dir=tmp_path,
    )
    assert result.answer == "Halloran memo"
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["config"]["reasoning"] == reasoning
    folder = json.loads((result.run_dir / "reasoning" / "result.json").read_text())
    assert folder["path"] == reasoning


@pytest.mark.parametrize("reasoning", REASONING_PATHS)
def test_the_answer_cites_the_documents_it_used_in_the_result_and_the_run_folder(
    answering_llm, tmp_path, reasoning
):
    result = run(
        "Who?",
        ["Halloran memo", "Merrow memo"],
        reasoning=reasoning,
        completion=answering_llm,
        logs_dir=tmp_path,
    )
    assert result.answer == "Halloran memo; Merrow memo"
    assert without_trail(result.provenance) == {
        "summary_ids": [],
        "parse_ids": result.document_ids,
        "quotes": [],
    }
    assert result.relevance_provenance == [[], []]
    assert result.parsing_provenance == [[], []]
    folder = json.loads((result.run_dir / "reasoning" / "result.json").read_text())
    assert (folder["path"], folder["provenance"]) == (reasoning, result.provenance)


def test_a_reasoning_path_r3con_does_not_have_is_refused_before_any_request(
    llm, tmp_path
):
    with pytest.raises(ValueError, match="`reasoning='agent'`: Input should be"):
        run("Who?", ["memo"], reasoning="agent", completion=llm, logs_dir=tmp_path)
    assert llm.requests == []
    assert list(tmp_path.iterdir()) == []


def test_run_answers_through_the_callers_completion(answering_llm):
    result = run(
        "Who?", ["Halloran memo"], completion=answering_llm, save_artifacts=False
    )
    assert result.answer == "Halloran memo"


def test_a_run_leaves_the_callers_metadata_as_it_was(answering_llm, own_trace, logged):
    metadata = {"trace": own_trace}
    run(
        "Who?",
        DOCS[:2],
        params={"metadata": metadata},
        completion=answering_llm,
        save_artifacts=False,
    )
    for _ in answering_llm.requests:
        assert logged.acquire(timeout=LOGGED_TIMEOUT_SECONDS)
    assert metadata == {"trace": own_trace}


@pytest.mark.parametrize("save_artifacts", [True, False])
def test_save_artifacts_decides_whether_a_run_folder_is_written(
    answering_llm, tmp_path, save_artifacts
):
    logs = tmp_path / "logs"
    run(
        "Who?",
        ["memo"],
        completion=answering_llm,
        logs_dir=logs,
        save_artifacts=save_artifacts,
    )
    assert logs.exists() is save_artifacts


def test_the_run_folder_goes_under_logs_dir_and_is_returned(answering_llm, tmp_path):
    result = run("Who?", ["memo"], completion=answering_llm, logs_dir=tmp_path / "out")
    assert result.run_dir.parent == tmp_path / "out"
    assert (result.run_dir / "manifest.json").is_file()


def test_half_a_surrogate_pair_in_a_pdf_reads_as_a_replacement_and_the_run_goes_on(
    answering_llm, tmp_path
):
    pdf = tmp_path / "memo.pdf"
    # pypdf reads a font's ToUnicode CMap with surrogatepass: D800 stays half a pair.
    write_pdf(pdf, "Memo A here", to_unicode={"A": "D800"})
    documents = read_documents(pdf)
    assert documents == {"memo.pdf": "Memo \ufffd here"}
    answer = finished_run(answering_llm, tmp_path, documents)
    assert answering_llm.requests_for("relevance")
    assert answer.answer == "Memo \ufffd here"


def test_a_folder_read_with_read_documents_can_be_answered_over(
    answering_llm, tmp_path
):
    write_tree(tmp_path / "docs")
    result = run("Who?", read_documents(tmp_path / "docs"), completion=answering_llm)
    assert result.answer == "alpha document; beta document; gamma document"


@pytest.mark.skipif(os.name != "posix", reason="a file name of any bytes is POSIX's")
def test_a_file_name_that_is_not_utf8_reads_with_a_replacement_and_runs(
    answering_llm, tmp_path
):
    Path("memos").mkdir()
    for name in [b"caf\xe8.txt", b"caf\xe9.txt"]:
        try:
            (Path("memos") / os.fsdecode(name)).write_text("A memo.", encoding="utf-8")
        except OSError:
            pytest.skip("this file system takes UTF-8 names only")
    documents = read_documents("memos")
    names = ["memos/caf\ufffd.txt", "memos/caf\ufffd.txt (2)"]
    assert list(documents) == names
    result = finished_run(answering_llm, tmp_path, documents)
    assert result.document_names == names
    recorded = json.loads((result.run_dir / "names.json").read_text())
    assert list(recorded.values()) == names


@pytest.mark.parametrize(
    ("documents", "names"),
    [(NAMED_DOCS, NAMES), (DOCS, ["0", "1", "2"])],
    ids=["by-name", "a-plain-list"],
)
def test_a_run_names_each_document_by_its_key_or_else_its_position(
    answering_llm, tmp_path, documents, names
):
    events = []
    result = finished_run(answering_llm, tmp_path, documents, on_event=events.append)
    assert result.answer == JOINED
    assert result.document_names == names
    assert events[0]["document_names"] == names
    recorded = json.loads((result.run_dir / "names.json").read_text())
    assert recorded == dict(zip(result.document_ids, names, strict=True))


def test_a_run_folder_without_names_can_be_asked_and_names_documents_by_index(
    answering_llm, quoted_run
):
    (quoted_run.run_dir / "names.json").unlink()
    asked = ask(quoted_run.run_dir, "Who else?", completion=answering_llm)
    assert asked.answer == "Halloran memo"
    assert asked.document_names == ["0", "1", "2"]
    assert [quote["name"] for quote in asked.provenance["quotes"]] == ["1"]


@pytest.mark.parametrize("rounds", [0, 2])
def test_an_ask_answers_over_the_runs_representation_citing_only_its_ids(
    answering_llm, tmp_path, rounds
):
    result = finished_run(answering_llm, tmp_path, relevance_rounds=rounds)
    asked = ask(result, "Who else?", completion=answering_llm)
    assert asked.answer == JOINED
    cited = cited_documents(asked.provenance)
    assert cited
    assert cited <= set(result.document_ids)
    assert asked.run_dir.parent == result.run_dir / "asks"
    the_runs = replace(
        asked,
        answer=result.answer,
        provenance=result.provenance,
        run_dir=result.run_dir,
    )
    assert the_runs == result


@pytest.mark.parametrize(
    "source",
    [lambda asked: asked, lambda asked: asked.run_dir],
    ids=["answer", "folder"],
)
def test_an_ask_over_an_ask_asks_its_run_again_in_a_folder_of_its_own(
    answering_llm, tmp_path, source
):
    result = finished_run(answering_llm, tmp_path)
    first = ask(result, "Who else?", completion=answering_llm)
    second = ask(source(first), "Who, then?", completion=answering_llm)
    assert second.answer == JOINED
    assert first.run_dir.parent == second.run_dir.parent == result.run_dir / "asks"
    assert first.run_dir != second.run_dir
    manifest = json.loads((second.run_dir / "manifest.json").read_text())
    assert manifest["ask_of"] == result.run_dir.name
    _, asked_first, asked_second = answering_llm.requests_for("reasoning")
    assert asked_second["messages"][0] == asked_first["messages"][0]


def test_an_ask_sends_the_runs_reasoning_prompt_with_the_new_question(
    answering_llm, tmp_path
):
    result = finished_run(answering_llm, tmp_path)
    ask(result.run_dir, "Which memo came first?", completion=answering_llm)
    ran, asked = answering_llm.requests_for("reasoning")
    system, user = asked["messages"]
    assert system == ran["messages"][0]
    assert all(f"Notes on {doc}." in system["content"] for doc in DOCS)
    assert all(f'"who": "{doc}"' in system["content"] for doc in DOCS)
    assert user["content"] == "Input:\n<task>\nWhich memo came first?\n</task>"
    assert asked["response_format"] == ran["response_format"]


def test_each_ask_writes_a_folder_of_its_own_and_leaves_the_runs_files_as_they_were(
    answering_llm, tmp_path
):
    result = finished_run(answering_llm, tmp_path)
    before = the_runs_own_files(result.run_dir)
    ask(result, "Who else?", completion=answering_llm)
    ask(result, "Who else?", model=QWEN, completion=answering_llm)
    folders = list((result.run_dir / "asks").iterdir())
    assert [{str(p.relative_to(f)) for p in f.rglob("*.*")} for f in folders] == [
        ASK_FILES,
        ASK_FILES,
    ]
    assert the_runs_own_files(result.run_dir) == before


def test_an_asks_manifest_records_its_question_its_run_and_the_config_it_ran_under(
    answering_llm, tmp_path
):
    result = finished_run(answering_llm, tmp_path, model=QWEN, params=RUN_PARAMS)
    ask(result, "Who else?", reasoning="coding", completion=answering_llm)
    [folder] = (result.run_dir / "asks").iterdir()
    ran = json.loads((result.run_dir / "manifest.json").read_text())
    asked = json.loads((folder / "manifest.json").read_text())
    assert (asked["task"], asked["ask_of"]) == ("Who else?", result.run_dir.name)
    assert asked["run_folder"] == folder.name
    assert asked["litellm_model"] == QWEN
    assert asked["config"] == {
        **ran["config"],
        "reasoning": "coding",
        "overrides": {"reasoning": "coding"},
    }
    assert [asked[key] for key in ("sources", "settings", "r3con_version")] == [
        ran[key] for key in ("sources", "settings", "r3con_version")
    ]
    assert "ask_of" not in ran
    reasoned = json.loads((folder / "reasoning" / "result.json").read_text())
    assert (reasoned["answer"], reasoned["path"]) == (JOINED, "coding")


@pytest.mark.parametrize(
    ("options", "model", "params"),
    [
        ({}, GPT_4O, RUN_PARAMS),
        ({"model": GPT_4O}, GPT_4O, RUN_PARAMS),
        ({"model": "gpt-4o"}, "gpt-4o", RUN_PARAMS),
        ({"model": QWEN}, QWEN, {}),
        ({"model": QWEN, "params": {"temperature": 0.9}}, QWEN, {"temperature": 0.9}),
    ],
    ids=[
        "the-runs",
        "the-runs-named",
        "the-runs-by-its-bare-name",
        "another",
        "another-with-params",
    ],
)
def test_an_ask_sends_the_runs_params_only_on_the_runs_model_unless_given_its_own(
    answering_llm, tmp_path, options, model, params
):
    result = finished_run(answering_llm, tmp_path, model=GPT_4O, params=RUN_PARAMS)
    ask(result, "Who else?", completion=answering_llm, **options)
    ran, asked = answering_llm.requests_for("reasoning")
    assert ran["model"] == GPT_4O
    assert asked["model"] == model
    sent = {
        key: asked[key] for key in RUN_PARAMS.keys() | params.keys() if key in asked
    }
    assert sent == params
    [folder] = (result.run_dir / "asks").iterdir()
    manifest = json.loads((folder / "manifest.json").read_text())
    assert (manifest["litellm_model"], manifest["config"]["params"]) == (model, params)


def test_a_date_among_a_runs_params_is_sent_as_it_is_and_saved_as_its_iso_date(
    answering_llm, tmp_path
):
    result = finished_run(answering_llm, tmp_path, params=DATED_PARAMS)
    [sent] = answering_llm.requests_for("reasoning")
    assert sent["api_version"] == DATED_PARAMS["api_version"]
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["config"]["params"] == SAVED_DATED_PARAMS


def test_an_unquoted_date_in_a_config_files_params_is_saved_as_its_iso_date(
    answering_llm, configs, tmp_path
):
    default = load_config().model_dump(exclude={"name", "overrides"})
    configs(dated={**default, "params": DATED_PARAMS})
    written = (tmp_path / "configs" / "dated.yaml").read_text()
    assert "api_version: 2024-10-21\n" in written
    result = finished_run(answering_llm, tmp_path, config="dated")
    manifest = json.loads((result.run_dir / "manifest.json").read_text())
    assert manifest["config"]["params"] == SAVED_DATED_PARAMS


def test_an_ask_over_a_run_with_a_date_among_its_params_sends_its_iso_date(
    answering_llm, tmp_path
):
    built = finished_run(answering_llm, tmp_path, params=DATED_PARAMS)
    ask(built, "Who else?", completion=answering_llm)
    asked = answering_llm.requests_for("reasoning")[-1]
    assert asked["api_version"] == SAVED_DATED_PARAMS["api_version"]


def test_an_ask_over_a_run_recorded_without_its_litellm_model_uses_the_bare_name(
    answering_llm, tmp_path
):
    result = finished_run(answering_llm, tmp_path, model=GPT_4O)
    manifest_path = result.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["litellm_model"]
    manifest_path.write_text(json.dumps(manifest))
    ask(result, "Who else?", completion=answering_llm)
    ran, asked = answering_llm.requests_for("reasoning")
    assert (ran["model"], asked["model"]) == (GPT_4O, "gpt-4o")


def test_each_coding_turn_records_its_replys_body_and_how_long_it_took(
    answering_llm, tmp_path
):
    result = finished_run(answering_llm, tmp_path, reasoning="coding")

    calls = reasoning_calls(result.run_dir)
    assert what_each_call_records(calls) == {
        ("turn-1", "gpt-6-luna", FINGERPRINT, json.dumps(USAGE), True)
    }


def test_each_call_of_an_ask_records_its_replys_body_and_how_long_it_took(
    answering_llm, tmp_path
):
    result = finished_run(answering_llm, tmp_path)
    ask(result, "Who else?", reasoning="coding", completion=answering_llm)

    [folder] = (result.run_dir / "asks").iterdir()
    assert what_each_call_records(reasoning_calls(folder)) == {
        ("turn-1", "gpt-6-luna", FINGERPRINT, json.dumps(USAGE), True)
    }


def test_an_ask_reasons_on_the_path_given_instead_of_the_runs(answering_llm, tmp_path):
    result = finished_run(answering_llm, tmp_path, reasoning="direct")
    asked = ask(result, "Who else?", reasoning="coding", completion=answering_llm)
    assert asked.answer == JOINED
    _, turn = answering_llm.requests_for("reasoning")
    assert "final_answer(" in turn["messages"][0]["content"]


def test_a_document_read_in_parts_keeps_its_part_headings_in_an_ask(
    answering_llm, grown_registry, window, tmp_path
):
    model = window(5_000)
    documents = ["Halloran memo", grown_registry(40_000), "Merrow memo"]
    result = finished_run(answering_llm, tmp_path, documents, model=model)
    sent = len(answering_llm.requests)
    ask(result, "How?", model=model, completion=answering_llm)
    [ran, *_] = answering_llm.requests_for("reasoning")
    asked = answering_llm.requests[sent]
    assert asked["messages"][0] == ran["messages"][0]
    registry = result.document_ids[1]
    system = asked["messages"][0]["content"]
    assert f"### Document ID: {registry}.1\n" in system
    assert f"### Document ID: {registry}.2\n" in system


@pytest.mark.parametrize("rounds", [1, 3])
def test_a_run_with_a_document_in_parts_is_asked_through_each_of_its_rounds(
    answering_llm, shown_ids, grown_registry, window, tmp_path, rounds
):
    answering_llm.answers(
        relevance=quoting_its_first_line_and_citing_all_shown(shown_ids),
        reasoning=citing("Halloran memo", slice(None), slice(None)),
    )
    model = window(5_000)
    documents = ["Halloran memo", grown_registry(40_000), "Merrow memo"]
    result = finished_run(
        answering_llm, tmp_path, documents, model=model, relevance_rounds=rounds
    )
    asked = ask(result, "Who else?", model=model, completion=answering_llm)
    assert asked.answer == "Halloran memo"
    assert f"{result.document_ids[1]}.2" in asked.provenance["summary_ids"]
    trail = asked.provenance["trail"]["nodes"]
    reached = {node.split(":")[1] for node in trail if node.startswith("summary:")}
    assert reached == {str(k) for k in range(1, rounds + 1)}
    assert asked.provenance["quotes"]
    assert trace(asked) == asked.provenance


@pytest.mark.parametrize("folder", ["missing", ""], ids=["missing", "empty"])
def test_a_folder_that_is_not_a_run_folder_cannot_be_asked(llm, tmp_path, folder):
    with pytest.raises(FileNotFoundError) as refused:
        ask(tmp_path / folder, "Who?", completion=llm)
    assert str(refused.value) == (
        f"{tmp_path / folder} is not a run folder: it has no manifest.json."
    )
    assert llm.requests == []
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("make_older", "error"),
    [
        (in_json("relevance/result.json", snippets_without_provenance), "TypeError"),
        (in_json("relevance/result.json", snippets_as_plain_text), "TypeError"),
        (
            in_json("structuring/parsing/result.json", without_provenance),
            "KeyError: 'provenance'",
        ),
        (
            in_json("manifest.json", with_one_reasoning_prompt),
            "`prompts.reasoning`: Input should be 'relevance/first-round'",
        ),
        (
            in_json("manifest.json", with_a_seed),
            "`seed`: Extra inputs are not permitted",
        ),
        (lambda run_dir: (run_dir / "ids.json").unlink(), "FileNotFoundError"),
    ],
    ids=[
        "snippets-without-provenance",
        "snippets-as-plain-text",
        "parse-without-provenance",
        "one-reasoning-prompt",
        "config-with-a-seed",
        "no-ids",
    ],
)
def test_a_run_folder_an_older_r3con_wrote_is_refused_in_one_line(
    answering_llm, tmp_path, make_older, error
):
    result = finished_run(answering_llm, tmp_path)
    make_older(result.run_dir)
    sent = len(answering_llm.requests)
    with pytest.raises(ValueError) as refused:
        ask(result, "Who else?", completion=answering_llm)
    message = str(refused.value)
    assert message.startswith(f"can't read {result.run_dir} as a run folder: {error}")
    assert "\n" not in message
    assert len(answering_llm.requests) == sent
    assert not (result.run_dir / "asks").exists()


def test_an_ask_quotes_the_round_one_summaries_its_answer_reaches_through_citations(
    answering_llm, quoted_run
):
    events = []
    asked = ask(
        quoted_run.run_dir,
        "Who else?",
        completion=answering_llm,
        on_event=events.append,
    )
    ids = quoted_run.document_ids
    merrow = {
        "document": ids[1],
        "name": "memos/merrow.txt",
        "start": 0,
        "end": 11,
        "text": "Merrow memo",
    }
    assert without_trail(asked.provenance) == {
        "summary_ids": [ids[0]],
        "parse_ids": [],
        "quotes": [merrow],
    }
    assert asked.provenance == quoted_run.provenance
    assert asked.document_names == quoted_run.document_names == NAMES
    folder = json.loads((asked.run_dir / "reasoning" / "result.json").read_text())
    assert folder["provenance"] == asked.provenance
    [finished] = [e for e in events if e["type"] == "reasoning_finished"]
    assert finished["provenance"] == asked.provenance


def test_a_run_folder_whose_snippets_have_no_quotes_can_be_asked_and_quotes_nothing(
    answering_llm, quoted_run
):
    assert quoted_run.provenance["quotes"]
    in_json("relevance/result.json", snippets_without_quotes)(quoted_run.run_dir)
    asked = ask(quoted_run.run_dir, "Who else?", completion=answering_llm)
    assert asked.answer == "Halloran memo"
    ran, asked_request = answering_llm.requests_for("reasoning")
    assert asked_request["messages"][0] == ran["messages"][0]
    ids = quoted_run.document_ids
    assert without_trail(asked.provenance) == {
        "summary_ids": [ids[0]],
        "parse_ids": [],
        "quotes": [],
    }


def test_a_run_that_never_finished_parsing_cannot_be_asked(answering_llm, tmp_path):
    answering_llm.answers(parsing=bad_key())
    with pytest.raises(litellm.AuthenticationError):
        finished_run(answering_llm, tmp_path)
    [run_dir] = (tmp_path / "logs").iterdir()
    sent = len(answering_llm.requests)
    with pytest.raises(FileNotFoundError) as refused:
        ask(run_dir, "Who else?", completion=answering_llm)
    assert str(refused.value) == (
        f"{run_dir}: the run never finished parsing (structuring/parsing/result.json "
        "is missing), and ask needs its relevance, schema and parsing."
    )
    assert len(answering_llm.requests) == sent
    assert not (run_dir / "asks").exists()


def test_a_run_that_failed_in_reasoning_can_be_asked_over_the_folder_it_started_in(
    answering_llm, tmp_path
):
    reply = {"answer": "Calder memo", "provenance_ids": {}}
    answering_llm.answers(reasoning=[bad_key(), reply])
    events = []
    with pytest.raises(litellm.AuthenticationError):
        finished_run(answering_llm, tmp_path, on_event=events.append)
    started = events[0]
    assert started["type"] == "run_started"
    [only] = (tmp_path / "logs").iterdir()
    assert only == Path(started["run_dir"])
    asked = ask(started["run_dir"], "Who else?", completion=answering_llm)
    assert asked.answer == "Calder memo"


def test_an_answer_without_a_run_folder_cannot_be_asked(answering_llm):
    result = run("Who?", DOCS, completion=answering_llm, save_artifacts=False)
    sent = len(answering_llm.requests)
    with pytest.raises(ValueError) as refused:
        ask(result, "Who else?", completion=answering_llm)
    assert str(refused.value) == (
        "ask needs the run folder, and this Answer has none: its run was made with "
        "save_artifacts=False."
    )
    assert len(answering_llm.requests) == sent


@pytest.fixture
def cited_run(answering_llm, shown_ids, tmp_path) -> Answer:
    """A finished run over ``TWO_LINE_MEMOS`` whose summaries quote a line a round and
    cite the first summary shown, and whose answer cites the first summary and the
    last parse, which cites nothing; ``answering_llm`` answering so from then on."""
    answering_llm.answers(
        relevance=quoting_a_line_a_round(shown_ids),
        reasoning=citing("Halloran memo", slice(1), slice(-1, None)),
    )
    return finished_run(answering_llm, tmp_path, TWO_LINE_MEMOS)


def saved_provenance(folder: Path) -> dict:
    return json.loads((folder / "reasoning" / "result.json").read_text())["provenance"]


def test_trace_gives_the_saved_provenance_and_a_parse_rule_adds_its_documents_only(
    cited_run,
):
    assert trace(cited_run.run_dir) == trace(cited_run) == cited_run.provenance
    assert saved_provenance(cited_run.run_dir) == cited_run.provenance
    last = cited_run.document_ids[-1]
    lit = trace(str(cited_run.run_dir), "parses-light-documents")
    trail = cited_run.provenance["trail"]
    assert lit["trail"]["rule"] == "parses-light-documents"
    assert [e for e in lit["trail"]["edges"] if e not in trail["edges"]] == [
        [f"parse:{last}", f"document:{last}", "parses"]
    ]
    assert [n for n in lit["trail"]["nodes"] if n not in trail["nodes"]] == [
        f"document:{last}"
    ]
    assert without_trail(lit) == without_trail(cited_run.provenance)


def test_trace_of_an_asks_folder_follows_the_asks_citations_through_the_runs_records(
    answering_llm, cited_run
):
    answering_llm.answers(reasoning=citing("Merrow memo", slice(-1, None)))
    asked = ask(cited_run, "Who else?", completion=answering_llm)
    assert trace(asked.run_dir) == trace(asked) == asked.provenance
    assert asked.provenance["trail"] != cited_run.provenance["trail"]


def test_trace_of_a_folder_whose_snippets_have_no_quotes_quotes_nothing(cited_run):
    assert cited_run.provenance["quotes"]
    in_json("relevance/result.json", snippets_without_quotes)(cited_run.run_dir)
    assert trace(cited_run.run_dir) == {**cited_run.provenance, "quotes": []}


def pinning_a_missing_prompt(manifest: dict) -> None:
    manifest["config"]["prompts"]["reasoning/coding"] = "v9"


@pytest.mark.parametrize(
    "make_older",
    [with_one_reasoning_prompt, with_a_seed, pinning_a_missing_prompt],
    ids=["one-reasoning-prompt", "config-with-a-seed", "a-missing-prompt"],
)
def test_trace_reads_a_run_whose_config_or_prompts_this_r3con_no_longer_runs(
    cited_run, make_older
):
    in_json("manifest.json", make_older)(cited_run.run_dir)
    assert trace(cited_run.run_dir) == cited_run.provenance


def a_quote_without_text(result: dict) -> None:
    del result["rounds"][0]["snippets"][0]["quotes"][0]["text"]


@pytest.mark.parametrize(
    ("name", "make_malformed", "error"),
    [
        (
            "structuring/parsing/result.json",
            without_provenance,
            "KeyError: 'provenance'",
        ),
        ("reasoning/result.json", without_provenance, "KeyError: 'provenance'"),
        (
            "structuring/parsing/result.json",
            a_parse_citation_short,
            "ValueError: zip() argument 2 is shorter than argument 1",
        ),
        ("relevance/result.json", a_quote_without_text, "KeyError: 'text'"),
    ],
    ids=[
        "parse-without-provenance",
        "answer-without-provenance",
        "a-parse-citation-short",
        "a-quote-without-text",
    ],
)
def test_trace_refuses_malformed_saved_records_in_one_line(
    cited_run, name, make_malformed, error
):
    in_json(name, make_malformed)(cited_run.run_dir)
    with pytest.raises(ValueError) as refused:
        trace(cited_run.run_dir)
    assert str(refused.value) == (
        f"can't read {cited_run.run_dir} as a run folder: {error}"
    )


def a_quote_placed_by_a_list(result: dict) -> None:
    """A quote's place that breaks only the provenance of an answer whose trail
    reaches it: the first document's last-round quote, which ``cited_run``'s answer
    reaches."""
    quote = result["rounds"][-1]["snippets"][0]["quotes"][0]
    quote["start"] = [quote["start"]]


@pytest.mark.parametrize(
    ("name", "make_malformed", "error", "reason"),
    [
        (
            "structuring/parsing/result.json",
            a_parse_citation_short,
            "ValueError",
            "zip() argument 2 is shorter than argument 1",
        ),
        ("relevance/result.json", a_quote_without_text, "KeyError", "'text'"),
        (
            "relevance/result.json",
            a_quote_placed_by_a_list,
            "TypeError",
            "unhashable type: 'list'",
        ),
    ],
    ids=["a-parse-citation-short", "a-quote-without-text", "a-quote-placed-by-a-list"],
)
def test_an_ask_refuses_malformed_saved_records_in_one_line_before_any_request(
    answering_llm, cited_run, name, make_malformed, error, reason
):
    in_json(name, make_malformed)(cited_run.run_dir)
    sent = len(answering_llm.requests)
    with pytest.raises(ValueError) as refused:
        ask(cited_run, "Who else?", completion=answering_llm)
    message = str(refused.value)
    refusal = f"can't read {cited_run.run_dir} as a run folder: {error}: "
    assert message.startswith(refusal)
    # Python words the reason, and 3.14 reworded the TypeError around these words.
    assert reason in message.removeprefix(refusal)
    assert "\n" not in message
    assert len(answering_llm.requests) == sent
    assert not (cited_run.run_dir / "asks").exists()


@pytest.mark.parametrize(
    ("name", "in_the_ask"),
    [("structuring/parsing/result.json", False), ("reasoning/result.json", True)],
    ids=["the-runs-parse", "the-asks-answer"],
)
def test_trace_of_an_ask_names_the_folder_that_holds_the_malformed_file(
    answering_llm, cited_run, name, in_the_ask
):
    asked = ask(cited_run, "Who else?", completion=answering_llm)
    holder = asked.run_dir if in_the_ask else cited_run.run_dir
    in_json(name, without_provenance)(holder)
    with pytest.raises(ValueError) as refused:
        trace(asked.run_dir)
    assert str(refused.value) == (
        f"can't read {holder} as a run folder: KeyError: 'provenance'"
    )


def test_trace_from_inside_an_asks_folder_reads_the_run_it_asked_over(
    answering_llm, cited_run, monkeypatch
):
    asked = ask(cited_run, "Who else?", completion=answering_llm)
    monkeypatch.chdir(asked.run_dir)
    assert trace(".") == asked.provenance


def test_an_asks_folder_moved_out_of_its_runs_asks_is_refused(
    answering_llm, cited_run, tmp_path
):
    asked = ask(cited_run, "Who else?", completion=answering_llm)
    # In the asks/ of another run folder, which has a manifest but another name.
    other = tmp_path / "other"
    moved = shutil.copytree(asked.run_dir, other / "asks" / asked.run_dir.name)
    shutil.copy(cited_run.run_dir / "manifest.json", other / "manifest.json")
    refusal = (
        f"{moved} is an ask's folder, but not in the asks/ of {cited_run.run_dir.name}, "
        "the run folder it asked over."
    )
    with pytest.raises(FileNotFoundError) as refused:
        trace(moved)
    assert str(refused.value) == refusal
    with pytest.raises(FileNotFoundError) as refused:
        ask(moved, "Who else?", completion=answering_llm)
    assert str(refused.value) == refusal


def test_trace_refuses_an_unknown_rule_before_reading_the_folder(tmp_path):
    with pytest.raises(
        ValueError, match="no provenance rule is named 'through-parses'"
    ):
        trace(tmp_path / "missing", "through-parses")


def test_a_folder_that_is_not_a_run_folder_cannot_be_traced(tmp_path):
    with pytest.raises(FileNotFoundError) as refused:
        trace(tmp_path)
    assert str(refused.value) == (
        f"{tmp_path} is not a run folder: it has no manifest.json."
    )


def test_a_run_whose_reasoning_failed_has_no_answer_to_trace(answering_llm, tmp_path):
    answering_llm.answers(reasoning=bad_key())
    with pytest.raises(litellm.AuthenticationError):
        finished_run(answering_llm, tmp_path)
    [run_dir] = (tmp_path / "logs").iterdir()
    with pytest.raises(FileNotFoundError) as refused:
        trace(run_dir)
    assert str(refused.value) == (
        f"{run_dir} holds no answer: reasoning/result.json is missing."
    )


def test_an_answer_without_a_run_folder_cannot_be_traced(answering_llm):
    result = run("Who?", DOCS, completion=answering_llm, save_artifacts=False)
    with pytest.raises(ValueError) as refused:
        trace(result)
    assert str(refused.value) == (
        "trace needs the run folder, and this Answer has none: its run was made with "
        "save_artifacts=False."
    )


def test_a_reasoning_path_r3con_does_not_have_is_refused_before_an_ask_folder(
    answering_llm, tmp_path
):
    result = finished_run(answering_llm, tmp_path)
    with pytest.raises(ValueError) as refused:
        ask(result, "Who else?", reasoning="agent", completion=answering_llm)
    assert str(refused.value) == (
        "override: `reasoning='agent'`: Input should be 'direct' or 'coding'."
    )
    assert not (result.run_dir / "asks").exists()


def test_a_failed_ask_leaves_its_folder_with_its_error_and_names_it(
    answering_llm, tmp_path
):
    result = finished_run(answering_llm, tmp_path)
    answering_llm.answers(reasoning=bad_key())
    with pytest.raises(litellm.AuthenticationError) as failed:
        ask(result, "Who else?", completion=answering_llm)
    [folder] = (result.run_dir / "asks").iterdir()
    assert failed.value.__notes__ == [f"r3con: partial artifacts in {folder}"]
    assert (folder / "manifest.json").is_file()
    assert "bad key" in (folder / "reasoning" / "error.txt").read_text()
    assert not (folder / "reasoning" / "result.json").exists()


def test_an_ask_reports_its_own_events_into_its_folder(
    answering_llm, events_in, tmp_path
):
    ran = []
    built = finished_run(answering_llm, tmp_path, on_event=ran.append)
    asked = []
    answer = ask(built, "Who else?", completion=answering_llm, on_event=asked.append)
    assert [event["type"] for event in asked] == [
        "ask_started",
        "reasoning_started",
        "reasoning_finished",
        "ask_finished",
    ]
    started, reasoning, answered, finished = asked
    folder = Path(started["ask_dir"])
    assert folder == answer.run_dir
    assert folder.parent == built.run_dir / "asks"
    assert (started["question"], started["model"]) == ("Who else?", "openai/gpt-6-luna")
    assert (reasoning["question"], reasoning["model"]) == (
        "Who else?",
        "openai/gpt-6-luna",
    )
    assert answered["answer"] == JOINED
    assert finished["ask_dir"] == started["ask_dir"]
    assert finished["tokens"] == {"prompt": 10, "completion": 20, "total": 30}
    assert events_in(folder) == asked
    assert events_in(built.run_dir) == ran


def test_a_failed_ask_reports_ask_failed_into_its_folder(
    answering_llm, events_in, tmp_path
):
    built = finished_run(answering_llm, tmp_path)
    answering_llm.answers(reasoning=bad_key())
    events = []
    with pytest.raises(litellm.AuthenticationError):
        ask(built, "Who else?", completion=answering_llm, on_event=events.append)
    assert [event["type"] for event in events] == [
        "ask_started",
        "reasoning_started",
        "ask_failed",
    ]
    assert (events[-1]["stage"], events[-1]["error"]) == (
        "reasoning",
        "AuthenticationError: litellm.AuthenticationError: bad key",
    )
    assert events_in(Path(events[0]["ask_dir"])) == events


@pytest.mark.parametrize(
    "options",
    [
        {"model": COPILOT},
        {"model": "gpt-x", "params": {"custom_llm_provider": "github_copilot"}},
    ],
    ids=["by-its-prefix", "by-custom-llm-provider"],
)
def test_a_first_run_on_a_model_that_signs_in_starts_one_sign_in(
    answering_llm, tmp_path, monkeypatch, options
):
    sign_ins = []
    cached = []

    def get_api_key(self: CopilotAuthenticator) -> str:
        """litellm's Copilot key: the cached one, else a new device sign-in's, once
        the user has entered its code."""
        if cached:
            return cached[0]
        sign_ins.append(threading.current_thread().name)
        time.sleep(USER_ENTERS_CODE_SECONDS)
        cached.append("copilot-key")
        return cached[0]

    monkeypatch.setattr(CopilotAuthenticator, "get_api_key", get_api_key)
    result = finished_run(answering_llm, tmp_path, **options)
    assert len(sign_ins) == 1
    assert result.answer == JOINED


def test_an_ordinary_models_first_calls_are_all_in_flight_at_once(
    answering_llm, tmp_path
):
    all_sent = threading.Barrier(len(DOCS), timeout=BARRIER_TIMEOUT_SECONDS)

    def note_once_all_are_sent(request: dict[str, Any]) -> str:
        all_sent.wait()
        return "Notes."

    answering_llm.answers(relevance=note_once_all_are_sent)
    assert finished_run(answering_llm, tmp_path).answer == JOINED


def test_a_run_cancelled_during_relevance_sends_no_further_call_and_fails_there(
    answering_llm, tmp_path
):
    cancel = threading.Event()
    # Cancelled once every document's first read is in flight.
    in_flight = threading.Barrier(
        len(DOCS), action=cancel.set, timeout=BARRIER_TIMEOUT_SECONDS
    )

    def read_while_cancelled(request: dict[str, Any]) -> str:
        in_flight.wait()
        return "Notes."

    answering_llm.answers(relevance=read_while_cancelled)
    events = []
    with pytest.raises(r3con.Cancelled) as cancelled:
        finished_run(answering_llm, tmp_path, on_event=events.append, cancel=cancel)
    assert len(answering_llm.requests) == len(DOCS)
    types = [event["type"] for event in events]
    assert types.count("relevance_finished") == len(DOCS)
    assert {**events[-1], "time": None} == {
        "type": "run_failed",
        "time": None,
        "stage": "relevance",
        "error": CANCELLED,
    }
    run_dir = Path(events[0]["run_dir"])
    assert cancelled.value.__notes__ == [f"r3con: partial artifacts in {run_dir}"]
    calls = json.loads((run_dir / "relevance" / "calls.json").read_text())
    assert [call["error"] for call in calls] == [None] * len(DOCS)
    assert CANCELLED in (run_dir / "relevance" / "error.txt").read_text()


def test_a_run_cancelled_during_a_stages_last_call_stops_as_the_next_one_starts(
    answering_llm, tmp_path
):
    cancel = threading.Event()

    def schema_while_cancelled(request: dict[str, Any]) -> str:
        cancel.set()
        return ANSWERING_SCHEMA

    answering_llm.answers(schema=schema_while_cancelled)
    events = []
    with pytest.raises(r3con.Cancelled):
        finished_run(answering_llm, tmp_path, on_event=events.append, cancel=cancel)
    types = [event["type"] for event in events]
    assert types[types.index("schema_finished") :] == ["schema_finished", "run_failed"]
    assert events[-1]["stage"] == "parsing"
    parsing = Path(events[0]["run_dir"]) / "structuring" / "parsing"
    assert CANCELLED in (parsing / "error.txt").read_text()


def test_an_ask_cancelled_during_reasoning_sends_no_further_call_and_fails_there(
    answering_llm, tmp_path
):
    built = finished_run(answering_llm, tmp_path)
    sent = len(answering_llm.requests)
    cancel = threading.Event()

    def answer_while_cancelled(request: dict[str, Any]) -> str:
        cancel.set()
        return "not an answer"  # a reply that does not fit, which is sent again

    answering_llm.answers(reasoning=answer_while_cancelled)
    events = []
    with pytest.raises(r3con.Cancelled):
        ask(
            built,
            "Who else?",
            completion=answering_llm,
            on_event=events.append,
            cancel=cancel,
        )
    assert len(answering_llm.requests) == sent + 1
    failed = events[-1]
    assert (failed["type"], failed["stage"], failed["error"]) == (
        "ask_failed",
        "reasoning",
        CANCELLED,
    )
    folder = Path(events[0]["ask_dir"])
    assert CANCELLED in (folder / "reasoning" / "error.txt").read_text()


def test_without_a_run_folder_only_the_callback_gets_events(answering_llm, tmp_path):
    events = []
    run(
        "Who?",
        DOCS,
        completion=answering_llm,
        save_artifacts=False,
        on_event=events.append,
    )
    assert (events[0]["type"], events[-1]["type"]) == ("run_started", "run_finished")
    assert all(json.loads(json.dumps(event)) == event for event in events)
    finished = events[-1]
    assert (finished["run_dir"], finished["tokens"], finished["usd"]) == (None,) * 3
    assert list(tmp_path.iterdir()) == []


def test_run_is_reachable_from_the_package_and_from_its_module():
    assert r3con.run is namespace.run
    assert r3con.ask is namespace.ask
    assert r3con.trace is namespace.trace
    assert r3con.read_documents is namespace.read_documents
    assert Answer is namespace.Answer
    assert r3con.Cancelled is namespace.Cancelled


def test_the_version_lookup_leaves_nothing_in_the_package_namespace():
    assert not hasattr(r3con, "version")
    assert not hasattr(r3con, "PackageNotFoundError")


def test_the_version_is_the_installed_distributions():
    assert r3con.__version__ == importlib.metadata.version("r3context")


@pytest.mark.parametrize("given", ["api_key", "environment"])
def test_a_failed_run_writes_and_logs_no_key_but_keeps_the_documents_text(
    answering_llm, events_in, monkeypatch, tmp_path, caplog, given
):
    caplog.set_level(logging.WARNING, logger="r3con")
    if given == "environment":
        monkeypatch.setenv("OPENAI_API_KEY", KEY)
    answering_llm.replies(
        litellm.InternalServerError(**ECHOING), litellm.AuthenticationError(**ECHOING)
    )
    monkeypatch.setattr(litellm, "completion", answering_llm)
    options = {"api_key": KEY} if given == "api_key" else {}

    with pytest.raises(litellm.AuthenticationError):
        run("Who?", [NAMES_A_KEY], logs_dir=tmp_path / "logs", **options)

    [run_dir] = (tmp_path / "logs").iterdir()
    assert files_holding(KEY, run_dir) == []
    assert KEY not in caplog.text
    [call] = json.loads((run_dir / "relevance" / "calls.json").read_text())
    assert call["error"] == REFUSED_LINE
    assert events_in(run_dir)[-1]["error"] == REFUSED_LINE
    assert REFUSED_LINE in (run_dir / "relevance" / "error.txt").read_text()
    assert "Incorrect API key provided: REDACTED" in caplog.text
    assert NAMES_A_KEY in call["prompt"][1]["content"]


@pytest.mark.parametrize("sent_by", ["router", "extra_headers"])
def test_a_key_a_router_or_a_header_sends_is_in_no_file_or_event_of_a_failed_run(
    answering_llm, monkeypatch, router, tmp_path, sent_by
):
    monkeypatch.setattr(litellm, "completion", answering_llm)
    answering_llm.replies(litellm.AuthenticationError(**ECHOING))
    if sent_by == "router":
        routing = router({"model": "openai/gpt-6-luna", "api_key": KEY})
        options: dict[str, Any] = {"completion": routing.completion}
    else:
        options = {"params": {"extra_headers": {"x-gateway-auth": KEY}}}
    events = []

    with pytest.raises(litellm.AuthenticationError):
        run(
            "Who?",
            DOCS[:1],
            logs_dir=tmp_path / "logs",
            on_event=events.append,
            **options,
        )

    [run_dir] = (tmp_path / "logs").iterdir()
    assert files_holding(KEY, run_dir) == []
    assert KEY not in repr(events)
    assert events[-1]["error"].startswith(REFUSED_LINE)


def test_an_asks_manifest_holds_no_credential_of_its_params(answering_llm, tmp_path):
    built = finished_run(answering_llm, tmp_path)
    headers = {"extra_headers": {"x-gateway-auth": KEY}}
    asked = ask(built, "Who else?", params=headers, completion=answering_llm)
    assert files_holding(KEY, asked.run_dir) == []
    manifest = json.loads((asked.run_dir / "manifest.json").read_text())
    assert manifest["config"]["params"] == {
        "extra_headers": {"x-gateway-auth": "REDACTED"}
    }


def test_an_ask_refuses_to_send_a_header_its_runs_manifest_saved_masked(
    answering_llm, tmp_path
):
    built = finished_run(answering_llm, tmp_path, params=GATEWAY_HEADER)
    sent = len(answering_llm.requests)
    with pytest.raises(ValueError) as refused:
        ask(built, "Who else?", completion=answering_llm)
    assert str(refused.value) == MASKED_HEADER_REFUSAL
    assert answering_llm.requests[sent:] == []
    assert not (built.run_dir / "asks").exists()


@pytest.mark.parametrize(
    ("saved", "options"),
    [
        (GATEWAY_HEADER, {"params": GATEWAY_HEADER}),
        ({"api_key": KEY}, {"api_key": KEY}),
    ],
    ids=["a-header-in-params", "an-api-key-as-its-own-argument"],
)
def test_an_ask_sends_the_real_value_its_caller_gives_for_a_masked_param(
    answering_llm, tmp_path, saved, options
):
    built = finished_run(answering_llm, tmp_path, params=saved)
    manifest = json.loads((built.run_dir / "manifest.json").read_text())
    assert "REDACTED" in json.dumps(manifest["config"]["params"])
    ask(built, "Who else?", completion=answering_llm, **options)
    asked = answering_llm.requests_for("reasoning")[-1]
    assert {name: asked[name] for name in saved} == saved


@pytest.mark.parametrize(
    ("saved", "path"),
    [
        ({"extra_body": {"api_key": "REDACTED"}}, "extra_body.api_key"),
        ({"api_base": "https://REDACTED@gateway.example"}, "api_base"),
        ({"fallbacks": [{"api_key": "REDACTED"}]}, "fallbacks.0.api_key"),
        (
            {
                "vertex_credentials": {
                    "private_key": "REDACTED",
                    "client_id": "REDACTED",
                }
            },
            "vertex_credentials",
        ),
    ],
    ids=["nested-in-a-dict", "inside-a-longer-value", "in-a-list", "twice-named-once"],
)
def test_an_ask_names_the_path_to_a_masked_param_or_the_param_alone_if_it_holds_more(
    answering_llm, tmp_path, saved, path
):
    built = finished_run(answering_llm, tmp_path)
    manifest_path = built.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["config"]["params"] = saved
    manifest_path.write_text(json.dumps(manifest))
    sent = len(answering_llm.requests)
    with pytest.raises(ValueError) as refused:
        ask(built, "Who else?", completion=answering_llm)
    assert str(refused.value).endswith(f" Masked params: `{path}`.")
    assert answering_llm.requests[sent:] == []


# --- resume ---

MEMOS_BY_NAME = {
    f"memos/{line.split()[0].lower()}.txt": line
    for line in [
        "Halloran memo",
        "Merrow memo",
        "Calder memo",
        "Dunmore memo",
        "Easton memo",
    ]
}
LAST_THREE = ["Calder memo", "Dunmore memo", "Easton memo"]
BUILT_STAGES = ("relevance", "structuring/schema", "structuring/parsing")
# A request this long is refused as too long: the grown registry is read in 4 parts.
REFUSED_OVER_CHARS = 18_000
# Enough of an event's line to leave it cut short, as a kill does.
TORN_CHARS = 60
STRICT_DATED_SCHEMA = """Thought: one row per memo, with its date.
<schema>
from datetime import date
from pydantic import ConfigDict


class Row(BaseModel):
    model_config = ConfigDict(strict=True)
    who: str
    on: date


class Parse(BaseModel):
    model_config = ConfigDict(strict=True)
    rows: list[Row]
</schema>"""


@pytest.fixture
def one_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Documents read one after another, so a failure leaves those before it done."""
    monkeypatch.setenv("R3CON_DOC_WORKERS", "1")


def failed_run(llm, tmp_path: Path, documents=MEMOS_BY_NAME, **options) -> Path:
    """The folder of a run over ``documents`` that ``llm`` fails."""
    with pytest.raises(litellm.AuthenticationError):
        finished_run(llm, tmp_path, documents, **options)
    [run_dir] = (tmp_path / "logs").iterdir()
    return run_dir


def resumed(llm, run_dir: Path, documents=MEMOS_BY_NAME, **options):
    """The answer of resuming ``run_dir`` over ``documents``, and the requests it sent,
    each as its stage and, for a document's, the document's first line."""
    sent = len(llm.requests)
    answer = resume(run_dir, documents, completion=llm, **options)
    requests = [
        (stage, first_line(r) if stage in ("relevance", "parsing") else None)
        for r in llm.requests[sent:]
        if (stage := stage_of(r))
    ]
    return answer, requests


def as_named(answer: Answer) -> tuple:
    """What ``answer`` says, its documents named rather than given by id, which each
    run draws afresh."""
    names = dict(zip(answer.document_ids, answer.document_names, strict=True))
    rows = [
        {**row, "document": names[row["document"]]}
        for row in answer.structured_context["rows"]
    ]
    return (
        answer.answer,
        answer.relevant_context,
        answer.source_docs,
        answer.schema_code,
        rows,
        answer.document_names,
    )


def as_killed(run_dir: Path, stage: str, keep: int) -> list[str]:
    """``run_dir`` as a kill during ``stage`` leaves it: the stage's folder without its
    calls or error, and ``events.jsonl`` its first ``keep`` lines, which it returns."""
    for name in ("calls.json", "error.txt"):
        (run_dir / stage / name).unlink()
    kept = (run_dir / "events.jsonl").read_text().splitlines()[:keep]
    rewritten(run_dir, kept)
    return kept


def line_of(events: list[dict], kind: str, nth: int = 0, **fields) -> int:
    """The position among ``events`` of the ``nth`` one of type ``kind`` with
    ``fields``."""
    return [
        at
        for at, event in enumerate(events)
        if event["type"] == kind and fields.items() <= event.items()
    ][nth]


def rewritten(run_dir: Path, lines: list[str]) -> None:
    """``run_dir``'s ``events.jsonl`` written again as ``lines``."""
    (run_dir / "events.jsonl").write_text("".join(f"{line}\n" for line in lines))


def untouched(run_dir: Path, stages) -> dict[str, tuple]:
    """Each file of ``stages``' folders, by path, with its bytes and when it was last
    written."""
    return {
        str(path.relative_to(run_dir)): (path.read_bytes(), path.stat().st_mtime_ns)
        for stage in stages
        for path in (run_dir / stage).iterdir()
    }


@pytest.mark.parametrize(
    ("stage", "when", "resent", "finished"),
    [
        (
            "relevance",
            reading("Calder memo", later_round=True),
            [
                *[("relevance", line) for line in LAST_THREE],
                ("schema", None),
                *[("parsing", line) for line in MEMOS_BY_NAME.values()],
                ("reasoning", None),
            ],
            (),
        ),
        ("reasoning", lambda request: True, [("reasoning", None)], BUILT_STAGES),
    ],
    ids=["in-relevance-round-2", "in-reasoning"],
)
def test_a_failed_run_resumes_sending_only_what_it_lacks_and_answers_as_a_whole_run(
    answering_llm, one_worker, tmp_path, stage, when, resent, finished
):
    whole = finished_run(answering_llm, tmp_path / "whole", MEMOS_BY_NAME)
    llm = answering_llm.answers(**{stage: failing_once(when, ANSWERING[stage])})
    run_dir = failed_run(llm, tmp_path / "failed")
    before = untouched(run_dir, finished)
    answer, requests = resumed(llm, run_dir)
    assert requests == resent
    assert as_named(answer) == as_named(whole)
    assert answer.run_dir == run_dir
    assert untouched(run_dir, finished) == before


def test_a_resumed_stage_keeps_its_earlier_calls_and_leaves_the_finished_ones_alone(
    answering_llm, one_worker, tmp_path
):
    failing = failing_once(reading("Calder memo"), ANSWERING["parsing"])
    llm = answering_llm.answers(parsing=failing)
    run_dir = failed_run(llm, tmp_path)
    before = untouched(run_dir, BUILT_STAGES[:2])
    _, requests = resumed(llm, run_dir)
    assert requests == [
        *[("parsing", line) for line in LAST_THREE],
        ("reasoning", None),
    ]
    assert untouched(run_dir, BUILT_STAGES[:2]) == before
    calls = json.loads((run_dir / "structuring/parsing/calls.json").read_text())
    assert [(c["step"], c["kind"], c["error"] is None) for c in calls] == [
        (1, "parse-d0", True),
        (2, "parse-d1", True),
        (3, "parse-d2", False),
        (4, "parse-d2", True),
        (5, "parse-d3", True),
        (6, "parse-d4", True),
    ]
    result = json.loads((run_dir / "structuring/parsing/result.json").read_text())
    assert result["totals"]["n_steps"] == 5
    assert not (run_dir / "structuring/parsing/error.txt").exists()
    [resumed_at] = json.loads((run_dir / "manifest.json").read_text())["resumes"]
    assert resumed_at["r3con_version"] == r3con.__version__


def test_a_resume_reports_what_it_reuses_then_the_events_that_reported_it(
    answering_llm, one_worker, events_in, tmp_path
):
    failing = failing_once(reading("Calder memo"), ANSWERING["parsing"])
    llm = answering_llm.answers(parsing=failing)
    run_dir = failed_run(llm, tmp_path)
    before = events_in(run_dir)
    calder = before[0]["document_ids"][2]
    seen = []
    resume(run_dir, MEMOS_BY_NAME, completion=llm, on_event=seen.append)
    started, reuse, *rest = seen
    assert (started["type"], started["run_dir"]) == ("run_started", str(run_dir))
    assert reuse["type"] == "run_resumed"
    assert reuse["reused"] == {"relevance": [5, 5], "schema": 1, "parsing": 2}
    replayed = [
        event
        for event in before
        if event["type"]
        in ("relevance_finished", "documents_pruned", "schema_finished")
        or (event["type"] == "parsing_finished" and event["doc_id"] != calder)
    ]
    assert rest[: len(replayed)] == replayed
    new = rest[len(replayed) :]
    assert new[-1]["type"] == "run_finished"
    assert events_in(run_dir) == [*before, started, reuse, *new]


def test_an_event_that_differs_from_the_result_reused_is_not_replayed(
    answering_llm, events_in, tmp_path
):
    run_dir = failed_run(answering_llm.answers(parsing=bad_key()), tmp_path)
    events = events_in(run_dir)
    at = line_of(events, "schema_finished")
    events[at] = {**events[at], "schema_code": "class Parse: ..."}
    rewritten(run_dir, [json.dumps(event) for event in events])
    seen = []
    answering_llm.answers(parsing=ANSWERING["parsing"])
    resume(run_dir, MEMOS_BY_NAME, completion=answering_llm, on_event=seen.append)
    replayed = [event["type"] for event in seen]
    assert "schema_finished" not in replayed
    assert replayed.count("relevance_finished") == len(MEMOS_BY_NAME) * 2


def test_a_killed_run_resumes_from_its_events_and_reads_a_line_cut_short_again(
    answering_llm, one_worker, events_in, tmp_path, caplog
):
    failing = failing_once(reading("Easton memo"), ANSWERING["parsing"])
    llm = answering_llm.answers(parsing=failing)
    run_dir = failed_run(llm, tmp_path)
    dunmore = line_of(events_in(run_dir), "parsing_finished", 3)
    kept = as_killed(run_dir, "structuring/parsing", keep=dunmore + 1)
    torn = kept[-1][:TORN_CHARS]
    (run_dir / "events.jsonl").write_text("\n".join([*kept[:-1], torn]))
    answer, requests = resumed(llm, run_dir)
    assert requests == [
        ("parsing", "Dunmore memo"),
        ("parsing", "Easton memo"),
        ("reasoning", None),
    ]
    assert answer.answer == "; ".join(MEMOS_BY_NAME.values())
    written = (run_dir / "events.jsonl").read_text().splitlines()
    assert written[dunmore] == torn
    assert json.loads(written[dunmore + 1])["type"] == "run_started"
    calls = json.loads((run_dir / "structuring/parsing/calls.json").read_text())
    assert [(c["step"], c["kind"]) for c in calls] == [(1, "parse-d3"), (2, "parse-d4")]
    assert "1 line(s) of events.jsonl don't parse" in caplog.text


def test_a_snippet_holding_other_line_separators_is_reused(
    answering_llm, one_worker, tmp_path
):
    def notes(request):
        """Halloran's note holds a line separator and a next-line character."""
        line = first_line(request)
        across = ",\u2028across\x85lines" if line == "Halloran memo" else ""
        return f"Notes on {line}{across}."

    failing = failing_once(reading("Calder memo", later_round=True), notes)
    run_dir = failed_run(answering_llm.answers(relevance=failing), tmp_path)
    _, requests = resumed(answering_llm, run_dir)
    assert [r for r in requests if r[0] == "relevance"] == [
        ("relevance", line) for line in LAST_THREE
    ]


def reading_in_round(line: str, round_: int):
    """Whether a relevance request reads the document whose first line is ``line`` in
    round ``round_``, from 2, counting its reads in later rounds as they come."""
    later = itertools.count(2)
    return lambda request: (
        reading(line, later_round=True)(request) and next(later) == round_
    )


UNPRUNED = [line for line in MEMOS_BY_NAME.values() if line != "Dunmore memo"]
DUNMORE_SILENT_AFTER_ROUND_1 = silent_when(reading("Dunmore memo", later_round=True))


def as_left_before_pruning(result: dict) -> None:
    """Relevance's ``result.json`` over MEMOS_BY_NAME as r3context 0.6.25 wrote it:
    no pruned documents listed, and Dunmore's memo, empty in round 2, read again in
    round 3."""
    del result["pruned"]
    result["rounds"][2]["snippets"][3]["summary"] = "Notes on Dunmore memo."


@pytest.mark.parametrize(
    ("rounds", "stage", "when", "resent"),
    [
        (
            2,
            "parsing",
            reading("Calder memo"),
            [
                ("parsing", "Calder memo"),
                ("parsing", "Easton memo"),
                ("reasoning", None),
            ],
        ),
        (
            4,
            "relevance",
            reading_in_round("Calder memo", 4),
            [
                ("relevance", "Calder memo"),
                ("relevance", "Easton memo"),
                ("schema", None),
                *[("parsing", line) for line in UNPRUNED],
                ("reasoning", None),
            ],
        ),
    ],
    ids=["in-parsing", "in-relevance-round-4"],
)
def test_a_resume_prunes_what_the_run_pruned_and_never_reads_or_parses_it(
    answering_llm, one_worker, tmp_path, rounds, stage, when, resent
):
    llm = answering_llm.answers(relevance=DUNMORE_SILENT_AFTER_ROUND_1)
    whole = finished_run(
        llm, tmp_path / "whole", MEMOS_BY_NAME, relevance_rounds=rounds
    )
    replies = {**ANSWERING, "relevance": DUNMORE_SILENT_AFTER_ROUND_1}
    llm.answers(**{stage: failing_once(when, replies[stage])})
    run_dir = failed_run(llm, tmp_path / "failed", relevance_rounds=rounds)
    answer, requests = resumed(llm, run_dir)
    assert requests == resent
    assert as_named(answer) == as_named(whole)
    assert answer.answer == "; ".join(UNPRUNED)


def test_a_run_from_before_pruning_resumes_parsing_every_document(
    answering_llm, one_worker, events_in, tmp_path
):
    llm = answering_llm.answers(
        relevance=silent_when(reading_in_round("Dunmore memo", 2)),
        parsing=failing_once(reading("Calder memo"), ANSWERING["parsing"]),
    )
    run_dir = failed_run(llm, tmp_path, relevance_rounds=3)
    events = [e for e in events_in(run_dir) if e["type"] != "documents_pruned"]
    rewritten(run_dir, [json.dumps(event) for event in events])
    in_json("relevance/result.json", as_left_before_pruning)(run_dir)
    answer, requests = resumed(llm, run_dir)
    assert requests == [
        *[("parsing", line) for line in LAST_THREE],
        ("reasoning", None),
    ]
    assert answer.answer == "; ".join(MEMOS_BY_NAME.values())


def test_a_run_from_before_pruning_stopped_in_round_3_resumes_pruning_nothing(
    answering_llm, one_worker, events_in, tmp_path
):
    failing = failing_once(
        reading_in_round("Calder memo", 3),
        silent_when(reading_in_round("Dunmore memo", 2)),
    )
    llm = answering_llm.answers(relevance=failing)
    run_dir = failed_run(llm, tmp_path, relevance_rounds=3)
    events = [e for e in events_in(run_dir) if e["type"] != "documents_pruned"]
    rewritten(run_dir, [json.dumps(event) for event in events])
    seen = []
    answer, requests = resumed(llm, run_dir, on_event=seen.append)
    assert requests == [
        *[("relevance", line) for line in LAST_THREE],
        ("schema", None),
        *[("parsing", line) for line in MEMOS_BY_NAME.values()],
        ("reasoning", None),
    ]
    assert answer.answer == "; ".join(MEMOS_BY_NAME.values())
    written = [e["type"] for e in events_in(run_dir)]
    assert "documents_pruned" not in [e["type"] for e in seen] + written


def test_a_resume_after_round_2_replays_its_pruning_once_before_the_later_rounds(
    answering_llm, one_worker, events_in, tmp_path
):
    failing = failing_once(
        reading_in_round("Calder memo", 3), DUNMORE_SILENT_AFTER_ROUND_1
    )
    llm = answering_llm.answers(relevance=failing)
    run_dir = failed_run(llm, tmp_path, relevance_rounds=3)
    seen = []
    resume(run_dir, MEMOS_BY_NAME, completion=llm, on_event=seen.append)
    steps = [(e["type"], e.get("round")) for e in seen]
    pruning = ("documents_pruned", None)
    assert steps.count(pruning) == 1
    assert steps.index(pruning) < steps.index(("relevance_finished", 3))
    written = [e["type"] for e in events_in(run_dir)]
    assert written.count("documents_pruned") == 1


def test_a_snippet_read_over_a_round_redone_since_is_read_again(
    answering_llm, one_worker, events_in, tmp_path
):
    llm = answering_llm.answers(
        relevance=failing_once(
            reading("Calder memo", later_round=True), ANSWERING["relevance"]
        )
    )
    run_dir = failed_run(llm, tmp_path)
    # Easton's round-1 line is lost: a resume reads it again, and round 2 over it,
    # until it fails at Merrow, after Halloran.
    events = events_in(run_dir)
    del events[line_of(events, "relevance_finished", 4, round=1)]
    rewritten(run_dir, [json.dumps(event) for event in events])
    llm.answers(
        relevance=failing_once(
            reading("Merrow memo", later_round=True), ANSWERING["relevance"]
        )
    )
    with pytest.raises(litellm.AuthenticationError):
        resume(run_dir, MEMOS_BY_NAME, completion=llm)
    llm.answers(relevance=ANSWERING["relevance"])
    _, requests = resumed(llm, run_dir)
    # Merrow's first round-2 snippet was read over the Easton since redone.
    assert [r for r in requests if r[0] == "relevance"] == [
        ("relevance", line) for line in ["Merrow memo", *LAST_THREE]
    ]


def test_a_document_read_in_parts_resumes_in_the_same_parts_with_the_same_ids(
    answering_llm, one_worker, grown_registry, tmp_path
):
    documents = ["Halloran memo", grown_registry(40_000), "Merrow memo"]
    failing = failing_once(
        reading("Merrow memo", later_round=True), ANSWERING["relevance"]
    )
    llm = answering_llm.answers(relevance=failing).refuses_over(REFUSED_OVER_CHARS)
    run_dir = failed_run(llm, tmp_path, documents, model=QWEN)
    sent = len(llm.requests)
    answer, requests = resumed(llm, run_dir, documents)
    assert [r for r in requests if r[0] == "relevance"] == [
        ("relevance", "Merrow memo")
    ]
    assert all(
        sum(len(m["content"]) for m in r["messages"]) <= REFUSED_OVER_CHARS
        for r in llm.requests[sent:]
    )
    registry = answer.document_ids[1]
    rounds = json.loads((run_dir / "relevance/result.json").read_text())["rounds"]
    parts = [f"{registry}.{k}" for k in range(1, 5)]
    for round_ in rounds:
        ids = [snippet["id"] for snippet in round_["snippets"]]
        assert ids == [answer.document_ids[0], *parts, answer.document_ids[2]]


def test_a_document_whose_parts_were_not_all_saved_is_read_again_in_that_round(
    answering_llm, one_worker, events_in, grown_registry, tmp_path
):
    registry = grown_registry(40_000)
    documents = ["Halloran memo", registry, "Merrow memo"]
    failing = failing_once(
        reading("Merrow memo", later_round=True), ANSWERING["relevance"]
    )
    llm = answering_llm.answers(relevance=failing).refuses_over(REFUSED_OVER_CHARS)
    run_dir = failed_run(llm, tmp_path, documents, model=QWEN)
    third_part = line_of(events_in(run_dir), "relevance_finished", 2, round=2)
    as_killed(run_dir, "relevance", keep=third_part)
    sent = len(llm.requests)
    resume(run_dir, documents, completion=llm)
    round_2 = [
        r["messages"][1]["content"]
        for r in llm.requests[sent:]
        if stage_of(r) == "relevance" and is_later_round(r)
    ]
    assert sum(part in registry for part in round_2) == 4
    assert [part for part in round_2 if part not in registry] == ["Merrow memo"]


def test_a_round_whose_snippets_were_not_all_saved_has_every_later_round_read_again(
    answering_llm, one_worker, events_in, tmp_path, caplog
):
    failing = failing_once(
        reading("Calder memo", later_round=True), ANSWERING["relevance"]
    )
    llm = answering_llm.answers(relevance=failing)
    run_dir = failed_run(llm, tmp_path)
    # Easton's round-1 line is lost, as when writing it failed: round 2's saved
    # snippets were written over a round 1 the resume reads again.
    events = events_in(run_dir)
    del events[line_of(events, "relevance_finished", 4, round=1)]
    rewritten(run_dir, [json.dumps(event) for event in events])
    _, requests = resumed(llm, run_dir)
    assert [r for r in requests if r[0] == "relevance"] == [
        ("relevance", "Easton memo"),
        *[("relevance", line) for line in MEMOS_BY_NAME.values()],
    ]
    assert "2 saved result(s) can't be reused" in caplog.text


def test_a_snippet_with_a_quote_r3con_did_not_write_is_read_again(
    answering_llm, one_worker, events_in, tmp_path
):
    failing = failing_once(
        reading("Calder memo", later_round=True), ANSWERING["relevance"]
    )
    llm = answering_llm.answers(relevance=failing)
    run_dir = failed_run(llm, tmp_path)
    events = events_in(run_dir)
    at = line_of(events, "relevance_finished", 0, round=2)
    events[at] = {**events[at], "quotes": [{"text": "Halloran", "start": "0"}]}
    rewritten(run_dir, [json.dumps(event) for event in events])
    _, requests = resumed(llm, run_dir)
    assert [r for r in requests if r[0] == "relevance"] == [
        ("relevance", line) for line in ["Halloran memo", *LAST_THREE]
    ]


def test_a_parse_saved_under_a_strict_schema_is_reused(
    answering_llm, one_worker, tmp_path
):
    def dated_row(request):
        return {"rows": [{"who": first_line(request), "on": "2026-03-01"}]}

    llm = answering_llm.answers(
        schema=STRICT_DATED_SCHEMA,
        parsing=failing_once(reading("Calder memo"), dated_row),
    )
    run_dir = failed_run(llm, tmp_path)
    _, requests = resumed(llm, run_dir)
    assert [r for r in requests if r[0] == "parsing"] == [
        ("parsing", line) for line in LAST_THREE
    ]


@pytest.mark.parametrize("stage", ["relevance", "parsing"])
def test_a_resume_on_a_model_that_signs_in_starts_one_sign_in(
    answering_llm, one_worker, tmp_path, monkeypatch, stage
):
    sign_ins, cached = [], []

    def get_api_key(self: CopilotAuthenticator) -> str:
        if not cached:
            sign_ins.append(threading.current_thread().name)
            time.sleep(USER_ENTERS_CODE_SECONDS)
            cached.append("copilot-key")
        return cached[0]

    monkeypatch.setattr(CopilotAuthenticator, "get_api_key", get_api_key)
    when = reading("Calder memo", later_round=stage == "relevance")
    llm = answering_llm.answers(**{stage: failing_once(when, ANSWERING[stage])})
    run_dir = failed_run(llm, tmp_path, model=COPILOT)
    # A new process: no token yet, and the documents read at once.
    sign_ins.clear()
    cached.clear()
    monkeypatch.setattr("r3con.runtime.llm._signed_in", set())
    monkeypatch.delenv("R3CON_DOC_WORKERS")
    resume(run_dir, MEMOS_BY_NAME, completion=llm)
    assert len(sign_ins) == 1


def test_a_run_still_going_in_its_folder_is_not_resumed(answering_llm, tmp_path):
    refusals = []

    def resume_meanwhile(event):
        if event["type"] != "schema_started":
            return
        [run_dir] = (tmp_path / "logs").iterdir()
        with pytest.raises(RuntimeError) as refused:
            resume(run_dir, MEMOS_BY_NAME, completion=answering_llm)
        refusals.append(str(refused.value))

    answer = finished_run(
        answering_llm, tmp_path, MEMOS_BY_NAME, on_event=resume_meanwhile
    )
    assert refusals == [
        (
            f"{answer.run_dir}: a run or resume is going in it; carry it on once that "
            "one has stopped."
        )
    ]


@pytest.mark.parametrize(
    "error",
    [NotImplementedError("flock"), OSError(errno.ENOSYS, "Function not implemented")],
    ids=["filelock-without-a-fallback", "the-file-system"],
)
def test_a_run_on_a_file_system_without_locks_goes_on_with_a_warning(
    answering_llm, tmp_path, monkeypatch, caplog, error
):
    class Unlockable:
        def __init__(self, path, timeout):
            pass

        def acquire(self):
            raise error

    monkeypatch.setattr("r3con.runs.FileLock", Unlockable)
    answer = finished_run(answering_llm, tmp_path, MEMOS_BY_NAME)
    assert answer.answer == "; ".join(MEMOS_BY_NAME.values())
    assert "can't be locked" in caplog.text


def test_a_run_that_finished_while_its_resume_waited_is_not_resumed(
    answering_llm, tmp_path, monkeypatch
):
    run_dir = failed_run(answering_llm.answers(reasoning=bad_key()), tmp_path)
    sent = len(answering_llm.requests)

    class FinishingFirst(FileLock):
        """The run's own lock, held until the run finished."""

        def acquire(self, *args, **kwargs):
            shutil.copy(run_dir / "manifest.json", run_dir / "reasoning/result.json")
            return super().acquire(*args, **kwargs)

    monkeypatch.setattr("r3con.runs.FileLock", FinishingFirst)
    with pytest.raises(ValueError, match="this run finished"):
        resume(run_dir, MEMOS_BY_NAME, completion=answering_llm)
    assert answering_llm.requests[sent:] == []
    assert "resumes" not in json.loads((run_dir / "manifest.json").read_text())


def test_a_lock_a_process_left_behind_does_not_stop_a_resume(answering_llm, tmp_path):
    run_dir = failed_run(answering_llm.answers(parsing=bad_key()), tmp_path)
    (run_dir / "run.lock").write_text("999999999")
    answering_llm.answers(parsing=ANSWERING["parsing"])
    answer = resume(run_dir, MEMOS_BY_NAME, completion=answering_llm)
    assert answer.answer == "; ".join(MEMOS_BY_NAME.values())


@pytest.mark.parametrize(
    ("documents", "said"),
    [
        (
            {k: v for k, v in MEMOS_BY_NAME.items() if k != "memos/calder.txt"},
            "these are not the run's documents: missing memos/calder.txt.",
        ),
        (
            {**MEMOS_BY_NAME, "memos/fenwick.txt": "Fenwick memo"},
            "these are not the run's documents: extra memos/fenwick.txt.",
        ),
        (
            {**MEMOS_BY_NAME, "memos/calder.txt": "Calder memo, revised"},
            "these are not the run's documents: changed memos/calder.txt.",
        ),
        (
            dict(reversed(MEMOS_BY_NAME.items())),
            (
                "these are the run's documents in another order; pass them in its "
                "order, as its names.json lists them."
            ),
        ),
    ],
    ids=["missing", "extra", "changed", "reordered"],
)
def test_a_resume_over_other_documents_is_refused_before_any_request(
    answering_llm, tmp_path, documents, said
):
    run_dir = failed_run(answering_llm.answers(parsing=bad_key()), tmp_path)
    before, sent = the_runs_own_files(run_dir), len(answering_llm.requests)
    with pytest.raises(ValueError) as refused:
        resume(run_dir, documents, completion=answering_llm)
    assert str(refused.value) == f"{run_dir}: {said}"
    assert answering_llm.requests[sent:] == []
    assert the_runs_own_files(run_dir) == before


def test_a_resume_lists_a_few_of_many_changed_documents_and_counts_the_rest(
    answering_llm, tmp_path
):
    documents = [f"Memo {i}" for i in range(6)]
    run_dir = failed_run(answering_llm.answers(parsing=bad_key()), tmp_path, documents)
    with pytest.raises(ValueError, match=r"changed 0, 1, 2 and 3 more\.$"):
        resume(
            run_dir, [f"Memo {i}, revised" for i in range(6)], completion=answering_llm
        )


def test_a_resume_runs_under_the_runs_own_reasoning_turns_and_records_its_settings(
    answering_llm, tmp_path, monkeypatch
):
    monkeypatch.setattr(r3con.settings, "REASONING_MAX_TURNS", 7)
    run_dir = failed_run(answering_llm.answers(parsing=bad_key()), tmp_path)
    monkeypatch.undo()
    monkeypatch.setenv("R3CON_DOC_WORKERS", "3")
    answering_llm.answers(parsing=ANSWERING["parsing"])
    resume(run_dir, MEMOS_BY_NAME, completion=answering_llm)
    [resumed_at] = json.loads((run_dir / "manifest.json").read_text())["resumes"]
    used = resumed_at["settings"]
    assert (used["reasoning_max_turns"], used["doc_workers"]) == (7, 3)


def with_a_margin_out_of_range(run_dir: Path) -> None:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    manifest["settings"]["window_margin_percent"] = 150
    (run_dir / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    ("change", "error", "said"),
    [
        (
            predating_resume,
            FileNotFoundError,
            (
                "{run_dir}: this run predates resume: it has no documents.json to "
                "check the documents against."
            ),
        ),
        (
            reading_moved_prompts,
            ValueError,
            (
                "{run_dir}: the run's prompts resolve to other files than it read: "
                "structuring/parsing now reads "
            ),
        ),
        (
            with_a_margin_out_of_range,
            ValueError,
            "can't read {run_dir} as a run folder: ValueError: ",
        ),
    ],
    ids=["predating-resume", "prompts-moved", "margin-out-of-range"],
)
def test_a_folder_resume_cannot_carry_on_is_refused_before_any_request_or_write(
    answering_llm, tmp_path, change, error, said
):
    run_dir = failed_run(answering_llm.answers(parsing=bad_key()), tmp_path)
    change(run_dir)
    before, sent = the_runs_own_files(run_dir), len(answering_llm.requests)
    with pytest.raises(error) as refused:
        resume(run_dir, MEMOS_BY_NAME, completion=answering_llm)
    assert str(refused.value).startswith(said.format(run_dir=run_dir))
    assert answering_llm.requests[sent:] == []
    assert the_runs_own_files(run_dir) == before


def test_a_finished_run_is_not_resumed_but_asked(answering_llm, tmp_path):
    answer = finished_run(answering_llm, tmp_path, MEMOS_BY_NAME)
    with pytest.raises(ValueError) as refused:
        resume(answer.run_dir, MEMOS_BY_NAME, completion=answering_llm)
    assert str(refused.value) == (
        f"{answer.run_dir}: this run finished; ask it a new question instead (r3con "
        "ask, or r3con.ask)."
    )


PROXY = {"api_base": f"https://{KEY}@proxy-a.example/v1"}
OTHER_PARAMS = (
    "these params are not the run's: resume runs with its own, and takes them only "
    "for the real value of each credential its manifest saved as REDACTED."
)


@pytest.mark.parametrize(
    ("saved", "given", "said"),
    [
        (GATEWAY_HEADER, None, MASKED_HEADER_REFUSAL),
        (GATEWAY_HEADER, {**GATEWAY_HEADER, "temperature": 0.9}, OTHER_PARAMS),
        (PROXY, {"api_base": f"https://{KEY}@proxy-b.example/v1"}, OTHER_PARAMS),
    ],
    ids=["masked-and-not-given", "other-params", "another-host-around-the-key"],
)
def test_a_resume_needs_the_runs_own_params_with_their_real_credentials(
    answering_llm, tmp_path, saved, given, said
):
    llm = answering_llm.answers(parsing=bad_key())
    run_dir = failed_run(llm, tmp_path, params=saved)
    sent = len(llm.requests)
    with pytest.raises(ValueError) as refused:
        resume(run_dir, MEMOS_BY_NAME, completion=llm, params=given)
    assert str(refused.value).endswith(said)
    assert llm.requests[sent:] == []


@pytest.mark.parametrize("params", [GATEWAY_HEADER, PROXY], ids=["header", "url"])
def test_a_resume_given_the_runs_params_sends_their_real_credentials(
    answering_llm, tmp_path, params
):
    failing = failing_once(reading("Calder memo"), ANSWERING["parsing"])
    run_dir = failed_run(
        answering_llm.answers(parsing=failing), tmp_path, params=params
    )
    resume(run_dir, MEMOS_BY_NAME, completion=answering_llm, params=params)
    [*_, reasoned] = answering_llm.requests
    assert {name: reasoned[name] for name in params} == params
