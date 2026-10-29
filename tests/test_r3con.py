import importlib.metadata
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import litellm
import pytest

import r3con
from r3con import Answer, RunConfig
from r3con import r3con as namespace
from r3con.config import PROMPT_STAGES, REASONING_PATHS
from r3con.r3con import ask, read_documents, run

PINS_A_MISSING_FILE = {**dict.fromkeys(PROMPT_STAGES, "v1"), "reasoning/coding": "v9"}
DOCS = ["Halloran memo", "Merrow memo", "Calder memo"]
JOINED = "Halloran memo; Merrow memo; Calder memo"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
GPT_4O = "openai/gpt-4o"
RUN_PARAMS = {"temperature": 0.3, "extra_body": {"top_k": 20}}
ASK_FILES = {
    "manifest.json",
    "reasoning/calls.json",
    "reasoning/result.json",
    "reasoning/transcript.yaml",
}


def finished_run(llm, tmp_path: Path, documents=DOCS, **options) -> Answer:
    """A run of "Who?" over ``documents``, ``llm`` answering, its folder under
    ``tmp_path / "logs"``."""
    logs = tmp_path / "logs"
    return run("Who?", documents, completion=llm, logs_dir=logs, **options)


def the_runs_own_files(run_dir: Path) -> dict[str, bytes]:
    """Every file in ``run_dir`` but those under ``asks/``, by path."""
    return {
        str(path.relative_to(run_dir)): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file() and path.relative_to(run_dir).parts[0] != "asks"
    }


def in_json(name: str, change: Callable[[Any], None]) -> Callable[[Path], None]:
    """Rewrites the JSON file ``name`` of a run folder after ``change``."""

    def rewrite(run_dir: Path) -> None:
        data = json.loads((run_dir / name).read_text())
        change(data)
        (run_dir / name).write_text(json.dumps(data))

    return rewrite


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


def bad_key() -> litellm.AuthenticationError:
    return litellm.AuthenticationError(
        message="bad key", llm_provider="openai", model="gpt-6-luna"
    )


def write_tree(root: Path) -> None:
    (root / "nested").mkdir(parents=True)
    (root / "a.md").write_text("alpha document", encoding="utf-8")
    (root / "b.txt").write_text("beta document", encoding="utf-8")
    (root / "nested" / "c.md").write_text("gamma document", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n not text")
    (root / "empty.txt").write_text("   \n", encoding="utf-8")


def write_pdf(path: Path, text: str | None = None) -> None:
    """A minimal valid PDF (correct xref offsets and %%EOF), with or without a text
    layer: generated, so the fixture stays readable."""
    objs = ["<</Type/Catalog/Pages 2 0 R>>", "<</Type/Pages/Kids[3 0 R]/Count 1>>"]
    if text:
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET"
        objs += [
            (
                "<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Contents 4 0 R"
                "/Resources<</Font<</F1 5 0 R>>>>>>"
            ),
            f"<</Length {len(stream)}>>stream\n{stream}\nendstream",
            "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        ]
    else:
        objs.append("<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]>>")
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj{body}endobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offsets:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer<</Size {len(objs) + 1}/Root 1 0 R>>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


@pytest.mark.parametrize(
    ("documents", "error", "message"),
    [
        ("./docs", TypeError, r"not a single string or path.*\[text\].*read_documents"),
        ("just some document text", TypeError, "not a single string or path"),
        (Path("./docs"), TypeError, "not a single string or path"),
        (42, TypeError, "must be a sequence of strings — got int"),
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
    assert result.provenance == {"summary_ids": [], "parse_ids": result.document_ids}
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


def test_a_directory_is_read_recursively_in_path_order_skipping_binaries_and_blanks(
    tmp_path,
):
    write_tree(tmp_path)
    assert read_documents(tmp_path) == [
        "alpha document",
        "beta document",
        "gamma document",
    ]


def test_pdfs_are_extracted_and_one_that_cannot_be_read_is_dropped(tmp_path):
    write_pdf(tmp_path / "a_memo.pdf", "Incidents this quarter: 7.")
    write_pdf(tmp_path / "b_scanned.pdf")
    (tmp_path / "c_corrupt.pdf").write_bytes(b"%PDF-1.4 truncated garbage")
    (tmp_path / "d_plain.txt").write_text("a plain memo", encoding="utf-8")
    documents = read_documents(tmp_path)
    assert len(documents) == 2
    assert "Incidents this quarter: 7." in documents[0]
    assert documents[1] == "a plain memo"


def test_a_glob_is_expanded_from_the_working_directory(tmp_path):
    write_tree(tmp_path)
    assert read_documents("*.md") == ["alpha document"]


def test_several_sources_are_read_in_the_order_given(tmp_path):
    (tmp_path / "second.txt").write_text("second", encoding="utf-8")
    (tmp_path / "first.txt").write_text("first", encoding="utf-8")
    sources = [tmp_path / "first.txt", tmp_path / "second.txt"]
    assert read_documents(sources) == ["first", "second"]


@pytest.mark.parametrize(
    "source", [Path("/definitely/not/here/at/all"), "this is document text, not a path"]
)
def test_a_source_that_matches_nothing_is_an_error_not_a_document(source):
    with pytest.raises(FileNotFoundError, match="No documents found at"):
        read_documents(source)


def test_undecodable_bytes_are_replaced_not_fatal(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"good text \xff\xfe more text")
    [document] = read_documents(tmp_path / "bad.txt")
    assert document.startswith("good text ")
    assert document.endswith(" more text")


def test_a_folder_read_with_read_documents_can_be_answered_over(
    answering_llm, tmp_path
):
    write_tree(tmp_path / "docs")
    result = run("Who?", read_documents(tmp_path / "docs"), completion=answering_llm)
    assert result.answer == "alpha document; beta document; gamma document"


@pytest.mark.parametrize("rounds", [0, 2])
def test_an_ask_answers_over_the_runs_representation_citing_only_its_ids(
    answering_llm, tmp_path, rounds
):
    result = finished_run(answering_llm, tmp_path, relevance_rounds=rounds)
    asked = ask(result, "Who else?", completion=answering_llm)
    assert asked.answer == JOINED
    cited = {id_.partition(".")[0] for ids in asked.provenance.values() for id_ in ids}
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


def test_an_ask_over_a_run_recorded_without_its_litellm_model_uses_the_bare_name(
    answering_llm, tmp_path
):
    result = finished_run(answering_llm, tmp_path)
    manifest_path = result.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["litellm_model"]
    manifest_path.write_text(json.dumps(manifest))
    ask(result, "Who else?", completion=answering_llm)
    ran, asked = answering_llm.requests_for("reasoning")
    assert (ran["model"], asked["model"]) == ("openai/gpt-6-luna", "gpt-6-luna")


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


def test_a_run_that_failed_in_reasoning_can_still_be_asked(answering_llm, tmp_path):
    reply = {"answer": "Calder memo", "provenance_ids": {}}
    answering_llm.answers(reasoning=[bad_key(), reply])
    with pytest.raises(litellm.AuthenticationError):
        finished_run(answering_llm, tmp_path)
    [run_dir] = (tmp_path / "logs").iterdir()
    assert ask(run_dir, "Who else?", completion=answering_llm).answer == "Calder memo"


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


def test_run_is_reachable_from_the_package_and_from_its_module():
    assert r3con.run is namespace.run
    assert r3con.ask is namespace.ask
    assert r3con.read_documents is namespace.read_documents
    assert Answer is namespace.Answer


def test_the_version_lookup_leaves_nothing_in_the_package_namespace():
    assert not hasattr(r3con, "version")
    assert not hasattr(r3con, "PackageNotFoundError")


def test_the_version_is_the_installed_distributions():
    assert r3con.__version__ == importlib.metadata.version("r3context")
