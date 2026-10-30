import base64
import importlib.metadata
import json
import logging
import os
import re
import threading
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
from tests.documents import (
    CORRUPT_DOCX,
    PNG,
    paragraph,
    table,
    write_docx,
    write_pdf,
)

# A paragraph whose second run Word tracks as inserted, and third as deleted.
TRACKED_CHANGES = (
    '<w:p><w:r><w:t xml:space="preserve">Revenue was </w:t></w:r>'
    '<w:ins w:id="1" w:author="a"><w:r><w:t>12 million</w:t></w:r></w:ins>'
    '<w:del w:id="2" w:author="a"><w:r><w:delText>9 million</w:delText></w:r></w:del>'
    '<w:r><w:t xml:space="preserve"> in Q3.</w:t></w:r></w:p>'
)
# A content control around a paragraph, and one around a run.
CONTENT_CONTROLS = (
    f"<w:sdt><w:sdtContent>{paragraph('Inside a block control.')}</w:sdtContent></w:sdt>"
    '<w:p><w:r><w:t xml:space="preserve">Name: </w:t></w:r>'
    "<w:sdt><w:sdtContent><w:r><w:t>Jane Doe</w:t></w:r></w:sdtContent></w:sdt></w:p>"
)
MERGED_HEADER = (
    '<w:tbl><w:tr><w:tc><w:tcPr><w:gridSpan w:val="3"/></w:tcPr>'
    f"{paragraph('Total')}</w:tc></w:tr><w:tr>"
    + "".join(f"<w:tc>{paragraph(text)}</w:tc>" for text in ["Site A", "Q2", "5"])
    + "</w:tr></w:tbl>"
)
NESTED_TABLE = (
    f"<w:tbl><w:tr><w:tc>{paragraph('outer')}</w:tc>"
    f"<w:tc>{table([['inner1', 'inner2']])}{paragraph('')}</w:tc></w:tr></w:tbl>"
)
PLAIN_TEXT = b"Plain words, saved under the wrong name."
# What Word writes beside a document someone has open: their name, padded.
WORD_OWNER_FILE = b"\x07Michael" + bytes(154)
PINS_A_MISSING_FILE = {**dict.fromkeys(PROMPT_STAGES, "v1"), "reasoning/coding": "v9"}
DOCS = ["Halloran memo", "Merrow memo", "Calder memo"]
JOINED = "Halloran memo; Merrow memo; Calder memo"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
GPT_4O = "openai/gpt-4o"
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
CANCELLED = "Cancelled: cancel was set, so no further model call is sent"
BARRIER_TIMEOUT_SECONDS = 10
SCHEMA_REPLY = """Thought: one row per memo.
<schema>
class Row(BaseModel):
    who: str


class Parse(BaseModel):
    rows: list[Row]
</schema>"""
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


def test_a_word_files_paragraphs_and_tables_come_out_in_order_as_markdown(tmp_path):
    docx = tmp_path / "report.docx"
    table = [["Site", "Incidents"], ["Halloran", "7"]]
    write_docx(docx, ["Incidents by site:", table, "Merrow's count is provisional."])
    lines = [
        "Incidents by site:",
        "",
        "| Site | Incidents |",
        "| --- | --- |",
        "| Halloran | 7 |",
        "",
        "Merrow's count is provisional.",
    ]
    assert read_documents(docx) == ["\n".join(lines)]


@pytest.mark.parametrize(
    ("xml", "text"),
    [
        (TRACKED_CHANGES, "Revenue was 12 million in Q3."),
        (CONTENT_CONTROLS, "Inside a block control.\n\nName: Jane Doe"),
        (
            MERGED_HEADER,
            "| Total | | |\n| --- | --- | --- |\n| Site A | Q2 | 5 |",
        ),
    ],
    ids=[
        "tracked-insertion-kept-deletion-dropped",
        "content-controls",
        "merged-header",
    ],
)
def test_a_word_files_text_is_what_word_shows(tmp_path, xml, text):
    docx = tmp_path / "report.docx"
    write_docx(docx, xml=xml)
    assert read_documents(docx) == [text]


def test_a_word_files_underscores_and_asterisks_are_kept_as_written(tmp_path):
    docx = tmp_path / "report.docx"
    write_docx(docx, ["account_id_2024 is *important* 5*3"])
    assert read_documents(docx) == ["account_id_2024 is *important* 5*3"]


@pytest.mark.parametrize("header", [False, True], ids=["plain-rows", "header-row"])
def test_a_pipe_in_a_word_tables_cell_is_escaped_so_each_row_keeps_its_cells(
    tmp_path, header: bool
):
    docx = tmp_path / "report.docx"
    write_docx(docx, xml=table([["Name|Alias", "Value"], ["a|b", "7"]], header=header))
    [text] = read_documents(docx)
    assert text == "| Name\\|Alias | Value |\n| --- | --- |\n| a\\|b | 7 |"
    cells = {len(re.split(r"(?<!\\)\|", line)) for line in text.splitlines()}
    assert cells == {len(["", "Name|Alias", "Value", ""])}


def test_a_file_called_just_docx_is_a_word_file_to_its_path_and_hidden_to_a_walk(
    tmp_path,
):
    write_docx(tmp_path / ".docx", ["Incidents this quarter: 7."])
    assert read_documents(tmp_path / ".docx") == ["Incidents this quarter: 7."]
    with pytest.raises(FileNotFoundError, match="No documents found at"):
        read_documents(tmp_path)


def test_a_table_nested_in_a_word_table_keeps_its_text(tmp_path):
    docx = tmp_path / "report.docx"
    write_docx(docx, xml=NESTED_TABLE)
    [text] = read_documents(docx)
    assert all(word in text for word in ["outer", "inner1", "inner2"])


def test_a_word_files_image_leaves_none_of_its_data_in_the_text(tmp_path):
    docx = tmp_path / "report.docx"
    write_docx(docx, ["Before the chart."], image=True)
    [text] = read_documents(docx)
    assert text.startswith("Before the chart.")
    assert "base64," not in text
    assert base64.b64encode(PNG).decode() not in text


@pytest.mark.parametrize(
    "content", [CORRUPT_DOCX, PLAIN_TEXT], ids=["corrupt", "plain-text"]
)
def test_a_word_file_that_cannot_be_read_is_logged_and_dropped_not_read_as_text(
    tmp_path, caplog, content: bytes
):
    unreadable = tmp_path / "a_memo.docx"
    unreadable.write_bytes(content)
    (tmp_path / "b_plain.txt").write_text("a plain memo", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        documents = read_documents(tmp_path)
    assert documents == ["a plain memo"]
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"{unreadable}: not readable as a Word file (BadZipFile); skipping it"
    ]


def test_a_file_that_cannot_be_opened_is_logged_and_skipped(tmp_path, caplog):
    write_pdf(tmp_path / "a_memo.pdf", "Incidents this quarter: 7.")
    write_docx(tmp_path / "b_report.docx", ["Incidents by site."])
    (tmp_path / "c_notes.txt").write_text("a locked note", encoding="utf-8")
    (tmp_path / "d_plain.txt").write_text("a plain memo", encoding="utf-8")
    locked = [
        tmp_path / name for name in ["a_memo.pdf", "b_report.docx", "c_notes.txt"]
    ]
    for path in locked:
        path.chmod(0)
    if os.access(locked[0], os.R_OK):
        pytest.skip("this user, root say, reads a file whatever its mode")
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        documents = read_documents(tmp_path)
    assert documents == ["a plain memo"]
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"{path}: not readable (PermissionError); skipping it" for path in locked
    ]


@pytest.mark.parametrize(
    "source",
    [lambda root: root, lambda root: str(root / "*.docx")],
    ids=["walk", "glob"],
)
def test_a_walk_or_a_glob_skips_office_lock_files_and_logs_a_file_with_no_text(
    tmp_path, caplog, source: Callable[[Path], str | Path]
):
    write_docx(tmp_path / "a_report.docx", ["Incidents this quarter: 7."])
    write_docx(tmp_path / "b_empty.docx")
    (tmp_path / "~$a_report.docx").write_bytes(WORD_OWNER_FILE)
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        documents = read_documents(source(tmp_path))
    assert documents == ["Incidents this quarter: 7."]
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"{tmp_path / 'b_empty.docx'}: no extractable text; skipping it"
    ]


def test_a_walk_skips_what_is_hidden_or_under_macosx_below_its_folder(tmp_path):
    texts = {
        "a_memo.md": "a memo",
        ".notes.md": "a hidden note",
        ".git/description.md": "a hidden folder's file",
        "q3/.cache/b_memo.md": "a file in a hidden folder further down",
        "__MACOSX/a_memo.md": "macOS's metadata, named like a memo",
    }
    for name, text in texts.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    assert read_documents(tmp_path) == ["a memo"]


@pytest.mark.parametrize(
    ("pattern", "documents"),
    [
        ("d/*.md", ["a memo"]),
        ("d/**/*.md", ["a memo", "a memo further down"]),
        ("d/.*", ["a hidden memo"]),
        ("**/.env.md", ["a hidden file further down"]),
        ("d/*/.env.md", ["a hidden file further down"]),
        ("d/*/../*.md", ["a memo"]),
    ],
    ids=[
        "wildcard",
        "recursive",
        "dot-wildcard",
        "recursive-to-a-dot-name",
        "wildcard-to-a-dot-name",
        "climbing-back-after-a-wildcard",
    ],
)
def test_a_globs_wildcards_match_no_hidden_name_but_a_dot_in_the_pattern_does(
    tmp_path, pattern: str, documents: list[str]
):
    texts = {
        "d/a.md": "a memo",
        "d/.x.md": "a hidden memo",
        "d/q3/b.md": "a memo further down",
        "d/q3/.env.md": "a hidden file further down",
        "d/.git/.env.md": "a hidden folder's hidden file",
    }
    for name, text in texts.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    assert read_documents(pattern) == documents


def test_a_recursive_glob_reads_a_file_once_past_a_link_to_its_own_folder(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.md").write_text("a memo", encoding="utf-8")
    (tmp_path / "d" / "loop").symlink_to(".")
    assert read_documents("d/**/*.md") == ["a memo"]


@pytest.mark.parametrize(
    ("source", "documents"),
    [
        (".env.md", ["a hidden file"]),
        (".github/*.yml", ["a hidden folder's file"]),
        (".github", ["a hidden folder's file"]),
    ],
    ids=["path", "glob-rooted-in-it", "walk-of-it"],
)
def test_a_hidden_file_or_folder_you_name_is_read(
    tmp_path, source: str, documents: list[str]
):
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "ci.yml").write_text(
        "a hidden folder's file", encoding="utf-8"
    )
    (tmp_path / ".env.md").write_text("a hidden file", encoding="utf-8")
    assert read_documents(source) == documents


FOLDER_DOCUMENTS = ["a received memo", "a memo stored as a blob"]
HUB_DOCUMENTS = ["the first memo in the hub", "the second memo in the hub"]


def write_folder_with_links(root: Path) -> None:
    for folder in ["folder/blobs", "outside_dir", "hub/blobs", "hub/snapshot"]:
        (root / folder).mkdir(parents=True)
    texts = {
        "secret.txt": "a secret outside the folder",
        "outside_dir/b.txt": "outside the folder",
        "folder/a.txt": "a received memo",
        "folder/blobs/7f3c": "a memo stored as a blob",
        "hub/blobs/1a2b": "the first memo in the hub",
        "hub/blobs/3c4d": "the second memo in the hub",
    }
    for name, text in texts.items():
        (root / name).write_text(text, encoding="utf-8")
    links = {
        "folder/link.txt": "../secret.txt",
        "folder/linked.txt": "blobs/7f3c",
        "folder/linkdir": "../outside_dir",
        "alias": "folder",
        "hub/snapshot/first.txt": "../blobs/1a2b",
        "hub/snapshot/second.txt": "../blobs/3c4d",
    }
    for name, target in links.items():
        (root / name).symlink_to(target)


@pytest.mark.parametrize(
    ("source", "documents"),
    [
        (lambda root: "folder", FOLDER_DOCUMENTS),
        (lambda root: "folder/*.txt", FOLDER_DOCUMENTS),
        (lambda root: "folder/**/*.txt", FOLDER_DOCUMENTS),
        (lambda root: str(root / "folder" / "*.txt"), FOLDER_DOCUMENTS),
        (lambda root: "alias", FOLDER_DOCUMENTS),
        (lambda root: "hub", HUB_DOCUMENTS),
    ],
    ids=[
        "walk",
        "glob",
        "recursive-glob",
        "absolute-glob",
        "folder-named-through-a-link",
        "folder-of-links-inside-it",
    ],
)
def test_a_walk_or_a_glob_reads_every_file_inside_the_folder_and_none_outside(
    tmp_path, source: Callable[[Path], str], documents: list[str]
):
    write_folder_with_links(tmp_path)
    assert read_documents(source(tmp_path)) == documents


def test_a_walk_logs_each_file_that_resolves_outside_the_folder(tmp_path, caplog):
    write_folder_with_links(tmp_path)
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        documents = read_documents("folder")
    assert documents == FOLDER_DOCUMENTS
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"folder/link.txt: resolves outside {tmp_path / 'folder'}; skipping it"
    ]


@pytest.mark.parametrize(
    ("pattern", "skipped"),
    [
        ("folder/*/*.txt", "folder/linkdir/b.txt"),
        ("folder/*/../../secret.txt", "folder/blobs/../../secret.txt"),
    ],
    ids=["through-a-linked-folder", "climbing-out-after-a-wildcard"],
)
def test_a_glob_that_leaves_its_folder_finds_nothing_and_logs_why(
    tmp_path, caplog, pattern: str, skipped: str
):
    write_folder_with_links(tmp_path)
    with (
        caplog.at_level(logging.WARNING, logger="r3con.documents"),
        pytest.raises(FileNotFoundError, match="No documents found at"),
    ):
        read_documents(pattern)
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"{skipped}: resolves outside {tmp_path / 'folder'}; skipping it"
    ]


def test_a_file_named_directly_is_read_even_through_a_link(tmp_path):
    write_folder_with_links(tmp_path)
    assert read_documents("folder/link.txt") == ["a secret outside the folder"]


def test_half_a_surrogate_pair_in_a_pdf_reads_as_a_replacement_and_the_run_goes_on(
    answering_llm, tmp_path
):
    pdf = tmp_path / "memo.pdf"
    # pypdf reads a font's ToUnicode CMap with surrogatepass: D800 stays half a pair.
    write_pdf(pdf, "Memo A here", to_unicode={"A": "D800"})
    documents = read_documents(pdf)
    assert documents == ["Memo \ufffd here"]
    answer = finished_run(answering_llm, tmp_path, documents)
    assert answering_llm.requests_for("relevance")
    assert answer.answer == "Memo \ufffd here"


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
    result = finished_run(answering_llm, tmp_path, model=GPT_4O)
    manifest_path = result.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    del manifest["litellm_model"]
    manifest_path.write_text(json.dumps(manifest))
    ask(result, "Who else?", completion=answering_llm)
    ran, asked = answering_llm.requests_for("reasoning")
    assert (ran["model"], asked["model"]) == (GPT_4O, "gpt-4o")


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
        return SCHEMA_REPLY

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
