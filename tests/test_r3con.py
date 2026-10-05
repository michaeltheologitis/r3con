import importlib.metadata
from pathlib import Path

import pytest

import r3con
from r3con import Answer, RunConfig
from r3con import r3con as namespace
from r3con.config import PROMPT_STAGES
from r3con.r3con import read_documents, run


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


def test_a_cap_below_its_floor_is_refused_before_a_run_folder_exists(
    llm, tmp_path, monkeypatch
):
    monkeypatch.setattr(r3con.settings, "REASONING_MAX_TURNS", 0)
    with pytest.raises(ValueError, match="REASONING_MAX_TURNS must be >= 1, got 0."):
        run("Who?", ["memo"], completion=llm, logs_dir=tmp_path / "logs")
    assert not (tmp_path / "logs").exists()
    assert llm.requests == []


@pytest.mark.parametrize(
    "model", ["anthropic/claude-sonnet-5-5", "gemini/gemini-3.8-flash"]
)
def test_a_run_on_a_provider_that_takes_no_seed_completes(answering_llm, model):
    result = run("Who?", ["Halloran memo"], model=model, completion=answering_llm)
    assert result.answer == "Halloran memo"
    assert {request["model"] for request in answering_llm.requests} == {model}


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


def test_run_is_reachable_from_the_package_and_from_its_module():
    assert r3con.run is namespace.run
    assert r3con.read_documents is namespace.read_documents
    assert Answer is namespace.Answer


def test_the_version_is_the_installed_distributions():
    assert r3con.__version__ == importlib.metadata.version("r3context")
