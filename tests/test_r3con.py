"""`r3con.r3con` — the public surface.

The contract these pin: **`documents` is a sequence of document texts**, and reading the
filesystem is a separate, explicit job. Nothing sniffs a string to guess which you meant.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r3con.r3con import TEXT_SUFFIXES, _check_documents, read_documents, run  # noqa: E402


def _tree(root: Path) -> None:
    (root / "a.md").write_text("alpha document", encoding="utf-8")
    (root / "b.txt").write_text("beta document", encoding="utf-8")
    (root / "nested").mkdir()
    (root / "nested" / "c.md").write_text("gamma document", encoding="utf-8")
    (root / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n not text")
    (root / "empty.txt").write_text("   \n", encoding="utf-8")


# ---------- documents are texts ----------


def test_documents_is_a_list_of_strings() -> None:
    assert _check_documents(["one", "two"]) == ["one", "two"]
    assert _check_documents(("one", "two")) == ["one", "two"]      # any sequence
    assert _check_documents(iter(["one", "two"])) == ["one", "two"]  # any iterable


def test_a_document_is_never_read_off_disk_behind_your_back() -> None:
    """The regression this contract exists to prevent: a short document whose text
    happens to match a filename used to be silently replaced by that file's contents."""
    with tempfile.TemporaryDirectory() as tmp:
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            Path("Q3 was strong.").write_text("SOMETHING ELSE ENTIRELY", encoding="utf-8")
            assert _check_documents(["Q3 was strong."]) == ["Q3 was strong."]
        finally:
            os.chdir(cwd)


def test_a_bare_string_is_refused_not_guessed() -> None:
    """Ambiguous input gets an error that says what to do, rather than a guess."""
    for bad in ("./docs", "just some document text"):
        try:
            _check_documents(bad)
        except TypeError as e:
            assert "read_documents" in str(e) and "[text]" in str(e)
        else:
            raise AssertionError(f"expected TypeError for {bad!r}")
    try:
        _check_documents(Path("./docs"))
    except TypeError:
        pass
    else:
        raise AssertionError("expected TypeError for a Path")


def test_non_strings_are_refused_with_the_offending_index() -> None:
    try:
        _check_documents(["fine", 42])
    except TypeError as e:
        assert "item 1" in str(e) and "int" in str(e)
        return
    raise AssertionError("expected TypeError")


def test_blank_documents_are_dropped_and_all_blank_is_an_error() -> None:
    assert _check_documents(["real", "   ", ""]) == ["real"]
    try:
        _check_documents(["  ", ""])
    except ValueError as e:
        assert "empty" in str(e)
        return
    raise AssertionError("expected ValueError")


def test_run_validates_documents_before_spending_anything() -> None:
    for bad, exc in (("./docs", TypeError), ([], ValueError), (["  "], ValueError)):
        try:
            run("q", bad)
        except exc:
            continue
        raise AssertionError(f"expected {exc.__name__} for {bad!r}")


# ---------- reading the filesystem is a separate job ----------


def test_read_documents_walks_a_directory_in_sorted_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root)
        docs = read_documents(root)
    # a.md, b.txt, nested/c.md — sorted by path; the PNG skipped, the blank file dropped
    assert docs == ["alpha document", "beta document", "gamma document"]


def test_read_documents_skips_binaries() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root)
        assert not any("PNG" in d for d in read_documents(root))
    assert ".png" not in TEXT_SUFFIXES


def test_read_documents_expands_a_glob() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root)
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            assert read_documents("*.md") == ["alpha document"]
        finally:
            os.chdir(cwd)


def test_read_documents_accepts_several_sources_in_order() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "second.txt").write_text("second", encoding="utf-8")
        (root / "first.txt").write_text("first", encoding="utf-8")
        assert read_documents([root / "first.txt", root / "second.txt"]) == ["first", "second"]


def test_read_documents_raises_when_a_source_matches_nothing() -> None:
    try:
        read_documents(Path("/definitely/not/here/at/all"))
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_read_documents_never_treats_its_argument_as_text() -> None:
    """It is a filesystem reader; a string that is not a path is an error, not a document."""
    try:
        read_documents("this is document text, not a path")
    except FileNotFoundError:
        return
    raise AssertionError("expected FileNotFoundError")


def test_undecodable_bytes_do_not_sink_the_read() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / "bad.txt"
        f.write_bytes(b"good text \xff\xfe more text")
        docs = read_documents(f)
    assert len(docs) == 1 and "good text" in docs[0] and "more text" in docs[0]


def test_the_two_compose() -> None:
    """The documented folder path: read, then answer."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _tree(root)
        assert _check_documents(read_documents(root)) == [
            "alpha document", "beta document", "gamma document",
        ]


# ---------- the rest of the surface ----------


def test_the_namespace_import_shape_works() -> None:
    from r3con import r3con as ns

    assert callable(ns.run) and callable(ns.read_documents)
    import r3con as pkg

    assert pkg.run is ns.run
    from r3con import Answer, run as flat
    assert flat is ns.run
    assert Answer is ns.Answer


def test_answer_object_exposes_the_intermediate_views() -> None:
    from r3con.pipeline import Answer

    a = Answer(answer="42", relevance=["doc one's note", ""], struct_data={"rows": [{"document": 1}]},
               schema_code="class Parse(BaseModel): ...", source_docs={"rows": [0]}, run_dir=None)
    assert a.answer == "42" and str(a) == "42"
    assert a.relevance == ["doc one's note", ""]
    assert a.struct_data["rows"][0]["document"] == 1
    assert "Parse" in a.schema_code and a.source_docs == {"rows": [0]}


def test_run_rejects_overrides_that_a_prebuilt_config_would_swallow() -> None:
    from r3con.config import RunConfig

    cfg = RunConfig(model="openai/m", prompts={k: "v1" for k in
                    ("relevance", "structuring/schema", "structuring/parsing", "reasoning")})
    try:
        run("q", ["a document"], config=cfg, model="openai/other")
    except ValueError as e:
        assert "RunConfig" in str(e) and "model" in str(e)
        return
    raise AssertionError("expected ValueError")


if __name__ == "__main__":
    tests = [
        test_documents_is_a_list_of_strings,
        test_a_document_is_never_read_off_disk_behind_your_back,
        test_a_bare_string_is_refused_not_guessed,
        test_non_strings_are_refused_with_the_offending_index,
        test_blank_documents_are_dropped_and_all_blank_is_an_error,
        test_run_validates_documents_before_spending_anything,
        test_read_documents_walks_a_directory_in_sorted_order,
        test_read_documents_skips_binaries,
        test_read_documents_expands_a_glob,
        test_read_documents_accepts_several_sources_in_order,
        test_read_documents_raises_when_a_source_matches_nothing,
        test_read_documents_never_treats_its_argument_as_text,
        test_undecodable_bytes_do_not_sink_the_read,
        test_the_two_compose,
        test_the_namespace_import_shape_works,
        test_answer_object_exposes_the_intermediate_views,
        test_run_rejects_overrides_that_a_prebuilt_config_would_swallow,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
