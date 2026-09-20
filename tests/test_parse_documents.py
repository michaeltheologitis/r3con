"""Tests for ``r3con.stages.structuring.parsing.parse_documents`` — the per-document fan-out
and the merge.

The parsing step no longer chunks: every document is fed **whole** through one
``parse_one_document`` call, all documents in parallel, each call's system prompt
carrying the corpus-wide relevance snippets. The per-document parses merge into one
(list fields concatenated in document order), and each merged record is tagged
with its source-document index (``ParseResult.source_docs``).

``parse_one_document`` (the real per-doc LLM call) is monkeypatched with a scripted
fake for the merge / order / source-tag tests; the prompt-contract test patches
``litellm_chat_completion`` instead so the real ``parse_one_document`` renders the
parsing prompt.

Run with:  uv run python tests/test_parse_documents.py
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from typing import Any

from pydantic import BaseModel

from r3con.stages.structuring import parsing as parsing_mod

DOC_A = "Alpha document about whales."
DOC_B = "Beta document about ships and the sea."
DOC_C = "Gamma document about harbors."


class Item(BaseModel):
    text: str


class Parse(BaseModel):
    items: list[Item]


@contextlib.contextmanager
def _patched_extract_one(
    fake: Callable[..., BaseModel],
) -> Iterator[list[dict[str, Any]]]:
    """Swap ``parse_one_document`` for a scripted fake; yield captured kwargs (call order
    is nondeterministic under parallelism — filter by ``document``)."""
    original = parsing_mod.parse_one_document
    calls: list[dict[str, Any]] = []

    def wrapper(**kwargs: Any) -> BaseModel:
        calls.append(dict(kwargs))
        return fake(**kwargs)

    parsing_mod.parse_one_document = wrapper  # type: ignore[assignment]
    try:
        yield calls
    finally:
        parsing_mod.parse_one_document = original  # type: ignore[assignment]


@contextlib.contextmanager
def _patched_llm(fake: Callable[..., BaseModel]) -> Iterator[list[dict[str, Any]]]:
    """Swap the raw LLM call so the real ``parse_one_document`` renders the prompt."""
    original = parsing_mod.litellm_chat_completion
    calls: list[dict[str, Any]] = []

    def wrapper(**kwargs: Any) -> BaseModel:
        calls.append(dict(kwargs))
        return fake(**kwargs)

    parsing_mod.litellm_chat_completion = wrapper  # type: ignore[assignment]
    try:
        yield calls
    finally:
        parsing_mod.litellm_chat_completion = original  # type: ignore[assignment]


def _run(
    fake: Callable[..., BaseModel],
    *,
    documents: list[str],
    relevance_snippets: list[str] | None = None,
    doc_ids: list[str] | None = None,
    workers: int = 1,
) -> tuple[Any, list[dict[str, Any]]]:
    with _patched_extract_one(fake) as calls:
        result = parsing_mod.parse_documents(
            documents=documents,
            schema_code="SCHEMA",
            parse_cls=Parse,
            task="q",
            relevance_snippets=relevance_snippets,
            doc_ids=doc_ids,
            model="m",
            prompt_version="v1",
            workers=workers,
        )
    return result, calls


# --------------------------------------------------------------------------- #
# One call per whole document (no chunking)
# --------------------------------------------------------------------------- #


def test_one_call_per_document_whole_doc() -> None:
    """Each document is one call; the whole document is the input (no chunking)."""
    def fake(*, document: str, **_: Any) -> Parse:
        return Parse(items=[Item(text=document)])

    result, calls = _run(fake, documents=[DOC_A, DOC_B, DOC_C])
    assert len(calls) == 3
    seen_docs = sorted(c["document"] for c in calls)
    assert seen_docs == sorted([DOC_A, DOC_B, DOC_C])
    # Each call got a whole document, never a fragment.
    for c in calls:
        assert c["document"] in (DOC_A, DOC_B, DOC_C)


def test_each_call_carries_doc_index_kind() -> None:
    """The per-doc call kind encodes the source document index (for the cost ledger)."""
    def fake(*, document: str, **_: Any) -> Parse:
        return Parse(items=[])

    _, calls = _run(fake, documents=[DOC_A, DOC_B])
    kinds = sorted(c["kind"] for c in calls)
    assert kinds == ["parse-d0", "parse-d1"]


# --------------------------------------------------------------------------- #
# Summaries injection
# --------------------------------------------------------------------------- #


def test_summaries_passed_to_each_doc() -> None:
    """The corpus-wide relevance snippets reaches every per-document parsing call."""
    summ = ["Doc A is about whales.", "Doc B is about ships."]

    def fake(*, document: str, **_: Any) -> Parse:
        return Parse(items=[])

    _, calls = _run(fake, documents=[DOC_A, DOC_B], relevance_snippets=summ)
    for c in calls:
        assert c["relevance_snippets"] == summ


def test_summaries_rendered_into_system_prompt() -> None:
    """The real parse_one_document renders the relevance snippets into the system prompt's
    '## Document summaries' block; the document is the user message."""
    summ = ["Doc A is about whales.", "Doc B is about ships."]

    def fake(*, system_prompt: str, user_prompt: str, **_: Any) -> Parse:
        # capture by stashing on the function's attribute via the calls list below
        return Parse(items=[])

    with _patched_llm(fake) as calls:
        parsing_mod.parse_documents(
            documents=[DOC_A], schema_code="SCHEMA", parse_cls=Parse,
            task="What is discussed?", relevance_snippets=summ, model="m", prompt_version="v1", workers=1,
        )
    sys_msg = calls[0]["system_prompt"]
    # The injected block heading (distinct from the example's "Document summaries:" label).
    assert "## Task-conditioned document summaries" in sys_msg
    assert "Doc A is about whales." in sys_msg
    assert "Doc B is about ships." in sys_msg
    assert "What is discussed?" in sys_msg  # task in the system prompt
    assert calls[0]["user_prompt"] == DOC_A  # document is the user message


def test_no_summaries_omits_block() -> None:
    def fake(*, system_prompt: str, user_prompt: str, **_: Any) -> Parse:
        return Parse(items=[])

    with _patched_llm(fake) as calls:
        parsing_mod.parse_documents(
            documents=[DOC_A], schema_code="SCHEMA", parse_cls=Parse,
            task="q", relevance_snippets=None, model="m", prompt_version="v1", workers=1,
        )
    # The injected block (not the example's label) is omitted when there are no summaries.
    assert "## Task-conditioned document summaries" not in calls[0]["system_prompt"]


# --------------------------------------------------------------------------- #
# Merge + source-doc tagging
# --------------------------------------------------------------------------- #


def test_merge_concatenates_list_fields_in_doc_order() -> None:
    """Records from each document concatenate in document order."""
    def fake(*, document: str, **_: Any) -> Parse:
        return Parse(items=[Item(text=f"{document}#a"), Item(text=f"{document}#b")])

    result, _ = _run(fake, documents=[DOC_A, DOC_B], workers=4)
    texts = [it.text for it in result.parse.items]
    assert texts == [f"{DOC_A}#a", f"{DOC_A}#b", f"{DOC_B}#a", f"{DOC_B}#b"]


def test_source_docs_alignment_one_record_per_doc() -> None:
    """source_docs aligns 1:1 with the merged list, tagging each record's origin doc."""
    def fake(*, document: str, **_: Any) -> Parse:
        return Parse(items=[Item(text=document)])

    result, _ = _run(fake, documents=[DOC_A, DOC_B, DOC_C], workers=4)
    assert result.source_docs["items"] == [0, 1, 2]


def test_source_docs_multi_record_per_doc_shares_tag() -> None:
    """Multiple records from one document all carry that document's index."""
    def fake(*, document: str, **_: Any) -> Parse:
        return Parse(items=[Item(text=f"{document}#a"), Item(text=f"{document}#b")])

    result, _ = _run(fake, documents=[DOC_A, DOC_B], workers=4)
    assert result.source_docs["items"] == [0, 0, 1, 1]


def test_parallel_preserves_document_order() -> None:
    """Even with parallel workers, the merge stays in document order."""
    def fake(*, document: str, **_: Any) -> Parse:
        return Parse(items=[Item(text=document)])

    result, _ = _run(fake, documents=[DOC_A, DOC_B, DOC_C], workers=8)
    assert [it.text for it in result.parse.items] == [DOC_A, DOC_B, DOC_C]
    assert result.source_docs["items"] == [0, 1, 2]


# --------------------------------------------------------------------------- #
# Doc labels + edge cases
# --------------------------------------------------------------------------- #


def test_doc_ids_labels() -> None:
    def fake(*, document: str, **_: Any) -> Parse:
        return Parse(items=[Item(text=document)])

    result, _ = _run(fake, documents=[DOC_A, DOC_B], doc_ids=["fileA", "fileB"])
    assert result.doc_ids == ["fileA", "fileB"]
    assert result.doc_label(0) == "fileA"
    assert result.doc_label(1) == "fileB"
    assert result.doc_label(9) == "9"  # out of range → bare index


def test_empty_corpus_no_calls() -> None:
    """An empty corpus returns an empty parse and makes no extraction calls."""
    def fake(*, document: str, **_: Any) -> Parse:
        return Parse(items=[Item(text=document)])

    result, calls = _run(fake, documents=[])
    assert result.parse.items == []
    assert result.source_docs.get("items", []) == []
    assert len(calls) == 0


def test_exception_propagates() -> None:
    """An exception in any document's extraction surfaces out of parse_documents."""
    def fake(*, document: str, **_: Any) -> Parse:
        if document == DOC_B:
            raise RuntimeError("extract blew up")
        return Parse(items=[Item(text=document)])

    try:
        _run(fake, documents=[DOC_A, DOC_B, DOC_C], workers=4)
    except RuntimeError as e:
        assert "blew up" in str(e)
    else:
        raise AssertionError("expected RuntimeError to propagate")


def test_progress_logged_per_doc_at_info() -> None:
    """Parsing logs ``doc n/N`` per document via the `r3con` logger (INFO)."""
    import io
    import logging

    from r3con.logging_setup import get_logger

    def fake(*, document: str, **_: Any) -> Parse:
        return Parse(items=[])

    logger = get_logger("structuring.parsing")  # the exact logger the parsing stage uses
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        with _patched_extract_one(fake):
            parsing_mod.parse_documents(
                documents=[DOC_A, DOC_B], schema_code="SCHEMA", parse_cls=Parse,
                task="q", model="m", prompt_version="v1", workers=1,
            )
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    out = buf.getvalue()
    assert "parse doc 1/2" in out, out
    assert "parse doc 2/2" in out, out


if __name__ == "__main__":
    tests = [
        test_one_call_per_document_whole_doc,
        test_each_call_carries_doc_index_kind,
        test_summaries_passed_to_each_doc,
        test_summaries_rendered_into_system_prompt,
        test_no_summaries_omits_block,
        test_merge_concatenates_list_fields_in_doc_order,
        test_source_docs_alignment_one_record_per_doc,
        test_source_docs_multi_record_per_doc_shares_tag,
        test_parallel_preserves_document_order,
        test_doc_ids_labels,
        test_empty_corpus_no_calls,
        test_exception_propagates,
        test_progress_logged_per_doc_at_info,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
