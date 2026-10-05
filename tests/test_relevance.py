import re
from collections import Counter

import pytest

from r3con.stages.relevance import render_relevance, surface_relevance

MODEL = "openai/gpt-5.6-luna"
NOTE = re.compile(r"N\(DOC\d,r\d\)")


def notes_by_round():
    """A relevance reply naming the document and the round it was read in."""
    reads = Counter()

    def note(request) -> str:
        document = request["messages"][1]["content"]
        reads[document] += 1
        return f"N({document},r{reads[document]})"

    return note


def surface(llm, documents, rounds, **kwargs):
    return surface_relevance(
        task="Which filing reports higher revenue?",
        documents=documents,
        model=MODEL,
        prompt_version="v1",
        rounds=rounds,
        completion=llm.answers(relevance=notes_by_round()),
        **kwargs,
    )


@pytest.mark.parametrize(("n_docs", "rounds"), [(3, 1), (3, 2), (3, 3), (2, 3), (1, 3)])
def test_each_round_reads_a_document_with_only_the_others_last_notes(
    llm, n_docs, rounds
):
    documents = [f"DOC{d}" for d in range(n_docs)]
    surface(llm, documents, rounds, workers=4)
    for document in documents:
        reads = [r for r in llm.requests if r["messages"][1]["content"] == document]
        assert len(reads) == rounds
        for k, read in enumerate(reads, start=1):
            seen = set(NOTE.findall(read["messages"][0]["content"]))
            others = {f"N({d},r{k - 1})" for d in documents if d != document}
            assert seen == (others if k > 1 else set())


@pytest.mark.parametrize("rounds", [1, 3])
def test_the_last_rounds_notes_are_returned_aligned_with_the_documents(llm, rounds):
    documents = ["DOC0", "DOC1", "DOC2"]
    result = surface(llm, documents, rounds, workers=4)
    assert result.snippets == [f"N({d},r{rounds})" for d in documents]
    assert result.rounds == [
        [f"N({d},r{k})" for d in documents] for k in range(1, rounds + 1)
    ]


@pytest.mark.parametrize(("documents", "rounds"), [([], 3), (["a", "b"], 0)])
def test_no_documents_or_no_rounds_means_no_notes_and_no_request(
    llm, documents, rounds
):
    result = surface(llm, documents, rounds)
    assert (result.snippets, result.rounds) == ([], [])
    assert llm.requests == []


def test_the_system_prompt_holds_the_task_and_the_user_message_the_document(llm):
    surface(llm, ["AAA", "BBB"], rounds=2, workers=1)
    first, second = llm.requests[0]["messages"], llm.requests[2]["messages"]
    assert [first[1]["content"], second[1]["content"]] == ["AAA", "AAA"]
    task = "<task>\nWhich filing reports higher revenue?\n</task>"
    assert task in first[0]["content"]
    assert task in second[0]["content"]
    assert "summaries of the other documents" not in first[0]["content"]
    assert "summaries of the other documents" in second[0]["content"]


@pytest.mark.parametrize(
    ("snippets", "doc_ids", "rendered"),
    [
        (None, None, ""),
        ([], None, ""),
        (
            ["first note", ""],
            None,
            (
                "### Document 1\nfirst note\n\n"
                "### Document 2\n(no relevant summary for this task)"
            ),
        ),
        (["s0", "s1"], ["10-K", "10-Q"], "### 10-K\ns0\n\n### 10-Q\ns1"),
    ],
)
def test_the_notes_render_as_one_labelled_section_per_document(
    snippets, doc_ids, rendered
):
    assert render_relevance(snippets, doc_ids) == rendered
