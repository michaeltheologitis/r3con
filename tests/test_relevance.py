import re
from collections import Counter

import pytest

from r3con.stages.relevance import join_parts, render_relevance, surface_relevance

MODEL = "openai/gpt-6-luna"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
NOTE = re.compile(r"N\(\w+,r\d\)")
# 3,920 characters in 20 paragraphs, P0 to P19: refused whole under a 6,000-character
# limit beside the relevance prompt, read in halves that start at P0 and P10.
LONG = "\n\n".join(f"P{i}. " + "word " * 38 for i in range(20))


def notes_by_round():
    """A relevance reply naming the document, by its text up to the first full stop,
    and the round it was read in."""
    reads = Counter()

    def note(request) -> str:
        document = request["messages"][1]["content"]
        reads[document] += 1
        return f"N({document.split('.')[0]},r{reads[document]})"

    return note


def surface(llm, documents, rounds, model=MODEL, **kwargs):
    return surface_relevance(
        task="Which filing reports higher revenue?",
        documents=documents,
        model=model,
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
    ("snippets", "rendered"),
    [
        (None, ""),
        ([], ""),
        (
            ["first note", ""],
            (
                "### Document 1\nfirst note\n\n"
                "### Document 2\n(no relevant summary for this task)"
            ),
        ),
        (
            [["a", ""], "c"],
            (
                "### Document 1.1\na\n\n"
                "### Document 1.2\n(no relevant summary for this task)\n\n"
                "### Document 2\nc"
            ),
        ),
    ],
)
def test_the_notes_render_as_one_labelled_section_per_document(snippets, rendered):
    assert render_relevance(snippets) == rendered


@pytest.mark.parametrize(
    ("snippet", "joined"),
    [("a note", "a note"), (["a", " ", "b "], "a\n\nb"), (["", ""], "")],
)
def test_a_split_documents_notes_join_in_order_without_the_empty_ones(snippet, joined):
    assert join_parts(snippet) == joined


def test_a_refused_document_is_read_in_parts_and_keeps_one_note_per_part(llm):
    documents = ["DOC0", LONG, "DOC2"]
    result = surface(llm.refuses_over(6_000), documents, rounds=2, model=QWEN)
    assert result.snippets == ["N(DOC0,r2)", ["N(P0,r2)", "N(P10,r2)"], "N(DOC2,r2)"]
    assert result.rounds[0][1] == ["N(P0,r1)", "N(P10,r1)"]
    halves = [LONG[: LONG.index("P10. ")], LONG[LONG.index("P10. ") :]]
    reads = [r["messages"][1]["content"] for r in llm.requests]
    assert [read for read in reads if read.startswith("P")] == [LONG, *halves, *halves]


def test_a_parts_later_round_sees_the_other_documents_notes_but_never_its_own_parts(
    llm,
):
    surface(llm.refuses_over(6_000), ["DOC0", LONG, "DOC2"], rounds=2, model=QWEN)
    round_2 = {
        request["messages"][1]["content"].split(".")[0]: request["messages"][0]
        for request in llm.requests
        if "summaries of the other documents" in request["messages"][0]["content"]
    }
    seen = {
        reader: set(NOTE.findall(prompt["content"]))
        for reader, prompt in round_2.items()
    }
    assert seen == {
        "P0": {"N(DOC0,r1)", "N(DOC2,r1)"},
        "P10": {"N(DOC0,r1)", "N(DOC2,r1)"},
        "DOC0": {"N(P0,r1)", "N(P10,r1)", "N(DOC2,r1)"},
        "DOC2": {"N(DOC0,r1)", "N(P0,r1)", "N(P10,r1)"},
    }
    assert "N(P0,r1)\n\nN(P10,r1)" in round_2["DOC0"]["content"]
    assert not any("### Document" in prompt["content"] for prompt in round_2.values())
