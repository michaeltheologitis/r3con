import json
import re
from collections import Counter

import litellm
import pytest
import yaml

from r3con import settings
from r3con.notes import Budget, NotesTooLong
from r3con.prompts import load_prompt
from r3con.runs import StageRun, TaskLogger
from r3con.splitting import Splits
from r3con.stages.relevance import (
    RelevantContext,
    join_parts,
    note_texts,
    render_relevance,
    surface_relevance,
    takes_word_budget,
)

MODEL = "openai/gpt-6-luna"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
NOTE = re.compile(r"N\(\w+,r\d\)")
# 3,920 characters in 20 paragraphs, P0 to P19: refused whole under a 6,000-character
# limit beside the relevance prompt, read in halves that start at P0 and P10.
LONG = "\n\n".join(f"P{i}. " + "word " * 38 for i in range(20))
TASK = "Which filing reports higher revenue?"
SENTENCE = (
    " Keep it under 16 words: the collection is large, and every document's summary "
    "has to fit beside all the others."
)
# A 60-word note: nine of them make a round-2 request about 1,335 tokens, over the
# 1,200-token line of a 1,412-token window; at 30 words each, about 1,050.
SIXTY = " ".join(["word"] * 60)
WINDOW_1200 = 1_412
ROUTINE = "The site logged its routine checks and found nothing out of the ordinary. "


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


def kinds(run: StageRun) -> list[str]:
    return [step.kind for step in run.steps]


def budgeted(llm, documents, model, tmp_path, *, reply=SIXTY, **kwargs):
    """``surface_relevance`` on v2 under a fresh budget, the model keeping to it, one
    document at a time; returns the context, the stage's run and the budget."""
    logger = TaskLogger("run", root=tmp_path)
    run = StageRun(stage="relevance", task_logger=logger)
    splits = Splits(documents, model=model, task_logger=logger)
    budget = kwargs.pop("budget", None) or Budget(splits, task_logger=logger)
    context = surface_relevance(
        task=TASK,
        documents=documents,
        model=model,
        prompt_version="v2",
        run=run,
        workers=1,
        splits=splits,
        budget=budget,
        completion=llm.answers(relevance=reply),
        **kwargs,
    )
    return context, run, budget


@pytest.mark.parametrize("others", ["", "N(DOC1,r1)\n\n---\n\nN(DOC2,r1)"])
def test_v2_is_v1_byte_for_byte_without_a_budget(others):
    v1 = load_prompt("relevance", version="v1", task=TASK, other_snippets=others)
    assert (
        load_prompt("relevance", version="v2", task=TASK, other_snippets=others) == v1
    )
    assert (
        load_prompt(
            "relevance", version="v2", task=TASK, other_snippets=others, max_words=None
        )
        == v1
    )


def test_v2_asks_each_note_to_stay_under_its_budget():
    v1 = load_prompt("relevance", version="v1", task=TASK, other_snippets="")
    v2 = load_prompt(
        "relevance", version="v2", task=TASK, other_snippets="", max_words=16
    )
    assert v2.count(SENTENCE) == 1
    assert v2.replace(SENTENCE, "") == v1


def test_only_a_prompt_that_renders_the_budget_takes_one(tmp_path):
    shipped = yaml.safe_load((settings.PROMPTS_DIR / "relevance/v1.yaml").read_text())
    for version, extra in (("v8", ""), ("v9", " At most {{ max_words }} words.")):
        path = tmp_path / "prompts" / "relevance" / f"{version}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump({**shipped, "instructions": shipped["instructions"] + extra})
        )
    taken = {v: takes_word_budget(v) for v in ("v1", "v2", "v8", "v9")}
    assert taken == {"v1": False, "v2": True, "v8": False, "v9": True}


def test_note_texts_lists_each_document_or_part_note_without_the_empty_ones():
    assert note_texts(["a ", ["b", " ", "c\n"], "", "d"]) == ["a", "b", "c", "d"]
    assert note_texts([]) == []


def test_notes_too_long_for_the_next_round_are_read_again_first(
    llm, window, within_budget, tmp_path
):
    # The last document is the longest, so its round-2 request has the least room.
    documents = [f"DOC{i}" for i in range(9)] + ["DOC9, the longest of the ten"]
    context, run, _ = budgeted(
        llm, documents, window(WINDOW_1200), tmp_path, reply=within_budget(SIXTY)
    )
    assert kinds(run) == [
        *[f"relevance-r1-d{i}" for i in range(10)],
        *[f"relevance-r1-w30-d{i}" for i in range(10)],
        *[f"relevance-r2-w30-d{i}" for i in range(10)],
    ]
    assert context.words == [30, 30]
    assert context.snippets == [" ".join(["word"] * 30)] * 10
    asked = [r["messages"][0]["content"] for r in llm.requests[10:]]
    assert all(SENTENCE.replace("16", "30") in prompt for prompt in asked)
    [event] = json.loads((run.task_logger.dir / "notes.json").read_text())["events"]
    assert (event["call"], event["cause"], event["round"], event["words"]) == (
        "relevance-r2-d9",
        "estimate",
        1,
        30,
    )


def test_a_round_refused_for_its_notes_reads_the_previous_round_again(
    llm, within_budget, tmp_path
):
    # Every round-2 request is about 6,400 characters; the last document's is 6,600.
    documents = [f"DOC{i}" for i in range(9)] + ["DOC9 " + "x" * 200]
    context, run, _ = budgeted(
        llm.refuses_over(6_500), documents, QWEN, tmp_path, reply=within_budget(SIXTY)
    )
    assert context.words == [30, 30]
    [event] = json.loads((run.task_logger.dir / "notes.json").read_text())["events"]
    assert (event["call"], event["cause"], event["room"]) == (
        "relevance-r2-d9",
        "refusal",
        None,
    )
    assert (event["discarded"], event["round"], event["words"]) == (9, 1, 30)
    assert kinds(run)[10:19] == [f"relevance-r2-d{i}" for i in range(9)]
    assert kinds(run)[19:] == [
        *[f"relevance-r1-w30-d{i}" for i in range(10)],
        *[f"relevance-r2-w30-d{i}" for i in range(10)],
    ]


def test_reread_reads_only_the_last_round_again(llm, window, tmp_path):
    documents = ["DOC0", "DOC1", "DOC2"]
    model = window(1_000_000)
    first, _, budget = budgeted(llm, documents, model, tmp_path)
    budget.words = 16
    llm.requests.clear()
    again, run, _ = budgeted(
        llm, documents, model, tmp_path, budget=budget, reread=first
    )
    assert kinds(run) == [f"relevance-r2-w16-d{i}" for i in range(3)]
    assert again.rounds[0] == first.rounds[0]
    assert again.words == [None, 16]
    assert all(SENTENCE in r["messages"][0]["content"] for r in llm.requests)


def test_the_final_check_reads_the_last_round_again_until_it_passes(
    llm, window, within_budget, tmp_path
):
    checked: list[list[str]] = []

    def over_twenty_words(notes):
        checked.append(note_texts(notes))
        if max(len(note.split()) for note in note_texts(notes)) > 20:
            raise NotesTooLong(
                "reasoning did not fit.",
                model=QWEN,
                call="reasoning",
                cause="estimate",
                estimate=9_000,
                notes=note_texts(notes),
                notes_tokens=1_000,
            )

    documents = ["DOC0", "DOC1", "DOC2"]
    context, run, _ = budgeted(
        llm,
        documents,
        window(1_000_000),
        tmp_path,
        reply=within_budget(SIXTY),
        final_check=over_twenty_words,
    )
    assert [len(notes[0].split()) for notes in checked] == [60, 30, 15]
    assert kinds(run)[6:] == [
        *[f"relevance-r2-w30-d{i}" for i in range(3)],
        *[f"relevance-r2-w15-d{i}" for i in range(3)],
    ]
    assert context.words == [None, 15]
    assert context.rounds[0] == [SIXTY] * 3


def test_without_a_budget_a_round_stops_where_more_parts_cannot_help(
    llm, window, tmp_path
):
    reports = [f"Report {i:02d}. " + ROUTINE * 5 for i in range(40)]
    logger = TaskLogger("run", root=tmp_path)
    model = window(6_000)
    with pytest.raises(litellm.ContextWindowExceededError) as stop:
        surface_relevance(
            task=TASK,
            documents=reports,
            model=model,
            prompt_version="v2",
            splits=Splits(reports, model=model, task_logger=logger),
            completion=llm.answers(relevance=(ROUTINE * 14)[:1_000]),
        )
    [note] = stop.value.__notes__
    assert note.startswith(
        "r3con: reading documents[0] (Document 1) in more parts cannot help in "
        "relevance-r2-d0: "
    )
    assert len(llm.requests) == 40
    assert (logger.dir / "splits.json").exists()
    assert not (logger.dir / "notes.json").exists()


@pytest.mark.parametrize(
    "options",
    [
        {"reread": RelevantContext(snippets=["a"], rounds=[["a"]])},
        {"final_check": lambda notes: None},
    ],
    ids=["reread", "final-check"],
)
def test_reread_or_a_final_check_without_a_budget_is_refused(llm, options):
    with pytest.raises(ValueError, match="budget"):
        surface(llm, ["DOC0"], rounds=2, **options)
    assert llm.requests == []
