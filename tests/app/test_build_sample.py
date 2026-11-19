from collections.abc import Callable

import pytest

from r3con import Answer, ask, read_documents, run
from tests.app.build_sample import (
    APP_JS,
    MEMOS,
    app_block,
    ask_entry,
    build_sample,
    built_blocks,
)
from tests.fakes import (
    ANSWERING,
    FakeLLM,
    Raw,
    Request,
    citing,
    in_json,
    quoting_its_first_line_and_citing_all_shown,
    reading,
    replying_when,
)

EASTFIELD = "MEMO — Eastfield site, Q3 operations review"
REGISTRY = "APPROVED CONTRACTOR REGISTRY (extract, revised July)"
CUT_OFF = Raw('{"summary": "Notes on and on and', "length")


@pytest.fixture
def todays_run(
    answering_llm: FakeLLM, shown_ids: Callable[[Request], list[str]], tmp_path
) -> Answer:
    """A run over the memos as this r3con saves it, its answer citing the first summary
    and the last parse; ``answering_llm`` answering so from then on."""
    answering_llm.answers(
        relevance=quoting_its_first_line_and_citing_all_shown(shown_ids),
        reasoning=citing("Halloran memo", slice(1), slice(-1, None)),
    )
    documents = read_documents(MEMOS)
    return run("Who?", documents, completion=answering_llm, logs_dir=tmp_path)


@pytest.mark.parametrize("name", ["SAMPLE", "RECORDED_ASKS"])
def test_app_js_holds_the_block_its_run_folder_builds(name: str) -> None:
    app = APP_JS.read_text(encoding="utf-8")
    assert app_block(app, name) == built_blocks()[name]


def test_a_run_and_ask_this_r3con_saves_build_with_their_saved_provenance(
    todays_run: Answer, answering_llm: FakeLLM
) -> None:
    asked = ask(todays_run, "Who else?", completion=answering_llm)
    assert todays_run.run_dir is not None and asked.run_dir is not None
    assert todays_run.provenance["quotes"] and todays_run.provenance["trail"]["edges"]
    built = build_sample(todays_run.run_dir, MEMOS)
    assert built["answer"]["provenance"] == todays_run.provenance
    assert ask_entry(asked.run_dir)["provenance"] == asked.provenance


def test_a_saved_trail_trace_does_not_give_fails_the_build(todays_run: Answer) -> None:
    assert todays_run.run_dir is not None
    in_json(
        "reasoning/result.json",
        lambda result: result["provenance"]["trail"].update(rule="another rule"),
    )(todays_run.run_dir)
    with pytest.raises(AssertionError, match="trace gives the saved provenance"):
        build_sample(todays_run.run_dir, MEMOS)


def test_a_run_that_skipped_builds_with_its_skips_and_one_call_per_node(
    answering_llm: FakeLLM, shown_ids: Callable[[Request], list[str]], tmp_path
) -> None:
    relevance = quoting_its_first_line_and_citing_all_shown(shown_ids)
    answering_llm.answers(
        relevance=replying_when(
            reading(EASTFIELD, later_round=True), CUT_OFF, relevance
        ),
        parsing=replying_when(reading(REGISTRY), Raw("{}"), ANSWERING["parsing"]),
        reasoning=citing("Halloran memo", slice(1), slice(-1, None)),
    )
    answer = run(
        "Who?", read_documents(MEMOS), completion=answering_llm, logs_dir=tmp_path
    )
    assert answer.run_dir is not None
    built = build_sample(answer.run_dir, MEMOS)
    note = [{"part": None, "reason": "output context length"}]
    parse = [{"part": None, "reason": "schema error"}]
    rounds = [[s.get("skipped") for s in round_] for round_ in built["rounds"]]
    assert rounds == [[None] * 5, [None, note, None, None, None]]
    assert [p.get("skipped") for p in built["parsed"]] == [None] * 3 + [parse, None]
    # Each skipped call was sent three times.
    assert built["calls"]["r2-1"]["tokens"] == 3 * built["calls"]["r2-0"]["tokens"]
    assert (
        built["calls"]["parsed-3"]["tokens"] == 3 * built["calls"]["parsed-0"]["tokens"]
    )
