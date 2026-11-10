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
from tests.fakes import Content, FakeLLM, Request, citing, in_json, is_later_round


def quoting_the_first_line(
    shown_ids: Callable[[Request], list[str]],
) -> Callable[[Request], Content]:
    """A relevance reply quoting the document's first line, and in a later round citing
    the first id shown."""

    def reply(request: Request) -> Content:
        first = request["messages"][1]["content"].splitlines()[0]
        content = {"summary": f"Notes on {first}.", "quotes": [first]}
        if is_later_round(request):
            content["provenance_ids"] = shown_ids(request)[:1]
        return content

    return reply


@pytest.fixture
def todays_run(
    answering_llm: FakeLLM, shown_ids: Callable[[Request], list[str]], tmp_path
) -> Answer:
    """A run over the memos as this r3con saves it, its answer citing the first summary
    and the last parse; ``answering_llm`` answering so from then on."""
    answering_llm.answers(
        relevance=quoting_the_first_line(shown_ids),
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


def test_a_saved_provenance_trace_does_not_give_fails_the_build(
    todays_run: Answer,
) -> None:
    assert todays_run.run_dir is not None
    in_json(
        "reasoning/result.json", lambda result: result["provenance"]["quotes"].pop()
    )(todays_run.run_dir)
    with pytest.raises(AssertionError, match="trace gives the saved provenance"):
        build_sample(todays_run.run_dir, MEMOS)
