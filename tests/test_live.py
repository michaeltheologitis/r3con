import os
import re
from pathlib import Path
from typing import Any

import pytest

from r3con import ask, read_documents, run
from r3con.config import REASONING_PATHS

MEMOS = Path(__file__).resolve().parents[1] / "examples" / "memos"
QUESTION = (
    "Which maintenance contractor's sites logged the most equipment incidents in Q3 "
    "in total, and how many? Give the contractor's name, not its code."
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set"
    ),
]


def assert_cost_reported(events: list[dict[str, Any]], finished: str) -> None:
    """The last of ``events`` is ``finished``, with the tokens and dollars spent."""
    assert events[-1]["type"] == finished
    assert events[-1]["usd"] > 0
    assert events[-1]["tokens"]["total"] > 0


def cited_documents(provenance: dict[str, list[Any]]) -> set[str]:
    """The ids of the documents whose snippets or records ``provenance`` cites."""
    views = (provenance["summary_ids"], provenance["parse_ids"])
    return {id_.split(".")[0] for ids in views for id_ in ids}


@pytest.mark.parametrize("reasoning", REASONING_PATHS)
def test_the_answer_joins_evidence_that_no_single_memo_holds(
    tmp_path, events_in, reasoning
):
    documents = read_documents(MEMOS)
    result = run(QUESTION, documents, reasoning=reasoning, logs_dir=tmp_path)
    assert "halloran" in result.answer.lower()
    assert re.search(r"\b11\b", result.answer)
    used = cited_documents(result.provenance)
    cited = [documents[result.document_ids.index(id_)] for id_ in used]
    assert cited
    assert not any("Cafeteria refurbishment" in document for document in cited)
    assert result.run_dir is not None
    assert_cost_reported(events_in(result.run_dir), "run_finished")


def test_an_ask_answers_a_new_question_over_a_finished_run(tmp_path, events_in):
    result = run(QUESTION, read_documents(MEMOS), logs_dir=tmp_path)
    asked = ask(result, "Which site logged the fewest equipment incidents in Q3?")
    assert "northgate" in asked.answer.lower()
    used = cited_documents(asked.provenance)
    assert used
    assert used <= set(result.document_ids)
    assert asked.run_dir is not None
    assert_cost_reported(events_in(asked.run_dir), "ask_finished")
