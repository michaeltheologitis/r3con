import os
import re
from pathlib import Path

import pytest

from r3con import read_documents, run

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


def test_the_answer_joins_evidence_that_no_single_memo_holds(tmp_path):
    result = run(QUESTION, read_documents(MEMOS), logs_dir=tmp_path)
    assert "halloran" in result.answer.lower()
    assert re.search(r"\b11\b", result.answer)
