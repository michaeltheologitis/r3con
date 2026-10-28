import os
import re
from pathlib import Path

import pytest

from r3con import read_documents, run
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


@pytest.mark.parametrize("reasoning", REASONING_PATHS)
def test_the_answer_joins_evidence_that_no_single_memo_holds(tmp_path, reasoning):
    documents = read_documents(MEMOS)
    result = run(QUESTION, documents, reasoning=reasoning, logs_dir=tmp_path)
    assert "halloran" in result.answer.lower()
    assert re.search(r"\b11\b", result.answer)
    used = {id_.split(".")[0] for ids in result.provenance.values() for id_ in ids}
    cited = [documents[result.document_ids.index(id_)] for id_ in used]
    assert cited
    assert not any("Cafeteria refurbishment" in document for document in cited)
