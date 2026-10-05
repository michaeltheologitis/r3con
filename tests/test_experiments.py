import itertools
import json
import os
import re
import time
from pathlib import Path

import litellm
import pytest

from r3con import read_documents, run

MEMOS = Path(__file__).resolve().parents[1] / "examples" / "memos"
QUESTION = (
    "Which maintenance contractor's sites logged the most equipment incidents in Q3 "
    "in total, and how many? Give the contractor's name, not its code."
)
# The longest user message the wrapped model accepts, for the parts it is read in. The
# registry grown to 40,000 characters is refused whole at 24,000, and its halves (about
# 20,000) at 12,000; the model's own mapped window never splits these sizes.
LIMITS = {1: None, 2: 24_000, 4: 12_000}
STAGES = ("relevance", "structuring/schema", "structuring/parsing", "reasoning")

pytestmark = [
    pytest.mark.live,
    pytest.mark.experiment,
    pytest.mark.enable_socket,
    pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY is not set"
    ),
]


def refusing(limit: int | None):
    """``litellm.completion``, refusing a two-message request whose user message is
    longer than ``limit`` characters as a provider refuses one too long for its window."""

    def completion(**request):
        messages = request["messages"]
        if (
            limit is not None
            and len(messages) == 2
            and len(messages[1]["content"]) > limit
        ):
            raise litellm.ContextWindowExceededError(
                message=f"The user message is over {limit} characters.",
                model=request["model"],
                llm_provider="openai",
            )
        return litellm.completion(**request)

    return completion


def strings(value) -> list[str]:
    """Every string inside ``value``, however deeply nested."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, list):
        return [text for item in value for text in strings(item)]
    return []


def mapping_records(structured_context: dict) -> list[dict]:
    """The records from the registry (Document 4) that map CT-118 to Halloran."""
    return [
        record
        for records in structured_context.values()
        if isinstance(records, list)
        for record in records
        if isinstance(record, dict) and record.get("document") == 4
        if "CT-118" in " ".join(strings(record))
        and "halloran" in " ".join(strings(record)).lower()
    ]


@pytest.mark.parametrize(
    ("parts", "seed"), list(itertools.product((1, 2, 4), (1, 2, 3)))
)
def test_a_document_read_in_parts_still_answers(tmp_path, grown_registry, parts, seed):
    documents = read_documents(MEMOS)
    documents[3] = grown_registry(40_000, at=0.6)
    started = time.perf_counter()
    result = run(
        QUESTION,
        documents,
        params={"seed": seed},
        completion=refusing(LIMITS[parts]),
        logs_dir=tmp_path,
    )
    assert result.run_dir is not None
    splits = result.run_dir / "splits.json"
    read = json.loads(splits.read_text())["documents"]["3"] if splits.exists() else {}
    totals = [
        json.loads((result.run_dir / stage / "result.json").read_text())["totals"]
        for stage in STAGES
    ]
    tokens = [total["tokens"] or {"prompt": 0, "completion": 0} for total in totals]
    records = mapping_records(result.structured_context)
    # The measurement, one line per case in the job's log, before anything is asserted.
    print(
        "E6",
        json.dumps(
            {
                "parts": parts,
                "seed": seed,
                "parts_read": read.get("parts", {}),
                "refusals": len(read.get("events", [])),
                "answer_has_halloran": "halloran" in result.answer.lower(),
                "answer_has_11": bool(re.search(r"\b11\b", result.answer)),
                "mapping_records": len(records),
                "calls": sum(total["n_steps"] for total in totals),
                "prompt_tokens": sum(t["prompt"] for t in tokens),
                "completion_tokens": sum(t["completion"] for t in tokens),
                "seconds": round(time.perf_counter() - started),
                "answer": result.answer,
            }
        ),
    )
    assert read.get("parts", {}).get("relevance-r1", 1) == parts
    assert splits.exists() is (parts > 1)
    assert "halloran" in result.answer.lower()
    assert re.search(r"\b11\b", result.answer)
    assert len(records) == 1
