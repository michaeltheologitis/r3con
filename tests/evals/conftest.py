import json
from pathlib import Path
from typing import Any

import litellm
import pytest

from evals.loong import _LEGAL_VERDICT_MARKER

CORPUSQA_INSTRUCTION = "Output requirements:\nEnd with: The answer is: xxx."
LOONG_FILES = {
    "financial/report_7-2024-ACME INC.-Annual.txt": "ACME's 2024 revenue was 7 million.",
    "financial/report_6-2023-ACME INC.-Annual.txt": "ACME's 2023 revenue was 6 million.",
    "paper/2402.01739.md": "# Sparse Notes\nWe study notes.",
}
LOONG_LEGAL = {
    "case A": {"content": "Body A.", "result": " Verdict A."},
    "case B": {"content": "Body B.", "result": " Verdict B."},
}
LOONG_ROWS = [
    {
        "id": "loong-financial",
        "type": "financial",
        "level": 1,
        "doc": ["ACME INC."],
        "instruction": "Answer from the reports.",
        "question": "What was ACME's revenue?",
        "answer": "7 million",
    },
    {
        "id": "loong-financial-years",
        "type": "financial",
        "level": 4,
        "doc": ["2023-ACME INC."],
        "instruction": "Answer from the reports.",
        "question": "What was ACME's revenue in 2023?",
    },
    {
        "id": "loong-paper",
        "type": "paper",
        "level": 4,
        "doc": ["2402.01739.md"],
        "instruction": "List the citations.",
        "question": "",
        "answer": {"Reference": ["Sparse Notes"], "Citation": []},
    },
    {
        "id": "loong-legal",
        "type": "legal",
        "level": 4,
        "doc": ["case A", "case B"],
        "instruction": "Read the judgments.",
        "question": "Which case was appealed?",
    },
    {
        "id": "loong-verdicts",
        "type": "legal",
        "level": 4,
        "doc": ["case A", "case B"],
        "instruction": "Read the judgments.",
        "question": f"{_LEGAL_VERDICT_MARKER} Verdict A. Verdict B.",
    },
]


@pytest.fixture
def judging_llm(answering_llm, monkeypatch: pytest.MonkeyPatch):
    """``answering_llm`` as litellm's completion, judging too: a Loong answer rates 100
    when the question names ACME and 40 otherwise, and a CorpusQA answer is
    equivalent when the gold is 42."""

    def parse_or_judge(request: dict[str, Any]) -> str:
        schema = request["response_format"]["json_schema"]["name"]
        user = request["messages"][1]["content"]
        if schema == "Rating":
            rating = 100 if "ACME" in user else 40
            return json.dumps({"rationale": "Checked the gold.", "rating": rating})
        if schema == "Equivalence":
            equivalent = "Answer 2: 42\n" in user
            return json.dumps({"explanation": "Checked.", "equivalent": equivalent})
        first_line = user.splitlines()[0]
        return json.dumps({"rows": [{"who": first_line}]})

    answering_llm.answers(parsing=parse_or_judge)
    monkeypatch.setattr(litellm, "completion", answering_llm)
    return answering_llm


@pytest.fixture
def loong_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Loong cache, already downloaded, holding ``LOONG_ROWS`` and their documents."""
    cache = tmp_path / "cache" / "loong"
    for name, text in LOONG_FILES.items():
        (cache / "doc" / name).parent.mkdir(parents=True, exist_ok=True)
        (cache / "doc" / name).write_text(text, encoding="utf-8")
    (cache / "doc" / "legal").mkdir()
    legal = json.dumps(LOONG_LEGAL, ensure_ascii=False)
    (cache / "doc" / "legal" / "legal.json").write_text(legal, encoding="utf-8")
    template = "{docs}{instruction}\n\n{question}"
    write_jsonl(
        cache / "loong.jsonl",
        [{"prompt_template": template, **row} for row in LOONG_ROWS],
    )
    monkeypatch.setattr("evals.loong.CACHE", cache)
    return cache


@pytest.fixture
def corpusqa_instruction() -> str:
    return CORPUSQA_INSTRUCTION


@pytest.fixture
def corpusqa_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A CorpusQA cache, already downloaded, holding two tasks over two stores."""
    cache = tmp_path / "cache" / "corpusqa"
    stores = ["Store A sold 40 units.", "Store B sold 2 units."]
    write_jsonl(
        cache / "128k_4domains.jsonl",
        [
            corpusqa_row("sales_en_1", "How many units were sold?", 42, stores),
            corpusqa_row("sales_en_2", "How many stores sold units?", 7, stores),
        ],
    )
    monkeypatch.setattr("evals.corpusqa.CACHE", cache)
    return cache


def corpusqa_row(task_id: str, question: str, answer: Any, docs: list[str]) -> dict:
    """A CorpusQA row as upstream ships it: the documents, question and instruction
    baked into one user message."""
    body = "".join(f"# Document {i}:\n{doc}\n" for i, doc in enumerate(docs, 1))
    user = f"{body}\n# Question:\n{question}\n\n{CORPUSQA_INSTRUCTION}\n"
    return {
        "id": task_id,
        "question": question,
        "answer": answer,
        "prompt": [
            {"role": "system", "content": "You are a data analyst."},
            {"role": "user", "content": user},
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
