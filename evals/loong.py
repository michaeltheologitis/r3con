"""Loong (EMNLP 2024, arXiv 2406.17419): multi-document QA whose evidence is scattered
across every document of a task, in English and Chinese, judged on a 1–100 rating.

Ported from r3con-evaluation (commit d71a54aaf15be6a47c003df442fd01ef371c1714):
``evals/benchmarks/loong/loader.py`` for the documents, including its fix that hides
the verdicts in the level-4 legal task (upstream checks the wrong field);
``evals/benchmarks/loong/judge.py`` for the rubric; and the r3con harness's adapter,
which asks the instruction, then the question. Changed: the judge is
``openai/gpt-6-luna`` on default settings, with no seed, and both downloads are pinned
by SHA-256: the task file from that commit, the document pool from upstream's bucket.
"""

from __future__ import annotations

import functools
import json
import os
import statistics
import tempfile
import threading
import zipfile
from pathlib import Path
from typing import Any

import jinja2
from pydantic import BaseModel, Field

from evals import CACHE_DIR, ask_judge, download, rows_by_id
from r3con.runs import StageRun

TASKS_URL = (
    "https://raw.githubusercontent.com/michaeltheologitis/r3con-evaluation/"
    "d71a54aaf15be6a47c003df442fd01ef371c1714/evals/benchmarks/loong/data/loong.jsonl"
)
TASKS_SHA256 = "cd640d3ca0e41d3638153506e3caee109850b30de0a4e829862232a2b6302da8"
DOCS_URL = "https://alibaba-research.oss-cn-beijing.aliyuncs.com/loong/doc.zip"
DOCS_SHA256 = "ceb75be50f7f114f231e3a8069b51457745ea7db151c017f00de33d0789ab4cc"
CACHE = CACHE_DIR / "loong"
WORST_SCORE = 1
PERFECT_SCORE = 100
CHAIN_OF_REASONING = 4
SCORE_DIGITS = 2
RATE_DIGITS = 3

# Upstream gates the verdict on this marker in the instruction, but it is in the question.
_LEGAL_VERDICT_MARKER = "阅读以上判决文书，我将给你若干份判决结果："
_pool_lock = threading.Lock()

SYSTEM_PROMPT = (
    "We would like to request your feedback on the performance of the AI "
    "assistant in response to the user question displayed below according to the "
    "gold answer. Please use the following listed aspects and their descriptions "
    "as evaluation criteria:\n"
    "    - Accuracy and Hallucinations: The assistant's answer is semantically "
    "consistent with the gold answer; The numerical value and order need to be "
    "accurate, and there should be no hallucinations.\n"
    "    - Completeness: Referring to the reference answers, the assistant's "
    "answer should contain all the key points needed to answer the user's "
    "question; further elaboration on these key points can be omitted.\n"
    "Please rate whether this answer is suitable for the question. Please note "
    "that the gold answer can be considered as a correct answer to the question.\n\n"
    "The assistant receives an overall score on a scale of 1 to 100, where a "
    "higher score indicates better overall performance.\n"
    "Please note that if the assistant's answer and the gold answer fully meet "
    "the above criteria, its overall rating should be the full marks (100).\n"
    "Please first provide a comprehensive explanation of your evaluation, "
    "avoiding any potential bias.\n\n"
    "Return your evaluation as structured fields: a `rationale` (your evaluation "
    "explanation, no more than 100 words) and an integer `rating` on the 1-to-100 "
    "scale."
)
JUDGE_PROMPT = jinja2.Template(
    "[Question]\n"
    "{{ question }}\n\n"
    "[Gold Answer]\n"
    "{{ gold }}\n\n"
    "[The Start of Assistant's Predicted Answer]\n"
    "{{ answer }}\n"
    "[The End of Assistant's Predicted Answer]"
)


class Rating(BaseModel):
    rationale: str
    rating: int = Field(ge=WORST_SCORE, le=PERFECT_SCORE)


def rows() -> dict[str, dict[str, Any]]:
    """Every Loong task by id, after fetching the task file once."""
    return rows_by_id(download(TASKS_URL, CACHE / "loong.jsonl", TASKS_SHA256))


def inputs(row: dict[str, Any]) -> tuple[str, list[str]]:
    """The question r3con is asked, the instruction then the question, and the task's
    documents, in the order its gold answer refers to them."""
    doc_dir, legal = _pool()
    task = "\n\n".join(part for part in (row["instruction"], row["question"]) if part)
    documents = []
    for index, name in enumerate(row["doc"]):
        heading, content = _resolve_doc(doc_dir, legal, row, index, name)
        documents.append(f"{heading}\n{content}\n\n")
    return task, documents


def _pool() -> tuple[Path, dict[str, dict[str, str]]]:
    """The document pool and its legal cases, fetched and read once across threads."""
    with _pool_lock:
        return _load_pool(CACHE)


@functools.cache
def _load_pool(cache: Path) -> tuple[Path, dict[str, dict[str, str]]]:
    doc_dir = cache / "doc"
    if not doc_dir.is_dir():
        _extract(download(DOCS_URL, cache / "doc.zip", DOCS_SHA256), doc_dir)
    legal = json.loads((doc_dir / "legal" / "legal.json").read_text(encoding="utf-8"))
    return doc_dir, legal


def _extract(archive: Path, target: Path) -> None:
    # The zip stores its Chinese file names as UTF-8 without the flag that says so.
    with (
        tempfile.TemporaryDirectory(dir=target.parent) as staging,
        zipfile.ZipFile(archive, metadata_encoding="utf-8") as pool,
    ):
        pool.extractall(staging)
        os.replace(Path(staging) / "doc", target)


def _resolve_doc(
    doc_dir: Path, legal: dict[str, dict[str, str]], row: dict, index: int, name: str
) -> tuple[str, str]:
    """One document's heading and text, as upstream's ``get_content`` builds them."""
    doc_type = row["type"]
    type_dir = doc_dir / doc_type
    if doc_type == "financial":
        pattern = (
            f"*{name}*.txt"
            if row["level"] == CHAIN_OF_REASONING
            else f"*2024-{name}*.txt"
        )
        # 60 tasks name a company with two or more files: take the first, sorted.
        matches = sorted(type_dir.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No Loong financial document matching {pattern!r} under {type_dir}"
            )
        path = matches[0]
        return f"《{path.stem.split('-')[-1]}》", path.read_text(encoding="utf-8")
    if doc_type == "paper":
        content = (type_dir / name).read_text(encoding="utf-8")
        return content.split("\n", 1)[0].strip("#").strip(), content
    if doc_type == "legal":
        case = legal[name]
        hide_verdict = (
            row["level"] == CHAIN_OF_REASONING
            and _LEGAL_VERDICT_MARKER in row["question"]
        )
        content = case["content"] if hide_verdict else case["content"] + case["result"]
        return f"《判决文书{index + 1}》", content
    raise ValueError(
        f"Unsupported Loong doc_type {doc_type!r} "
        f"(expected 'financial', 'paper', or 'legal')."
    )


def judge(
    row: dict[str, Any], answer: str, *, run: StageRun | None = None
) -> tuple[int, str | None]:
    """The 1–100 rating of ``answer`` and the judge's rationale; an empty answer gets
    the worst rating without a call."""
    if not answer.strip():
        return WORST_SCORE, None
    gold = row["answer"]
    if not isinstance(gold, str):
        gold = json.dumps(gold, ensure_ascii=False, indent=2)
    user = JUDGE_PROMPT.render(question=_judge_question(row), gold=gold, answer=answer)
    verdict = ask_judge(SYSTEM_PROMPT, user, Rating, run)
    return verdict.rating, verdict.rationale


def _judge_question(row: dict[str, Any]) -> str:
    """The task's own upstream prompt template, filled without the documents."""
    return (
        row["prompt_template"]
        .replace("{docs}", "")
        .replace("{question}", row["question"])
        .replace("{instruction}", row["instruction"])
    )


def metrics(scores: list[int]) -> dict[str, float]:
    """Avg Score (the mean rating) and Perfect Rate (the share rated 100), as in Loong."""
    perfect = sum(score == PERFECT_SCORE for score in scores)
    return {
        "avg_score": round(statistics.fmean(scores), SCORE_DIGITS),
        "perfect_rate": round(perfect / len(scores), RATE_DIGITS),
    }
