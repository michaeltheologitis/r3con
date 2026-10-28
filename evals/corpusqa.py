"""CorpusQA (Tongyi-Zhiwen, 2025): analytical questions (filter, rank, aggregate) over a
bundle of table-rich documents, here the 128k tier, judged right or wrong.

Ported from r3con-evaluation (commit d71a54aaf15be6a47c003df442fd01ef371c1714):
``evals/benchmarks/corpusqa/loader.py`` for splitting each row's frozen prompt into its
instruction and documents; ``evals/benchmarks/corpusqa/judge.py`` for
``extract_answer`` and the equivalence rubric; and the r3con harness's adapter, which
asks the question, then the instruction. Changed: the judge is ``openai/gpt-6-luna`` on
default settings, with no seed; the tier is 128k rather than 1m; and the download is
pinned to a Hugging Face revision and its SHA-256.
"""

from __future__ import annotations

import re
import statistics
from typing import Any

import jinja2
from pydantic import BaseModel

from evals import CACHE_DIR, ask_judge, download, rows_by_id
from r3con.runs import StageRun

DATA_URL = (
    "https://huggingface.co/datasets/Tongyi-Zhiwen/CorpusQA/resolve/"
    "272dee3575d2b84e17890483253b92a08a39c87e/128k_4domains.jsonl"
)
DATA_SHA256 = "227fa5d6b0011d0e4e63ac27587421b13f500f73b9d468083ca9a323f5608ef6"
CACHE = CACHE_DIR / "corpusqa"
WORST_SCORE = 0
RATE_DIGITS = 3

_DOC_HEADER_RE = re.compile(r"# Document \d+:\n")
_QUESTION_SEP = "\n# Question:\n"
_ANSWER_RE = re.compile(r"The answer is: (.*)")

SYSTEM_PROMPT = (
    "You are an expert in verifying if two answers are the same.\n"
    "Your input is a problem and two answers, Answer 1 and Answer 2. You need to "
    "check if they are equivalent.\n"
    "Your task is to determine if two answers are equivalent, without attempting to "
    "solve the original problem.\n"
    "Compare the answers to verify they represent identical values or meaning, even "
    "when written in different forms or notations.\n\n"
    "First provide an explanation for why the answers are equivalent or not, then "
    "give your verdict as the boolean `equivalent` field."
)
JUDGE_PROMPT = jinja2.Template(
    "\nProblem: {{ problem }}\nAnswer 1: {{ answer_1 }}\nAnswer 2: {{ answer_2 }}\n",
    keep_trailing_newline=True,
)


class Equivalence(BaseModel):
    explanation: str
    equivalent: bool


def rows() -> dict[str, dict[str, Any]]:
    """Every 128k task by id, after fetching the tier file once."""
    return rows_by_id(download(DATA_URL, CACHE / "128k_4domains.jsonl", DATA_SHA256))


def inputs(row: dict[str, Any]) -> tuple[str, list[str]]:
    """The question r3con is asked, the question then the instruction, and the task's
    documents."""
    instruction, documents = _split_prompt(row)
    task = "\n\n".join(part for part in (row["question"], instruction) if part)
    return task, documents


def _split_prompt(row: dict[str, Any]) -> tuple[str, list[str]]:
    """The instruction and documents baked into the row's user message: each document
    behind a ``# Document N:`` header, then ``# Question:``, the question and the
    instruction."""
    user = next(
        message["content"] for message in row["prompt"] if message["role"] == "user"
    )
    body, _, tail = user.rpartition(_QUESTION_SEP)
    instruction = tail.removeprefix(row["question"]).strip()
    documents = [chunk.strip() for chunk in _DOC_HEADER_RE.split(body)[1:]]
    return instruction, documents


def extract_answer(response: str) -> str:
    """The text after "The answer is: ", the contract every task's instruction sets,
    else the whole response, as upstream's ``eval.py``."""
    match = _ANSWER_RE.search(response)
    if match:
        return match.group(1).strip()
    return response.strip()


def judge(
    row: dict[str, Any], answer: str, *, run: StageRun | None = None
) -> tuple[int, str | None]:
    """1 if ``answer`` is equivalent to the gold, else 0, and the judge's explanation;
    an empty answer scores 0 without a call."""
    extracted = extract_answer(answer)
    if not extracted:
        return WORST_SCORE, None
    user = JUDGE_PROMPT.render(
        problem=row["question"], answer_1=extracted, answer_2=str(row["answer"])
    )
    verdict = ask_judge(SYSTEM_PROMPT, user, Equivalence, run)
    return int(verdict.equivalent), verdict.explanation


def metrics(scores: list[int]) -> dict[str, float]:
    return {"accuracy": round(statistics.fmean(scores), RATE_DIGITS)}
