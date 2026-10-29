"""Run r3con on the pinned tasks, judge each answer, and track the scores over time.

    uv run python -m evals                      # a full run: writes its line to results.jsonl
    uv run python -m evals --limit 1            # one task per benchmark, writes nothing
    uv run python -m evals --resume evals/runs/<folder>

See ``evals/README.md``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import shutil
import subprocess
import time
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from types import ModuleType
from typing import Any

from litellm import exceptions

import r3con
from evals import JUDGE_MODEL, corpusqa, log, loong
from r3con.config import RunConfig, load_config
from r3con.logging_setup import configure_logging
from r3con.masking import masked
from r3con.runs import StageRun, TaskLogger, new_run_folder
from r3con.runtime.llm import is_temporary

EVALS_DIR = Path(__file__).resolve().parent
REPO_DIR = EVALS_DIR.parent
# The code a run's results depend on; a change here makes the tree dirty.
DIRTY_PATHS = ("src", "evals", ":(exclude)evals/results.jsonl")
TASKS_PATH = EVALS_DIR / "tasks.json"
RESULTS_PATH = EVALS_DIR / "results.jsonl"
RUNS_DIR = EVALS_DIR / "runs"
STARTED = "run.json"
ANSWER = "answer.json"
RECORD = "record.json"
BENCHMARKS: dict[str, ModuleType] = {"loong": loong, "corpusqa": corpusqa}
SAMPLE_SIZES = {"loong": 40, "corpusqa": 35}
SEED = 20261028
DEFAULT_WORKERS = 32
USD_DIGITS = 4
# A refused key, access or quota: the setup's problem, not r3con's.
SETUP_ERRORS = (
    exceptions.AuthenticationError,
    exceptions.PermissionDeniedError,
    exceptions.RateLimitError,
)


def sample(ids: Iterable[str], n: int) -> list[str]:
    """``n`` of ``ids`` drawn at random with the pinned seed: the same ids, in the same
    order, for the same input in any order."""
    return random.Random(SEED).sample(sorted(ids), n)


def draw() -> dict[str, Any]:
    """The contents of ``tasks.json``: the seed, and each benchmark's sample."""
    samples = {
        name: sample(BENCHMARKS[name].rows(), n) for name, n in SAMPLE_SIZES.items()
    }
    return {"seed": SEED, **samples}


def evaluate(
    pinned: dict[str, Any],
    folder: Path,
    results: Path,
    *,
    limit: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any] | None:
    """Run each pinned task, or the first ``limit`` of each benchmark, that has no
    record in ``folder`` yet. Once every one has a record, return the results line; a
    full run also writes it to ``results``, in place of an earlier line of the same
    run. Else log what is missing and return ``None``."""
    earlier = _results_lines(results) if limit is None else []
    config = load_config()
    header = _start(folder, config)
    selection = {name: pinned[name][:limit] for name in BENCHMARKS}
    pending = _pending(selection, folder)
    log.info("%s: %d task(s) to run, %d at a time", folder, len(pending), workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for name, row in pending:
            task_dir = folder / name / row["id"]
            future = pool.submit(run_task, name, row, task_dir, config)
            futures[future] = (name, row["id"])
        try:
            for future in as_completed(futures):
                _report(future, *futures[future])
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    paths = {
        name: [folder / name / task_id / RECORD for task_id in ids]
        for name, ids in selection.items()
    }
    missing = sum(not path.exists() for group in paths.values() for path in group)
    if missing:
        log.error("%d task(s) have no record; finish with --resume %s", missing, folder)
        return None
    records = {
        name: [json.loads(path.read_text(encoding="utf-8")) for path in group]
        for name, group in paths.items()
    }
    costs = [record["usd"] for group in records.values() for record in group]
    priced = [cost for cost in costs if cost is not None]
    line = {
        **header,
        "benchmarks": {name: _scores(name, group) for name, group in records.items()},
        "usd": round(sum(priced), USD_DIGITS) if len(priced) == len(costs) else None,
        "run": folder.name,
    }
    if limit is None:
        _save_line(results, earlier, line)
    return line


def run_task(
    name: str, row: dict[str, Any], task_dir: Path, config: RunConfig
) -> dict[str, Any]:
    """Answer one task with r3con unless its answer is saved already, judge the answer,
    and write the task's record."""
    if not (task_dir / ANSWER).exists():
        _write_json(task_dir / ANSWER, _answer(BENCHMARKS[name], row, task_dir, config))
    answered = json.loads((task_dir / ANSWER).read_text(encoding="utf-8"))
    judging = StageRun(
        stage="judge",
        task_logger=TaskLogger(task_dir.name, root=task_dir.parent),
        model=JUDGE_MODEL,
    )
    try:
        score, rationale = BENCHMARKS[name].judge(row, answered["answer"], run=judging)
    finally:
        judging.flush(write_transcript=False)
    record = {
        "benchmark": name,
        "task_id": row["id"],
        "score": score,
        "rationale": rationale,
        **answered,
        **spend(task_dir),
    }
    _write_json(task_dir / RECORD, record)
    return record


def _answer(
    benchmark: ModuleType, row: dict[str, Any], task_dir: Path, config: RunConfig
) -> dict[str, Any]:
    """r3con's answer to one task, from a fresh folder. A run that raises leaves its
    error and an empty answer, which the judge scores worst, unless the error
    ``is_temporary`` or is one of ``SETUP_ERRORS``: then it propagates, no answer is
    saved, and a resume retries it."""
    question, documents = benchmark.inputs(row)
    shutil.rmtree(task_dir, ignore_errors=True)
    started = time.monotonic()
    answer, error = "", None
    try:
        answer = r3con.run(question, documents, config=config, logs_dir=task_dir).answer
    except Exception as e:
        if is_temporary(e) or isinstance(e, SETUP_ERRORS):
            raise
        error = masked(f"{type(e).__name__}: {e}")
        log.warning("%s raised %s", row["id"], error)
    run_dir = next((path.parent for path in task_dir.glob("*/manifest.json")), None)
    return {
        "answer": answer,
        "error": error,
        "seconds": round(time.monotonic() - started, 1),
        "run_dir": str(run_dir) if run_dir else None,
    }


def spend(folder: Path) -> dict[str, Any]:
    """The tokens and cost of every call recorded under ``folder``, r3con's and the
    judge's. ``usd`` is ``None`` when litellm could not price one of the calls."""
    calls = [
        call
        for path in folder.rglob("calls.json")
        for call in json.loads(path.read_text(encoding="utf-8"))
    ]
    tokens = [call["tokens"] or {} for call in calls]
    priced = [call["usd"] for call in calls if call["usd"] is not None]
    return {
        "prompt_tokens": sum(used.get("prompt", 0) for used in tokens),
        "completion_tokens": sum(used.get("completion", 0) for used in tokens),
        "usd": sum(priced) if len(priced) == len(calls) else None,
    }


def _start(folder: Path, config: RunConfig) -> dict[str, Any]:
    """What the run is: written into ``folder`` when it starts, and checked against
    this checkout when it resumes, so one run never mixes two versions of r3con."""
    current = {
        "date": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(
            _git("status", "--porcelain", "--untracked-files=no", "--", *DIRTY_PATHS)
        ),
        "r3con_version": r3con.__version__,
        "label": config.label(),
        "judge_model": JUDGE_MODEL,
    }
    if not (folder / STARTED).exists():
        _write_json(folder / STARTED, current)
        return current
    started = json.loads((folder / STARTED).read_text(encoding="utf-8"))
    if started["dirty"] or current["dirty"]:
        when = "started with" if started["dirty"] else "now has"
        raise SystemExit(
            f"evals: cannot resume {folder}: the tree {when} uncommitted changes to "
            "src/ or evals/, so this checkout may not be the code it ran. Commit first, "
            "then start a new run."
        )
    if (started["commit"], started["label"]) != (current["commit"], current["label"]):
        raise SystemExit(
            f"evals: {folder} was started at commit {started['commit']} as "
            f"{started['label']}, but this checkout is {current['commit']} as "
            f"{current['label']}. Resume it from that commit, or start a new run."
        )
    return started


def _pending(
    selection: dict[str, list[str]], folder: Path
) -> list[tuple[str, dict[str, Any]]]:
    """Each selected task without a record, with its row. A benchmark with none is not
    loaded, and the rows of the others are dropped once these are picked."""
    pending = []
    for name, ids in selection.items():
        todo = [i for i in ids if not (folder / name / i / RECORD).exists()]
        if todo:
            rows = BENCHMARKS[name].rows()
            pending += [(name, rows[task_id]) for task_id in todo]
    return pending


def _results_lines(results: Path) -> list[str]:
    """The lines already in ``results``, blank ones skipped. A line that is not JSON
    stops the run before any task does, rather than after a paid run."""
    if not results.exists():
        return []
    lines = []
    for number, text in enumerate(results.read_text(encoding="utf-8").splitlines(), 1):
        if not text.strip():
            continue
        try:
            json.loads(text)
        except json.JSONDecodeError as error:
            raise SystemExit(
                f"evals: line {number} of {results} is not JSON ({error}); fix it first."
            ) from None
        lines.append(text)
    return lines


def _save_line(results: Path, lines: list[str], line: dict[str, Any]) -> None:
    """Write the run's line into ``results`` with the ``lines`` already there, in
    place of an earlier line of the same run, else after the others."""
    runs = [json.loads(text).get("run") for text in lines]
    text = json.dumps(line, ensure_ascii=False)
    if line["run"] in runs:
        lines[runs.index(line["run"])] = text
    else:
        lines.append(text)
    _write_atomically(results, "".join(f"{each}\n" for each in lines))
    log.info("wrote the results line to %s", results)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    _write_atomically(path, json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _report(future: Future, name: str, task_id: str) -> None:
    try:
        record = future.result()
    except Exception:  # noqa: BLE001 — logged, and resuming the run retries the task
        log.exception("%s %s has no record; resuming the run retries it", name, task_id)
        return
    log.info("%s %s: score %s, $%s", name, task_id, record["score"], record["usd"])


def _scores(name: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "errors": sum(record["error"] is not None for record in records),
        **BENCHMARKS[name].metrics([record["score"] for record in records]),
    }


def _git(*args: str) -> str:
    done = subprocess.run(
        ["git", *args], cwd=REPO_DIR, capture_output=True, text=True, check=True
    )
    return done.stdout.strip()


def positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, not {value}")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evals",
        description="Run r3con on the pinned Loong and CorpusQA tasks, judge each "
        "answer, and write the scores to evals/results.jsonl.",
    )
    parser.add_argument(
        "--limit",
        type=positive,
        metavar="N",
        help="Run only the first N pinned tasks of each benchmark. A partial run "
        "writes nothing to results.jsonl.",
    )
    parser.add_argument(
        "--workers",
        type=positive,
        default=DEFAULT_WORKERS,
        metavar="N",
        help=f"Tasks run at once (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        metavar="FOLDER",
        help="Finish the run in FOLDER: tasks without a record run, and a saved answer "
        "is judged again rather than recomputed.",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Re-draw evals/tasks.json from the seed, then exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging()
    log.setLevel(logging.INFO)
    if args.draw:
        _write_json(TASKS_PATH, draw())
        return 0
    pinned = json.loads(TASKS_PATH.read_text(encoding="utf-8"))
    folder = args.resume or RUNS_DIR / new_run_folder()
    line = evaluate(
        pinned, folder, RESULTS_PATH, limit=args.limit, workers=args.workers
    )
    if line is None:
        return 1
    print(json.dumps(line, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
