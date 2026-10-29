"""The ``r3con`` command line: run the pipeline over a folder of documents.

    r3con run "Which supplier missed the most delivery windows?" ./reports
    r3con ask ./logs/<run> "Which supplier delivered on time most often?"

Everything the pipeline did is left in a run folder under ``./logs`` — the relevance
snippets, the proposed schema, the parse, and the reasoning transcript — because the
answer alone rarely tells you whether to trust it.

``r3con ask`` reruns only reasoning over a finished run's folder (:func:`r3con.ask`).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from r3con.config import REASONING_PATHS, available_configs, load_config
from r3con.logging_setup import configure_logging, get_logger
from r3con.masking import masked
from r3con.pipeline import (
    ASKS_FOLDER,
    ask_config,
    ask_pipeline,
    read_representation,
    run_pipeline,
)
from r3con.r3con import read_documents
from r3con.runs import TaskLogger, new_run_folder
from r3con.settings import active_logs_dir, check_cap, settings_snapshot

_log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r3con",
        description="Just-in-time reasoning over a collection of documents.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    # --- transport (not part of the run identity) ---
    common.add_argument(
        "--api-key",
        default=None,
        help="Provider API key. Usually unnecessary — litellm reads the provider's own "
        "variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, …), and a .env here is loaded.",
    )
    common.add_argument(
        "--base-url",
        default=None,
        help="Provider base URL, e.g. a self-hosted endpoint.",
    )
    # --- runtime ---
    common.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Stream per-stage progress to stderr.",
    )

    run_cmd = sub.add_parser(
        "run",
        parents=[common],
        help="Answer a question over a collection of documents.",
        description=(
            "Answer a question over a collection of documents. Sources may be "
            "directories, globs, or individual files; a document too long for the "
            "model's context is read in parts."
        ),
    )
    run_cmd.add_argument("question", help="The question to answer.")
    run_cmd.add_argument(
        "documents",
        nargs="+",
        metavar="SOURCE",
        help="Directories, globs, or files to read as documents (one document per file).",
    )
    # --- run identity (these shape the answer, and appear in the run label) ---
    run_cmd.add_argument(
        "--config",
        choices=available_configs() or None,
        default="default",
        help="Run config: model, relevance rounds, reasoning path, prompt versions, "
        "generation params.",
    )
    run_cmd.add_argument(
        "--model",
        default=None,
        help="Override the config's model, e.g. openai/gpt-6-luna.",
    )
    run_cmd.add_argument(
        "--relevance-rounds",
        type=int,
        default=None,
        metavar="N",
        help="Override how many rounds of relevance surfacing to run "
        "(0 = skip relevance; 1 = each document read alone; 2+ = re-read in light of "
        "the others).",
    )
    run_cmd.add_argument(
        "--reasoning",
        choices=REASONING_PATHS,
        default=None,
        help="Override how reasoning answers (direct = one call over the summaries and "
        "the whole parse; coding = an agent that reads the parse with code, which a "
        "direct run also uses when its prompt does not fit the model's window).",
    )
    # --- runtime ---
    run_cmd.add_argument(
        "--doc-workers",
        type=int,
        default=None,
        metavar="N",
        help="Max documents processed concurrently within the run (default: 16, or "
        "R3CON_DOC_WORKERS). Each document is one LLM call per relevance round "
        "and one for parsing, so this is your main lever on endpoint load.",
    )
    run_cmd.add_argument(
        "--logs-dir",
        default=None,
        metavar="DIR",
        help="Where to write the run folder (default: ./logs, or R3CON_LOGS_DIR).",
    )
    run_cmd.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Don't write a run folder. You lose the record of what the run did.",
    )

    ask_cmd = sub.add_parser(
        "ask",
        parents=[common],
        help="Answer a new question over a finished run.",
        description=(
            "Answer a new question over what a finished run built, running only "
            "reasoning. Each ask is written to the run folder's asks/."
        ),
    )
    ask_cmd.add_argument("run_dir", metavar="RUN_DIR", help="The run's folder.")
    ask_cmd.add_argument("question", help="The question to answer.")
    ask_cmd.add_argument(
        "--model",
        default=None,
        help="The model to reason with (default: the run's), e.g. openai/gpt-6-luna.",
    )
    ask_cmd.add_argument(
        "--reasoning",
        choices=REASONING_PATHS,
        default=None,
        help="How reasoning answers (default: the run's).",
    )
    return p


def _run(args: argparse.Namespace) -> int:
    try:
        configure_logging("INFO" if args.verbose else None)
        config = load_config(
            args.config,
            model=args.model,
            relevance_rounds=args.relevance_rounds,
            reasoning=args.reasoning,
        )
        if args.doc_workers is not None:
            check_cap("doc_workers", args.doc_workers, name="--doc-workers")
            # The stages read this through settings.active_doc_workers() at call time.
            os.environ["R3CON_DOC_WORKERS"] = str(args.doc_workers)
        settings_snapshot()
        documents = read_documents(args.documents)
    except (FileNotFoundError, ValueError) as e:
        print(f"r3con: {e}", file=sys.stderr)
        return 2
    if not documents:
        print("r3con: no documents to read (every source was empty).", file=sys.stderr)
        return 2

    task_logger = None
    if not args.no_artifacts:
        root = Path(args.logs_dir) if args.logs_dir else active_logs_dir()
        task_logger = TaskLogger(new_run_folder(), root=root)

    # Progress and provenance go to stderr; only the answer goes to stdout, so the
    # command composes in a pipe.
    print(f"{len(documents)} document(s) · {config.label()}", file=sys.stderr)
    if task_logger is not None:
        print(f"artifacts: {task_logger.dir}", file=sys.stderr)

    try:
        result = run_pipeline(
            task=args.question,
            documents=documents,
            config=config,
            task_logger=task_logger,
            api_base=args.base_url,
            api_key=args.api_key,
        )
    except Exception as e:  # noqa: BLE001 — a CLI reports, it doesn't traceback at the user
        return _failed(e)
    print(result.answer)
    return 0


def _ask(args: argparse.Namespace) -> int:
    try:
        configure_logging("INFO" if args.verbose else None)
        settings_snapshot()
        built = read_representation(Path(args.run_dir))
        config = ask_config(built, model=args.model, reasoning=args.reasoning)
    except (FileNotFoundError, ValueError) as e:
        print(f"r3con: {e}", file=sys.stderr)
        return 2
    task_logger = TaskLogger(new_run_folder(), root=built.run_dir / ASKS_FOLDER)
    print(f"artifacts: {task_logger.dir}", file=sys.stderr)
    try:
        result = ask_pipeline(
            built,
            config,
            task=args.question,
            task_logger=task_logger,
            api_base=args.base_url,
            api_key=args.api_key,
        )
    except Exception as e:  # noqa: BLE001 — a CLI reports, it doesn't traceback at the user
        return _failed(e)
    print(result.answer)
    return 0


def _failed(error: Exception) -> int:
    """Report ``error`` and its notes, such as where its artifacts are, on stderr,
    keys masked."""
    lines = [
        f"r3con: {type(error).__name__}: {error}",
        *getattr(error, "__notes__", ()),
    ]
    print(masked("\n".join(lines)), file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    # An application may load the user's .env; a library may not (importing r3con must
    # not mutate the environment). usecwd=True searches from the user's working
    # directory rather than from this file, which lives in site-packages.
    load_dotenv(find_dotenv(usecwd=True))
    args = build_parser().parse_args(argv)
    command = {"run": _run, "ask": _ask}[args.command]
    try:
        return command(args)
    except KeyboardInterrupt:
        print("\nr3con: interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
