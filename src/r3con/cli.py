"""The ``r3con`` command line: run the pipeline over a folder of documents.

    r3con run "Which supplier missed the most delivery windows?" ./reports

Everything the pipeline did is left in a run folder under ``./logs`` — the relevance
snippets, the proposed schema, the parse, and the reasoning transcript — because the
answer alone rarely tells you whether to trust it.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

from r3con.r3con import read_documents
from r3con.config import available_configs, load_config
from r3con.logging_setup import configure_logging, get_logger
from r3con.pipeline import run_pipeline
from r3con.runs import TaskLogger, new_run_folder
from r3con.settings import active_logs_dir

_log = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r3con",
        description="Just-in-time reasoning over a collection of documents.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_cmd = sub.add_parser(
        "run",
        help="Answer a question over a collection of documents.",
        description=(
            "Answer a question over a collection of documents. Sources may be "
            "directories, globs, or individual files; each document must fit in the "
            "model's context (there is no chunking)."
        ),
    )
    run_cmd.add_argument("question", help="The question to answer.")
    run_cmd.add_argument(
        "documents", nargs="+", metavar="SOURCE",
        help="Directories, globs, or files to read as documents (one document per file).",
    )
    # --- run identity (these shape the answer, and appear in the run label) ---
    run_cmd.add_argument(
        "--config", choices=available_configs() or None, default="default",
        help="Run config: model, seed, relevance rounds, prompt versions, generation params.",
    )
    run_cmd.add_argument("--model", default=None, help="Override the config's model, e.g. openai/gpt-5.6-luna.")
    run_cmd.add_argument("--seed", type=int, default=None, help="Override the config's seed.")
    run_cmd.add_argument(
        "--relevance-rounds", type=int, default=None, metavar="N",
        help="Override how many rounds of relevance surfacing to run "
             "(1 = each document read alone; 2+ = re-read in light of the others).",
    )
    # --- transport (not part of the run identity) ---
    run_cmd.add_argument(
        "--api-key", default=None,
        help="Provider API key. Usually unnecessary — litellm reads the provider's own "
             "variable (OPENAI_API_KEY, ANTHROPIC_API_KEY, …), and a .env here is loaded.",
    )
    run_cmd.add_argument("--base-url", default=None, help="Provider base URL, e.g. a self-hosted endpoint.")
    # --- runtime ---
    run_cmd.add_argument(
        "--doc-workers", type=int, default=None, metavar="N",
        help="Max documents processed concurrently within the run (default: 16, or "
             "R3CON_DOC_WORKERS). Each document is one LLM call per relevance round "
             "and one for parsing, so this is your main lever on endpoint load.",
    )
    run_cmd.add_argument("--logs-dir", default=None, metavar="DIR",
                     help="Where to write the run folder (default: ./logs, or R3CON_LOGS_DIR).")
    run_cmd.add_argument("--no-artifacts", action="store_true",
                     help="Don't write a run folder. You lose the record of what the run did.")
    run_cmd.add_argument("-v", "--verbose", action="store_true", help="Stream per-stage progress to stderr.")
    return p


def _run(args: argparse.Namespace) -> int:
    if args.doc_workers is not None:
        # The stages read this through settings.active_doc_workers() at call time.
        os.environ["R3CON_DOC_WORKERS"] = str(args.doc_workers)
    try:
        documents = read_documents(args.documents)
    except FileNotFoundError as e:
        print(f"r3con: {e}", file=sys.stderr)
        return 2
    if not documents:
        print("r3con: no documents to read (every source was empty).", file=sys.stderr)
        return 2

    config = load_config(
        args.config, model=args.model, seed=args.seed, relevance_rounds=args.relevance_rounds,
    )

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
            task=args.question, documents=documents, config=config,
            task_logger=task_logger, api_base=args.base_url, api_key=args.api_key,
        )
    except Exception as e:  # noqa: BLE001 — a CLI reports, it doesn't traceback at the user
        print(f"r3con: {type(e).__name__}: {e}", file=sys.stderr)
        if task_logger is not None:
            print(f"r3con: partial artifacts in {task_logger.dir}", file=sys.stderr)
        return 1
    print(result.answer)
    return 0


def main(argv: list[str] | None = None) -> int:
    # An application may load the user's .env; a library may not (importing r3con must
    # not mutate the environment). usecwd=True searches from the user's working
    # directory rather than from this file, which lives in site-packages.
    load_dotenv(find_dotenv(usecwd=True))
    args = build_parser().parse_args(argv)
    configure_logging("INFO" if getattr(args, "verbose", False) else None)
    if args.command == "run":
        try:
            return _run(args)
        except KeyboardInterrupt:
            print("\nr3con: interrupted.", file=sys.stderr)
            return 130
    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
