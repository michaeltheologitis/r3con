"""The ``r3con`` command line: run the pipeline over a folder of documents.

    r3con run "Which supplier missed the most delivery windows?" ./reports
    r3con resume ./logs/<run> ./reports
    r3con ask ./logs/<run> "Which supplier delivered on time most often?"
    r3con app

Everything the pipeline did is left in a run folder under ``./logs`` — the relevance
snippets, the proposed schema, the parse, and the reasoning transcript — because the
answer alone rarely tells you whether to trust it.

``r3con resume`` carries on a run that failed or was stopped, in its folder, over the
same sources (:func:`r3con.resume`); a run that fails or is interrupted says so, with
the command to paste. ``r3con ask`` reruns only reasoning over a finished run's folder
(:func:`r3con.ask`).
``r3con app`` serves the app's page on 127.0.0.1 (:mod:`r3con.app`), when the ``app``
extra is installed.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from r3con.config import REASONING_PATHS, available_configs, load_config
from r3con.documents import read_documents
from r3con.logging_setup import configure_logging, get_logger
from r3con.masking import masked, masked_values
from r3con.pipeline import (
    ASKS_FOLDER,
    ResumeRefused,
    ask_config,
    ask_pipeline,
    read_representation,
    refuse_masked,
    resume_pipeline,
    run_pipeline,
)
from r3con.runs import RunFolderBusy, TaskLogger, new_run_folder
from r3con.runtime.llm import register_credentials
from r3con.settings import active_logs_dir, check_cap, settings_snapshot

_log = get_logger("cli")
# The packages of the app extra, and what to install when one is missing.
APP_PACKAGES = frozenset({"keyring", "platformdirs", "starlette", "uvicorn"})
APP_EXTRA_HINT = (
    'the app needs its extra: uv tool install "r3context[app]" '
    '(or pip install "r3context[app]")'
)
# How an ask refused for a masked param gets past it here, where it takes no params.
MASKED_PARAM_ADVICE = (
    "pass --api-key or --base-url in place of a masked `api_key` or `api_base`, or ask "
    "with another --model, which sends none of the run's params"
)
# How a resume refused for a masked param gets past it here, where it takes no params.
RESUME_MASKED_ADVICE = (
    "pass --api-key or --base-url in place of a masked `api_key` or `api_base`; any "
    "other masked param can be passed again only through r3con.resume's `params`"
)
NO_DOCUMENTS = "no documents to read (every source was empty)."


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
    workers = argparse.ArgumentParser(add_help=False)
    workers.add_argument(
        "--doc-workers",
        type=int,
        default=None,
        metavar="N",
        help="Max documents processed concurrently within the run (default: 16, or "
        "R3CON_DOC_WORKERS). Each document is one LLM call per relevance round "
        "and one for parsing, so this is your main lever on endpoint load.",
    )

    run_cmd = sub.add_parser(
        "run",
        parents=[common, workers],
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
        help="Override how reasoning answers (direct = one call over the relevance "
        "snippets and the whole parse; coding = an agent that reads the parse with "
        "code, which a direct run also uses when its prompt does not fit the model's "
        "window).",
    )
    # --- runtime ---
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

    resume_cmd = sub.add_parser(
        "resume",
        parents=[common, workers],
        help="Carry on a run that failed or was stopped.",
        description=(
            "Carry on a run that failed or was stopped, in its own folder and under its "
            "own settings: what it finished is reused, and only the rest is computed. "
            "It needs the run's documents, the same sources in the same order."
        ),
    )
    resume_cmd.add_argument("run_dir", metavar="RUN_DIR", help="The run's folder.")
    resume_cmd.add_argument(
        "documents",
        nargs="+",
        metavar="SOURCE",
        help="The sources the run was given: directories, globs, or files.",
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

    app_cmd = sub.add_parser(
        "app",
        help="Serve the app's page on 127.0.0.1.",
        description=(
            "Serve the app's page on 127.0.0.1 and open it, with this launch's token "
            f"in its URL; {APP_EXTRA_HINT}."
        ),
    )
    app_cmd.add_argument(
        "--port",
        type=int,
        default=0,
        help="The port to listen on (default: a free one).",
    )
    app_cmd.add_argument(
        "--logs-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Where run folders go (default: R3CON_LOGS_DIR, else runs/ in r3con's "
        "user data directory).",
    )
    app_cmd.add_argument(
        "--no-browser", action="store_true", help="Print the URL, but don't open it."
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
        documents = _sources(args)
    except (FileNotFoundError, ValueError) as e:
        _error(str(e))
        return 2
    if not documents:
        _error(NO_DOCUMENTS)
        return 2

    task_logger = None
    if not args.no_artifacts:
        root = Path(args.logs_dir) if args.logs_dir else active_logs_dir()
        task_logger = TaskLogger(new_run_folder(), root=root)

    # Progress and provenance go to stderr; only the answer goes to stdout, so the
    # command composes in a pipe. The label shows the params: their keys masked.
    register_credentials(config.params)
    shown = config.model_copy(update={"params": masked_values(config.params)})
    print(f"{len(documents)} document(s) · {shown.label()}", file=sys.stderr)
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
    except KeyboardInterrupt:
        code = _interrupted()
    except Exception as e:  # noqa: BLE001 — a CLI reports, it doesn't traceback at the user
        code = _failed(e)
    else:
        print(result.answer)
        return 0
    if task_logger is not None:
        _carry_on(task_logger.dir, args.documents)
    return code


def _resume(args: argparse.Namespace) -> int:
    try:
        configure_logging("INFO" if args.verbose else None)
        documents = _sources(args)
    except (FileNotFoundError, ValueError) as e:
        _error(str(e))
        return 2
    if not documents:
        _error(NO_DOCUMENTS)
        return 2

    run_dir = Path(args.run_dir)

    def started(event: dict[str, Any]) -> None:
        # Once its refusals are past, so a refused resume says one line.
        if event["type"] == "run_started":
            print(f"{len(documents)} document(s) · resuming the run", file=sys.stderr)
            print(f"artifacts: {run_dir}", file=sys.stderr)

    try:
        result = resume_pipeline(
            run_dir,
            documents,
            api_base=args.base_url,
            api_key=args.api_key,
            on_event=started,
            advice=RESUME_MASKED_ADVICE,
        )
    except (FileNotFoundError, ResumeRefused, RunFolderBusy) as e:
        _error(str(e))
        return 2
    except KeyboardInterrupt:
        code = _interrupted()
    except Exception as e:  # noqa: BLE001 — a CLI reports, it doesn't traceback at the user
        code = _failed(e)
    else:
        print(result.answer)
        return 0
    _carry_on(run_dir, args.documents)
    return code


def _sources(args: argparse.Namespace) -> dict[str, str]:
    """The documents ``args`` names, read once the runtime settings are checked and
    ``--doc-workers`` is in force."""
    if args.doc_workers is not None:
        check_cap("doc_workers", args.doc_workers, name="--doc-workers")
        # The stages read this through settings.active_doc_workers() at call time.
        os.environ["R3CON_DOC_WORKERS"] = str(args.doc_workers)
    settings_snapshot()
    return read_documents(args.documents)


def _carry_on(run_dir: Path, sources: Sequence[str]) -> None:
    """Say how to carry on the run in ``run_dir`` once it stopped, if a resume can:
    ``r3con resume`` over the same ``sources``, quoted so that the line pastes."""
    if (run_dir / "documents.json").is_file():
        command = shlex.join(["r3con", "resume", str(run_dir), *sources])
        _error(f"to carry on: {command}")


def _ask(args: argparse.Namespace) -> int:
    try:
        configure_logging("INFO" if args.verbose else None)
        settings_snapshot()
        built = read_representation(Path(args.run_dir))
        config = ask_config(built, model=args.model, reasoning=args.reasoning)
        refuse_masked(
            config.params,
            api_base=args.base_url,
            api_key=args.api_key,
            advice=MASKED_PARAM_ADVICE,
        )
    except (FileNotFoundError, ValueError) as e:
        _error(str(e))
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


def _app(args: argparse.Namespace) -> int:
    # Imported here: its dependencies are the app extra, which run and ask don't need.
    try:
        from r3con.app import server
    except ModuleNotFoundError as missing:
        if (missing.name or "").partition(".")[0] not in APP_PACKAGES:
            raise
        _error(APP_EXTRA_HINT)
        return 1
    try:
        listener = server.listen(args.port)
    except (OSError, OverflowError) as error:
        _error(f"can't listen on port {args.port}: {error}")
        return 2
    logs_dir = args.logs_dir or os.environ.get("R3CON_LOGS_DIR") or server.LOGS_DIR
    server.launch(listener, logs_dir=Path(logs_dir), browser=not args.no_browser)
    return 0


def _interrupted() -> int:
    print("\nr3con: interrupted.", file=sys.stderr)
    return 130


def _failed(error: Exception) -> int:
    """Report ``error`` and its notes, such as where its artifacts are."""
    notes = getattr(error, "__notes__", ())
    _error("\n".join([f"{type(error).__name__}: {error}", *notes]))
    return 1


def _error(text: str) -> None:
    """Print ``text`` on stderr after ``r3con: ``, keys masked: an error may echo one."""
    print(masked(f"r3con: {text}"), file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    # An application may load the user's .env; a library may not (importing r3con must
    # not mutate the environment). usecwd=True searches from the user's working
    # directory rather than from this file, which lives in site-packages.
    load_dotenv(find_dotenv(usecwd=True))
    args = build_parser().parse_args(argv)
    command = {"run": _run, "resume": _resume, "ask": _ask, "app": _app}[args.command]
    try:
        return command(args)
    except KeyboardInterrupt:
        return _interrupted()


if __name__ == "__main__":
    raise SystemExit(main())
