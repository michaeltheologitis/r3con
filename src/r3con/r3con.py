"""The public entry point: ask a question over a collection of documents, get an answer.

This is the whole surface most callers need::

    from r3con import r3con

    result = r3con.run("Who approved the Q3 budget increase?", [memo_a, memo_b, memo_c])
    print(result)                 # the answer text
    result.relevant_context              # what each document was found to contribute
    result.structured_context            # the structured parse the answer was computed over
    result.schema_code            # the schema proposed for this question

(``from r3con import run`` works too — same function, imported from the package root.)

``documents`` is a list of strings: the documents themselves. To read them off disk,
:func:`read_documents` turns a folder, a glob or a list of files into that list —
one job each, so nothing has to guess which you meant.

The pipeline itself lives in :mod:`r3con.pipeline`; this module only resolves inputs
and hands back the answer. Reach for :func:`r3con.pipeline.run_pipeline` directly when
you want to pass a prebuilt :class:`~r3con.config.RunConfig` or your own
:class:`~r3con.runs.TaskLogger`.

**Provider credentials work the way litellm's do.** ``model`` is a litellm model string,
and the key comes from the provider's own environment variable (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, …) or a ``.env`` in your working directory.
r3con invents no environment variable of its own for credentials; ``api_key`` /
``api_base`` are here for the cases litellm itself takes them explicitly, such as a
self-hosted endpoint.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from collections.abc import Callable, Iterable, Sequence
from typing import Any

from r3con.config import DEFAULT_CONFIG, RunConfig, load_config
from r3con.pipeline import Answer, run_pipeline
from r3con.runs import TaskLogger, new_run_folder
from r3con.settings import active_logs_dir

# Which files a folder walk picks up. A walk that swallowed binaries would produce
# garbage documents and confusing model errors, so keep it to text-ish files; pass
# explicit paths for anything else.
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {".txt", ".md", ".markdown", ".rst", ".json", ".jsonl", ".csv", ".tsv", ".yaml", ".yml", ".html", ".xml"}
)


def read_documents(source: str | Path | Iterable[str | Path]) -> list[str]:
    """Read documents off the filesystem into the list of strings :func:`run` wants.

    This is **only** a filesystem reader — it never treats its argument as document text.
    Each source may be a directory (every text file under it, recursively, sorted by
    path), a single file, or a glob such as ``"reports/*.md"``; pass several and they are
    concatenated in the order given.

    Files are read as UTF-8, replacing undecodable bytes rather than failing: one bad byte
    in one document should not sink a run. Empty documents are dropped — an empty document
    contributes nothing and costs a relevance call.

    Document order is preserved end to end (it is what ``Document N`` refers to
    throughout the artifacts), so a stable, sorted walk matters.

    Raises:
        FileNotFoundError: if a source matches nothing.
    """
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    texts: list[str] = []
    for src in sources:
        path = Path(src)
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES)
        elif path.is_file():
            files = [path]
        else:  # treat it as a glob, anchored if it is absolute
            pattern, anchor = str(src), Path(src).anchor
            files = sorted(
                p for p in (Path(anchor).glob(pattern[len(anchor):]) if anchor else Path().glob(pattern))
                if p.is_file()
            )
        if not files:
            raise FileNotFoundError(f"No documents found at {src!r}.")
        texts.extend(p.read_text(encoding="utf-8", errors="replace") for p in files)
    return [t for t in texts if t.strip()]


def _check_documents(documents: Any) -> list[str]:
    """``documents`` is the document *texts* — a sequence of strings, and nothing else.

    Deliberately no path-sniffing. Guessing whether a string is a filename or a document
    means a one-line document that happens to match a file on disk gets silently replaced
    by that file's contents. Reading the filesystem is :func:`read_documents`' job, and
    the caller says which they meant.
    """
    if isinstance(documents, (str, Path)):
        raise TypeError(
            "`documents` is a sequence of document TEXTS, not a single string or path. "
            "For one document pass [text]; to read a folder or glob use "
            "r3con.read_documents(...) and pass its result."
        )
    try:
        docs = list(documents)
    except TypeError:
        raise TypeError(f"`documents` must be a sequence of strings — got {type(documents).__name__}.") from None
    bad = next(((i, d) for i, d in enumerate(docs) if not isinstance(d, str)), None)
    if bad is not None:
        raise TypeError(
            f"`documents` must be a sequence of strings — item {bad[0]} is {type(bad[1]).__name__}."
        )
    kept = [d for d in docs if d.strip()]
    if not kept:
        raise ValueError("`documents` is empty — there is nothing to read.")
    return kept


def run(
    question: str,
    documents: Sequence[str],
    *,
    model: str | None = None,
    config: str | RunConfig = DEFAULT_CONFIG,
    seed: int | None = None,
    relevance_rounds: int | None = None,
    params: dict[str, Any] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    completion: Callable[..., Any] | None = None,
    logs_dir: str | Path | None = None,
    save_artifacts: bool = True,
) -> Answer:
    """Run the pipeline: answer ``question`` over ``documents``.

    Args:
        question: what to ask. It is the task the whole pipeline is built for — the
            schema and the reasoning are proposed *from it*, so a specific question
            gets a much better apparatus than a vague one.
        documents: the document **texts**, as a sequence of strings. To read them off
            disk instead, call :func:`read_documents` and pass its result. Each document
            must fit in the model's context: there is no chunking, and one that doesn't
            fit raises ``litellm.ContextWindowExceededError``.
        model: a litellm model string (e.g. ``"openai/gpt-5.6-luna"``,
            ``"anthropic/claude-sonnet-4"``, ``"hosted_vllm/Qwen/Qwen3.5-35B-A3B"``).
            Defaults to the config's.
        config: a bundled config name, or a :class:`RunConfig` you built yourself.
        seed, relevance_rounds: override single config fields.
        params: extra keyword arguments passed straight to ``litellm.completion`` —
            ``temperature``, ``top_p``, ``extra_body``, anything it accepts.
        api_key, api_base: passed through to litellm. Usually unnecessary: litellm reads
            the provider's own environment variable, and a ``.env`` in your working
            directory is loaded for you.
        completion: the callable that makes each request, defaulting to
            ``litellm.completion``. Supply your own when you already own the connection —
            a configured ``litellm.Router``'s ``.completion`` for fallbacks or load
            balancing, or a wrapper that adds caching, logging or an internal gateway.
            It is transport, so it does not change the run's identity.
        logs_dir: where the run folder goes. Defaults to ``./logs``
            (or ``R3CON_LOGS_DIR``).
        save_artifacts: write the per-stage artifacts — the relevance snippets, the
            proposed schema, the parse, the reasoning transcript — into a run folder.
            On by default: they are how you see what the run actually did. Set ``False``
            to run without touching the filesystem.

    Returns:
        An :class:`~r3con.pipeline.Answer`: the answer text (``.answer``, and
        ``str()`` of the result), the relevant context (``.relevant_context``), the
        structured parse (``.structured_context``), the schema proposed for this question
        (``.schema_code``), and where the artifacts were written (``.run_dir``).

    Raises:
        TypeError: if ``documents`` is not a sequence of strings (a bare string or a
            path is rejected rather than guessed at).
        ValueError: if ``documents`` is empty, or if ``config`` is a prebuilt
            :class:`RunConfig` *and* field overrides were also given (which would be
            silently ignored).
        Anything the underlying stages raise — most usefully
        ``litellm.ContextWindowExceededError`` when a document does not fit.
    """
    # `run` is the batteries-included path, so it loads the caller's .env (from THEIR
    # working directory) the way the CLI does. Importing r3con does not.
    load_dotenv(find_dotenv(usecwd=True))

    overrides = {"model": model, "seed": seed, "relevance_rounds": relevance_rounds, "params": params}
    given = {k: v for k, v in overrides.items() if v is not None}
    if isinstance(config, RunConfig) and given:
        raise ValueError(
            f"`config` is already a RunConfig, so {sorted(given)} would be ignored — set "
            "them on the RunConfig instead."
        )

    docs = _check_documents(documents)

    cfg = config if isinstance(config, RunConfig) else load_config(config, **given)

    task_logger = None
    if save_artifacts:
        root = Path(logs_dir) if logs_dir is not None else active_logs_dir()
        task_logger = TaskLogger(new_run_folder(), root=root)

    return run_pipeline(
        task=question,
        documents=docs,
        config=cfg,
        task_logger=task_logger,
        api_base=api_base,
        api_key=api_key,
        completion=completion,
    )


__all__ = ["Answer", "read_documents", "run", "TEXT_SUFFIXES"]
