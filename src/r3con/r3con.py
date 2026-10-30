"""The public entry point: ask a question over a collection of documents, get an answer.

This is the whole surface most callers need::

    from r3con import r3con

    result = r3con.run("Who approved the Q3 budget increase?", [memo_a, memo_b, memo_c])
    print(result)                 # the answer text
    result.relevant_context              # what each document was found to contribute
    result.structured_context            # the structured parse the answer was computed over
    result.provenance             # the ids of the relevance snippets and parses the answer used
    result.schema_code            # the schema proposed for this question

The answer cites its two views separately: the relevance snippets it used
(``result.provenance["summary_ids"]``) and the documents whose parsed records it used
(``result.provenance["parse_ids"]``). Each document's relevance snippet and its parse
also record the snippets they used (``result.relevance_provenance``,
``result.parsing_provenance``).

(``from r3con import run`` works too — same function, imported from the package root.)

:func:`ask` reruns only reasoning over a finished run, for a new question.

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

import io
import threading
from collections.abc import Callable, Iterable, Sequence
from fnmatch import fnmatch
from itertools import takewhile
from pathlib import Path
from typing import Any, BinaryIO

import mammoth
from dotenv import find_dotenv, load_dotenv
from markdownify import ATX, MarkdownConverter
from pypdf import PdfReader

from r3con.config import (
    DEFAULT_CONFIG,
    ReasoningPath,
    RunConfig,
    check_prompts,
    load_config,
)
from r3con.logging_setup import get_logger
from r3con.pipeline import (
    Answer,
    ask_config,
    ask_pipeline,
    read_representation,
    run_pipeline,
)
from r3con.runs import TaskLogger, new_run_folder
from r3con.runtime.llm import Cancelled
from r3con.settings import active_logs_dir, settings_snapshot

_log = get_logger("documents")
# Office's lock file for a document someone has open: ~$report.docx.
_LOCK_FILE_PREFIX = "~$"
# Where macOS's Finder puts each file's metadata in a zip it makes.
_MACOS_METADATA_FOLDER = "__MACOSX"
_GLOB_WILDCARDS = frozenset("*?[")
_RECURSIVE_WILDCARD = "**"


def _read_one(path: Path) -> str:
    """One file's text, as :func:`_read_stream` reads it. A file that can't be opened is
    logged, and gives ``""``, which the caller then drops."""
    try:
        file = path.open("rb")
    except OSError as e:
        _log.warning("%s: not readable (%s); skipping it", path, type(e).__name__)
        return ""
    with file:
        return _read_stream(str(path), file)


def _read_stream(name: str, file: BinaryIO) -> str:
    """The text of a file called ``name``, from a stream of its bytes.

    A file whose suffix is in :data:`_EXTRACTORS` is extracted, and an extractor gives
    ``""`` for a file it can't read; everything else is read as UTF-8, replacing
    undecodable bytes rather than failing — one bad byte in one document should not
    sink a run.
    """
    extract = _extractor(name)
    if extract is not None:
        return extract(name, file)
    return io.TextIOWrapper(file, encoding="utf-8", errors="replace").read()


def _pdf_text(name: str, pdf: BinaryIO) -> str:
    """A PDF's text, page by page, from a stream of its bytes; ``name`` is what the
    warnings call it.

    A PDF pypdf can't read gives ``""``, and so does one with no text layer (a scan),
    which the caller then drops: say so rather than letting it vanish. The fix is OCR,
    outside this package. Half a surrogate pair, which a font's ToUnicode map can give
    and no UTF-8 encoder takes, becomes U+FFFD.
    """
    try:
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
    except Exception as e:  # noqa: BLE001 — one unreadable file, not a dead run
        _log.warning(
            "%s: not readable as a PDF (%s); skipping it", name, type(e).__name__
        )
        return ""
    # pypdf decodes ToUnicode maps with surrogatepass, so a pair can come as two code
    # units, or half of one alone: UTF-16 joins the pairs and replaces the halves.
    text = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    if not text.strip():
        _log.warning(
            "%s: no extractable text — a scanned PDF needs OCR first; skipping it", name
        )
    return text


def _docx_text(name: str, docx: BinaryIO) -> str:
    """A Word file's text as Markdown, from a stream of its bytes; ``name`` is what the
    warnings call it.

    mammoth turns its body into HTML, and markdownify that into Markdown: headings,
    lists and tables keep their shape, a table's first row is its header, and a tracked
    insertion stays while a deletion goes. The text is kept as written, its ``_`` and
    ``*`` unescaped, but for a ``|`` in a table's cell. Images are left out. A file that
    can't be read gives ``""``, and so does one with no text, which the caller then
    drops: say so.
    """
    try:
        # An image converter that makes no element, so no image's bytes are read.
        converted = mammoth.convert_to_html(docx, convert_image=lambda image: [])
        text = _CellPipesEscaped(
            heading_style=ATX,
            table_infer_header=True,
            escape_asterisks=False,
            escape_underscores=False,
            escape_misc=False,
        ).convert(converted.value)
    except Exception as e:  # noqa: BLE001 — one unreadable file, not a dead run
        _log.warning(
            "%s: not readable as a Word file (%s); skipping it", name, type(e).__name__
        )
        return ""
    for message in converted.messages:
        _log.debug("%s: %s", name, message.message)
    if not text.strip():
        _log.warning("%s: no extractable text; skipping it", name)
    return text


class _CellPipesEscaped(MarkdownConverter):
    """markdownify, with a ``|`` in a table's cell escaped, so it can't split the cell."""

    # markdownify's stub leaves out the convert_* hooks its README says to override.
    def convert_td(self, el: Any, text: str, parent_tags: set[str]) -> str:
        cell = text.replace("|", r"\|")
        return super().convert_td(el, cell, parent_tags)  # pyright: ignore[reportAttributeAccessIssue]

    def convert_th(self, el: Any, text: str, parent_tags: set[str]) -> str:
        cell = text.replace("|", r"\|")
        return super().convert_th(el, cell, parent_tags)  # pyright: ignore[reportAttributeAccessIssue]


_EXTRACTORS: dict[str, Callable[[str, BinaryIO], str]] = {
    ".pdf": _pdf_text,
    ".docx": _docx_text,
}

# Which files a folder walk picks up. A walk that swallowed binaries would produce
# garbage documents and confusing model errors, so keep it to files we can turn into
# text — the text-ish ones, read directly, plus those in _EXTRACTORS. Pass explicit paths
# for anything else.
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".yaml",
        ".yml",
        ".html",
        ".xml",
        *_EXTRACTORS,
    }
)


def _extractor(name: str) -> Callable[[str, BinaryIO], str] | None:
    """How to extract the file called ``name``, by the suffix it ends with, so that a
    file called just ``.docx`` is a Word file; or ``None`` if it is read as text."""
    lower = name.lower()
    return next(
        (f for suffix, f in _EXTRACTORS.items() if lower.endswith(suffix)), None
    )


def _left_out_unsaid(parts: Sequence[str]) -> bool:
    """Whether a folder walk leaves out, without a word, the file at ``parts`` below the
    folder: Office's lock file, or one that is hidden or under a hidden folder or
    ``__MACOSX``."""
    return parts[-1].startswith(_LOCK_FILE_PREFIX) or any(
        part.startswith(".") or part == _MACOS_METADATA_FOLDER for part in parts
    )


def _walk_reads(name: str) -> bool:
    """Whether a folder walk reads a file called ``name`` by its suffix: one in
    :data:`TEXT_SUFFIXES`."""
    return name.lower().endswith(tuple(TEXT_SUFFIXES))


def _shell_matches(pattern: Sequence[str], parts: Sequence[str]) -> bool:
    """Whether the shell's glob matches the path ``parts`` to ``pattern``, part by part:
    as pathlib's does, but ``**`` takes no hidden name, nor does any other part that
    doesn't start with a dot."""
    if not pattern:
        return not parts
    if pattern[0] == _RECURSIVE_WILDCARD:
        return _shell_matches(pattern[1:], parts) or (
            bool(parts)
            and not parts[0].startswith(".")
            and _shell_matches(pattern, parts[1:])
        )
    return (
        bool(parts)
        and fnmatch(parts[0], pattern[0])
        and (pattern[0].startswith(".") or not parts[0].startswith("."))
        and _shell_matches(pattern[1:], parts[1:])
    )


def _resolves_inside(path: Path, root: Path) -> bool:
    """Whether ``path`` resolves inside ``root``, itself resolved; logs it if not."""
    if path.resolve().is_relative_to(root):
        return True
    _log.warning("%s: resolves outside %s; skipping it", path, root)
    return False


def read_documents(source: str | Path | Iterable[str | Path]) -> list[str]:
    """Read documents off the filesystem into the list of strings :func:`run` wants.

    This is **only** a filesystem reader — it never treats its argument as document text.
    Each source may be a directory (every text file under it, recursively, sorted by
    path), a single file, or a glob such as ``"reports/*.md"``; pass several and they are
    concatenated in the order given.

    A directory walk picks up only the suffixes in :data:`TEXT_SUFFIXES` — text-ish files,
    PDFs and Word files (``.docx``). An explicit path or a glob reads **whatever you
    name**, so that a ``.log``, a ``.py`` or an extensionless file is yours to pass; the
    flip side is that a glob aimed at binaries will happily turn them into garbage
    documents. A walk and a glob both skip Office's lock files (``~$report.docx``). A
    walk skips anything hidden or under ``__MACOSX`` below the folder you named; a glob
    follows the shell, where a wildcard matches no hidden name but a name in the pattern
    that starts with a dot does (``"d/.*"``, ``"**/.env"``). Both read only files that
    resolve inside the folder you named, or the glob's part before its first wildcard: a
    file that a link or a ``..`` takes outside it is logged and skipped, so a folder you
    received can't point r3con at files elsewhere. What you name, a file or that folder,
    is taken as named, even if hidden or a link.

    PDFs are extracted page by page. A PDF with no text layer — a scan — yields nothing,
    is logged and then dropped; OCR is outside this package. A Word file is converted to
    Markdown, its tables included; one that can't be read, or has no text, is logged and
    dropped. Everything else is read as UTF-8, replacing undecodable bytes rather than
    failing: one bad byte in one document should not sink a run. Empty documents are
    dropped — an empty document contributes nothing and costs a relevance call.

    Document order is preserved end to end (it is what ``Document N`` refers to in the
    structuring prompts, and ``document_ids`` aligns with it), so a stable, sorted walk
    matters.

    Raises:
        FileNotFoundError: if a source matches nothing.
    """
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    texts: list[str] = []
    for src in sources:
        path = Path(src)
        if path.is_dir():
            root = path.resolve()
            files = sorted(
                p
                for p in path.rglob("*")
                if not _left_out_unsaid(p.relative_to(path).parts)
                and _walk_reads(p.name)
                and p.is_file()
                and _resolves_inside(p, root)
            )
        elif path.is_file():
            files = [path]
        else:  # treat it as a glob, anchored if it is absolute
            pattern, anchor = str(src), Path(src).anchor
            base = Path(*takewhile(_GLOB_WILDCARDS.isdisjoint, path.parts))
            root, below = base.resolve(), len(base.parts)
            files = sorted(
                p
                for p in (
                    Path(anchor).glob(pattern[len(anchor) :])
                    if anchor
                    else Path().glob(pattern)
                )
                if not p.name.startswith(_LOCK_FILE_PREFIX)
                and _shell_matches(path.parts[below:], p.parts[below:])
                and p.is_file()
                and _resolves_inside(p, root)
            )
        if not files:
            raise FileNotFoundError(f"No documents found at {src!r}.")
        texts.extend(_read_one(p) for p in files)
    return [t for t in texts if t.strip()]


def _check_documents(documents: Any) -> list[str]:
    """``documents`` is the document *texts* — a sequence of strings, and nothing else.

    Deliberately no path-sniffing and no silent filtering. Guessing whether a string is a
    filename or a document means a one-line document that happens to match a file on disk
    gets silently replaced by that file's contents. Dropping a blank entry is the same
    mistake one step later: every index downstream is positional — ``relevant_context[i]``,
    the ``source_docs`` indices, and ``document_ids[i]``, which the answer's
    ``provenance`` cites — so a silent drop renumbers the caller's collection without
    telling them, and the wrong document has already left the library by the time anyone
    could notice.

    :func:`read_documents` drops blank *files*, which is a different thing: it drops them
    before the list ever reaches the caller, where no index can shift yet. Here the list is
    the caller's, so a blank is refused rather than quietly removed.
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
        raise TypeError(
            f"`documents` must be a sequence of strings — got {type(documents).__name__}."
        ) from None
    bad = next(((i, d) for i, d in enumerate(docs) if not isinstance(d, str)), None)
    if bad is not None:
        raise TypeError(
            f"`documents` must be a sequence of strings — item {bad[0]} is {type(bad[1]).__name__}."
        )
    # After the type scan, so `[" ", 42]` reports the type problem rather than the blank.
    blank = next((i for i, d in enumerate(docs) if not d.strip()), None)
    if blank is not None:
        raise ValueError(
            f"`documents` item {blank} is blank. Every entry must be a document — the "
            "indices are positional, so dropping it would silently renumber the rest: "
            "relevant_context[i], the source_docs indices and document_ids[i], which the "
            "answer's provenance cites, all refer to documents[i]. Drop the blanks "
            "yourself, or build the list with r3con.read_documents(...), which skips "
            "empty files."
        )
    if not docs:
        raise ValueError("`documents` is empty — there is nothing to read.")
    return docs


def run(
    question: str,
    documents: Sequence[str],
    *,
    model: str | None = None,
    config: str | RunConfig = DEFAULT_CONFIG,
    relevance_rounds: int | None = None,
    reasoning: ReasoningPath | None = None,
    params: dict[str, Any] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    completion: Callable[..., Any] | None = None,
    logs_dir: str | Path | None = None,
    save_artifacts: bool = True,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> Answer:
    """Run the pipeline: answer ``question`` over ``documents``.

    Args:
        question: what to ask. It is the task the whole pipeline is built for — the
            schema and the reasoning are proposed *from it*, so a specific question
            gets a much better apparatus than a vague one.
        documents: the document **texts**, as a sequence of strings; no entry may be
            blank, so that every index the run reports refers to *your* list. To read
            them off disk instead, call :func:`read_documents` and pass its result. A
            document too long for the model's window is read in parts.
        model: a litellm model string (e.g. ``"openai/gpt-6-luna"``,
            ``"anthropic/claude-sonnet-5-5"``, ``"hosted_vllm/Qwen/Qwen3.5-35B-A3B"``).
            Defaults to the config's.
        config: a bundled config name, or a :class:`RunConfig` you built yourself.
        relevance_rounds: overrides the config's.
        reasoning: overrides the config's reasoning path: ``"direct"``, one completion
            over the relevant context and the whole parse, or ``"coding"``, an agent
            that reads the parse with code. A direct run takes the coding path when its
            prompt does not fit the model's window.
        params: extra keyword arguments passed straight to ``litellm.completion`` —
            ``temperature``, ``top_p``, ``seed``, ``extra_body``, anything it accepts.
        api_key, api_base: passed through to litellm. Usually unnecessary: litellm reads
            the provider's own environment variable, and a ``.env`` in your working
            directory is loaded for you.
        completion: the callable that makes each request, defaulting to
            ``litellm.completion``. Supply your own when you already own the connection —
            a configured ``litellm.Router``'s ``.completion`` for fallbacks or load
            balancing, or a wrapper that adds caching, logging or an internal gateway.
            It is transport, so it does not change the run's identity. r3con calls it
            once per request and leaves any resending to it.
        logs_dir: where the run folder goes. Defaults to ``./logs``
            (or ``R3CON_LOGS_DIR``).
        save_artifacts: write the per-stage artifacts — the relevance snippets, the
            proposed schema, the parse, the reasoning transcript — into a run folder.
            On by default: they are how you see what the run actually did. Set ``False``
            to run without touching the filesystem.
        on_event: called with each progress event as it happens, a copy as a dict of
            JSON values with its ``type`` (``run_started``, ``relevance_started``, …,
            ``run_finished`` or ``run_failed``), its ``time`` and its fields, listed in
            :func:`~r3con.pipeline.run_pipeline`. It is called from worker threads,
            but never twice at once, so it needs no lock of its own; the workers wait
            for it, so it should return quickly (put the event on a queue, say). If it
            raises, the error is logged and the run goes on. The run folder's
            ``events.jsonl`` records the same events, whether or not it is given.
        cancel: a ``threading.Event`` that stops the run once it is set, from any
            thread. From then on no new model call is sent, no stage starts, and a
            wait to resend a rate-limited call ends at once; the stage under way, or
            the next one if the last call of one was in flight, raises
            :class:`Cancelled` and fails as any stage does, with ``run_failed``
            naming it and, with artifacts on, its folder holding the calls it made
            and ``error.txt``, beside the earlier stages' artifacts. What is already
            under way runs its course: a call already sent (litellm cannot stop one),
            which is recorded, so a run whose last call is in flight finishes as
            usual; the resends r3con does not make itself, a supplied
            ``completion``'s (a ``litellm.Router``'s, say) and those of
            ``num_retries``, ``max_retries`` or ``retry_policy`` in ``params``; and a
            step the coding path is running in its sandbox, to its end or its
            timeout.

    Returns:
        An :class:`~r3con.pipeline.Answer`: the answer text (``.answer``, and
        ``str()`` of the result), the ids of the relevance snippets and parsed records
        it used (``.provenance``: ids from ``.document_ids``, a part's snippet as
        ``<id>.<k>``, whose document is the part before the dot), the relevant context
        (``.relevant_context``), the structured parse (``.structured_context``), the
        schema proposed for this question (``.schema_code``), and where the artifacts
        were written (``.run_dir``).

    Raises:
        TypeError: if ``documents`` is not a sequence of strings (a bare string or a
            path is rejected rather than guessed at).
        ValueError: if ``documents`` is empty or any entry is blank (a blank is refused,
            not dropped — dropping it would renumber every index the run reports), if
            ``config`` is a prebuilt :class:`RunConfig` *and* field overrides were also
            given (which would be silently ignored), if the config pins no prompt
            version for a stage, if an override is not valid for its setting (a
            ``reasoning`` that is not a reasoning path, say), or if a setting in
            :mod:`r3con.settings` is below its floor.
        FileNotFoundError: if the config, by name or as a :class:`RunConfig`, pins a
            prompt version that has no file.
        Each of these is raised before any request and before a run folder exists, as
        is anything :func:`~r3con.config.load_config` refuses.
        Anything the underlying stages raise — most usefully
        ``litellm.ContextWindowExceededError`` when the prompt and notes sent beside a
        document leave it no room in the model's window. With
        artifacts on, an exception from a stage carries a note naming the run folder,
        where the stage left its completed calls and ``error.txt``.
        Cancelled: once ``cancel`` is set, in place of the next model call or stage.
    """
    # `run` is the batteries-included path, so it loads the caller's .env (from THEIR
    # working directory) the way the CLI does. Importing r3con does not.
    load_dotenv(find_dotenv(usecwd=True))

    overrides = {
        "model": model,
        "relevance_rounds": relevance_rounds,
        "reasoning": reasoning,
        "params": params,
    }
    given = {k: v for k, v in overrides.items() if v is not None}
    if isinstance(config, RunConfig) and given:
        raise ValueError(
            f"`config` is already a RunConfig, so {sorted(given)} would be ignored — set "
            "them on the RunConfig instead."
        )

    docs = _check_documents(documents)

    cfg = config if isinstance(config, RunConfig) else load_config(config, **given)
    # Refuse what run_pipeline would refuse, a RunConfig's prompts included, before a
    # run folder exists.
    check_prompts(cfg)
    settings_snapshot()

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
        on_event=on_event,
        cancel=cancel,
    )


def ask(
    source: Answer | str | Path,
    question: str,
    *,
    model: str | None = None,
    reasoning: ReasoningPath | None = None,
    params: dict[str, Any] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    completion: Callable[..., Any] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> Answer:
    """Answer a new ``question`` over what a finished run built, running only reasoning.

    The run's relevance snippets, schema and parse are reused as its folder holds them,
    so an ask costs one reasoning step rather than a run. Reasoning is sent what the
    run's was, with ``question`` in place of the run's question. Its prompts are the
    run's pins, resolved as a run resolves them (a ``./prompts`` overlay in the working
    directory included), and the ask's manifest records the files it used. It falls
    back to the coding path as a run's does.

    Args:
        source: an :class:`Answer` from :func:`run` or :func:`ask`, whose ``run_dir``
            is read, or the path of a run folder or of an ask's folder. An ask's folder
            stands for the run it asked over, so asking over an ask asks that run again.
        question: the question to answer; it may be the run's own.
        model: a litellm model string. Defaults to the one the run called; for a run
            recorded before r3con kept that string, its bare name (``gpt-6-luna``).
        reasoning: ``"direct"`` or ``"coding"``, as in :func:`run`. Defaults to the
            run's.
        params: as in :func:`run`. Defaults to the run's when the ask runs on the run's
            model (by its name without the provider prefix), and to none on another
            model, since one provider's params (``reasoning_effort``, ``extra_body``)
            can break another's request.
        api_key, api_base, completion: as in :func:`run`.
        on_event: as in :func:`run`, called with the ask's events: ``ask_started``,
            ``reasoning_started``, ``reasoning_finished``, then ``ask_finished`` or
            ``ask_failed`` (:func:`~r3con.pipeline.ask_pipeline`).
        cancel: as in :func:`run`: the ask fails with ``ask_failed``, its folder
            holding the calls reasoning made and ``error.txt``.

    Each ask is written to a new folder in the run folder's ``asks/``, named as a run
    folder is: a ``manifest.json`` as a run writes one, for the run's config with this
    ask's overrides (its ``task`` the question, ``ask_of`` the run folder's name),
    ``events.jsonl`` and ``reasoning/`` as a run writes them. The run's own files are
    left as they are.

    Returns:
        An :class:`Answer` whose ``answer`` and ``provenance`` come from the new
        reasoning, and whose other fields are the run's; ``run_dir`` is the ask's own
        folder, which holds its manifest, calls and transcript.

    Raises:
        FileNotFoundError: if ``source`` is not a run folder, if its run never finished
            parsing, or if a prompt version it pins has no file.
        ValueError: if ``source`` is an Answer without a ``run_dir`` (its run was made
            with ``save_artifacts=False``), if a file in its folder is not as this
            r3con writes it (one an older r3con wrote, say), if it leaves a prompt
            version unpinned, if an override is not valid for its setting, or if a
            setting in :mod:`r3con.settings` is out of range.
        Each of these is raised in one line, before any request and before the ask's
        folder exists.
        Anything reasoning raises, with a note naming the ask's folder, where it left
        its completed calls and ``error.txt``; :class:`Cancelled` among them, once
        ``cancel`` is set.
    """
    load_dotenv(find_dotenv(usecwd=True))
    run_dir = source.run_dir if isinstance(source, Answer) else Path(source)
    if run_dir is None:
        raise ValueError(
            "ask needs the run folder, and this Answer has none: its run was made with "
            "save_artifacts=False."
        )
    built = read_representation(run_dir)
    config = ask_config(built, model=model, reasoning=reasoning, params=params)
    return ask_pipeline(
        built,
        config,
        task=question,
        api_base=api_base,
        api_key=api_key,
        completion=completion,
        on_event=on_event,
        cancel=cancel,
    )


__all__ = ["TEXT_SUFFIXES", "Answer", "Cancelled", "ask", "read_documents", "run"]
