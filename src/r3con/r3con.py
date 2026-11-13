"""The public entry point: ask a question over a collection of documents, get an answer.

This is the whole surface most callers need::

    from r3con import r3con

    result = r3con.run("Who approved the Q3 budget increase?", [memo_a, memo_b, memo_c])
    print(result)                 # the answer text
    result.relevant_context              # what each document was found to contribute
    result.structured_context            # the structured parse the answer was computed over
    result.provenance             # the snippets, parses, quotes and trail behind it
    result.schema_code            # the schema proposed for this question

The answer cites its two views separately: the relevance snippets it used
(``result.provenance["summary_ids"]``) and the documents whose parsed records it used
(``result.provenance["parse_ids"]``). Each document's relevance snippet and its parse
also record the snippets they used (``result.relevance_provenance``,
``result.parsing_provenance``). ``result.provenance["trail"]`` is what the answer
reaches through those citations, and ``result.provenance["quotes"]`` the sentences
behind it, the quotes on that trail (:attr:`~r3con.pipeline.Answer.provenance`,
:mod:`r3con.provenance`).

(``from r3con import run`` works too — same function, imported from the package root.)

:func:`ask` reruns only reasoning over a finished run, for a new question,
:func:`resume` carries on a run that failed or was stopped from where it was, and
:func:`trace` builds a saved answer's provenance again, under any rule, calling no
model.

``documents`` is the documents themselves: their texts, each under its name, or a plain
list of them. To read them off disk, :func:`read_documents` turns a folder, a glob or a
list of files into ``{name: text}``, each named by its path — one job each, so nothing
has to guess which you meant. ``result.document_names`` lines up with
``result.document_ids``, and each quote in ``result.provenance["quotes"]`` names its
document.

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

import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import find_dotenv, load_dotenv

from r3con.config import (
    DEFAULT_CONFIG,
    ReasoningPath,
    RunConfig,
    check_prompts,
    load_config,
)
from r3con.documents import TEXT_SUFFIXES, read_documents, unique_names
from r3con.pipeline import (
    Answer,
    ask_config,
    ask_pipeline,
    names_and_texts,
    read_answer,
    read_representation,
    read_resume,
    resume_pipeline,
    run_pipeline,
)
from r3con.provenance import DEFAULT_RULE, Provenance
from r3con.runs import TaskLogger, new_run_folder
from r3con.runtime.llm import Cancelled
from r3con.settings import active_logs_dir, settings_snapshot


def run(
    question: str,
    documents: Mapping[str, str] | Sequence[str],
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
        documents: the document **texts**, each under its name, as
            :func:`read_documents` gives them, or as a sequence of strings, each then
            named by its index (``"0"``, ``"1"``, …, so ``documents[int(name)]`` is
            the document); no text may be blank, so that every index the run reports
            refers to *your* documents, in their order. The names are recorded, never
            sent to the model. A document too long for
            the model's window is read in parts.
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
        it used, its trail and the quotes on it (``.provenance``: ids from
        ``.document_ids``, a part's snippet as ``<id>.<k>``, whose document is the part
        before the dot, each quote with its document's name; see
        :attr:`~r3con.pipeline.Answer.provenance`), each document's name
        (``.document_names``), the relevant context
        (``.relevant_context``), the structured parse (``.structured_context``), the
        schema proposed for this question (``.schema_code``), and where the artifacts
        were written (``.run_dir``).

    Raises:
        TypeError: if ``documents`` is not a mapping of names to strings or a sequence
            of strings (a bare string or a path is rejected rather than guessed at).
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

    # Refuse what run_pipeline would refuse, the documents and a RunConfig's prompts
    # included, before a run folder exists.
    names, texts = names_and_texts(documents)
    cfg = config if isinstance(config, RunConfig) else load_config(config, **given)
    check_prompts(cfg)
    settings_snapshot()

    task_logger = None
    if save_artifacts:
        root = Path(logs_dir) if logs_dir is not None else active_logs_dir()
        task_logger = TaskLogger(new_run_folder(), root=root)

    return run_pipeline(
        task=question,
        documents=dict(zip(names, texts, strict=True)),
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
            can break another's request. A credential among the run's params is saved
            as ``REDACTED`` (:mod:`r3con.masking`), so pass its real value here.
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
        reasoning (its trail and quotes through the run's snippets of every round and
        its parses), and whose other fields are the run's; ``run_dir`` is the ask's own
        folder, which holds its manifest, calls and transcript.

    Raises:
        FileNotFoundError: if ``source`` is neither a run folder nor an ask's folder in
            the ``asks/`` of the run it asked over, if its run never finished parsing,
            or if a prompt version it pins has no file.
        ValueError: if ``source`` is an Answer without a ``run_dir`` (its run was made
            with ``save_artifacts=False``), if a file in its folder is not as this
            r3con writes it (one an older r3con wrote, say), if it leaves a prompt
            version unpinned, if an override is not valid for its setting, if a param
            it would send holds ``REDACTED``, at any depth, naming each one, or if a
            setting in :mod:`r3con.settings` is out of range.
        Each of these is raised in one line, before any request and before the ask's
        folder exists.
        Anything reasoning raises, with a note naming the ask's folder, where it left
        its completed calls and ``error.txt``; :class:`Cancelled` among them, once
        ``cancel`` is set.
    """
    load_dotenv(find_dotenv(usecwd=True))
    built = read_representation(_folder_of(source, "ask"))
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


def resume(
    run_dir: str | Path,
    documents: Mapping[str, str] | Sequence[str],
    *,
    params: dict[str, Any] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    completion: Callable[..., Any] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> Answer:
    """Carry on a run that failed or was stopped from where it was, computing only what
    it lacks, in its own folder.

    A stage that finished is taken as its ``result.json`` holds it, its files left as
    they are. An unfinished one reuses each document's result the run saved in
    ``events.jsonl`` (each relevance snippet a round wrote, all its parts there, and
    each parse) and computes the rest, as the run would have, under its question,
    config and document ids; reasoning always runs. The folder keeps one record of the
    whole run: an unfinished stage's ``calls.json`` keeps the earlier calls before the
    new ones, ``events.jsonl`` adds the resume's events, the manifest lists each resume
    under ``resumes``, and ``run_finished`` totals every stage. A call the run had made
    when it was killed outright, rather than failing or being cancelled, is not in its
    stage's ``calls.json``, nor in the totals; its result, when its events hold it, is
    reused all the same. Lines of ``events.jsonl`` or saved results it can't use are
    computed again, and a warning counts them.

    A run or resume holds its folder while it writes there (``run.lock``,
    :func:`r3con.runs.run_lock`), so a run still going, or another resume, is not
    resumed meanwhile.

    Args:
        run_dir: the run's folder, as the note on its exception names it.
        documents: the run's documents, as :func:`run` took them, in the same order:
            each is checked against the fingerprint the run recorded.
        params: only to pass again the real value of each credential among the run's
            params, which its manifest saved as ``REDACTED`` (:mod:`r3con.masking`):
            otherwise the run's own, and anything else in them is refused.
        api_key, api_base, completion: as in :func:`run`.
        on_event: as in :func:`run`, called with the resume's events: ``run_started``;
            ``run_resumed``, whose ``reused`` gives how many results each stage reuses
            (``relevance``, per round, ``schema``, 0 or 1, and ``parsing``); then the
            ``relevance_finished``, ``schema_finished`` and ``parsing_finished`` events
            that reported the results it reuses, as the folder saved them, their
            ``time`` the original's, in the order written, which ``events.jsonl`` does
            not get twice; then the events of what it computes, as a run's.
        cancel: as in :func:`run`; a cancelled resume can be resumed in turn.

    Returns:
        An :class:`Answer`, as :func:`run` returns it.

    Raises:
        FileNotFoundError: if ``run_dir`` is not a run folder, if it predates resume,
            or if a prompt version it pins has no file.
        ValueError: if the run finished, if ``documents`` are not the run's or not in
            its order, if a prompt it pins now resolves to another file than it read,
            if ``params`` are not the run's, if a param it would send holds
            ``REDACTED``, or if a file in its folder is not as this r3con writes it.
        RuntimeError: if a run or resume is going in its folder.
        Each of these is raised in one line, before the folder is written to and before
        any request. Anything the stages raise, as for :func:`run`.
    """
    load_dotenv(find_dotenv(usecwd=True))
    resumable = read_resume(
        Path(run_dir), documents, params=params, completion=completion
    )
    return resume_pipeline(
        resumable,
        api_base=api_base,
        api_key=api_key,
        completion=completion,
        on_event=on_event,
        cancel=cancel,
    )


def trace(source: Answer | str | Path, rule: str = DEFAULT_RULE) -> Provenance:
    """The provenance of the answer ``source`` saved, its trail walked under ``rule``
    (:mod:`r3con.provenance`), built from the files of its folder alone.

    It calls no model, and reads neither the run's config nor its prompts, so a run
    whose config or prompts this r3con no longer runs is traced all the same.

    Args:
        source: an :class:`Answer` from :func:`run` or :func:`ask`, whose ``run_dir``
            is read, or the path of a run folder or of an ask's folder: an ask's answer
            is its own, and its snippets and parses those of the run it asked over.
        rule: the name of a rule in :data:`r3con.provenance.RULES`. Under the default,
            the provenance is the one the answer was saved with.

    Raises:
        ValueError: if no rule is named ``rule``, before anything is read; if
            ``source`` is an Answer without a ``run_dir``; or, in one line naming the
            folder that holds it, if a file it reads is not as this r3con writes it.
        FileNotFoundError: if ``source`` is neither a run folder nor an ask's folder in
            the ``asks/`` of the run it asked over, or it holds no answer, as when its
            reasoning failed.
    """
    return read_answer(_folder_of(source, "trace"), rule)


def _folder_of(source: Answer | str | Path, reader: str) -> Path:
    """The folder ``source`` names for ``reader``: an Answer's ``run_dir``, or the path
    given.

    Raises:
        ValueError: if ``source`` is an Answer without a ``run_dir``.
    """
    if not isinstance(source, Answer):
        return Path(source)
    if source.run_dir is None:
        raise ValueError(
            f"{reader} needs the run folder, and this Answer has none: its run was "
            "made with save_artifacts=False."
        )
    return source.run_dir


__all__ = [
    "TEXT_SUFFIXES",
    "Answer",
    "Cancelled",
    "ask",
    "read_documents",
    "resume",
    "run",
    "trace",
    "unique_names",
]
