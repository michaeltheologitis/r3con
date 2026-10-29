"""End-to-end just-in-time solver: (task, documents) → answer.

Wires the three moves of the method — see :mod:`r3con.stages`:

1. **surfacing relevance** (:mod:`r3con.stages.relevance`) — each document gets a
   task-conditioned **relevance snippet**, rebuilt over ``relevance_rounds`` synchronous
   rounds; from round 2 on, each document is re-read in light of the *other* documents'
   previous snippets. The collection of them is the **relevant context**, and
   it is what every later stage reads.
2. **structuring** — two steps that are one idea: propose a per-task Pydantic schema
   from the task and the relevant context
   (:mod:`r3con.stages.structuring.schema`), then **parse** every document, in
   parallel, into instances of it (:mod:`r3con.stages.structuring.parsing`).
3. **reasoning** (:mod:`r3con.stages.reasoning`) — answer over the merged parse and
   the relevant context: in one completion (``reasoning: direct``), or in a multi-turn
   sandboxed Python loop (``reasoning: coding``, and a direct run whose prompt does not
   fit the model's window). The answer cites the summaries it used, and the documents
   whose parsed records it used, by id.

The document is the unit throughout: a document too long for the model's window is read
in parts (:mod:`r3con.splitting`), and the whole collection is never placed in one
prompt.

With a ``task_logger``, each stage writes its artifacts into one flat run-folder as soon
as that stage succeeds, so a later failure still leaves the earlier work on disk, and a
stage that fails leaves the calls it completed and its traceback (see
:mod:`r3con.runs`). No coordinator class; the routing is plain control flow here.

:func:`ask_pipeline` answers a new task over what a finished run built, read back from
its folder (:func:`read_representation`), by running reasoning alone.
"""

from __future__ import annotations

import datetime
import json
import re
import traceback
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from r3con.config import (
    ReasoningPath,
    RunConfig,
    check_prompts,
    check_reasoning,
    normalize_model_name,
)
from r3con.logging_setup import get_logger
from r3con.prompts import require_prompt_path
from r3con.runs import StageRun, TaskLogger, write_manifest
from r3con.runtime.codeact import DEFAULT_EXEC_TIMEOUT_S
from r3con.settings import settings_snapshot
from r3con.splitting import Splits
from r3con.stages import reasoning
from r3con.stages.relevance import (
    RelevantContext,
    SnippetRecord,
    draw_document_ids,
    join_parts,
    surface_relevance,
)
from r3con.stages.structuring.parsing import parse_documents
from r3con.stages.structuring.schema import propose_schema

_log = get_logger("pipeline")

# The run folder's subfolder that holds one folder per ask.
_ASKS_FOLDER = "asks"
# The stages whose results an ask reasons over, as folders of the run folder.
_BUILT_STAGES = ("relevance", "structuring/schema", "structuring/parsing")
_REASONING_PROMPTS = ("reasoning/direct", "reasoning/coding")
# Anything a model name holds that a folder name should not.
_NOT_IN_FOLDER_NAMES = re.compile(r"[^\w.-]+")


def _sources_as_ids(
    source_docs: dict[str, list[int]], document_ids: list[str]
) -> dict[str, list[str]]:
    """``source_docs`` with each document index replaced by that document's id."""
    return {
        name: [document_ids[i] for i in indices]
        for name, indices in source_docs.items()
    }


@dataclass(frozen=True)
class Answer:
    """What a run produced: the answer, and the two views it was derived from.

    The answer alone rarely tells you whether to trust it, and the intermediate views are
    often what you actually wanted — so a run hands back the whole thing rather than a
    bare string. ``str(result)`` is the answer text, so printing one still does the
    obvious thing.

    - ``answer`` — the answer text.
    - ``provenance`` — the ids the answer used, by view: ``provenance["summary_ids"]``,
      those of the final-round relevance snippets it used (``<id>.<k>`` for part ``k`` of
      a document read in parts), and ``provenance["parse_ids"]``, those of the
      documents whose records in ``structured_context`` it used. An id can be in both.
    - ``relevant_context`` — the **relevant context**: the final-round relevance snippet
      of each document, aligned 1:1 with the ``documents`` you passed in
      (``relevant_context[i]`` describes ``documents[i]``); a document read in parts has
      its parts' notes joined. A document that contributes nothing is an empty string,
      which is a real result rather than a failure.
    - ``structured_context`` — the merged parse: the per-task schema filled from every document,
      as plain JSON-able data. Each record carries a ``document`` key naming the id of the
      document it came from, as reasoning saw it.
    - ``schema_code`` — the Pydantic schema that was proposed *for this question*. The
      most characteristic artifact of the method: it is what the question was decided to
      be *about*.
    - ``source_docs`` — for each list field of the parse, the 0-based source-document
      index of each record, positionally aligned with it.
    - ``run_dir`` — the folder holding the full artifacts, or ``None`` if the run wasn't
      asked to write any.
    - ``document_ids`` — each document's id in this run, aligned 1:1 with
      ``documents``: random, drawn afresh for each run, and recorded in the run folder's
      ``ids.json``.
    - ``relevance_provenance`` — for each document, aligned 1:1 with
      ``relevant_context``, the ids of the previous round's relevance snippets its
      final-round snippet used: each a document's id, or ``<id>.<k>`` for part ``k`` of
      a document read in parts in that previous round. A document read in parts has its
      parts' ids combined, in order, without repeats.
    - ``parsing_provenance`` — for each document, aligned 1:1 with ``documents``, the
      ids of the final-round relevance snippets its parse used, its own document's
      among them when it helped: each a document's id, or ``<id>.<k>`` for part ``k``
      of a document read in parts in that round, combined across a document's parts
      as in ``relevance_provenance``.
    - ``source_ids`` — read-only, derived from ``source_docs`` and ``document_ids``:
      ``source_docs`` with each index replaced by that document's id, so it lines up
      with ``source_docs`` by position. A record parsed from one part of a document read
      in parts carries the document's id. The id lets a record be matched against the
      ids in ``provenance``, in ``relevance_provenance``, in ``parsing_provenance`` and
      in the run folder's ``relevance/result.json``, where a part's id is ``<id>.<k>``
      and matching uses the part before the dot.
    """

    answer: str
    relevant_context: list[str]
    structured_context: dict[str, Any]
    schema_code: str
    source_docs: dict[str, list[int]]
    run_dir: Path | None = None
    document_ids: list[str] = field(default_factory=list)
    provenance: dict[str, list[str]] = field(default_factory=reasoning.cites_nothing)
    relevance_provenance: list[list[str]] = field(default_factory=list)
    parsing_provenance: list[list[str]] = field(default_factory=list)

    @property
    def source_ids(self) -> dict[str, list[str]]:
        return _sources_as_ids(self.source_docs, self.document_ids)

    def __str__(self) -> str:
        return self.answer


@dataclass(frozen=True)
class Representation:
    """What a run built for its question, read back from its folder by
    :func:`read_representation`, for :func:`ask_pipeline` to reason over: the
    manifest's ``config`` block, the document ids, the final round of relevance
    snippets, the schema, and the parse as the folder holds it, with its
    ``source_docs`` and ``provenance``."""

    run_dir: Path
    config: dict[str, Any]
    document_ids: list[str]
    relevance: RelevantContext
    schema_code: str
    parsed: dict[str, Any]
    source_docs: dict[str, list[int]]
    parsing_provenance: list[list[str]]


@dataclass
class _StageRecord:
    """What one stage leaves in the run folder: its calls, and its ``result.json``."""

    run: StageRun | None
    result: dict[str, Any] = field(default_factory=dict)


@contextmanager
def _recorded_stage(
    task_logger: TaskLogger | None, stage: str, model: str, *, transcript: bool
) -> Iterator[_StageRecord]:
    """Record one stage's artifacts under ``<run-folder>/<stage>/``.

    The body passes ``record.run`` to the stage and sets ``record.result`` to the
    fields of its ``result.json``. On a clean exit this writes ``calls.json`` (and,
    with ``transcript``, ``transcript.yaml``), then ``result.json`` with the stage's
    token and cost ``totals``. On any exception, an interrupt included, it writes the calls
    that completed and ``error.txt`` with the traceback, adds a note naming the run
    folder to the exception, and re-raises. Without a logger it records nothing.
    """
    if task_logger is None:
        yield _StageRecord(run=None)
        return
    run = StageRun(stage=stage, task_logger=task_logger, model=model)
    record = _StageRecord(run=run)
    try:
        yield record
        run.flush(write_transcript=transcript)
        task_logger.write_json(
            f"{stage}/result", {**record.result, "totals": run.compute_totals()}
        )
    except BaseException as error:
        run.flush(write_transcript=transcript)
        task_logger.write_text(f"{stage}/error", traceback.format_exc())
        error.add_note(f"r3con: partial artifacts in {task_logger.dir}")
        raise


def _preview(text: str, n: int = 100) -> str:
    """One-line, truncated preview of an answer for the progress log."""
    s = " ".join((text or "").split())
    return s if len(s) <= n else s[:n] + "…"


def run_pipeline(
    *,
    task: str,
    documents: list[str],
    config: RunConfig,
    max_reasoning_turns: int | None = None,
    reasoning_timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    task_logger: TaskLogger | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    completion: Callable[..., Any] | None = None,
) -> Answer:
    """Run all three stages for one ``(task, documents)`` pair under one ``config``.

    Returns an :class:`Answer` — the answer text plus the relevant context,
    the structured parse, and the schema that was proposed for this question.

    ``config`` (a :class:`r3con.config.RunConfig`) carries everything that shapes the
    output — model, relevance rounds, the reasoning path, the prompt version of each
    stage, and any generation params — so nothing output-affecting is threaded ad-hoc.

    ``api_base``, ``api_key`` and ``completion`` are **transport**: routing, auth, and how
    the request is actually made. None of them changes what a correct answer is, so none
    appears in the run identity. ``completion`` replaces ``litellm.completion`` for every
    call this run makes — hand it a configured ``litellm.Router``'s ``.completion``, or any
    wrapper with the same ``(model, messages, **kwargs)`` shape. It is called once per
    request and owns its retrying.

    Before the first request, every stage's prompt is checked to exist
    (:func:`r3con.config.check_prompts`), and every runtime cap is read once, checked
    and recorded (:func:`r3con.settings.settings_snapshot`); ``max_reasoning_turns``
    replaces ``settings.REASONING_MAX_TURNS`` when it is not ``None``.

    If ``task_logger`` is provided, each stage's artifacts are written immediately
    after that stage succeeds, so a later failure still leaves earlier artifacts; a
    stage that raises writes its completed calls and ``error.txt``, and the exception
    carries a note naming the run folder.
    """
    check_prompts(config)
    caps = settings_snapshot(reasoning_max_turns=max_reasoning_turns)
    model = config.model
    rounds = config.relevance_rounds
    llm_kwargs = _llm_kwargs(
        config.params, api_base=api_base, api_key=api_key, completion=completion
    )

    document_ids = draw_document_ids(len(documents))
    # The identity card first, so even a run that dies in stage 1 says what it was.
    if task_logger is not None:
        write_manifest(
            task_logger,
            task=task,
            config=config,
            n_docs=len(documents),
            context_chars=sum(len(d) for d in documents),
            settings=caps,
        )
        task_logger.write_json("ids", {id_: i for i, id_ in enumerate(document_ids)})

    # Where each document is cut into parts, shared by every stage that reads one.
    splits = Splits(
        documents,
        model=model,
        margin_percent=caps["window_margin_percent"],
        task_logger=task_logger,
    )

    # --- Stage 1: surface relevance — the relevant context. ---
    _log.info(
        "stage 1/3 · surfacing relevance (%d round(s), %d doc(s))",
        rounds,
        len(documents),
    )
    with _recorded_stage(task_logger, "relevance", model, transcript=False) as record:
        relevance = surface_relevance(
            task=task,
            documents=documents,
            document_ids=document_ids,
            model=model,
            prompts=config.prompts,
            rounds=rounds,
            max_attempts=caps["relevance_max_attempts"],
            workers=caps["doc_workers"],
            run=record.run,
            splits=splits,
            **llm_kwargs,
        )
        record.result = {
            "n_rounds": rounds,
            "n_docs": len(documents),
            "rounds": [
                {
                    "round": k + 1,
                    "snippets": [asdict(r) for parts in per_doc for r in parts],
                }
                for k, per_doc in enumerate(relevance.records)
            ],
        }
    # the relevant context feeds every later stage
    relevance_snippets = relevance.snippets

    # --- Stage 2a: structuring — propose the schema (task + relevant context). ---
    _log.info("stage 2/3 · structuring · proposing the schema")
    with _recorded_stage(
        task_logger, "structuring/schema", model, transcript=True
    ) as record:
        proposal = propose_schema(
            task=task,
            relevance_snippets=relevance_snippets,
            model=model,
            prompt_version=config.prompts["structuring/schema"],
            max_attempts=caps["schema_max_attempts"],
            run=record.run,
            **llm_kwargs,
        )
        record.result = {
            "schema_code": proposal.schema_code,
            "thought": proposal.attempts[-1].thought if proposal.attempts else None,
            "attempts": [
                {"schema_code": a.schema_code, "error": a.error, "thought": a.thought}
                for a in proposal.attempts
            ],
        }

    # --- Stage 2b: structuring — parse every document, in parallel. ---
    _log.info("stage 2/3 · structuring · parsing %d doc(s)", len(documents))
    with _recorded_stage(
        task_logger, "structuring/parsing", model, transcript=False
    ) as record:
        extraction = parse_documents(
            documents=documents,
            schema_code=proposal.schema_code,
            parse_cls=proposal.parse_cls,
            task=task,
            prompt_version=config.prompts["structuring/parsing"],
            relevance_snippets=relevance_snippets,
            document_ids=document_ids,
            model=model,
            max_attempts=caps["parsing_max_attempts"],
            run=record.run,
            workers=caps["doc_workers"],
            splits=splits,
            **llm_kwargs,
        )
        record.result = {
            "parsed": extraction.parse,
            "source_docs": extraction.source_docs,
            "source_ids": _sources_as_ids(extraction.source_docs, document_ids),
            "provenance": extraction.provenance,
        }
    parsed = extraction.parse

    # --- Stage 3: reasoning over the parse + the relevant context. ---
    _log.info("stage 3/3 · reasoning")
    result = _reasoning_stage(
        task_logger,
        model,
        task=task,
        schema_code=proposal.schema_code,
        parsed=parsed,
        source_docs=extraction.source_docs,
        relevance_snippets=relevance_snippets,
        prompts=config.prompts,
        path=config.reasoning,
        document_ids=document_ids,
        max_attempts=caps["reasoning_max_attempts"],
        max_turns=caps["reasoning_max_turns"],
        timeout_s=reasoning_timeout_s,
        splits=splits,
        **llm_kwargs,
    )
    return _answer(
        result,
        relevance=relevance,
        schema_code=proposal.schema_code,
        parsed=parsed,
        source_docs=extraction.source_docs,
        parsing_provenance=extraction.provenance,
        document_ids=document_ids,
        run_dir=task_logger.dir if task_logger is not None else None,
    )


def read_representation(run_dir: Path) -> Representation:
    """Read back what the run in ``run_dir`` built, for :func:`ask_pipeline`.

    The final round's relevance snippets are kept as the run left them, one per part
    for a document read in parts. ``reasoning/`` is not read, so a run that failed in
    reasoning can still be asked.

    Raises:
        FileNotFoundError: in one line naming ``run_dir``, if it is not a run folder or
            its run never finished parsing.
    """
    if not (run_dir / "manifest.json").is_file():
        raise FileNotFoundError(
            f"{run_dir} is not a run folder: it has no manifest.json."
        )
    unbuilt = [s for s in _BUILT_STAGES if not (run_dir / s / "result.json").is_file()]
    if unbuilt:
        raise FileNotFoundError(
            f"{run_dir}: the run never finished parsing ({unbuilt[0]}/result.json is "
            "missing), and ask needs its relevance, schema and parsing."
        )
    ids: dict[str, int] = _read_json(run_dir / "ids.json")
    rounds = _read_json(run_dir / "relevance" / "result.json")["rounds"]
    final_round: list[list[SnippetRecord]] = [[] for _ in ids]
    for snippet in rounds[-1]["snippets"] if rounds else []:
        document = ids[snippet["id"].partition(".")[0]]
        final_round[document].append(SnippetRecord(**snippet))
    parsing = _read_json(run_dir / "structuring" / "parsing" / "result.json")
    schema = _read_json(run_dir / "structuring" / "schema" / "result.json")
    return Representation(
        run_dir=run_dir,
        config=_read_json(run_dir / "manifest.json")["config"],
        document_ids=sorted(ids, key=ids.__getitem__),
        relevance=RelevantContext(records=[final_round] if rounds else []),
        schema_code=schema["schema_code"],
        parsed=parsing["parsed"],
        source_docs=parsing["source_docs"],
        parsing_provenance=parsing["provenance"],
    )


def ask_pipeline(
    built: Representation,
    *,
    task: str,
    model: str | None = None,
    path: ReasoningPath | None = None,
    params: Mapping[str, Any] | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    completion: Callable[..., Any] | None = None,
) -> Answer:
    """Answer ``task`` over ``built`` by running reasoning alone, as
    :func:`run_pipeline` runs it, under the run's reasoning prompt pins.

    ``model``, ``path`` and ``params`` default to the run's; the model as the manifest
    records it, without its provider prefix. ``api_base``, ``api_key`` and
    ``completion`` are transport, as in :func:`run_pipeline`.

    The ask is written to a new ``<run_dir>/asks/<NNN>-<model>/``: ``ask.json`` first,
    then ``reasoning/`` as a run writes it, or, if reasoning raises, its completed calls
    and ``error.txt``, with a note naming the folder on the exception.

    Raises:
        ValueError: if ``path`` is not a reasoning path, or a setting is out of range.
        FileNotFoundError: if a reasoning prompt the run pins has no file.
        Each of these is raised before any request and before the ask's folder exists.
    """
    config = built.config
    model = str(config["model"]) if model is None else model
    path = check_reasoning(config["reasoning"] if path is None else path, "ask")
    prompts = {stage: config["prompts"][stage] for stage in _REASONING_PROMPTS}
    for stage, version in prompts.items():
        require_prompt_path(stage, version)
    caps = settings_snapshot()
    task_logger = _ask_logger(built.run_dir, model)
    task_logger.write_json(
        "ask",
        {
            "question": task,
            "model": normalize_model_name(model),
            "reasoning": path,
            "prompts": prompts,
            "created": datetime.datetime.now(datetime.UTC).isoformat(
                timespec="seconds"
            ),
        },
    )
    _log.info("asking over %s · artifacts: %s", built.run_dir, task_logger.dir)
    result = _reasoning_stage(
        task_logger,
        model,
        task=task,
        schema_code=built.schema_code,
        parsed=built.parsed,
        source_docs=built.source_docs,
        relevance_snippets=built.relevance.snippets,
        prompts=prompts,
        path=path,
        document_ids=built.document_ids,
        max_attempts=caps["reasoning_max_attempts"],
        max_turns=caps["reasoning_max_turns"],
        **_llm_kwargs(
            config.get("params") or {} if params is None else params,
            api_base=api_base,
            api_key=api_key,
            completion=completion,
        ),
    )
    return _answer(
        result,
        relevance=built.relevance,
        schema_code=built.schema_code,
        parsed=built.parsed,
        source_docs=built.source_docs,
        parsing_provenance=built.parsing_provenance,
        document_ids=built.document_ids,
        run_dir=built.run_dir,
    )


def _llm_kwargs(
    params: Mapping[str, Any],
    *,
    api_base: str | None,
    api_key: str | None,
    completion: Callable[..., Any] | None,
) -> dict[str, Any]:
    """What every request is sent with: the output knobs in ``params``, then the
    transport."""
    llm_kwargs: dict[str, Any] = {**params}
    if api_base:
        llm_kwargs["api_base"] = api_base
    if api_key:
        llm_kwargs["api_key"] = api_key
    if completion is not None:
        # A named parameter of `litellm_chat_completion`, so it binds there rather than
        # being forwarded into the provider request.
        llm_kwargs["completion"] = completion
    return llm_kwargs


def _reasoning_stage(
    task_logger: TaskLogger | None, model: str, **reason_kwargs: Any
) -> reasoning.ReasoningResult:
    """:func:`~r3con.stages.reasoning.reason` on ``model`` with ``reason_kwargs``,
    recorded under ``reasoning/``."""
    with _recorded_stage(task_logger, "reasoning", model, transcript=True) as record:
        result = reasoning.reason(model=model, run=record.run, **reason_kwargs)
        _log.info(
            "reasoning done on the %s path (%s, %d turn(s)): %s",
            result.path,
            result.terminated_by,
            len(result.turns),
            _preview(result.answer),
        )
        record.result = {
            "answer": result.answer,
            "provenance": result.provenance,
            "path": result.path,
            "fallback": result.fallback,
            "terminated_by": result.terminated_by,
            "n_turns": len(result.turns),
            "turns": [
                {
                    "response": t.response,
                    "raw_response": t.raw_response,
                    "code": t.code,
                    "observation": t.observation,
                    "error": t.error,
                    "is_final_answer": t.is_final_answer,
                }
                for t in result.turns
            ],
        }
    return result


def _answer(
    result: reasoning.ReasoningResult,
    *,
    relevance: RelevantContext,
    schema_code: str,
    parsed: BaseModel | dict[str, Any],
    source_docs: dict[str, list[int]],
    parsing_provenance: list[list[str]],
    document_ids: list[str],
    run_dir: Path | None,
) -> Answer:
    """The :class:`Answer` that reasoning's ``result`` makes with what it reasoned
    over."""
    return Answer(
        answer=result.answer,
        relevant_context=[join_parts(s) for s in relevance.snippets],
        # the parse reasoning saw, as plain data
        structured_context=reasoning.tag_source_documents(
            parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed,
            source_docs,
            document_ids,
        ),
        schema_code=schema_code,
        source_docs=source_docs,
        run_dir=run_dir,
        document_ids=document_ids,
        provenance=result.provenance,
        relevance_provenance=relevance.provenance,
        parsing_provenance=parsing_provenance,
    )


def _ask_logger(run_dir: Path, model: str) -> TaskLogger:
    """A logger for a new ask's folder, ``<run_dir>/asks/<NNN>-<model>/``, ``NNN`` one
    past the highest number there."""
    asks = run_dir / _ASKS_FOLDER
    model_name = _NOT_IN_FOLDER_NAMES.sub("-", str(normalize_model_name(model)))
    while True:
        numbers = [
            int(n) for p in asks.glob("*") if (n := p.name.partition("-")[0]).isdigit()
        ]
        folder = f"{max(numbers, default=0) + 1:03d}-{model_name}"
        try:
            (asks / folder).mkdir(parents=True)
        except FileExistsError:
            continue  # an ask running alongside took this number first
        return TaskLogger(folder, root=asks)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
