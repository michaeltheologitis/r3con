"""End-to-end just-in-time solver: (task, documents) → answer.

Wires the three moves of the method — see :mod:`r3con.stages`:

1. **surfacing relevance** (:mod:`r3con.stages.relevance`) — each document gets a
   task-conditioned **relevance snippet**, rebuilt over ``relevance_rounds`` synchronous
   rounds; from round 2 on, each document is re-read in light of the *other* documents'
   previous snippets. The collection of them is the **relevant context**, and
   it is what every later stage reads.
2. **structuring** — two steps that are one idea: propose a per-task Pydantic schema
   from the task and the relevant context
   (:mod:`r3con.stages.structuring.schema`), then **parse** every document, whole and
   in parallel, into instances of it (:mod:`r3con.stages.structuring.parsing`).
3. **reasoning** (:mod:`r3con.stages.reasoning`) — answer over the merged parse and
   the relevant context, in a multi-turn sandboxed Python loop.

The document is the unit throughout: nothing is chunked, and the whole collection is
never placed in one prompt.

With a ``task_logger``, each stage writes its artifacts into one flat run-folder as soon
as that stage succeeds, so a later failure still leaves the earlier work on disk, and a
stage that fails leaves the calls it completed and its traceback (see
:mod:`r3con.runs`). No coordinator class; the routing is plain control flow here.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from r3con.config import RunConfig, check_prompts
from r3con.logging_setup import get_logger
from r3con.runs import StageRun, TaskLogger, write_manifest
from r3con.runtime.codeact import DEFAULT_EXEC_TIMEOUT_S
from r3con.settings import settings_snapshot
from r3con.stages import reasoning
from r3con.stages.relevance import join_parts, surface_relevance
from r3con.stages.structuring.parsing import parse_documents
from r3con.stages.structuring.schema import propose_schema

_log = get_logger("pipeline")


@dataclass(frozen=True)
class Answer:
    """What a run produced: the answer, and the two views it was derived from.

    The answer alone rarely tells you whether to trust it, and the intermediate views are
    often what you actually wanted — so a run hands back the whole thing rather than a
    bare string. ``str(result)`` is the answer text, so printing one still does the
    obvious thing.

    - ``answer`` — the committed answer text.
    - ``relevant_context`` — the **relevant context**: the final-round relevance snippet
      of each document, aligned 1:1 with the ``documents`` you passed in
      (``relevant_context[i]`` describes ``documents[i]``); a document read in parts has
      its parts' notes joined. A document that contributes nothing is an empty string,
      which is a real result rather than a failure.
    - ``structured_context`` — the merged parse: the per-task schema filled from every document,
      as plain JSON-able data. Each record carries a ``document`` key naming the 1-based
      document it came from.
    - ``schema_code`` — the Pydantic schema that was proposed *for this question*. The
      most characteristic artifact of the method: it is what the question was decided to
      be *about*.
    - ``source_docs`` — for each list field of the parse, the 0-based source-document
      index of each record, positionally aligned with it.
    - ``run_dir`` — the folder holding the full artifacts, or ``None`` if the run wasn't
      asked to write any.
    """

    answer: str
    relevant_context: list[str]
    structured_context: dict[str, Any]
    schema_code: str
    source_docs: dict[str, list[int]]
    run_dir: Path | None = None

    def __str__(self) -> str:
        return self.answer


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
    token ``totals``. On any exception, an interrupt included, it writes the calls
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
    output — model, relevance rounds, the prompt version of each stage, and any
    generation params — so nothing output-affecting is threaded ad-hoc.

    ``api_base``, ``api_key`` and ``completion`` are **transport**: routing, auth, and how
    the request is actually made. None of them changes what a correct answer is, so none
    appears in the run identity. ``completion`` replaces ``litellm.completion`` for every
    call this run makes — hand it a configured ``litellm.Router``'s ``.completion``, or any
    wrapper with the same ``(model, messages, **kwargs)`` shape.

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
    # The output knobs flow from the config; api_base/api_key are transport only.
    llm_kwargs: dict[str, Any] = {**config.params}
    if api_base:
        llm_kwargs["api_base"] = api_base
    if api_key:
        llm_kwargs["api_key"] = api_key
    if completion is not None:
        # A named parameter of `litellm_chat_completion`, so it binds there rather than
        # being forwarded into the provider request.
        llm_kwargs["completion"] = completion

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
            model=model,
            prompt_version=config.prompts["relevance"],
            rounds=rounds,
            workers=caps["doc_workers"],
            run=record.run,
            **llm_kwargs,
        )
        # Per-round, per-document — `round` is the refinement depth; the LAST round
        # is what downstream stages consume. snippets[doc_i] aligns to documents[i].
        record.result = {
            "n_rounds": rounds,
            "n_docs": len(documents),
            "rounds": [
                {"round": k + 1, "snippets": per_doc}
                for k, per_doc in enumerate(relevance.rounds)
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

    # --- Stage 2b: structuring — parse every document, whole, in parallel. ---
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
            model=model,
            max_attempts=caps["parsing_max_attempts"],
            run=record.run,
            workers=caps["doc_workers"],
            **llm_kwargs,
        )
        record.result = {
            "parsed": extraction.parse,
            "source_docs": extraction.source_docs,
        }
    parsed = extraction.parse

    # --- Stage 3: reasoning over the parse + the relevant context. ---
    _log.info("stage 3/3 · reasoning")
    with _recorded_stage(task_logger, "reasoning", model, transcript=True) as record:
        result = reasoning.reason(
            task=task,
            schema_code=proposal.schema_code,
            parsed=parsed,
            source_docs=extraction.source_docs,
            relevance_snippets=relevance_snippets,
            model=model,
            prompt_version=config.prompts["reasoning"],
            max_turns=caps["reasoning_max_turns"],
            timeout_s=reasoning_timeout_s,
            run=record.run,
            **llm_kwargs,
        )
        _log.info(
            "reasoning done (%s, %d turn(s)): %s",
            result.terminated_by,
            len(result.turns),
            _preview(result.answer),
        )
        record.result = {
            "answer": result.answer,
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
    return Answer(
        answer=result.answer,
        relevant_context=[join_parts(s) for s in relevance_snippets],
        # the parse exactly as the agent saw it: plain data, each record stamped with the
        # 1-based document it came from
        structured_context=reasoning.tag_source_documents(
            parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed,
            extraction.source_docs,
        ),
        schema_code=proposal.schema_code,
        source_docs=extraction.source_docs,
        run_dir=task_logger.dir if task_logger is not None else None,
    )
