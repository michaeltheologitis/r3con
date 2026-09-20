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
as that stage succeeds, so a later failure still leaves the earlier work on disk (see
:mod:`r3con.runs`). No coordinator class; the routing is plain control flow here.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from r3con.config import RunConfig
from r3con.logging_setup import get_logger
from r3con.runs import StageRun, TaskLogger, write_manifest
from r3con.runtime.codeact import DEFAULT_EXEC_TIMEOUT_S
from r3con.settings import settings
from r3con.stages import reasoning
from r3con.stages.structuring.parsing import parse_documents
from r3con.stages.structuring.schema import propose_schema
from r3con.stages.relevance import surface_relevance

_log = get_logger("pipeline")


@dataclass(frozen=True)
class Answer:
    """What a run produced: the answer, and the two views it was derived from.

    The answer alone rarely tells you whether to trust it, and the intermediate views are
    often what you actually wanted — so a run hands back the whole thing rather than a
    bare string. ``str(result)`` is the answer text, so printing one still does the
    obvious thing.

    - ``answer`` — the committed answer text.
    - ``relevance`` — the **relevant context**: the final-round relevance snippet
      of each document, aligned 1:1 with the ``documents`` you passed in
      (``relevance[i]`` describes ``documents[i]``). A document that contributes nothing
      is an empty string, which is a real result rather than a failure.
    - ``struct_data`` — the merged parse: the per-task schema filled from every document,
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
    relevance: list[str]
    struct_data: dict[str, Any]
    schema_code: str
    source_docs: dict[str, list[int]]
    run_dir: Path | None = None

    def __str__(self) -> str:
        return self.answer


def _preview(text: str, n: int = 100) -> str:
    """One-line, truncated preview of an answer for the progress log."""
    s = " ".join((text or "").split())
    return s if len(s) <= n else s[:n] + "…"


def run_pipeline(
    *,
    task: str,
    documents: list[str],
    config: RunConfig,
    max_reasoning_turns: int = settings.REASONING_MAX_TURNS,
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
    output — model, seed, relevance rounds, the prompt version of each stage, and any
    generation params — so nothing output-affecting is threaded ad-hoc.

    ``api_base``, ``api_key`` and ``completion`` are **transport**: routing, auth, and how
    the request is actually made. None of them changes what a correct answer is, so none
    appears in the run identity. ``completion`` replaces ``litellm.completion`` for every
    call this run makes — hand it a configured ``litellm.Router``'s ``.completion``, or any
    wrapper with the same ``(model, messages, **kwargs)`` shape.


    If ``task_logger`` is provided, each stage's artifacts are written immediately
    after that stage succeeds, so a later failure still leaves earlier artifacts.
    """
    model = config.model
    rounds = config.relevance_rounds
    seed = config.seed
    # The output knobs flow from the config; api_base/api_key are transport only.
    llm_kwargs: dict[str, Any] = {**config.params, "seed": seed}
    if api_base:
        llm_kwargs["api_base"] = api_base
    if api_key:
        llm_kwargs["api_key"] = api_key
    if completion is not None:
        # A named parameter of `litellm_chat_completion`, so it binds there rather than
        # being forwarded into the provider request.
        llm_kwargs["completion"] = completion

    def _run(stage: str, model_for_log: str) -> StageRun | None:
        if task_logger is None:
            return None
        return StageRun(stage=stage, task_logger=task_logger, model=model_for_log, seed=seed)

    # The identity card first, so even a run that dies in stage 1 says what it was.
    if task_logger is not None:
        write_manifest(
            task_logger, task=task, config=config,
            n_docs=len(documents), context_chars=sum(len(d) for d in documents),
        )

    # --- Stage 1: surface relevance — the relevant context. ---
    _log.info("stage 1/3 · surfacing relevance (%d round(s), %d doc(s))", rounds, len(documents))
    relevance_run = _run("relevance", model)
    relevance = surface_relevance(
        task=task, documents=documents, model=model, prompt_version=config.prompts["relevance"],
        rounds=rounds, run=relevance_run, **llm_kwargs,
    )
    relevance_snippets = relevance.snippets  # the relevant context feeds every later stage
    if relevance_run is not None:
        relevance_run.flush(write_transcript=False)
    if task_logger is not None:
        # Per-round, per-document — `round` is the refinement depth; the LAST round
        # is what downstream stages consume. snippets[doc_i] aligns to documents[i].
        task_logger.write_json(
            "relevance/result",
            {
                "n_rounds": rounds,
                "n_docs": len(documents),
                "rounds": [
                    {"round": k + 1, "snippets": per_doc}
                    for k, per_doc in enumerate(relevance.rounds)
                ],
                "totals": relevance_run.compute_totals() if relevance_run else None,
            },
        )

    # --- Stage 2a: structuring — propose the schema (task + relevant context). ---
    _log.info("stage 2/3 · structuring · proposing the schema")
    schema_run = _run("structuring/schema", model)
    proposal = propose_schema(
        task=task, relevance_snippets=relevance_snippets, model=model, prompt_version=config.prompts["structuring/schema"],
        run=schema_run, **llm_kwargs,
    )
    if schema_run is not None:
        schema_run.flush()
    if task_logger is not None:
        task_logger.write_json(
            "structuring/schema/result",
            {
                "schema_code": proposal.schema_code,
                "thought": proposal.attempts[-1].thought if proposal.attempts else None,
                "attempts": [
                    {"schema_code": a.schema_code, "error": a.error, "thought": a.thought}
                    for a in proposal.attempts
                ],
                "totals": schema_run.compute_totals() if schema_run else None,
            },
        )

    # --- Stage 2b: structuring — parse every document, whole, in parallel. ---
    _log.info("stage 2/3 · structuring · parsing %d doc(s)", len(documents))
    parsing_run = _run("structuring/parsing", model)
    extraction = parse_documents(
        documents=documents, schema_code=proposal.schema_code, parse_cls=proposal.parse_cls,
        task=task, prompt_version=config.prompts["structuring/parsing"], relevance_snippets=relevance_snippets, model=model,
        run=parsing_run, **llm_kwargs,
    )
    parsed = extraction.parse
    if parsing_run is not None:
        parsing_run.flush(write_transcript=False)
    if task_logger is not None:
        task_logger.write_json(
            "structuring/parsing/result",
            {
                "parsed": parsed,
                "source_docs": extraction.source_docs,
                "totals": parsing_run.compute_totals() if parsing_run else None,
            },
        )

    # --- Stage 3: reasoning over the parse + the relevant context. ---
    _log.info("stage 3/3 · reasoning")
    reasoning_run = _run("reasoning", model)
    try:
        result = reasoning.reason(
            task=task, schema_code=proposal.schema_code, parsed=parsed,
            source_docs=extraction.source_docs, relevance_snippets=relevance_snippets,
            model=model, prompt_version=config.prompts["reasoning"],
            max_turns=max_reasoning_turns, timeout_s=reasoning_timeout_s,
            run=reasoning_run, **llm_kwargs,
        )
        _log.info("reasoning done (%s, %d turn(s)): %s",
                  result.terminated_by, len(result.turns), _preview(result.answer))
        if reasoning_run is not None:
            reasoning_run.flush()
        if task_logger is not None:
            task_logger.write_json(
                "reasoning/result",
                {
                    "answer": result.answer,
                    "terminated_by": result.terminated_by,
                    "n_turns": len(result.turns),
                    "turns": [
                        {
                            "response": t.response, "raw_response": t.raw_response,
                            "code": t.code, "observation": t.observation,
                            "error": t.error, "is_final_answer": t.is_final_answer,
                        }
                        for t in result.turns
                    ],
                    "totals": reasoning_run.compute_totals() if reasoning_run else None,
                },
            )
    except Exception:
        # Re-raise, but leave the traceback on disk first: the earlier stages' artifacts
        # are already written, so the run folder should also say why the run ended. The
        # whole block is guarded, not just the LLM call — a failure while flushing the
        # transcript or writing result.json is exactly as worth recording. Written from
        # inside the `except` so format_exc() sees the active exception.
        if task_logger is not None:
            err_dir = task_logger.dir / "reasoning"
            err_dir.mkdir(parents=True, exist_ok=True)
            (err_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    return Answer(
        answer=result.answer,
        relevance=list(relevance_snippets),
        # the parse exactly as the agent saw it: plain data, each record stamped with the
        # 1-based document it came from
        struct_data=reasoning.tag_source_documents(
            parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed,
            extraction.source_docs,
        ),
        schema_code=proposal.schema_code,
        source_docs=extraction.source_docs,
        run_dir=task_logger.dir if task_logger is not None else None,
    )
