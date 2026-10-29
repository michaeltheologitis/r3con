"""Per-task structured logging at ``logs/<run-folder>/``.

Each task-run gets **one flat, opaque, uniquely-named folder** — no ``run-dir/task-id``
nesting. ``<run-folder>`` (:func:`new_run_folder`) is ``<UTC-timestamp>_<hex>`` and
carries **no identity at all**: what was run, and under which config, lives *inside* it
in ``manifest.json``. Every run is its own folder (timestamp + a random suffix never
collide), so re-runs **accumulate** rather than overwrite; runs are grouped after the
fact by reading each manifest (e.g. by the config's run label,
:meth:`r3con.config.RunConfig.label`).

Layout per ``logs/<run-folder>/``:

- ``manifest.json`` (:func:`write_manifest`) — the identity card + the full parameter
  snapshot — the task, a ``config`` block (the resolved :class:`RunConfig` — model,
  relevance_rounds, prompts, params, …), ``litellm_model`` (the model string exactly as
  litellm was called with it, where ``config`` keeps only the bare name), and a
  ``settings`` block (the runtime knobs). Written first, so a crashed task is still
  discoverable. The run label is **not** stored — it is recomputed from the ``config``
  block (:meth:`r3con.config.RunConfig.label`).
- ``ids.json`` — each document's id in this run, mapped to its 0-based index in
  ``documents``. Written with the manifest.
- ``relevance/`` (stage 1, surfacing relevance) — ``result.json`` (``{n_rounds, n_docs,
  rounds: [{round, snippets: [{id, summary, provenance}]}], totals}`` — within a round
  every relevance snippet in document order, a document read in parts as one per part,
  each with its id (the document's, or ``<id>.<k>`` for part ``k`` in that round) and
  the ids of the previous round's snippets it used; the last round is the relevant
  context that feeds downstream) + ``calls.json`` (one per ``relevance_snippet`` call,
  tagged ``relevance-r{round}-d{doc}``, or ``relevance-r{round}-d{doc}c{part}`` for a
  part, and ``-retry-{n}`` added to the n-th resend of a reply that did not fit its
  schema).
- ``structuring/schema/`` (stage 2, the schema proposal) — ``result.json``
  (``{schema_code, thought, attempts, totals}``) + ``calls.json`` + ``transcript.yaml``.
- ``structuring/parsing/`` (stage 2, filling that schema per document) — ``result.json``
  (``{parsed, source_docs, source_ids, provenance, totals}`` — for each list field of
  ``parsed``, the 0-based index and the id of the document each record came from, a
  part's records carrying their document's; and per document, in document order, the
  ids of the relevance snippets its parse used) + ``calls.json`` (one entry per
  ``parse-d{doc}`` call, or ``parse-d{doc}c{part}`` for a part, ``-retry-{n}`` added to
  a resend).
- ``reasoning/`` (stage 3) — ``result.json`` (``{answer, provenance, path, fallback,
  terminated_by, n_turns, turns, totals}`` — ``provenance`` is ``{summary_ids,
  parse_ids}``, the ids of the relevance snippets the answer used (``<id>.<k>`` for a
  part) and of the documents whose records it used, ``path`` is ``direct`` or
  ``coding``, and ``fallback`` says why the coding path ran when the config asked for
  direct, else ``null``) + ``calls.json`` + ``transcript.yaml``.
- ``splits.json`` — only when a document was read in parts or splitting stopped
  (:mod:`r3con.splitting`): the model's window, and per document its cuts, the parts
  each call read, and every split and stop with its cause.
- ``events.jsonl`` — the run's progress events (:class:`EventRecorder`), one JSON
  object per line, in the order and at the wall-clock time they happened, so the run
  can be replayed: each stage's start and finish, each document's (or part's) in
  relevance and parsing, and the run's own start and its finish or failure.
- ``asks/<folder>/`` — one per :func:`r3con.ask` over the run, named as a run folder
  is: ``manifest.json`` as above, its ``task`` the ask's question and ``ask_of`` the
  run folder's name, ``events.jsonl`` (the ask's own events) and ``reasoning/`` as
  above.

A stage that raises writes no ``result.json``: its folder holds ``calls.json`` (every
call that answered or raised an error, plus ``transcript.yaml`` where the stage writes
one) and ``error.txt``, the traceback.

Error text is written with keys masked (:mod:`r3con.masking`): ``calls.json``'s
``error``, ``error.txt``, the ``error`` of ``run_failed`` and ``ask_failed`` in
``events.jsonl``, each split's ``error`` in ``splits.json`` and the ``error`` of
reasoning's ``fallback`` say ``REDACTED`` where a provider echoed a key, as does
``manifest.json``'s ``config`` in place of a credential in the run's ``params``, an
extra header say. Prompts, replies and documents are written as they are.

Two layers in code:

- ``TaskLogger`` writes structured artifacts under one run-folder. ``write_json("a/b",
  data)`` writes ``<folder>/a/b.json`` and auto-creates any missing parent directories
  — so callers spell stage paths inline (``"structuring/schema/result"``,
  ``"reasoning/result"``).
- ``StageRun`` accumulates per-LLM-call ``StepRecord`` instances and on ``flush()``
  writes ``calls.json`` (the full per-call record — ``{step, kind, doc, chunk,
  tokens, usd, sends, resend_wait_seconds, finish_reason, error, prompt, output}``,
  where ``chunk`` is the part index of a document read in parts, ``usd`` is litellm's
  price for the call, or ``null`` when it cannot price it, ``sends`` is the times the
  request was sent, more than 1 when a temporary error made r3con resend it, and
  ``resend_wait_seconds`` the seconds, to hundredths, waited between those sends, both
  ``null`` when r3con did not resend it itself, and ``error`` is ``null``, or, for a
  call that raised one, the error's class and the first non-empty line of its message,
  keys masked: such a call has ``tokens``, ``finish_reason`` and ``output`` ``null``
  and ``usd`` 0, as r3con got no reply to price) and, unless skipped,
  ``transcript.yaml`` (the readable message thread). Aggregate token and cost totals
  are available via ``compute_totals()`` so callers fold them into the stage's
  ``result.json``; their ``n_steps`` counts only the calls that answered.
"""

from __future__ import annotations

import datetime
import json
import logging
import re
import secrets
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel

# ``normalize_model_name`` lives in ``r3con.config`` — it's a model-identity helper
# ``RunConfig.label()`` uses too; shared here so the transcript header below records the
# same normalized name the run label does.
from r3con.config import normalize_model_name
from r3con.logging_setup import get_logger
from r3con.masking import masked
from r3con.settings import active_logs_dir

_log = get_logger("runs")

# Pulls the source-document index (and, for a document read in parts, the part index)
# out of a call kind so calls.json carries per-call provenance. Matches ``parse-d0``,
# ``relevance-r2-d3`` and ``parse-d3c1``.
_DOC_CHUNK_RE = re.compile(r"-d(\d+)(?:c(\d+))?\b")
_EVENTS = "events.jsonl"

Emit = Callable[..., None]
"""Reports one progress event, ``emit(event_type, **fields)``, as
:class:`EventRecorder` does."""


def new_run_folder() -> str:
    """A fresh, opaque, unique log-folder name for one task-run:
    ``<UTC-timestamp>_<hex>`` (e.g. ``20260613T142233Z_a1b2c3d4``).

    Carries **no identity** — what was run, and under which config, lives inside the
    folder in ``manifest.json``. Every call is unique (timestamp + a random suffix), so
    runs **accumulate** rather than overwrite; grouping them is a read-the-manifests
    job, done after the fact and never encoded in a path.
    """
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{secrets.token_hex(4)}"


class TaskLogger:
    """Writes per-stage artifacts under one flat run-folder ``logs/<folder>/``.

    Callers spell stage paths inline:

    - ``write_json("manifest", {...})`` → ``<folder>/manifest.json``
    - ``write_json("structuring/schema/result", {...})`` →
      ``<folder>/structuring/schema/result.json``
    - ``write_yaml("reasoning/transcript", msgs)`` →
      ``<folder>/reasoning/transcript.yaml``

    ``folder`` is the unique run-folder name (see :func:`new_run_folder`); ``task_id``
    is the caller's own name for the task, recorded in transcript headers (defaults to
    ``folder``).
    Parent directories are auto-created. The extension is appended from the method
    name. ``data`` may be a Pydantic ``BaseModel`` / class, a ``dict``/``list``, or
    any JSON-serializable value; non-serializable values fall back to ``repr``.
    """

    def __init__(
        self, folder: str, *, task_id: str | None = None, root: Path | None = None
    ) -> None:
        self.folder = folder
        self.task_id = task_id if task_id is not None else folder
        base = root or active_logs_dir()
        self.dir = base / folder
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, name: str, ext: str) -> Path:
        path = self.dir / f"{name}.{ext}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, name: str, data: Any) -> Path:
        path = self._path(name, "json")
        # ensure_ascii=False keeps non-ASCII (e.g. CJK) readable in the logs rather than
        # as \uXXXX escapes; the explicit utf-8 write makes that safe on any host locale.
        path.write_text(
            json.dumps(_jsonable(data), indent=2, default=repr, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def write_yaml(self, name: str, data: Any) -> Path:
        path = self._path(name, "yaml")
        path.write_text(_dump_transcript_yaml(data), encoding="utf-8")
        return path

    def write_text(self, name: str, text: str) -> Path:
        path = self._path(name, "txt")
        path.write_text(text, encoding="utf-8")
        return path


def no_events(event_type: str, /, **fields: Any) -> None:
    """An :data:`Emit` that drops every event."""


class EventRecorder:
    """An :data:`Emit` for one run or ask: each event is ``{"type": event_type,
    "time": ..., **fields}``, ``time`` the wall clock in ISO 8601, written as one line
    of JSON (a value JSON can't hold as its ``str``) to ``events.jsonl`` in
    ``task_logger``'s folder, if there is one, and handed to ``on_event``, if there is
    one, as that line decoded: a copy, so changing it changes nothing in the run.

    Stages emit from worker threads. Each event is stamped, written and handed over
    under one lock, so ``on_event`` is never entered twice at once and the file's lines
    are in time order; workers wait while it runs, so it should return quickly (put the
    event on a queue, say). Recording never fails the run: when writing the file or
    ``on_event`` raises, the error is logged, at warning level with its traceback the
    first time its type is raised there and at debug level after, and the run goes on.
    """

    def __init__(
        self,
        on_event: Callable[[dict[str, Any]], None] | None,
        task_logger: TaskLogger | None,
    ) -> None:
        self._on_event = on_event
        self._path = None if task_logger is None else task_logger.dir / _EVENTS
        self._lock = threading.Lock()
        self._raised: set[tuple[str, type[Exception]]] = set()

    def __call__(self, event_type: str, /, **fields: Any) -> None:
        with self._lock:
            now = datetime.datetime.now(datetime.UTC)
            event = {
                "type": event_type,
                "time": now.isoformat(timespec="milliseconds"),
                **fields,
            }
            line = json.dumps(event, ensure_ascii=False, default=str)
            if self._path is not None:
                where = f"writing {self._path}"
                self._guarded(where, event_type, _append_line, self._path, line)
            if self._on_event is not None:
                self._guarded("on_event", event_type, self._on_event, json.loads(line))

    def _guarded(
        self, what: str, event_type: str, action: Callable[..., None], *args: Any
    ) -> None:
        try:
            action(*args)
        except Exception as error:  # noqa: BLE001 — recording a paid run must not kill it
            raised = (what, type(error))
            _log.log(
                logging.DEBUG if raised in self._raised else logging.WARNING,
                "%s raised %s on a %s event; the run goes on",
                what,
                type(error).__name__,
                event_type,
                exc_info=True,
            )
            self._raised.add(raised)


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as lines:
        lines.write(line + "\n")


def write_manifest(
    task_logger: TaskLogger,
    *,
    task: str,
    config: Any,
    n_docs: int,
    context_chars: int | None = None,
    settings: Mapping[str, int] | None = None,
    ask_of: str | None = None,
) -> Path:
    """Write ``<run-folder>/manifest.json`` — the run's identity card.

    The folder name is opaque **by design**, so this file is the only thing that says
    what the run was: the question, the resolved :class:`~r3con.config.RunConfig`, the
    model string litellm is called with (``litellm_model``), the runtime caps it ran
    under (``settings``, else :func:`r3con.settings.settings_snapshot` taken now), and
    the package version; for an ask (:func:`r3con.ask`), ``ask_of``, the name of the
    run folder it answered over.
    Written before the first stage, so even a crashed run is identifiable.

    ``r3con_version`` matters more than it looks: the prompts ship *inside* the
    installed package, so a prompt version string like ``v1`` identifies content only
    relative to a release. Without the version recorded here, "which prompt actually
    ran?" is unanswerable after an upgrade.
    """
    from r3con import __version__
    from r3con.config import PROMPT_STAGES, resolve_config_path
    from r3con.prompts import resolve_prompt_path
    from r3con.settings import settings_snapshot

    config_block = (
        config.model_dump() if hasattr(config, "model_dump") else dict(config)
    )
    litellm_model = config_block.get("model")
    if "model" in config_block:
        config_block["model"] = normalize_model_name(config_block["model"])
    overrides = config_block.get("overrides") or {}
    if "model" in overrides:
        overrides["model"] = normalize_model_name(overrides["model"])
    # Its params may hold a credential, an extra header say, registered at run start.
    as_json = json.dumps(config_block, default=repr, ensure_ascii=False)
    config_block = json.loads(masked(as_json))
    manifest = {
        "r3con_version": __version__,
        "task": task,
        "n_docs": n_docs,
        "context_chars": context_chars,
        "run_folder": task_logger.folder,
        # --- what shapes the output (the run's identity) ---
        "config": config_block,
        "litellm_model": litellm_model,
        # --- where the config and each prompt actually came from ---
        # A user overlay (./configs, ./prompts) shadows the packaged copy, and an
        # overlaid file is invisible in the run label — so record the winning path.
        "sources": _sources(
            config_block.get("name"),
            config_block.get("prompts") or {},
            PROMPT_STAGES,
            resolve_config_path,
            resolve_prompt_path,
        ),
        # --- runtime knobs (parallelism / resilience) ---
        "settings": dict(settings) if settings is not None else settings_snapshot(),
        "created": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    if ask_of is not None:
        manifest["ask_of"] = ask_of
    return task_logger.write_json("manifest", manifest)


def _sources(
    config_name, prompts, stages, resolve_config, resolve_prompt
) -> dict[str, Any]:
    """Which file on disk each of the run's inputs resolved to."""
    cfg = resolve_config(config_name) if config_name else None
    return {
        "config": str(cfg) if cfg else None,
        "prompts": {
            stage: (
                str(path) if (path := resolve_prompt(stage, prompts[stage])) else None
            )
            for stage in stages
            if stage in prompts
        },
    }


@dataclass
class StepRecord:
    """One step in a stage run — typically one LLM call (or retry, or loop turn)."""

    step: int
    kind: str  # "llm_call" | "retry" | "turn-N" | "relevance-r2-d3" | "parse-d0" | ...
    messages: list[dict[str, Any]]
    response: dict[str, Any]  # {"role": ..., "content": ..., "finish_reason": ...}
    schema: dict[str, Any] | None = None
    # {"prompt": int, "completion": int, "total": int}
    tokens: dict[str, int] | None = None
    usd: float | None = None
    sends: int | None = None
    resend_wait_seconds: float | None = None
    error: str | None = None


def step_totals(steps: Sequence[StepRecord]) -> dict[str, Any]:
    """The count of ``steps`` that answered, leaving out those that raised an error
    (``calls.json`` lists them), the summed tokens of ``steps`` (``None`` when none
    counted any), and their ``usd``: the sum of theirs, or ``None`` when litellm could
    not price one of them."""
    prompt = completion = total = 0
    any_tokens = False
    costs = [s.usd for s in steps]
    for s in steps:
        if s.tokens is not None:
            any_tokens = True
            prompt += s.tokens.get("prompt", 0)
            completion += s.tokens.get("completion", 0)
            total += s.tokens.get("total", 0)
    return {
        "n_steps": sum(s.error is None for s in steps),
        "tokens": {"prompt": prompt, "completion": completion, "total": total}
        if any_tokens
        else None,
        "usd": None if None in costs else float(sum(cast("list[float]", costs))),
    }


class StageRun:
    """Accumulates per-LLM-call records for one stage; flushes calls.json (+ transcript).

    ``add_step`` records one LLM call (or one turn of a multi-turn loop), and
    ``flush`` writes them. ``add_step`` is thread-safe, so calls fanned out across
    documents in parallel can share one run.
    """

    def __init__(
        self,
        *,
        stage: str,
        task_logger: TaskLogger,
        model: str | None = None,
    ) -> None:
        self.stage = stage
        self.task_logger = task_logger
        self.model = model
        self.steps: list[StepRecord] = []
        self._lock = threading.Lock()
        self.dir = task_logger.dir / stage
        self.dir.mkdir(parents=True, exist_ok=True)

    def add_step(
        self,
        *,
        kind: str = "llm_call",
        messages: list[dict[str, Any]],
        response: dict[str, Any],
        schema: dict[str, Any] | None = None,
        tokens: dict[str, int] | None = None,
        usd: float | None = None,
        sends: int | None = None,
        resend_wait_seconds: float | None = None,
        error: str | None = None,
    ) -> StepRecord:
        """Record one step (thread-safe). Step numbers are assigned under a lock, in
        completion order, and identify a call: a kind can repeat, as when a refused
        turn is sent again under the same kind."""
        with self._lock:
            step = StepRecord(
                step=len(self.steps) + 1,
                kind=kind,
                messages=list(messages),
                response=dict(response),
                schema=schema,
                tokens=tokens,
                usd=usd,
                sends=sends,
                resend_wait_seconds=resend_wait_seconds,
                error=error,
            )
            self.steps.append(step)
        return step

    def flush(self, *, write_transcript: bool = True) -> Path:
        """Write ``calls.json`` (always) and, unless ``write_transcript=False``,
        ``transcript.yaml``; return the path of the primary file written.

        ``write_transcript=False`` is used by stages whose many parallel
        independent calls (surfacing relevance, parsing) make a flattened transcript
        confusing — their ``calls.json`` is the readable source.
        """
        calls_path = self._write_calls()
        if not write_transcript:
            return calls_path
        transcript = {
            "stage": self.stage,
            "task_id": self.task_logger.task_id,
            "model": normalize_model_name(self.model),
            "messages": self._build_transcript_messages(),
        }
        transcript_path = self.dir / "transcript.yaml"
        transcript_path.write_text(_dump_transcript_yaml(transcript), encoding="utf-8")
        return transcript_path

    def _write_calls(self) -> Path:
        """Write ``calls.json`` — one entry per LLM call, with the fields the module
        docstring defines. ``doc``/``chunk`` are read out of the call ``kind`` (e.g.
        ``parse-d0``, ``relevance-r1-d2``) when present."""
        calls: list[dict[str, Any]] = []
        for s in self.steps:
            m = _DOC_CHUNK_RE.search(s.kind)
            resp = s.response if isinstance(s.response, dict) else {}
            calls.append(
                {
                    "step": s.step,
                    "kind": s.kind,
                    "doc": int(m.group(1)) if m else None,
                    "chunk": int(m.group(2)) if (m and m.group(2)) else None,
                    "tokens": s.tokens,
                    "usd": s.usd,
                    "sends": s.sends,
                    "resend_wait_seconds": s.resend_wait_seconds,
                    "finish_reason": resp.get("finish_reason"),
                    "error": s.error,
                    "prompt": s.messages,
                    "output": resp.get("content"),
                }
            )
        path = self.dir / "calls.json"
        path.write_text(
            json.dumps(calls, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    def compute_totals(self) -> dict[str, Any]:
        """The stage's :func:`step_totals`, for its ``result.json``."""
        return step_totals(self.steps)

    def _build_transcript_messages(self) -> list[dict[str, Any]]:
        """The readable conversation = the final step's full message thread + its
        reply, if it got one. For a multi-turn loop (codeact) the last step's
        ``messages`` already contain the whole grown conversation; for a retry loop (the
        schema proposal) they are the final attempt — earlier attempts live in
        ``calls.json``. A stage that ended on a call that raised shows the thread that
        call sent, with no reply. Empty run → ``[]``. (Stages with many parallel calls
        flush with ``write_transcript=False`` and never reach here.)"""
        if not self.steps:
            return []
        last = self.steps[-1]
        thread = list(last.messages)
        if last.response.get("content"):
            thread.append(last.response)
        return thread


def _str_representer(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    """Render strings with newlines as literal ``|`` block scalars (readable
    transcripts); rstrip each line so trailing whitespace doesn't force PyYAML's
    escaped single-line fallback."""
    if "\n" in data:
        cleaned = "\n".join(line.rstrip() for line in data.split("\n"))
        return dumper.represent_scalar("tag:yaml.org,2002:str", cleaned, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


class _BlockDumper(yaml.SafeDumper):
    """Local SafeDumper subclass so the literal-block representer stays local."""


_BlockDumper.add_representer(str, _str_representer)


def _dump_transcript_yaml(payload: Any) -> str:
    return yaml.dump(
        _jsonable(payload),
        Dumper=_BlockDumper,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
        width=10000,
    )


def _jsonable(data: Any) -> Any:
    """Coerce Pydantic instances → ``model_dump``, classes → JSON schema; recurse
    lists/dicts; leave everything else for ``json.dumps`` (``default=repr`` net)."""
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")
    if isinstance(data, type) and issubclass(data, BaseModel):
        return data.model_json_schema()
    if isinstance(data, dict):
        return {k: _jsonable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [_jsonable(v) for v in data]
    return data
