"""Runtime knobs and the paths the package reads and writes.

Everything here is RUNTIME only — parallelism, retry caps, timeouts, where files live.
None of it changes a correct answer. Everything that shapes the OUTPUT (model,
relevance rounds, prompt versions, generation params) lives in a
:class:`r3con.config.RunConfig`, not here.

Each knob is a module constant, read when the call that uses it runs, so setting
``r3con.settings.REASONING_MAX_TURNS = 3`` bounds every run started afterwards.

Loading a ``.env`` is deliberately NOT done here: a library must not mutate the
process environment as a side effect of being imported.

**Two kinds of path, and they must never be confused:**

- **Shipped, read-only data** — the prompts and the bundled configs — lives *inside the
  installed package* (``PKG_DIR``). It travels in the wheel.
- **Written output** — run artifacts — goes under the *user's* working directory. The
  package directory is not writable (and on a shared install it belongs to no one), so
  nothing here may ever write next to the code.

There is deliberately no "repo root". Walking up from ``__file__`` to a parent directory
works in a source checkout and silently points at the interpreter's library directory
once installed.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- shipped package data (read-only, inside the wheel) ---
PKG_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = PKG_DIR / "prompts"  # prompts/<stage>/<version>.yaml
CONFIGS_DIR = PKG_DIR / "configs"  # configs/<name>.yaml

# Max documents processed concurrently within ONE task — the relevance fan-out (each
# round) and the per-document parsing fan-out. A caller running several tasks at once
# should keep tasks × this under their endpoint's concurrency cap. Override via
# R3CON_DOC_WORKERS.
DOC_WORKERS = 16

# Hard cap on the reasoning agent's turns.
REASONING_MAX_TURNS = 30

# Cap on the schema-proposal validation retry loop.
SCHEMA_MAX_ATTEMPTS = 5

# Cap on the parsing structured-output retry loop (one document).
PARSING_MAX_ATTEMPTS = 5

# Transport-level retries passed to every litellm call on transient errors: connection
# refused/reset, timeouts, 5xx.
LLM_NUM_RETRIES = 10

# Re-rolls of an empty structured-output response (a 200 with no JSON, e.g. a
# reasoning model that spent its budget on thinking).
LLM_EMPTY_CONTENT_RETRIES = 3

# Show the WHOLE parse in the reasoning system prompt below this many tokens (tiktoken
# cl100k_base); above it (a record flood — e.g. a 1735-row catalog) fall back to one
# sample per field + a prominent note, so the prompt can't blow up.
REASONING_PARSE_MAX_TOKS = 16000

# The smallest value of each cap a run can execute under, keyed as in the snapshot.
_FLOORS: dict[str, int] = {
    "doc_workers": 1,
    "reasoning_max_turns": 1,
    "schema_max_attempts": 1,
    "parsing_max_attempts": 1,
    "llm_num_retries": 0,
    "llm_empty_content_retries": 0,
    "reasoning_parse_max_toks": 0,
}


def active_doc_workers() -> int:
    """Max documents processed concurrently within one task —
    ``R3CON_DOC_WORKERS`` else ``DOC_WORKERS``."""
    raw = os.environ.get("R3CON_DOC_WORKERS")
    return int(raw) if raw else DOC_WORKERS


def active_logs_dir() -> Path:
    """Where run artifacts go: ``R3CON_LOGS_DIR`` if set, else ``./logs`` under the
    current working directory. Resolved per call, never at import, so it follows the
    caller rather than freezing whoever imported first. Never inside the package."""
    return Path(os.environ.get("R3CON_LOGS_DIR") or Path.cwd() / "logs")


def settings_snapshot(*, reasoning_max_turns: int | None = None) -> dict[str, int]:
    """The caps one run executes under, each read now and checked against its floor.

    An explicit ``reasoning_max_turns`` replaces ``REASONING_MAX_TURNS``. These are
    the runtime knobs a run's manifest records, distinct from the output-shaping
    ``RunConfig``, which is the run's identity.

    Raises:
        ValueError: naming the first cap that is not an integer (a ``bool`` is not)
            or is below its floor, e.g. ``REASONING_MAX_TURNS must be >= 1, got 0.``
    """
    snapshot = {
        "doc_workers": active_doc_workers(),
        "reasoning_max_turns": (
            REASONING_MAX_TURNS if reasoning_max_turns is None else reasoning_max_turns
        ),
        "schema_max_attempts": SCHEMA_MAX_ATTEMPTS,
        "parsing_max_attempts": PARSING_MAX_ATTEMPTS,
        "llm_num_retries": LLM_NUM_RETRIES,
        "llm_empty_content_retries": LLM_EMPTY_CONTENT_RETRIES,
        "reasoning_parse_max_toks": REASONING_PARSE_MAX_TOKS,
    }
    for key, value in snapshot.items():
        floor = _FLOORS[key]
        if type(value) is not int:  # a bool is not a cap
            raise ValueError(
                f"{key.upper()} must be an integer >= {floor}, got {value!r}."
            )
        if value < floor:
            raise ValueError(f"{key.upper()} must be >= {floor}, got {value}.")
    return snapshot
