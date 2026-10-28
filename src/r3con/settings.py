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

# Cap on the sends of one relevance request whose reply does not fit its schema.
RELEVANCE_MAX_ATTEMPTS = 3

# Cap on the schema-proposal validation retry loop.
SCHEMA_MAX_ATTEMPTS = 5

# Cap on the parsing structured-output retry loop (one document).
PARSING_MAX_ATTEMPTS = 5

# Cap on the sends of the direct reasoning request whose reply does not fit its schema.
REASONING_MAX_ATTEMPTS = 3

# Most resends of one request after temporary errors (the rule is in r3con.runtime.llm).
LLM_TEMPORARY_ERROR_RETRIES = 10

# Cap on the exponential backoff before one resend, in seconds (not on a Retry-After).
LLM_RETRY_MAX_WAIT_SECONDS = 60

# Seconds from a request's first attempt after which no resend starts.
LLM_RETRY_MAX_TOTAL_SECONDS = 300

# Re-rolls of an empty structured-output response (a 200 with no content, e.g. a
# reasoning model that spent its budget on thinking).
LLM_EMPTY_CONTENT_RETRIES = 3

# On the coding path, show the WHOLE parse in the reasoning system prompt below this many
# cl100k_base tokens; above it (a record flood — e.g. a 1735-row catalog) fall back to
# one sample per field + a prominent note, so the prompt can't blow up. The direct path
# always shows the whole parse.
REASONING_PARSE_MAX_TOKS = 16000

# The share of the model's input window kept free when r3con estimates whether a
# request fits; a request estimated over the rest is split before it is sent.
WINDOW_MARGIN_PERCENT = 15

# The smallest value of each cap a run can execute under, keyed as in the snapshot.
_FLOORS: dict[str, int] = {
    "doc_workers": 1,
    "reasoning_max_turns": 1,
    "relevance_max_attempts": 1,
    "schema_max_attempts": 1,
    "parsing_max_attempts": 1,
    "reasoning_max_attempts": 1,
    "llm_temporary_error_retries": 0,
    "llm_retry_max_wait_seconds": 0,
    "llm_retry_max_total_seconds": 0,
    "llm_empty_content_retries": 0,
    "reasoning_parse_max_toks": 0,
    "window_margin_percent": 0,
}

# The largest value of a cap that has one, keyed as in the snapshot.
_CEILINGS: dict[str, int] = {"window_margin_percent": 99}


def active_doc_workers() -> int:
    """Max documents processed concurrently within one task —
    ``R3CON_DOC_WORKERS`` else ``DOC_WORKERS``.

    Raises:
        ValueError: naming whichever of the two is set to anything but an integer >= 1.
    """
    raw = os.environ.get("R3CON_DOC_WORKERS")
    if not raw:
        return check_cap("doc_workers", DOC_WORKERS)
    try:
        workers = int(raw)
    except ValueError:
        workers = raw
    return check_cap("doc_workers", workers, name="R3CON_DOC_WORKERS")


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
        ValueError: naming the first cap that is not an integer (a ``bool`` is not),
            is below its floor, or is above its ceiling, e.g.
            ``REASONING_MAX_TURNS must be >= 1, got 0.``
    """
    snapshot = {
        "doc_workers": active_doc_workers(),
        "reasoning_max_turns": (
            REASONING_MAX_TURNS if reasoning_max_turns is None else reasoning_max_turns
        ),
        "relevance_max_attempts": RELEVANCE_MAX_ATTEMPTS,
        "schema_max_attempts": SCHEMA_MAX_ATTEMPTS,
        "parsing_max_attempts": PARSING_MAX_ATTEMPTS,
        "reasoning_max_attempts": REASONING_MAX_ATTEMPTS,
        "llm_temporary_error_retries": LLM_TEMPORARY_ERROR_RETRIES,
        "llm_retry_max_wait_seconds": LLM_RETRY_MAX_WAIT_SECONDS,
        "llm_retry_max_total_seconds": LLM_RETRY_MAX_TOTAL_SECONDS,
        "llm_empty_content_retries": LLM_EMPTY_CONTENT_RETRIES,
        "reasoning_parse_max_toks": REASONING_PARSE_MAX_TOKS,
        "window_margin_percent": WINDOW_MARGIN_PERCENT,
    }
    for key, value in snapshot.items():
        check_cap(key, value)
    return snapshot


def check_cap(key: str, value: object, name: str | None = None) -> int:
    """Return ``value`` if it is an integer from the floor of the cap ``key`` (keyed as
    in the snapshot) to its ceiling, if it has one; else raise ``ValueError`` naming
    ``name``, the setting the value came from (by default the cap's constant)."""
    name = name or key.upper()
    floor, ceiling = _FLOORS[key], _CEILINGS.get(key)
    if type(value) is not int:  # a bool is not a cap
        raise ValueError(f"{name} must be an integer >= {floor}, got {value!r}.")
    if value < floor:
        raise ValueError(f"{name} must be >= {floor}, got {value}.")
    if ceiling is not None and value > ceiling:
        raise ValueError(f"{name} must be <= {ceiling}, got {value}.")
    return value
