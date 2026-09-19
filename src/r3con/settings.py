"""Runtime knobs and the paths the package reads and writes.

Everything here is RUNTIME only — parallelism, retry caps, timeouts, where files live.
None of it changes a correct answer. Everything that shapes the OUTPUT (model, seed,
relevance rounds, prompt versions, generation params) lives in a
:class:`r3con.config.RunConfig`, not here.

Loading a ``.env`` is deliberately NOT done here: a library must not mutate the
process environment as a side effect of being imported. The CLI and :func:`r3con.run`
load one explicitly; a library caller manages their own environment.

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
from typing import Any



class Settings:
    # --- shipped package data (read-only, inside the wheel) ---
    PKG_DIR: Path = Path(__file__).resolve().parent
    PROMPTS_DIR: Path = PKG_DIR / "prompts"  # prompts/<stage>/<version>.yaml
    CONFIGS_DIR: Path = PKG_DIR / "configs"  # configs/<name>.yaml

    # NOTE: there is deliberately no LOGS_DIR attribute. A class attribute is evaluated
    # once at import, which would freeze the working directory of whoever imported first
    # and ignore a later R3CON_LOGS_DIR. Use active_logs_dir().

    # Max documents processed concurrently within ONE task — the relevance fan-out
    # (each round) and the per-document parsing fan-out. A caller running several tasks
    # at once should keep tasks × this under their endpoint's concurrency cap.
    # Override via R3CON_DOC_WORKERS.
    DOC_WORKERS: int = 16

    # Hard cap on the reasoning agent's multi-turn loop.
    REASONING_MAX_TURNS: int = 30

    # Cap on the schema-proposal validation retry loop.
    SCHEMA_MAX_ATTEMPTS: int = 5

    # Cap on the parsing structured-output retry loop (one document).
    PARSING_MAX_ATTEMPTS: int = 5

    # Transport-level retries passed to every litellm call (exponential backoff) on
    # transient errors: connection refused/reset, timeouts, 5xx. NOTE: litellm imports
    # `tenacity` lazily to service this, which is why tenacity is a declared dependency
    # even though nothing in this package imports it.
    LLM_NUM_RETRIES: int = 10

    # Content-level re-rolls for an empty structured-output response (a 200 with
    # no JSON — e.g. a reasoning model that spent its budget on thinking). Each
    # re-roll perturbs the seed so a pinned-seed call gets a different roll.
    LLM_EMPTY_CONTENT_RETRIES: int = 3

    # Show the WHOLE parse in the reasoning system prompt below this many tokens (tiktoken
    # cl100k_base); above it (a record flood — e.g. a 1735-row catalog) fall back to one
    # sample per field + a prominent note, so the prompt can't blow up. Most parses are
    # small, so the whole thing shows; this is just the flood guard.
    REASONING_PARSE_MAX_TOKS: int = 16000


settings = Settings()


# ---------- runtime resolvers ----------


def active_doc_workers() -> int:
    """Max documents processed concurrently within one task —
    ``R3CON_DOC_WORKERS`` else the default."""
    raw = os.environ.get("R3CON_DOC_WORKERS")
    return int(raw) if raw else settings.DOC_WORKERS


def active_logs_dir() -> Path:
    """Where run artifacts go: ``R3CON_LOGS_DIR`` if set, else ``./logs`` under the
    current working directory. Resolved per call, never at import, so it follows the
    caller rather than freezing whoever imported first. Never inside the package."""
    return Path(os.environ.get("R3CON_LOGS_DIR") or Path.cwd() / "logs")


def settings_snapshot() -> dict[str, Any]:
    """The runtime knobs (parallelism + resilience caps), for a caller that wants to
    record what a run was executed under — distinct from the output-shaping
    ``RunConfig``, which is the run's identity."""
    return {
        "doc_workers": active_doc_workers(),
        "reasoning_max_turns": settings.REASONING_MAX_TURNS,
        "schema_max_attempts": settings.SCHEMA_MAX_ATTEMPTS,
        "parsing_max_attempts": settings.PARSING_MAX_ATTEMPTS,
        "llm_num_retries": settings.LLM_NUM_RETRIES,
        "llm_empty_content_retries": settings.LLM_EMPTY_CONTENT_RETRIES,
        "reasoning_parse_max_toks": settings.REASONING_PARSE_MAX_TOKS,
    }
