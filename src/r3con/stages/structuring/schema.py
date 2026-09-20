"""Stage 2 (structuring), step 1: the per-task schema-proposal driver.

Calls the model with the task — and, when available, the relevant
context from stage 1 — validates the emitted code with
:func:`r3con.stages.structuring.parsing.check_schema`, and on failure feeds the
validator's error back so the model can correct itself. That retry-feedback channel
is the whole point of ``check_schema`` raising :class:`SchemaError` with a shaped
message: a rejection has to say *what* was wrong precisely enough to act on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import BaseModel

from r3con.logging_setup import get_logger
from r3con.runtime.llm import litellm_chat_completion
from r3con.stages.structuring.parsing import SchemaError, check_schema
from r3con.stages.relevance import render_relevance
from r3con.prompts import load_prompt
from r3con.runs import StageRun
from r3con.settings import settings

_log = get_logger("structuring.schema")


# The schema prompts instruct the model to wrap the Python schema source
# in ``<schema>...</schema>`` tags (matching the in-codebase convention used
# by the reasoning agent's ``<code>``/``<observation>`` blocks). XML-style
# tags are less ambiguous than markdown fences (no collision with backticks
# inside string literals or docstrings) and easier for humans to spot in
# transcripts. ``check_schema`` runs on the *body* between the tags.
_SCHEMA_TAG_RE = re.compile(
    r"<schema>\s*\n?(?P<body>.*?)\n?\s*</schema>",
    re.DOTALL | re.IGNORECASE,
)

# Fallback: some models wrap code in markdown fences (e.g. because they
# missed the tag instructions or because they emit both). Search for
# *any* ```python``` (or bare ``` ```) block anywhere in the response —
# the model might prefix it with a `Thought:` line or other prose.
_FENCE_RE = re.compile(
    r"```(?:python|py)?\s*\n(?P<body>.*?)\n\s*```",
    re.DOTALL,
)

# The schema prompts ask for a brief ``Thought:`` snippet before the
# ``<schema>`` block. We pull it out for ``structuring/schema/result.json`` so
# the schema-design reasoning is readable without reading the raw transcript.
# Captures everything after ``Thought:`` until the schema body delimiter
# (the ``<schema>`` tag, a markdown fence, or end-of-text).
_THOUGHT_RE = re.compile(
    r"Thought:\s*(?P<body>.*?)(?=\n\s*<schema>|\n\s*```|\Z)",
    re.DOTALL | re.IGNORECASE,
)


def _extract_schema(text: str) -> str:
    """Pull the Python schema source out of the model's response.

    Resolution order:

    1. ``<schema>...</schema>`` tag body — what the prompts ask for.
    2. First markdown ```python``` (or bare ``` ```) fence found
       anywhere in the response — fallback when the model uses
       fences (often after a `Thought:` preamble) instead.
    3. Raw text — final fallback.

    The returned string is what ``check_schema`` will ``exec``. The
    ``Thought:`` preamble the prompt asks for (and any other prose
    outside the schema tags / fence) is automatically discarded.
    """
    if m := _SCHEMA_TAG_RE.search(text):
        return m.group("body").strip()
    if m := _FENCE_RE.search(text):
        return m.group("body").strip()
    return text.strip()


def _extract_thought(text: str) -> str | None:
    """Pull the ``Thought:`` snippet out of the model's response.

    Returns the prose after ``Thought:`` up to the schema body
    delimiter (the ``<schema>`` tag or a markdown fence), stripped.
    Returns ``None`` when no ``Thought:`` marker is present or when
    the captured body is empty.
    """
    m = _THOUGHT_RE.search(text)
    if not m:
        return None
    body = m.group("body").strip()
    return body or None


@dataclass
class ProposalAttempt:
    schema_code: str
    error: str | None  # ``None`` for the (final) successful attempt
    thought: str | None = None  # Optional 1-2 sentence rationale prefacing the schema


@dataclass
class ProposalResult:
    """Outcome of a successful :func:`propose_schema` call.

    ``attempts`` is the full history including the final successful attempt
    (its ``error`` is ``None``). Useful for logging — the schema prompt and
    its retries are the most interesting signal when iterating on stage 2.
    """

    schema_code: str
    parse_cls: type[BaseModel]
    attempts: list[ProposalAttempt] = field(default_factory=list)


def _retry_prompt(task: str, prev_code: str, error: str) -> str:
    """Build the next user prompt after a rejected attempt.

    Shows the model its previous attempt and the validator's error so it can
    fix the specific failure rather than start over blindly.
    """
    return (
        f"Input:\n<task>\n{task}\n</task>\n"
        "Output:\n\n"
        "# Previous attempt (rejected by validator)\n"
        f"{prev_code}\n\n"
        "# Validator error\n"
        f"{error}\n\n"
        "# Emit a corrected version that addresses the error above. "
        "Output only the Python code defining the schema."
    )


def propose_schema(
    *,
    task: str,
    relevance_snippets: list[str] | None = None,
    model: str,
    prompt_version: str,
    max_attempts: int = settings.SCHEMA_MAX_ATTEMPTS,
    run: StageRun | None = None,
    **llm_kwargs: Any,
) -> ProposalResult:
    """Propose a validated Pydantic schema for ``task``.

    Calls the model up to ``max_attempts`` times; after each rejection,
    the prior attempt and the validator error are appended to the next
    user prompt so the model can correct itself. Default comes from
    ``settings.SCHEMA_MAX_ATTEMPTS``.

    Args:
        task: The task the schema must capture information for.
        relevance_snippets: The relevant context from stage 1 — the
            final per-document notes. Rendered by :func:`render_relevance` into
            the system prompt so the schema is grounded in what the documents
            actually surfaced, not the task's surface words alone. ``None``/empty
            → no relevance block (the model sees only the task).
        model: LiteLLM provider-prefixed model string (e.g. ``"openai/gpt-5.6-luna"``).
        max_attempts: Cap on model calls. Defaults to
            ``settings.SCHEMA_MAX_ATTEMPTS``; pass a smaller value
            for a faster fail in tests, or a larger one if you expect
            the model to need more retry cycles.
        **llm_kwargs: Forwarded to :func:`litellm_chat_completion`.

    Returns:
        A :class:`ProposalResult` with the validated source, its ``Parse`` class,
        and the full attempt history.

    Raises:
        SchemaError: if every attempt fails validation. The exception's
            message references the last attempt's error.
        ValueError: if ``max_attempts`` is not positive.
    """
    if max_attempts < 1:
        raise ValueError(
            f"max_attempts must be >= 1, got {max_attempts}. "
            f"Override via settings.SCHEMA_MAX_ATTEMPTS (current default: "
            f"{settings.SCHEMA_MAX_ATTEMPTS})."
        )

    system_prompt = load_prompt("structuring/schema", version=prompt_version, relevance=render_relevance(relevance_snippets))
    user_prompt = f"Input:\n<task>\n{task}\n</task>\nOutput:"

    attempts: list[ProposalAttempt] = []
    last_error: str = ""
    for attempt_idx in range(max_attempts):
        _log.info("schema attempt %d/%d", attempt_idx + 1, max_attempts)
        raw_response = cast(
            str,
            litellm_chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=model,
                run=run,
                kind="llm_call" if attempt_idx == 0 else "retry",
                **llm_kwargs,
            ),
        )
        schema_code = _extract_schema(raw_response)
        thought = _extract_thought(raw_response)
        try:
            parse_cls = check_schema(schema_code)
        except SchemaError as e:
            last_error = str(e)
            _log.info("schema attempt %d rejected — %s", attempt_idx + 1, last_error.splitlines()[0][:120])
            attempts.append(
                ProposalAttempt(schema_code=schema_code, error=last_error, thought=thought)
            )
            user_prompt = _retry_prompt(task, schema_code, last_error)
            continue
        _log.info("schema accepted on attempt %d (fields: %s)",
                  attempt_idx + 1, ", ".join(parse_cls.model_fields))
        attempts.append(ProposalAttempt(schema_code=schema_code, error=None, thought=thought))
        return ProposalResult(
            schema_code=schema_code,
            parse_cls=parse_cls,
            attempts=attempts,
        )

    raise SchemaError(
        f"propose_schema exhausted {max_attempts} attempts. "
        f"Last validator error: {last_error}"
    )
