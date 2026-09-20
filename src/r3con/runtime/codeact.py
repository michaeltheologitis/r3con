"""Generic CodeAct agent loop — the reusable multi-turn code-execution core.

A CodeAct agent solves a task by writing Python in ``<code>...</code>`` blocks
that run in a persistent, sandboxed in-process interpreter; the runtime feeds
the captured ``print`` output back as ``<observation>...</observation>`` on the
next turn, and the agent commits by calling ``final_answer(x)`` from inside the
sandbox. This module owns that loop and nothing task-specific:

- Each turn: call the LLM with the running message history, clip any
  hallucinated tail past ``</code>``, extract the code, run it, and feed the
  observation back.
- State (variable bindings) survives across turns — the agent builds a
  candidate list in turn 1 and reads it in turn 2.
- ``final_answer`` is registered as a static tool; calling it raises a
  ``FinalAnswerException`` upstream, which surfaces here on
  ``CodeOutput.is_final_answer`` as the termination signal.
- ``max_turns`` caps the loop; on exhaustion we make one final plain-prose LLM call
  that synthesizes an answer from the work done (falling back to the last non-empty
  observation only if that synthesis is empty), rather than refusing.

Callers supply the rendered ``system_prompt`` + ``user_message``, the variables
to bind in the sandbox (e.g. ``{"parse": ...}``), and the usual knobs. The
stage-3 reasoning agent (``r3con.stages.reasoning.reason``) is the
only consumer; the loop itself knows nothing about parses, schemas, or
relevance snippets.

Execution errors (runtime, timeout, unauthorized import, missing ``<code>``
block) do **not** end the loop — they are fed back as observations so the
model can correct itself, the same way a human at a REPL would.
"""

from __future__ import annotations

import ast
import functools
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

import litellm

from r3con.logging_setup import get_logger
from r3con.runtime.llm import litellm_chat_completion
from r3con.runtime.python_executor import (
    ExecutionTimeoutError,
    InterpreterError,
    LocalPythonExecutor,
    fix_final_answer_code,
)
from r3con.runs import StageRun
from r3con.settings import settings

_log = get_logger("codeact")

DEFAULT_EXEC_TIMEOUT_S: float = 30.0


# Matches the smolagents/CodeAct convention: the model emits prose ("thought")
# and wraps each program inside a `<code>...</code>` block. Multiple blocks
# in one response are concatenated with double newlines. We fall back to ```
# markdown fences when the model ignores the tag convention.
_CODE_TAG_RE = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
_MARKDOWN_FENCE_RE = re.compile(
    r"```(?:python|py|code)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)

# Stop sequences for the LLM call. These are the runtime's own delimiters: the
# model emits up to `</code>` and then must wait for the real `<observation>`
# we feed back — it should never generate either tag's downstream content
# itself. Used both as a server-side `stop` (when the model supports it) and as
# the client-side truncation boundary in `_clip_assistant_response`.
_STOP_SEQUENCES = ["</code>", "<observation>"]

# Fed back when a response carries no runnable code. Reminds BOTH things: how
# to run code (so an explore-but-forgot-the-tags turn recovers) and how to
# commit (so a model that wrote its answer as prose — the dominant failure —
# learns that printing/describing doesn't submit; only final_answer() inside a
# <code> block does). We don't gate on keywords: in this loop a no-code
# response almost always means the model answered in prose, and the commit
# reminder is harmless when it was actually just exploring.
_NO_CODE_MESSAGE = (
    "Your response had no `<code>...</code>` block, so nothing ran. In this "
    "loop you act only by running Python inside `<code>` and `</code>` tags. "
    "For example:\n"
    "<code>\n"
    "x = 5\n"
    "print(x)\n"
    "</code>\n"
    "When you are ready to answer, call final_answer(...) from inside the code block:\n"
    "<code>\n"
    'final_answer("YOUR ANSWER HERE")\n'
    "</code>"
)

# Appended as a final user message when the loop exhausts ``max_turns`` without
# a commit. One plain LLM call over the whole conversation then synthesizes a
# best-effort answer from everything the agent printed and reasoned, instead of
# returning the last raw observation.
_MAX_TURNS_SYNTHESIS_PROMPT = (
    "You've run out of code turns. Based on everything above — what you printed "
    "and reasoned through — give your final answer now, in plain prose. Do not "
    "write code or call final_answer; just state the answer directly."
)


@functools.lru_cache(maxsize=None)
def _supports_stop_parameter(model: str) -> bool:
    """Whether ``model`` accepts the ``stop`` parameter, per litellm's param map.

    Delegating to litellm (rather than hardcoding a model list) keeps this
    current as new models ship. An unrecognized model string makes litellm log
    a provider-list warning and return an empty list; we silence that and treat
    'unknown' as unsupported — then skip ``stop`` and rely on the client-side
    truncation in :func:`_clip_assistant_response` (the correctness-bearing
    path) instead.
    """
    logger = logging.getLogger("LiteLLM")
    prev_suppress = litellm.suppress_debug_info
    prev_level = logger.level
    litellm.suppress_debug_info = True
    logger.setLevel(logging.CRITICAL)
    try:
        params = litellm.get_supported_openai_params(model=model) or []
    except Exception:
        return False
    finally:
        litellm.suppress_debug_info = prev_suppress
        logger.setLevel(prev_level)
    return "stop" in params


def _clip_assistant_response(response: str) -> str:
    """Normalize one assistant turn: cut at the first stop sequence, re-close.

    The model tends to keep generating past its own ``</code>`` — fabricating
    the ``<observation>`` the runtime would return and a ``final_answer`` after
    it (which the runtime never executes, so the model never actually commits).
    We cut the response at the first stop sequence — whether or not the provider
    honored a server-side ``stop`` — then re-append the closing ``</code>`` tag
    the cut/stop stripped, so the stored + re-fed history is well-formed and the
    ``<code>...</code>`` matcher still matches. Cutting at the first ``</code>``
    means one code block per turn (the CodeAct norm).
    """
    for stop in _STOP_SEQUENCES:
        response = response.split(stop, 1)[0]
    if "<code>" in response and not response.rstrip().endswith("</code>"):
        response += "\n</code>"
    return response


# A bare data-literal "answer" (the model emitting raw JSON/data like `{"k": "v"}` or
# `[1, 2]` instead of code) parses as valid Python but is a no-op when executed — no
# `print`, no `final_answer`. If we ran it, the runtime would feed back "(no output)" and
# the model, never told what went wrong, would repeat the same non-answer until max_turns.
# We detect it so such a response routes to the no-code reminder instead.
_BARE_LITERAL_NODES = (ast.Dict, ast.List, ast.Set, ast.Tuple, ast.Constant)


def _is_bare_data_literal(tree: ast.Module) -> bool:
    """True when ``tree`` is a single expression statement whose value is a plain data
    literal (dict / list / set / tuple / constant) — a prose-style answer, not code."""
    return (
        len(tree.body) == 1
        and isinstance(tree.body[0], ast.Expr)
        and isinstance(tree.body[0].value, _BARE_LITERAL_NODES)
    )


def _extract_code_from_response(text: str) -> str:
    """Pull the Python program out of the LLM's response.

    Convention (matches smolagents' CodeAct format):
      1. Model emits free-form prose ("Thought: ...").
      2. Final program inside one or more ``<code>...</code>`` blocks.

    Resolution order (mirrors smolagents' ``parse_code_blobs``):
      1. One or more ``<code>...</code>`` blocks (concatenated with double
         newlines so the executor runs them as one program).
      2. A markdown ``` fence.
      3. The whole response, if it is itself valid Python *code* (tag-less) — but
         NOT a bare data literal (see :func:`_is_bare_data_literal`): a model that
         "answers" with raw JSON like ``{"a": 1}`` is answering in prose, not coding,
         so it routes to the no-code reminder instead of being silently run to no output.

    Raises:
        CodeExecutionError: when none match. The message (`_NO_CODE_MESSAGE`)
            reminds the model of both how to run code and how to commit via
            `final_answer(...)` inside a `<code>` block — a no-code response is
            almost always the model answering in prose, and it needs to be told
            how to submit. Fed back into the loop so the LLM can recover.
    """
    matches = _CODE_TAG_RE.findall(text)
    if not matches:
        matches = _MARKDOWN_FENCE_RE.findall(text)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    stripped = text.strip()
    if stripped:
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            pass
        else:
            if not _is_bare_data_literal(tree):
                return stripped
    raise CodeExecutionError(message=_NO_CODE_MESSAGE, timed_out=False)


class CodeExecutionError(RuntimeError):
    """Raised when a generated program fails to run cleanly.

    The ``message`` is what the retry loop feeds back to the LLM as
    diagnostic text. ``timed_out`` distinguishes wall-clock kills from
    everything else so the retry prompt can be specific.
    """

    def __init__(self, *, message: str, timed_out: bool) -> None:
        self.message = message
        self.timed_out = timed_out
        super().__init__(message)


@dataclass
class ExecutionResult:
    stdout: str  # captured print output


@dataclass
class CodeActTurn:
    """One turn of the CodeAct loop.

    Records the model's response, what code we extracted, and what we fed
    back as the next observation (or recorded as an error). Exactly one of
    ``execution`` / ``error`` is populated for any turn that reached the
    executor; when extraction itself failed, ``code`` and ``execution`` are
    both ``None`` and the reason is recorded under ``error``.
    """

    response: str  # clipped assistant text stored in history (what the model re-sees)
    raw_response: str  # unmodified model output before _clip_assistant_response (debug)
    code: str | None  # extracted code, or None if extraction failed
    execution: ExecutionResult | None  # exec result when the code ran cleanly
    observation: str  # what was fed back as the next user message (or final stdout)
    error: str | None  # extraction or execution error text, if any
    is_final_answer: bool  # True when final_answer(x) was called this turn


@dataclass
class CodeActResult:
    """Full record of one CodeAct solve.

    ``answer`` is the value the agent committed via ``final_answer(x)``
    (stringified). When the loop exhausts ``max_turns`` without a commit,
    ``answer`` is a final synthesized best-effort (one plain-prose call over the whole
    conversation), or the last non-empty observation if that synthesis comes back empty.
    """

    answer: str
    turns: list[CodeActTurn] = field(default_factory=list)
    terminated_by: str = "final_answer"  # "final_answer" | "max_turns"


# Sentinel registered with the executor. Calling ``final_answer(x)`` from
# inside the sandbox raises ``FinalAnswerException`` (upstream behavior),
# which surfaces here as ``CodeOutput.is_final_answer=True``. The function
# itself just returns its argument unchanged — the exception is what
# carries the value out.
def _identity_final_answer(x: Any = None) -> Any:
    return x


def _format_observation(text: str) -> str:
    """Wrap captured stdout (or an error message) for the next user message."""
    body = text if text.strip() else "(no output)"
    return f"<observation>\n{body}\n</observation>"


def _stringify_final_answer(value: Any) -> str:
    """Coerce whatever was passed to ``final_answer`` into an answer string.

    ``str()`` returns a string unchanged, so this also passes strings through.
    """
    return str(value)


def run_codeact(
    *,
    system_prompt: str,
    user_message: str,
    model: str,
    variables: dict[str, Any] | None = None,
    tools: dict[str, Callable[..., Any]] | None = None,
    max_turns: int = settings.REASONING_MAX_TURNS,
    timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    additional_authorized_imports: list[str] | None = None,
    run: StageRun | None = None,
    **llm_kwargs: Any,
) -> CodeActResult:
    """Run the multi-turn CodeAct loop and return the committed answer.

    Each turn the model emits ``Thought:`` prose + a ``<code>...</code>`` block;
    we execute the code in a persistent sandbox and feed the captured ``print``
    output back as the next user message wrapped in
    ``<observation>...</observation>``. The agent commits by calling
    ``final_answer(x)`` inside its code; that returns control here.

    Variable bindings persist across turns inside the executor — the agent can
    ``rows = [...]`` in turn 1 and refer to ``rows`` in turn 2.

    Args:
        system_prompt: the rendered system message (instructions, any
            task-specific context). Stays cached across the loop's turns.
        user_message: the initial user message (e.g. the bare question).
        model: LiteLLM provider-prefixed model string.
        variables: names bound into the sandbox before the first turn (e.g.
            ``{"parse": parse_dict}``).
        tools: extra callables exposed to the sandbox as functions the agent
            can call from its code (e.g. ``{"web_search": fn}``), registered
            via the executor's ``additional_functions``. ``final_answer`` is
            always registered as the terminator and is a reserved name —
            passing it here raises ``ValueError``.
        max_turns: hard cap on the number of LLM calls. On exhaustion we make one
            final plain-prose call to synthesize a best-effort answer from the work
            done (falling back to the last non-empty observation if it comes back empty).
        timeout_s: per-execution wall-clock cap (in seconds). ``None`` disables
            the cap.
        additional_authorized_imports: extra modules the program is allowed
            to import beyond ``BASE_BUILTIN_MODULES``.
        run: optional :class:`r3con.runs.StageRun` to record each LLM call.

    Returns:
        :class:`CodeActResult` with the committed answer, the full turn
        history (response / code / observation / is_final_answer per turn),
        and the termination reason.
    """
    if max_turns < 1:
        raise ValueError(
            f"max_turns must be >= 1, got {max_turns}. "
            f"Override via settings.REASONING_MAX_TURNS (current default: "
            f"{settings.REASONING_MAX_TURNS})."
        )
    if tools and "final_answer" in tools:
        raise ValueError(
            "'final_answer' is reserved as the CodeAct terminator and cannot be "
            "passed as a tool; choose a different name."
        )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Persistent executor across turns. The agent can build up variables
    # in early turns and read them in later turns. ``final_answer`` is
    # registered as a static tool; calling it raises FinalAnswerException
    # upstream, which surfaces here as ``CodeOutput.is_final_answer=True``.
    timeout_int = int(timeout_s) if timeout_s is not None else None
    executor = LocalPythonExecutor(
        additional_authorized_imports=additional_authorized_imports or [],
        timeout_seconds=timeout_int,
        # final_answer is registered last so it can never be shadowed by a tool.
        additional_functions={**(tools or {}), "final_answer": _identity_final_answer},
    )
    if variables:
        executor.send_variables(variables)

    turns: list[CodeActTurn] = []
    last_nonempty_observation: str | None = None

    # Send `</code>`/`<observation>` as a server-side `stop` only when the
    # model accepts it (reasoning models like the gpt-5 family reject it).
    # Either way `_clip_assistant_response` truncates client-side below — that
    # is the correctness-bearing step; the server-side `stop` is just a
    # token/latency optimization. A caller-supplied `stop` (via llm_kwargs)
    # wins.
    call_kwargs = dict(llm_kwargs)
    if _supports_stop_parameter(model):
        call_kwargs.setdefault("stop", _STOP_SEQUENCES)

    for turn_idx in range(max_turns):
        _log.info("codeact turn %d/%d", turn_idx + 1, max_turns)
        raw_response = cast(
            str,
            litellm_chat_completion(
                messages=messages,
                model=model,
                run=run,
                kind=f"turn-{turn_idx + 1}",
                **call_kwargs,
            ),
        )
        # Cut any generation past the first stop sequence (the model tends to
        # hallucinate the <observation> + a final_answer after </code>, which
        # the runtime would otherwise discard while the model believes it
        # committed) and re-close the code block the stop/cut stripped.
        # ``raw_response`` keeps the unmodified output for the logs.
        response = _clip_assistant_response(raw_response)

        # Extract code from the response. On failure, feed the wrapping
        # instructions back as an observation and continue (don't end).
        try:
            code = fix_final_answer_code(_extract_code_from_response(response))
        except CodeExecutionError as e:
            _log.info("codeact turn %d: no runnable code, re-prompting", turn_idx + 1)
            observation = _format_observation(e.message)
            turns.append(
                CodeActTurn(
                    response=response,
                    raw_response=raw_response,
                    code=None,
                    execution=None,
                    observation=observation,
                    error=e.message,
                    is_final_answer=False,
                )
            )
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": observation})
            continue

        # Execute the code in the persistent sandbox. The smolagents-vendored
        # executor raises FinalAnswerException when ``final_answer`` is
        # called; upstream catches it and surfaces it as
        # ``is_final_answer=True`` on the CodeOutput, not as an exception
        # we see here.
        try:
            out = executor(code)
        except (ExecutionTimeoutError, InterpreterError) as e:
            err = (
                f"Program timed out after {timeout_s}s"
                if isinstance(e, ExecutionTimeoutError)
                else str(e)
            )
            _log.info("codeact turn %d: execution error — %s", turn_idx + 1, err.splitlines()[0][:100])
            observation = _format_observation(err)
            turns.append(
                CodeActTurn(
                    response=response,
                    raw_response=raw_response,
                    code=code,
                    execution=None,
                    observation=observation,
                    error=err,
                    is_final_answer=False,
                )
            )
            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": observation})
            continue

        execution = ExecutionResult(stdout=out.logs)

        if out.is_final_answer:
            # The model committed. ``out.output`` is the value handed to
            # final_answer (carried out via FinalAnswerException upstream).
            _log.info("codeact: committed final_answer on turn %d/%d", turn_idx + 1, max_turns)
            answer = _stringify_final_answer(out.output)
            observation = _format_observation(
                f"final_answer received: {answer}"
            )
            turns.append(
                CodeActTurn(
                    response=response,
                    raw_response=raw_response,
                    code=code,
                    execution=execution,
                    observation=observation,
                    error=None,
                    is_final_answer=True,
                )
            )
            return CodeActResult(
                answer=answer,
                turns=turns,
                terminated_by="final_answer",
            )

        observation = _format_observation(execution.stdout)
        if execution.stdout.strip():
            last_nonempty_observation = execution.stdout.strip()
        turns.append(
            CodeActTurn(
                response=response,
                raw_response=raw_response,
                code=code,
                execution=execution,
                observation=observation,
                error=None,
                is_final_answer=False,
            )
        )
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": observation})

    # Loop exhausted max_turns without a final_answer commit. The agent has
    # usually done real work across the turns; make ONE plain LLM call over the
    # whole conversation asking for its best answer in prose — no code, no stop
    # sequence — and use that. This recovers a coherent answer from the work
    # done rather than returning the last raw print. Falls back to the last
    # non-empty observation only if the synthesis comes back empty, so we never
    # regress to nothing. The call is recorded in the transcript (it is not a
    # CodeActTurn — it has no code/observation).
    _log.info("codeact: hit max_turns (%d) without a commit — synthesizing a final answer", max_turns)
    messages.append({"role": "user", "content": _MAX_TURNS_SYNTHESIS_PROMPT})
    synthesis = cast(
        str,
        litellm_chat_completion(
            messages=messages,
            model=model,
            run=run,
            kind="max-turns-fallback",
            **llm_kwargs,  # NOTE: not call_kwargs — no stop sequence; we want full prose
        ),
    ).strip()

    if synthesis:
        answer = synthesis
    elif last_nonempty_observation is not None:
        answer = last_nonempty_observation
    else:
        answer = "Cannot determine — the reasoning loop hit max_turns with no committed answer."

    return CodeActResult(
        answer=answer,
        turns=turns,
        terminated_by="max_turns",
    )
