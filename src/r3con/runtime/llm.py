"""The LiteLLM call every stage makes, plus structured-output enforcement.

**The connection is litellm's, not ours.** ``model`` is a litellm model string, credentials
come from the provider's own environment variable, and ``**kwargs`` reaches
``litellm.completion`` verbatim. There is deliberately no model registry, no provider
abstraction and no r3con-branded credential variable here — litellm already is that
abstraction, and re-implementing it would be building a framework inside a package that
exists not to be one.

The one escape hatch is ``completion=``, the callable that makes each request (default
``litellm.completion``): hand it a configured ``litellm.Router``'s ``.completion``, or any
wrapper with the same shape. It is **transport**, so it never enters a run's identity.

Two entry points. :func:`litellm_chat_completion_full` returns the raw response object;
:func:`litellm_chat_completion` — what the stages actually call — returns the text, or a
validated Pydantic instance when a ``schema`` is given, re-rolling with a perturbed seed if
the model answers with empty content (a 200 with no JSON, which transport retries never see).

Structured output goes out in OpenAI strict mode, which demands more of a JSON schema than
Pydantic emits; :func:`_enforce_strict_objects` closes that gap.

``num_retries`` is set here, and that is why **``tenacity`` is a declared dependency even
though nothing in this package imports it** — litellm imports it lazily, on the retry path
only. Drop it and every call that hits a transient error fails instead of retrying, on the
one path you are least likely to exercise before shipping.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import litellm
from pydantic import BaseModel

from r3con.logging_setup import get_logger
from r3con.settings import settings

if TYPE_CHECKING:
    from r3con.runs import StageRun

_log = get_logger("llm")


def _enforce_strict_objects(schema: Any) -> Any:
    """Make ``schema`` compliant with OpenAI strict mode.

    OpenAI's strict structured-output mode requires every object node to:
    1. Set ``additionalProperties: false`` (no extra keys allowed); and
    2. Mark *every* property key as required, even ones that are logically
       optional. Pydantic emits ``Optional[X]`` fields with no ``required``
       entry — strict mode rejects that with ``"required" is required to be
       supplied and to be an array including every key in properties``.

    For (2), we overwrite ``required`` to the full set of property keys. The
    field type itself can still allow ``null`` (via Pydantic's
    ``anyOf: [..., {"type": "null"}]``), so optionality is preserved at the
    value level; only the *key* is forced to appear.
    """
    if isinstance(schema, dict):
        if schema.get("type") == "object" and "properties" in schema:
            schema.setdefault("additionalProperties", False)
            schema["required"] = list(schema["properties"].keys())
        for v in schema.values():
            _enforce_strict_objects(v)
    elif isinstance(schema, list):
        for item in schema:
            _enforce_strict_objects(item)
    return schema


def litellm_chat_completion_full(
    *,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    schema: type[BaseModel] | None = None,
    seed: int | None = None,
    completion: Callable[..., Any] | None = None,
    run: "StageRun | None" = None,
    kind: str = "llm_call",
    **kwargs: Any,
) -> Any:
    """Call an LLM via LiteLLM and return the full response object.

    ``model`` uses LiteLLM's provider-prefixed form, e.g.
    ``"hosted_vllm/Qwen/Qwen3-8B"`` against a vLLM endpoint,
    or ``"anthropic/claude-sonnet-4"`` against Anthropic.

    Two mutually-exclusive ways to supply the conversation:

    - ``system_prompt`` + ``user_prompt``: the convenience path for a single
      system + single user turn (the vast majority of calls).
    - ``messages``: a pre-built ``[{"role": ..., "content": ...}, ...]`` list
      for callers that need richer turn structure (e.g. the reasoning agent's
      iterative loop in :mod:`r3con.runtime.codeact`).

    If ``schema`` is provided, it must be a Pydantic ``BaseModel`` class.

    If ``run`` is provided, a :class:`r3con.runs.StepRecord` is appended
    to it with the request messages, response content, tokens, and ``kind``
    label. Pass ``kind="retry"`` (or another custom label) on retry attempts
    so the transcript renders them as a labeled block.

    This is where ``num_retries`` is set, which pulls in a dependency that is
    invisible to every import in this package: litellm imports ``tenacity``
    *lazily*, on the retry path only. ``tenacity`` is therefore **required** —
    without it the first transient error raises a "tenacity import failed"
    error instead of being retried (empirically confirmed).
    """

    if messages is not None:
        if system_prompt is not None or user_prompt is not None:
            raise ValueError(
                "pass either `messages=` or `system_prompt=`/`user_prompt=`, not both"
            )
        request_messages = messages
    else:
        if system_prompt is None or user_prompt is None:
            raise ValueError(
                "must pass either `messages=` or both `system_prompt=` and `user_prompt=`"
            )
        request_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    request: dict[str, Any] = {
        "model": model,
        "messages": request_messages,
        **kwargs,
    }
    if api_base is not None:
        request["api_base"] = api_base
    if api_key is not None:
        request["api_key"] = api_key
    if seed is not None:
        request["seed"] = seed
    if schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": _enforce_strict_objects(schema.model_json_schema()),
                "strict": True,
            },
        }

    # Transport-level retries (exponential backoff via tenacity) on transient
    # vLLM/API errors — connection refused/reset, timeout, 5xx — so one blip
    # doesn't abort the whole task. setdefault so a caller can still override.
    request.setdefault("num_retries", settings.LLM_NUM_RETRIES)

    response = (completion or litellm.completion)(**request)

    if run is not None:
        run.add_step(
            kind=kind,
            messages=request_messages,
            response=_extract_response_dict(response),
            schema=request.get("response_format"),
            tokens=_extract_tokens(response),
        )

    return response


def litellm_chat_completion(
    *,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    schema: type[BaseModel] | None = None,
    seed: int | None = None,
    completion: Callable[..., Any] | None = None,
    run: "StageRun | None" = None,
    kind: str = "llm_call",
    max_empty_retries: int | None = None,
    **kwargs: Any,
) -> str | BaseModel:
    """Call an LLM via LiteLLM and optionally parse structured output.

    See :func:`litellm_chat_completion_full` for the two prompt-input paths
    (``system_prompt``/``user_prompt`` vs ``messages``) and the ``run`` /
    ``kind`` recording behavior.

    If ``schema`` is provided, it must be a Pydantic ``BaseModel`` class.

    **Empty structured-output re-roll.** When ``schema`` is set and the model returns
    *empty* content, the call is **re-rolled** up to ``max_empty_retries`` times
    (default ``settings.LLM_EMPTY_CONTENT_RETRIES``) before raising — an empty body is
    a recoverable bad generation, not a hard error, and it's a successful HTTP 200 so
    the transport ``num_retries`` never catches it. Each re-roll **perturbs the seed**
    (``seed + attempt``) so a pinned-seed call produces a genuinely different roll
    (deterministic → reproducible). This recovers the common transient case where a
    reasoning model spent its budget on the thinking block; a *deterministic* empty
    still raises after the re-rolls. (A non-empty-but-malformed JSON raises a Pydantic
    ``ValidationError`` instead — that's the *caller's* retry to handle, e.g. the
    parsing stage's per-document retry loop.)
    """
    if max_empty_retries is None:
        max_empty_retries = settings.LLM_EMPTY_CONTENT_RETRIES

    attempt = 0
    while True:
        # Perturb the seed on a re-roll so a pinned seed doesn't reproduce the same
        # empty output (deterministic offset → runs stay reproducible).
        call_seed = seed + attempt if (seed is not None and attempt > 0) else seed
        response = litellm_chat_completion_full(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=messages,
            model=model,
            api_base=api_base,
            api_key=api_key,
            schema=schema,
            seed=call_seed,
            completion=completion,
            run=run,
            kind=kind,
            **kwargs,
        )

        text = response.choices[0].message.content or ""

        if schema is None:
            return text
        if text:
            return schema.model_validate_json(text)

        attempt += 1
        if attempt > max_empty_retries:
            raise ValueError(
                "Structured output was requested, but the model returned empty text "
                f"after {attempt} attempt(s) (seed-perturbed re-rolls). For a reasoning "
                "model this usually means thinking consumed the response — try a "
                "no-thinking config, a higher max_tokens, or raising "
                "settings.LLM_EMPTY_CONTENT_RETRIES."
            )
        _log.warning("empty structured output, re-rolling (%d/%d)", attempt, max_empty_retries)


def _extract_response_dict(response: Any) -> dict[str, Any]:
    """Pull a serializable {role, content, finish_reason} dict out of a LiteLLM response.

    LiteLLM normalizes most providers' completion responses to OpenAI-shape:
    ``response.choices[0].message`` has ``.role`` and ``.content``,
    ``response.choices[0].finish_reason``. Tolerant of missing fields.
    """
    try:
        choice = response.choices[0]
        msg = choice.message
        return {
            "role": getattr(msg, "role", "assistant"),
            "content": getattr(msg, "content", "") or "",
            "finish_reason": getattr(choice, "finish_reason", None),
        }
    except (AttributeError, IndexError):
        return {"role": "assistant", "content": "", "finish_reason": None}


def _extract_tokens(response: Any) -> dict[str, int] | None:
    """Pull token counts out of a LiteLLM response, normalized to {prompt, completion, total}."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return {
        "prompt": int(getattr(usage, "prompt_tokens", 0) or 0),
        "completion": int(getattr(usage, "completion_tokens", 0) or 0),
        "total": int(getattr(usage, "total_tokens", 0) or 0),
    }
