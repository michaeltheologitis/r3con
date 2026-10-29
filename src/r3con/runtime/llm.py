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
:func:`litellm_chat_completion` returns the text, or a validated Pydantic instance when
a ``schema`` is given, re-rolling the same request if the model answers with empty
content (a 200 with no JSON, which transport retries never see).
:func:`send_until_valid` sends a structured request again, with the validator's error,
when its reply does not fit the schema.

Structured output goes out in OpenAI strict mode, which demands more of a JSON schema than
Pydantic emits; :func:`_enforce_strict_objects` closes that gap, and
:func:`_drop_refused_constraints` removes what it refuses: enums, the largest first, until
those left hold no more than 1,000 values together, and a pattern with a lookaround.
:func:`schema_as_sent` builds the schema as sent, for the request and for the token
budgets.

Two helpers ask litellm, not the model: :func:`count_tokens`, offline, and
:func:`quiet_litellm`, which silences a model lookup. For some local or Hugging Face
models, litellm's ``get_model_info`` lookup asks a server.

**Temporary errors are resent; refusals are not.** When r3con calls ``litellm.completion``
itself, tenacity resends a request whose error has the status 408, 409, 429, or 500 or
above (which a timeout or a refused connection also carries), unless it says the quota is
spent (OpenAI's 429 ``insufficient_quota``). Any other error is raised at once: a bad
request, a refused key, an exhausted quota, or a request too long for the model, which
:mod:`r3con.splitting` catches to read the document in parts. Before each resend a warning
is logged and r3con waits for the error's Retry-After, ``0`` included, plus up to a second
of jitter. The Retry-After is its ``retry-after-ms`` header in milliseconds, else its
``retry-after`` header in seconds or as an HTTP date; when it is missing, unreadable or
negative, r3con waits instead for a jittered exponential backoff of at most
``settings.LLM_RETRY_MAX_WAIT_SECONDS``. No wait is longer than what is left of
``settings.LLM_RETRY_MAX_TOTAL_SECONDS`` since the first attempt. The last error is raised
after ``settings.LLM_TEMPORARY_ERROR_RETRIES`` resends, once that total time has passed,
or at once when the Retry-After asks for longer than the time left. So no resend starts
after the total cap, and a request that itself runs past it is not resent: one that hits
litellm's default timeout of 600 seconds, say, since r3con sets none. Each request goes
out with ``num_retries=0``, because litellm's own retries resend every error, refusals
included; that also turns off the OpenAI SDK's, though a ``litellm.num_retries`` set by
the host application still applies inside each attempt. A request that carries the
caller's own ``num_retries``, ``max_retries`` or ``retry_policy`` goes to
``litellm.completion`` as it is, to be retried by those. A supplied ``completion=`` is
called as it is, once per request, and owns its retrying: a ``litellm.Router`` resends by
the same status rule, with its own cooldowns and fallbacks. A recorded call's ``sends``
and ``resend_wait_seconds`` come from tenacity, and are ``None`` for a call r3con did not
resend itself. A call that raises an error, once r3con stops resending it, is recorded
too, with its error on one line, keys masked (:func:`r3con.masking.one_line`), and then
the error is raised as it was. Every credential a request carries is registered for
masking before it is sent (:func:`register_credentials`), so the warning logged before a
resend, and any error r3con writes later, mask it.

**A call is never sent once ``cancel`` is set.** Every model call r3con makes, from
every stage and every document's worker thread, goes through
:func:`litellm_chat_completion_full`, which checks its ``cancel``, a
``threading.Event``, right before it sends, and raises :class:`Cancelled` instead once
it is set. A resend waits on that event (:func:`_sleep`), so the wait ends the moment
it is set, and raises :class:`Cancelled` instead of resending. A call already sent is
not interrupted, since litellm cannot stop one: it finishes and is recorded. Resends
r3con does not make itself run their course: a supplied ``completion=``'s, such as a
``litellm.Router``'s, and litellm's own for a request that carries the caller's
``num_retries``, ``max_retries`` or ``retry_policy``.
"""

from __future__ import annotations

import contextlib
import logging
import random
import re
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Any

import litellm
import tenacity
from litellm.router import Router
from pydantic import BaseModel, ValidationError

from r3con import settings
from r3con.logging_setup import get_logger
from r3con.masking import is_credential_name, one_line, register_key

if TYPE_CHECKING:
    from r3con.runs import StageRun

_log = get_logger("llm")
_quiet_lock = threading.RLock()

_CALLER_RETRY_SETTINGS = frozenset({"num_retries", "max_retries", "retry_policy"})
_QUOTA_EXHAUSTED = "insufficient_quota"
_TEMPORARY_CLIENT_ERRORS = frozenset({408, 409, 429})
_FIRST_SERVER_ERROR = 500
_MILLISECONDS_PER_SECOND = 1000
_RETRY_AFTER_JITTER_SECONDS = 1.0
_WAIT_DIGITS = 2
# OpenAI Structured Outputs: "up to 1000 enum values across all enum properties".
_MAX_ENUM_VALUES = 1000
# OpenAI strict mode refuses a pattern with a lookaround: (?=, (?!, (?<= or (?<!.
_LOOKAROUND = re.compile(r"\(\?<?[=!]")
_CANCELLED = "cancel was set, so no further model call is sent"


class Cancelled(Exception):
    """Raised in place of a model call once the ``cancel`` event it was given is set."""


def raise_if_cancelled(cancel: threading.Event | None) -> None:
    """Raise :class:`Cancelled` if ``cancel`` is set."""
    if cancel is not None and cancel.is_set():
        raise Cancelled(_CANCELLED)


def _sleep(seconds: float, cancel: threading.Event | None) -> None:
    """Wait ``seconds`` before a resend, or with ``cancel``, until it is set if that is
    sooner."""
    if cancel is None:
        time.sleep(seconds)
    else:
        cancel.wait(seconds)


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
    for node in _dicts_in(schema):
        if node.get("type") == "object" and "properties" in node:
            node.setdefault("additionalProperties", False)
            node["required"] = list(node["properties"].keys())
    return schema


def _drop_refused_constraints(schema: Any) -> Any:
    """Remove from ``schema`` what OpenAI refuses: ``enum`` lists, the largest first,
    until those left hold no more than :data:`_MAX_ENUM_VALUES` values together, and
    every ``pattern`` with a lookaround, such as the one Pydantic writes for a
    ``Decimal``. A field keeps its type, so the caller checks the values once the reply
    is in."""
    nodes = list(_dicts_in(schema))
    enums = sorted(
        (node for node in nodes if isinstance(node.get("enum"), list)),
        key=lambda node: len(node["enum"]),
    )
    values = sum(len(node["enum"]) for node in enums)
    while values > _MAX_ENUM_VALUES:
        values -= len(enums.pop().pop("enum"))
    for node in nodes:
        pattern = node.get("pattern")
        if isinstance(pattern, str) and _LOOKAROUND.search(pattern):
            del node["pattern"]
    return schema


def _dicts_in(schema: Any) -> Iterator[dict[str, Any]]:
    """``schema``, if it is a dict, and every dict nested in it."""
    if isinstance(schema, dict):
        yield schema
        schema = list(schema.values())
    if isinstance(schema, list):
        for item in schema:
            yield from _dicts_in(item)


def schema_as_sent(schema: type[BaseModel]) -> dict[str, Any]:
    """``schema``'s JSON schema as a structured request sends it, built afresh on each
    call: strict, without what OpenAI refuses."""
    return _drop_refused_constraints(
        _enforce_strict_objects(schema.model_json_schema())
    )


def litellm_chat_completion_full(
    *,
    system_prompt: str | None = None,
    user_prompt: str | None = None,
    messages: list[dict[str, str]] | None = None,
    model: str,
    api_base: str | None = None,
    api_key: str | None = None,
    schema: type[BaseModel] | None = None,
    completion: Callable[..., Any] | None = None,
    run: StageRun | None = None,
    kind: str = "llm_call",
    cancel: threading.Event | None = None,
    **kwargs: Any,
) -> Any:
    """Call an LLM via LiteLLM and return the full response object.

    ``model`` uses LiteLLM's provider-prefixed form, e.g.
    ``"hosted_vllm/Qwen/Qwen3-8B"`` against a vLLM endpoint,
    or ``"anthropic/claude-sonnet-5-5"`` against Anthropic.

    Two mutually-exclusive ways to supply the conversation:

    - ``system_prompt`` + ``user_prompt``: the convenience path for a single
      system + single user turn (the vast majority of calls).
    - ``messages``: a pre-built ``[{"role": ..., "content": ...}, ...]`` list
      for callers that need richer turn structure (e.g. the reasoning agent's
      iterative loop in :mod:`r3con.runtime.codeact`).

    If ``schema`` is provided, it must be a Pydantic ``BaseModel`` class.

    If ``run`` is provided, a :class:`r3con.runs.StepRecord` is appended
    to it with the request messages, response content, tokens, cost in USD, and
    ``kind`` label. Pass ``kind="retry"`` (or another custom label) on retry attempts
    so the transcript renders them as a labeled block. A call that raises an error is
    recorded with it, no reply and a cost of 0, as r3con got no reply to price, and the
    error is raised unchanged.

    If ``cancel`` is set, :class:`Cancelled` is raised before the request is sent, and
    nothing is recorded; if it is set while r3con waits to resend, the wait ends at
    once and :class:`Cancelled` is raised and recorded as the call's error.
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
    if schema is not None:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": schema_as_sent(schema),
                "strict": True,
            },
        }

    resends: dict[str, Any] = {"sends": None, "resend_wait_seconds": None}
    register_credentials(request, completion)

    def record(**outcome: Any) -> None:
        if run is not None:
            run.add_step(
                kind=kind,
                messages=request_messages,
                schema=request.get("response_format"),
                **resends,
                **outcome,
            )

    raise_if_cancelled(cancel)
    try:
        if completion is None and _CALLER_RETRY_SETTINGS.isdisjoint(request):
            response = _resend_temporary_errors(request, resends, cancel)
        else:
            response = (completion or litellm.completion)(**request)
    except Exception as error:
        record(response={}, usd=0.0, error=one_line(error))
        raise

    record(
        response=_extract_response_dict(response),
        tokens=_extract_tokens(response),
        usd=_extract_usd(response),
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
    completion: Callable[..., Any] | None = None,
    run: StageRun | None = None,
    kind: str = "llm_call",
    max_empty_retries: int | None = None,
    cancel: threading.Event | None = None,
    **kwargs: Any,
) -> str | BaseModel:
    """Call an LLM via LiteLLM and optionally parse structured output.

    See :func:`litellm_chat_completion_full` for the two prompt-input paths
    (``system_prompt``/``user_prompt`` vs ``messages``), the ``run`` /
    ``kind`` recording behavior and ``cancel``.

    If ``schema`` is provided, it must be a Pydantic ``BaseModel`` class.

    **Empty re-roll.** When ``schema`` is set and the model returns *empty* content
    (nothing, or only whitespace), the call is **re-rolled** up to
    ``max_empty_retries`` times (default ``settings.LLM_EMPTY_CONTENT_RETRIES``)
    before raising — an empty body is a recoverable bad generation, not a hard error,
    and it's a successful HTTP 200 so the resends of temporary errors never catch it.
    Each re-roll re-sends the same request, so it is an independent sample (a ``seed``
    the caller passes in ``kwargs`` is re-sent unchanged). This recovers the common
    transient case where a reasoning model spent its budget on the thinking block; a
    *deterministic* empty still raises after the re-rolls. (A non-empty-but-malformed
    JSON raises a Pydantic ``ValidationError`` instead — that's the *caller's* retry to
    handle, e.g. :func:`send_until_valid`.)
    """
    if max_empty_retries is None:
        max_empty_retries = settings.LLM_EMPTY_CONTENT_RETRIES

    attempt = 0
    while True:
        response = litellm_chat_completion_full(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            messages=messages,
            model=model,
            api_base=api_base,
            api_key=api_key,
            schema=schema,
            completion=completion,
            run=run,
            kind=kind,
            cancel=cancel,
            **kwargs,
        )

        text = response.choices[0].message.content or ""

        if schema is None:
            return text
        if text.strip():
            return schema.model_validate_json(text)

        attempt += 1
        if attempt > max_empty_retries:
            raise ValueError(
                f"The model returned empty text after {attempt} attempt(s). For a "
                "reasoning model this usually means thinking consumed the response — "
                "try a no-thinking config, a higher max_tokens, or raising "
                "settings.LLM_EMPTY_CONTENT_RETRIES."
            )
        _log.warning("empty reply, re-rolling (%d/%d)", attempt, max_empty_retries)


def _retry_prompt(user_prompt: str, error: str, rewrite: str) -> str:
    """The user message that resends ``user_prompt`` after a reply failed its schema:
    the message again, the validator's error, and ``rewrite``, which asks for the reply
    again."""
    return (
        f"{user_prompt}\n\n"
        "---\n\n"
        "Your previous response failed schema validation:\n\n"
        f"{error}\n\n"
        f"{rewrite}, fixing only the offending field(s) so the output validates "
        "against the schema. Output the corrected JSON."
    )


def send_until_valid(
    system_prompt: str,
    user_prompt: str,
    *,
    schema: type[BaseModel] | None,
    rewrite: str,
    kind: str,
    max_attempts: int,
    **llm_call: Any,
) -> str | BaseModel:
    """The reply to ``user_prompt`` beside ``system_prompt``, sending it again with the
    validator's error and ``rewrite`` when a reply does not fit ``schema`` (invalid
    JSON, a missing field), up to ``max_attempts`` sends; the last send's error is
    raised. Each resend is logged and recorded as ``<kind>-retry-<n>``."""

    def send(text: str, attempt: int) -> str | BaseModel:
        return litellm_chat_completion(
            system_prompt=system_prompt,
            user_prompt=text,
            schema=schema,
            kind=kind if attempt == 1 else f"{kind}-retry-{attempt - 1}",
            **llm_call,
        )

    text = user_prompt
    for attempt in range(1, max_attempts):
        try:
            return send(text, attempt)
        except ValidationError as e:
            error = f"{type(e).__name__}: {e}"
            _log.warning(
                "%s: sending again (%d/%d) after %s",
                kind,
                attempt,
                max_attempts - 1,
                error,
            )
            text = _retry_prompt(user_prompt, error, rewrite)
    return send(text, max_attempts)


def register_credentials(
    params: Mapping[str, Any], completion: Callable[..., Any] | None = None
) -> None:
    """Register for masking (:func:`r3con.masking.register_key`) every credential a
    request with ``params`` sends: each value of its ``extra_headers`` and of each
    parameter with a credential's name (:func:`r3con.masking.is_credential_name`),
    ``api_key`` among them, and the token alone of a ``<scheme> <token>`` one
    (``Bearer``, ``Basic``); and when ``completion`` is a ``litellm.Router``'s, the same
    of each of its deployments' ``litellm_params``, where the Router has already read
    an ``os.environ/NAME`` from the environment."""
    headers = params.get("extra_headers") or {}
    named = [value for name, value in params.items() if is_credential_name(name)]
    sent = [value for value in [*headers.values(), *named] if isinstance(value, str)]
    # A gateway may echo a "<scheme> <token>" value's token without its scheme word.
    schemed = [words for words in map(str.split, sent) if len(words) == 2]
    for value in [*sent, *(token for scheme, token in schemed if scheme.isalpha())]:
        register_key(value)
    router = getattr(completion, "__self__", None)
    if isinstance(router, Router):
        for deployment in router.model_list:
            register_credentials(deployment.get("litellm_params") or {})


def count_tokens(text: str) -> int:
    """The cl100k_base tokens in ``text``, from the vocabulary litellm ships: the same
    count whatever the model, and offline."""
    return len(litellm.encode(text=text))


@contextlib.contextmanager
def quiet_litellm() -> Iterator[None]:
    """Keep litellm from printing during the block, then restore its settings.

    litellm prints its provider list to stdout when asked about a model string it cannot
    place (a Router alias). Its settings are globals, so the block silences another
    thread's litellm calls too, and a block in another thread waits for this one to end.
    Keep the block to the lookup itself: a block that waits on another thread which also
    enters ``quiet_litellm`` deadlocks.
    """
    logger = logging.getLogger("LiteLLM")
    with _quiet_lock:
        suppress, level = litellm.suppress_debug_info, logger.level
        litellm.suppress_debug_info = True
        logger.setLevel(logging.CRITICAL)
        try:
            yield
        finally:
            litellm.suppress_debug_info = suppress
            logger.setLevel(level)


def _resend_temporary_errors(
    request: dict[str, Any],
    resends: dict[str, Any],
    cancel: threading.Event | None,
) -> Any:
    """The response to ``request``. Before each send, ``resends`` gets the times the
    request has been sent, this send included, and the seconds, to hundredths, waited
    between those sends, so it holds both whether the last send answers or raises.
    Each wait before a resend ends once ``cancel`` is set, and raises
    :class:`Cancelled` instead of resending."""

    def count(state: tenacity.RetryCallState) -> None:
        resends["sends"] = state.attempt_number
        resends["resend_wait_seconds"] = round(state.idle_for, _WAIT_DIGITS)

    def wait(seconds: float) -> None:
        _sleep(seconds, cancel)
        raise_if_cancelled(cancel)

    resending = tenacity.Retrying(
        sleep=wait,
        retry=tenacity.retry_if_exception(is_temporary),
        stop=_give_up,
        wait=_wait_before_resending,
        before=count,
        before_sleep=tenacity.before_sleep_log(_log, logging.WARNING),
        reraise=True,
    )
    return resending(litellm.completion, **request, num_retries=0)


def is_temporary(error: BaseException) -> bool:
    """Whether r3con resends ``error``: a 5xx, 408, 409 or 429, unless quota ran out."""
    status = getattr(error, "status_code", None)
    if not isinstance(status, int):
        return False
    if status < _FIRST_SERVER_ERROR and status not in _TEMPORARY_CLIENT_ERRORS:
        return False
    # litellm keeps OpenAI's code in body on the SDK path, in the message on its own.
    return _QUOTA_EXHAUSTED not in f"{error} {getattr(error, 'body', None)}"


def _give_up(retry_state: tenacity.RetryCallState) -> bool:
    left = _seconds_left(retry_state)
    retry_after = _retry_after(retry_state)
    return (
        retry_state.attempt_number > settings.LLM_TEMPORARY_ERROR_RETRIES
        or left <= 0
        or (retry_after is not None and retry_after > left)
    )


def _wait_before_resending(retry_state: tenacity.RetryCallState) -> float:
    retry_after = _retry_after(retry_state)
    if retry_after is None:
        backoff = tenacity.wait_random_exponential(
            max=settings.LLM_RETRY_MAX_WAIT_SECONDS
        )
        wait = backoff(retry_state)
    else:
        wait = retry_after + random.uniform(0, _RETRY_AFTER_JITTER_SECONDS)
    return min(wait, _seconds_left(retry_state))


def _seconds_left(retry_state: tenacity.RetryCallState) -> float:
    elapsed = retry_state.seconds_since_start or 0.0
    return settings.LLM_RETRY_MAX_TOTAL_SECONDS - elapsed


def _retry_after(retry_state: tenacity.RetryCallState) -> float | None:
    error = retry_state.outcome.exception() if retry_state.outcome else None
    headers = getattr(error, "litellm_response_headers", None) or {}
    seconds = _seconds_asked({name.lower(): value for name, value in headers.items()})
    return seconds if seconds is not None and seconds >= 0 else None


def _seconds_asked(headers: dict[str, Any]) -> float | None:
    with contextlib.suppress(KeyError, ValueError):
        return float(headers["retry-after-ms"]) / _MILLISECONDS_PER_SECOND
    with contextlib.suppress(KeyError, ValueError):
        return float(headers["retry-after"])
    with contextlib.suppress(KeyError, ValueError):
        return parsedate_to_datetime(headers["retry-after"]).timestamp() - time.time()
    return None


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


def _extract_usd(response: Any) -> float | None:
    """The call's price in USD as litellm reckons it, cached input and long-request rates
    included, or ``None`` when litellm cannot price it, as for a self-hosted model."""
    return (getattr(response, "_hidden_params", None) or {}).get("response_cost")


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
