"""Keys masked out of the error text r3con writes and logs.

A provider may echo the key it was sent in its error, and litellm passes the text on: for
``openai/``, ``hosted_vllm/`` and every OpenAI-compatible provider, ``str(error)`` holds
it as the provider sent it, and every traceback holds it through the chained SDK error.
So each error r3con writes (``calls.json``'s ``error``, ``error.txt``, the ``error`` of
``run_failed`` and ``ask_failed``, ``splits.json``, reasoning's ``fallback``) goes
through :func:`masked` or :func:`one_line` first, and so does every record an r3con
logger logs (:func:`r3con.logging_setup.get_logger`). Prompts, replies and documents are
data, and are written as they are: a document may well say ``api_key=…``.

:func:`masked` replaces each known key with ``REDACTED``, the longest first, then
whatever litellm's own redactor takes for a credential: a ``Bearer`` token, an ``sk-``
key, an ``api_key=`` value and so on. The known keys are those given, the ``api_key`` of
every model call this process has made (:func:`register_key`), and the value of every
environment variable whose name ends in ``_KEY``, ``_TOKEN`` or ``_SECRET``, where
litellm finds a provider's key. A key shorter than :data:`_MIN_MASKED_KEY_CHARS` is a
placeholder, such as vLLM's ``EMPTY``, and is left alone.
"""

from __future__ import annotations

import os
import threading

# litellm's internal module, not its API: if it moves, known keys are still masked.
try:
    from litellm.litellm_core_utils.secret_redaction import redact_string
except ImportError:

    def redact_string(value: str) -> str:
        return value


_MASK = "REDACTED"
# A shorter key is a placeholder, and masking one as short as "x" garbles "maximum".
_MIN_MASKED_KEY_CHARS = 8
_ERROR_CHARS = 300
_SECRET_NAME_ENDINGS = ("_KEY", "_TOKEN", "_SECRET")
_registered: set[str] = set()
_lock = threading.Lock()


def register_key(key: str | None) -> None:
    """Have :func:`masked` mask ``key`` from now on, for as long as the process runs."""
    if key and len(key) >= _MIN_MASKED_KEY_CHARS:
        with _lock:
            _registered.add(key)


def masked(text: str, *keys: str | None) -> str:
    """``text`` with each known key, ``keys`` among them, and then whatever litellm's
    redactor takes for a credential, replaced by ``REDACTED``."""
    known = {*keys, *_environment_secrets()}
    with _lock:
        known.update(_registered)
    long_enough = (key for key in known if key and len(key) >= _MIN_MASKED_KEY_CHARS)
    for key in sorted(long_enough, key=len, reverse=True):
        text = text.replace(key, _MASK)
    return redact_string(text)


def one_line(error: BaseException, *keys: str | None) -> str:
    """``error``'s class and the first non-empty line of its message, :func:`masked`
    and then cut to :data:`_ERROR_CHARS` characters; its class alone when the message is
    empty."""
    line = masked(str(error), *keys).strip().partition("\n")[0]
    text = f"{type(error).__name__}: {line}" if line else type(error).__name__
    return text[:_ERROR_CHARS]


def _environment_secrets() -> list[str]:
    return [
        value
        for name, value in os.environ.items()
        if name.upper().endswith(_SECRET_NAME_ENDINGS)
    ]
