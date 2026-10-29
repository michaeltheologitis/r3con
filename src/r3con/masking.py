"""Keys masked out of the error text r3con writes and logs.

A provider may echo the key it was sent in its error, and litellm passes the text on: for
``openai/``, ``hosted_vllm/`` and every OpenAI-compatible provider, ``str(error)`` holds
it as the provider sent it, and every traceback holds it through the chained SDK error.
So each error r3con writes (``calls.json``'s ``error``, ``error.txt``, the ``error`` of
``run_failed`` and ``ask_failed``, ``splits.json``, reasoning's ``fallback``) goes
through :func:`masked` or :func:`one_line` first, as do the run's ``params`` in
``manifest.json`` and every record an r3con logger logs
(:func:`r3con.logging_setup.get_logger`). Prompts, replies and documents are data, and
are written as they are: a document may well say ``api_key=…``.

:func:`masked` replaces each known key with ``REDACTED``, the longest first, and nothing
else, so it never garbles a path or a name that merely looks like a key. The known keys
are:

- every key registered with :func:`register_key`, which r3con does for each credential
  it sends (:func:`r3con.runtime.llm.register_credentials`): a call's ``api_key``, the
  values of ``extra_headers`` and of any parameter named like a credential, and the
  keys of a ``litellm.Router`` whose ``.completion`` is the run's ``completion``. A
  ``completion`` wrapper that adds a key of its own must register it;
- the value of every environment variable whose name says ``KEY``, ``TOKEN``,
  ``SECRET``, ``PASSWORD`` or ``CREDENTIAL``, where litellm finds a provider's key,
  when it is at least :data:`_MIN_ENVIRONMENT_SECRET_CHARS` characters with no
  whitespace, so that ``SORT_KEY=timestamp`` is not one.

A registered key shorter than :data:`_MIN_MASKED_KEY_CHARS` is a placeholder, such as
vLLM's ``EMPTY``, and is left alone.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

_MASK = "REDACTED"
# A shorter key is a placeholder, and masking one as short as "x" garbles "maximum".
_MIN_MASKED_KEY_CHARS = 8
_MIN_ENVIRONMENT_SECRET_CHARS = 16
_ERROR_CHARS = 300
_SECRET_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
_registered: set[str] = set()
_lock = threading.Lock()


def register_key(key: str | None) -> None:
    """Have :func:`masked` mask ``key`` from now on, for as long as the process runs,
    unless it is shorter than :data:`_MIN_MASKED_KEY_CHARS`."""
    if key and len(key) >= _MIN_MASKED_KEY_CHARS:
        with _lock:
            _registered.add(key)


def masked(text: str) -> str:
    """``text`` with each known key replaced by ``REDACTED``."""
    known = set(_environment_secrets())
    with _lock:
        known |= _registered
    for key in sorted(known, key=len, reverse=True):
        text = text.replace(key, _MASK)
    return text


def one_line(error: BaseException) -> str:
    """``error`` on one line, masked and then cut to :data:`_ERROR_CHARS` characters:
    its class, then, for a pydantic ``ValidationError``, each error's ``loc: msg``
    joined by ``; ``, and for any other, its message, whitespace collapsed; its class
    alone when that is empty."""
    if isinstance(error, ValidationError):
        details = error.errors(include_input=False, include_url=False)
        text = "; ".join(_located(detail) for detail in details)
    else:
        text = str(error)
    text = " ".join(masked(text).split())
    line = f"{type(error).__name__}: {text}" if text else type(error).__name__
    return line[:_ERROR_CHARS]


def _located(detail: Mapping[str, Any]) -> str:
    where = ".".join(map(str, detail["loc"]))
    return f"{where}: {detail['msg']}" if where else detail["msg"]


def _environment_secrets() -> list[str]:
    return [
        value
        for name, value in os.environ.items()
        if any(part in name.upper() for part in _SECRET_NAME_PARTS)
        and len(value) >= _MIN_ENVIRONMENT_SECRET_CHARS
        and not any(map(str.isspace, value))
    ]
