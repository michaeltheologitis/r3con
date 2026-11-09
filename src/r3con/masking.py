"""Keys masked out of the error text r3con writes and logs.

A provider may echo the key it was sent in its error, and litellm passes the text on: for
``openai/``, ``hosted_vllm/`` and every OpenAI-compatible provider, ``str(error)`` holds
it as the provider sent it, and every traceback holds it through the chained SDK error.
So each error r3con writes (``calls.json``'s ``error``, ``error.txt``, the ``error`` of
``run_failed`` and ``ask_failed``, ``splits.json``, reasoning's ``fallback``) goes
through :func:`masked` or :func:`one_line` first, as do the run's ``params`` in
``manifest.json`` and in the run label ``r3con run`` prints (:func:`masked_values`),
and every record an r3con logger logs (:func:`r3con.logging_setup.get_logger`).
Prompts, replies and documents are data, and are written as they are: a document may
well say ``api_key=…``.

:func:`masked` replaces each known key with ``REDACTED``, the longest first, and nothing
else, so it never garbles a path or a name that merely looks like a key. The known keys
are:

- every key registered with :func:`register_key`, which r3con does, before a run's
  first request, for each credential it sends
  (:func:`r3con.runtime.llm.register_credentials`): at any depth of the request's
  parameters, each string in ``extra_headers`` or ``headers``, whatever the header's
  name, and in a parameter with a credential's name (:func:`is_credential_name`), such
  as ``api_key``, or a dict-valued ``vertex_credentials``, all of whose strings are;
  the password and each key's value in the query of a URL, such as ``api_base``; and
  the keys of a ``litellm.Router`` whose ``.completion`` is the run's ``completion``.
  A ``completion`` wrapper that adds a key of its own must register it;
- the value of every environment variable with a credential's name, where litellm finds
  a provider's key, unless it holds whitespace or starts with ``/`` or ``~``, as a path
  does (``GOOGLE_APPLICATION_CREDENTIALS``), or is shorter than
  :data:`_MIN_ENVIRONMENT_SECRET_CHARS`, or :data:`_MIN_MASKED_KEY_CHARS` for a name
  that ends in ``api_key`` or ``apikey``, a provider's key.

A key shorter than :data:`_MIN_MASKED_KEY_CHARS` is a placeholder, such as vLLM's
``EMPTY``, and is left alone.
"""

from __future__ import annotations

import os
import re
import threading
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

_MASK = "REDACTED"
# A shorter key is a placeholder, and masking one as short as "x" garbles "maximum".
_MIN_MASKED_KEY_CHARS = 8
_MIN_ENVIRONMENT_SECRET_CHARS = 16
_ERROR_CHARS = 300
_PROVIDER_KEY_NAME = re.compile("api_?key$", re.IGNORECASE)
_CREDENTIAL_NAME = re.compile(
    "api_?key$|^oci_key$|token|secret|password|credential|auth", re.IGNORECASE
)
_PATH_STARTS = ("/", "~")
_registered: set[str] = set()
_lock = threading.Lock()


def register_key(key: str | None) -> None:
    """Have :func:`masked` mask ``key`` from now on, for as long as the process runs,
    unless it is shorter than :data:`_MIN_MASKED_KEY_CHARS`."""
    if key and len(key) >= _MIN_MASKED_KEY_CHARS:
        with _lock:
            _registered.add(key)


def is_credential_name(name: str) -> bool:
    """Whether a parameter or environment variable called ``name`` holds a credential:
    its name, in any case, ends in ``api_key`` or ``apikey``, is ``oci_key``, OCI's
    private key and the one credential litellm reads under a bare ``key``, or says
    token, secret, password, credential or auth. Any other bare ``key``, as in
    ``prompt_cache_key`` or litellm's ``cache_key``, does not."""
    return bool(_CREDENTIAL_NAME.search(name))


def masked(text: str) -> str:
    """``text`` with each known key replaced by ``REDACTED``."""
    known = set(_environment_secrets())
    with _lock:
        known |= _registered
    for key in sorted(known, key=len, reverse=True):
        text = text.replace(key, _MASK)
    return text


def masked_values(data: Any) -> Any:
    """``data`` with each string among its dicts' values and its lists' items masked
    (:func:`masked`), so a key that JSON or a ``repr`` would escape is still found."""
    if isinstance(data, str):
        return masked(data)
    if isinstance(data, dict):
        return {name: masked_values(value) for name, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [masked_values(item) for item in data]
    return data


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
    return [value for name, value in os.environ.items() if _is_secret(name, value)]


def _is_secret(name: str, value: str) -> bool:
    if not is_credential_name(name) or value.startswith(_PATH_STARTS):
        return False
    if any(map(str.isspace, value)):
        return False
    provider_key = _PROVIDER_KEY_NAME.search(name)
    floor = _MIN_MASKED_KEY_CHARS if provider_key else _MIN_ENVIRONMENT_SECRET_CHARS
    return len(value) >= floor
