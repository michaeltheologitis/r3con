"""Keys masked out of the error text r3con writes and logs.

A provider may echo the key it was sent in its error, and litellm passes the text on: for
``openai/``, ``hosted_vllm/`` and every OpenAI-compatible provider, ``str(error)`` holds
it as the provider sent it, and every traceback holds it through the chained SDK error.
So each error r3con writes (``calls.json``'s ``error``, ``error.txt``, the ``error`` of
``run_failed`` and ``ask_failed``, ``splits.json``, reasoning's ``fallback``) goes
through :func:`masked` or :func:`one_line` first, as does every record an r3con logger
logs (:func:`r3con.logging_setup.get_logger`). The parameters r3con writes, the
``config`` in ``manifest.json`` and each call's ``params`` in ``calls.json``, go
through :func:`masked_params`, which also masks a credential by its name, so one too
short to register is masked too. Prompts, replies and documents are data, and are
written as they are: a document may well say ``api_key=…``.

:func:`masked` replaces each known key with ``REDACTED``, the longest first, and nothing
else, so it never garbles a path or a name that merely looks like a key. The known keys
are:

- every key registered with :func:`register_key`, which r3con does for each credential
  it sends (:func:`r3con.runtime.llm.register_credentials`), at any depth of its
  parameters: a call's ``api_key``, each string with a credential's name
  (:func:`is_credential_name`), each header's value, the password and key in an
  ``api_base``, and the keys of a ``litellm.Router`` whose ``.completion`` is the run's
  ``completion``. A ``completion`` wrapper that adds a key of its own must register it;
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
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from pydantic import ValidationError

_MASK = "REDACTED"
# A shorter key is a placeholder, and masking one as short as "x" garbles "maximum".
_MIN_MASKED_KEY_CHARS = 8
_MIN_ENVIRONMENT_SECRET_CHARS = 16
_ERROR_CHARS = 300
_PROVIDER_KEY_NAME = re.compile("api_?key$", re.IGNORECASE)
_CREDENTIAL_NAME = re.compile(
    "api_?key$|token|secret|password|credential|auth", re.IGNORECASE
)
_PATH_STARTS = ("/", "~")
_HEADERS = frozenset({"headers", "extra_headers"})
_URLS = frozenset({"api_base", "base_url"})
# The query parameter Google's APIs read a key from, as in "?key=...".
_QUERY_KEY = "key"
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
    its name, in any case, ends in ``api_key`` or ``apikey``, or says token, secret,
    password, credential or auth. A bare ``key``, as in ``prompt_cache_key``, does
    not."""
    return bool(_CREDENTIAL_NAME.search(name))


def masked(text: str) -> str:
    """``text`` with each known key replaced by ``REDACTED``."""
    return _replaced(text, _known_keys())


def masked_params(params: Any) -> Any:
    """``params``, a request's parameters as JSON, with each credential in it replaced
    by ``REDACTED``, at any depth: the string, list or dict a name says is a credential
    holds (:func:`is_credential_name`; a number under one, ``max_tokens`` say, is
    kept), each header's value (``headers``, ``extra_headers``), and each known key in
    any other string, as :func:`masked` does, the keys read once. An ``api_base`` or a
    ``base_url`` is written without its user, password, query or fragment."""
    return _redacted(params, _known_keys())


def credentials_in(params: Any, name: str = "") -> Iterator[str]:
    """Each string a request sends as a credential in ``params``, its parameters, at
    any depth: each string with a credential's name (:func:`is_credential_name`), each
    header's value, and the password and the key (``key`` or a credential's name) in
    the query of an ``api_base`` or ``base_url``. A dict under a credential's name, a
    service account's say, is left to :func:`masked_params`: its strings include names
    that are not keys."""
    if isinstance(params, str):
        if is_credential_name(name):
            yield params
        elif name in _URLS:
            yield from _url_secrets(params)
    elif isinstance(params, Mapping):
        for key, value in params.items():
            header = name in _HEADERS and isinstance(value, str)
            yield from [value] if header else credentials_in(value, str(key))
    elif isinstance(params, (list, tuple)):
        for value in params:
            yield from credentials_in(value, name)


def _redacted(value: Any, known: list[str], name: str = "") -> Any:
    if is_credential_name(name) and isinstance(value, (str, dict, list)):
        return _MASK
    if name in _HEADERS and isinstance(value, dict):
        return dict.fromkeys(value, _MASK)
    if isinstance(value, dict):
        return {key: _redacted(item, known, str(key)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redacted(item, known, name) for item in value]
    if isinstance(value, str):
        return _replaced(_bare_url(value) if name in _URLS else value, known)
    return value


def _bare_url(url: str) -> str:
    """``url`` without its user, password, query or fragment."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return _MASK
    host = parts.netloc.rpartition("@")[2]
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _url_secrets(url: str) -> list[str]:
    try:
        parts = urlsplit(url)
        password = parts.password
    except ValueError:
        return []
    query = parse_qsl(parts.query)
    keys = [v for k, v in query if k == _QUERY_KEY or is_credential_name(k)]
    return [secret for secret in [password, *keys] if secret]


def _known_keys() -> list[str]:
    """Every key :func:`masked` masks, the longest first."""
    known = set(_environment_secrets())
    with _lock:
        known |= _registered
    return sorted(known, key=len, reverse=True)


def _replaced(text: str, keys: list[str]) -> str:
    for key in keys:
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
    return [value for name, value in os.environ.items() if _is_secret(name, value)]


def _is_secret(name: str, value: str) -> bool:
    if not is_credential_name(name) or value.startswith(_PATH_STARTS):
        return False
    if any(map(str.isspace, value)):
        return False
    provider_key = _PROVIDER_KEY_NAME.search(name)
    floor = _MIN_MASKED_KEY_CHARS if provider_key else _MIN_ENVIRONMENT_SECRET_CHARS
    return len(value) >= floor
