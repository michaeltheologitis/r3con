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
  (:func:`r3con.runtime.llm.register_credentials`). It looks at every depth of a plain
  copy of the request's parameters (:func:`plain`), but not into the request's data
  (``messages``, ``tools``, a schema, ``metadata``) wherever it sits, and takes each
  string or integer in ``extra_headers`` or ``headers``, whatever the header's name,
  and in a parameter with a credential's name (:func:`is_credential_name`), such as
  ``api_key``; in a URL, such as ``api_base``, its password, or its user name when it
  has none, and every value in its query but a public one (``api-version``); and the
  keys of a ``litellm.Router`` whose ``.completion`` is the run's ``completion``. Every
  string in a dict-valued ``vertex_credentials`` is masked, the public ``token_uri``
  too, since a service account's field names do not say which are secret. A
  ``completion`` wrapper that adds a key of its own must register it;
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
from pydantic_core import to_jsonable_python

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


def plain(data: Any) -> Any:
    """``data`` in JSON's dicts, lists, strings and numbers, so each walk over it sees
    the same shapes: a pydantic model, a dataclass or any other mapping as a dict, a set
    or a tuple as a list, and a value pydantic has no JSON form for as its ``repr``."""
    return to_jsonable_python(data, fallback=_plain_fallback)


def masked_values(data: Any) -> Any:
    """``data`` made plain (:func:`plain`), with each string and integer among its
    dicts' values and its lists' items masked (:func:`masked`), so a key that JSON or a
    ``repr`` would escape is still found, as is one YAML read as a number."""
    return _masked_leaves(plain(data))


def _masked_leaves(data: Any) -> Any:
    if isinstance(data, dict):
        return {name: _masked_leaves(value) for name, value in data.items()}
    if isinstance(data, list):
        return [_masked_leaves(item) for item in data]
    if isinstance(data, (str, int)):
        shown = masked(str(data))
        return data if shown == str(data) else shown
    return data


def _plain_fallback(value: Any) -> Any:
    return dict(value) if isinstance(value, Mapping) else repr(value)


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
