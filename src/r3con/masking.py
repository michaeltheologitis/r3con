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
  (:func:`r3con.runtime.llm.register_credentials`). A value is a credential by where
  it sits, at any depth of a plain copy of the request's parameters (:func:`plain`),
  but in ``messages``, the documents: in a header dict (``extra_headers``,
  ``headers``, an MCP tool's ``headers``), whatever the header's name, but a
  known-public header (``Content-Type``, ``anthropic-beta``); in a parameter with a
  credential's name (:func:`is_credential_name`), such as ``api_key`` or ``metadata``'s
  ``langfuse_secret_key``, unless it is a path that exists (:func:`is_path`); and in a
  URL, such as ``api_base``, as its password, or its user name when it has none, or as
  any value in its query but a public one (Azure's ``api-version``, watsonx's
  ``version``). The keys of a dict under a key that JSON Schema names fields or
  patterns with (``properties``, ``patternProperties``, ``$defs``, …) are not
  parameters' names. The rule goes by that key's name, not by whether the dict is a
  schema, so it holds wherever the key sits: in a tool's ``parameters``, in
  ``response_format`` or ``guided_json``, or anywhere else. The keys of a
  ``litellm.Router`` whose ``.completion`` is the run's ``completion`` are registered
  too. Every string in a dict-valued ``vertex_credentials`` is masked, the public
  ``token_uri`` too, since a service account's field names do not say which are
  secret. A ``completion`` wrapper that adds a key of its own must register it;
- the value of every environment variable with a credential's name, where litellm finds
  a provider's key, unless it holds whitespace or is a path that exists
  (:func:`is_path`), as ``GOOGLE_APPLICATION_CREDENTIALS`` and
  ``GITHUB_COPILOT_TOKEN_DIR`` hold, or is shorter than
  :data:`_MIN_ENVIRONMENT_SECRET_CHARS`, or :data:`_MIN_MASKED_KEY_CHARS` for a name
  that ends in ``api_key`` or ``apikey``, a provider's key.

A key shorter than :data:`_MIN_MASKED_KEY_CHARS` is a placeholder, such as vLLM's
``EMPTY``, and is left alone.
"""

from __future__ import annotations

import dataclasses
import datetime
import os
import re
import threading
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, ValidationError

MASK = "REDACTED"
# A shorter key is a placeholder, and masking one as short as "x" garbles "maximum".
_MIN_MASKED_KEY_CHARS = 8
_MIN_ENVIRONMENT_SECRET_CHARS = 16
_ERROR_CHARS = 300
_PROVIDER_KEY_NAME = re.compile("api_?key$", re.IGNORECASE)
_CREDENTIAL_NAME = re.compile(
    "api_?key$|^oci_key$|token|secret|password|credential|webhook", re.IGNORECASE
)
# A name's words, whatever its case: snake_case, kebab-case, camelCase, SCREAMING_CASE.
_NAME_WORDS = re.compile("[A-Z]?[a-z]+|[A-Z]+(?![a-z])")
# Whole words only: "auth" begins "author" and "authority", which name no credential.
_AUTH_WORDS = frozenset({"auth", "authentication", "authorization"})
# Each auth word holds it, so a name without it needs no splitting into words.
_AUTH = "auth"
# Not "~name": expanding one looks the user up, through NSS or LDAP on some hosts.
_PATH_STARTS = ("/", "~/")
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
    private key and the one credential litellm reads under a bare ``key``, says token,
    secret, password, credential or webhook (``slack_webhook_url``, whose secret is its
    URL's path), or has auth, authentication or authorization as a word of its own
    (``x-auth-token``, ``basicAuth``, Azure On Your Data's ``authentication``), not
    within another (``author``, ``authority``). Any other bare ``key``, as in
    ``prompt_cache_key`` or litellm's ``cache_key``, does not."""
    if _CREDENTIAL_NAME.search(name):
        return True
    if _AUTH not in name.lower():
        return False
    words = {word.lower() for word in _NAME_WORDS.findall(name)}
    return not _AUTH_WORDS.isdisjoint(words)


def is_path(value: str) -> bool:
    """Whether ``value``, held under a credential's name, is the path of a file or
    folder litellm reads a credential from or keeps one in (``vertex_credentials``,
    ``GOOGLE_APPLICATION_CREDENTIALS``, ``GITHUB_COPILOT_TOKEN_DIR``) rather than the
    credential: it starts with ``/`` or ``~/`` and the path exists. A base64 key may
    start with ``/``, as about one AWS secret key in 64 does, but names no path, so it
    is masked, as is a path not made yet, and any ``~name…``, which is never looked
    up: a path masked, never a key shown. A value too long or otherwise invalid as a
    path is not one."""
    if not value.startswith(_PATH_STARTS):
        return False
    return os.path.exists(os.path.expanduser(value))


def masked(text: str) -> str:
    """``text`` with each known key replaced by ``REDACTED``."""
    known = set(_environment_secrets())
    with _lock:
        known |= _registered
    for key in sorted(known, key=len, reverse=True):
        text = text.replace(key, MASK)
    return text


def plain(data: Any) -> Any:
    """``data`` in dicts with string keys, lists, strings, numbers, booleans and
    ``None``, so each walk over it sees the same shapes: a date, time or datetime as
    its ISO 8601 text (``isoformat()``), a mapping as a dict, a list,
    tuple, set or frozenset as a list, a pydantic model as its ``model_dump()``, a
    dataclass as a dict of its fields, and anything else, an iterator or a generator
    among them, as its ``repr``, never iterated, so a request still sends all of it."""
    if data is None or isinstance(data, (str, int, float)):
        return data
    if isinstance(data, (datetime.date, datetime.time)):
        return data.isoformat()
    if isinstance(data, Mapping):
        return {str(name): plain(value) for name, value in data.items()}
    if isinstance(data, (list, tuple, set, frozenset)):
        return [plain(item) for item in data]
    if isinstance(data, BaseModel):
        return plain(data.model_dump())
    if dataclasses.is_dataclass(data) and not isinstance(data, type):
        # Each field as it is: dataclasses.asdict deep-copies them, and fails on a lock.
        fields = dataclasses.fields(data)
        return {field.name: plain(getattr(data, field.name)) for field in fields}
    return repr(data)


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
    if not is_credential_name(name) or any(map(str.isspace, value)):
        return False
    provider_key = _PROVIDER_KEY_NAME.search(name)
    floor = _MIN_MASKED_KEY_CHARS if provider_key else _MIN_ENVIRONMENT_SECRET_CHARS
    return len(value) >= floor and not is_path(value)
