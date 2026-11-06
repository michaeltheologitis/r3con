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
short to register is masked too, and leaves every other setting as it was: an ask
sends a run's ``params`` again from its manifest, and refuses one that holds a masked
value (:func:`masked_settings`). Prompts, replies and documents are data, and are
written as they are: a document may well say ``api_key=…``.

:func:`masked` replaces each known key with ``REDACTED``, the longest first, and nothing
else, so it never garbles a path or a name that merely looks like a key. The known keys
are:

- every key registered with :func:`register_key`, which r3con does for each credential
  it sends (:func:`r3con.runtime.llm.register_credentials`), at any depth of its
  parameters (:func:`credentials_in`): a call's ``api_key``, each string with a
  credential's name (:func:`is_credential_name`), a header's among them, the password
  and each credential in the query of a URL setting, and the keys of a
  ``litellm.Router`` whose ``.completion`` is the run's ``completion``. A
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
from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import ValidationError

_MASK = "REDACTED"
# A shorter key is a placeholder, and masking one as short as "x" garbles "maximum".
_MIN_MASKED_KEY_CHARS = 8
_MIN_ENVIRONMENT_SECRET_CHARS = 16
_ERROR_CHARS = 300
_PROVIDER_KEY_NAME = re.compile("api_?key$", re.IGNORECASE)
# A name is read as its words: "x-api-key", "X_API_KEY" and "xApiKey" alike.
_CAMEL_HUMP = re.compile("([a-z0-9])([A-Z])")
_NOT_A_WORD = re.compile("[^a-z0-9]+")
# A credential's name has one of these words, or ends in one of the next, or in "key"
# after one of the last: not "stop_token_ids", "max_tokens", "author" or "auth_method".
_SECRET_WORDS = frozenset(
    {"secret", "password", "passwd", "credential", "credentials"}
    | {"authorization", "cookie", "bearer"}
)
_LAST_WORDS = frozenset({"token", "auth", "apikey"})
_KEY_KINDS = frozenset({"api", "subscription", "private", "auth"})
_KEY = "key"
# Where a bare "key" is one: the query Google's APIs read a key from ("?key=...").
_QUERY = "extra_query"
_URLS = frozenset({"api_base", "base_url", "azure_endpoint"})
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
    """Whether a parameter, header, field or environment variable called ``name``
    holds a credential, by the words of its name in any case and spelling
    (``x-api-key``, ``X_API_KEY``, ``xApiKey``): one says secret, password, credential,
    authorization, cookie or bearer, or the last is token, auth or apikey, or key after api,
    subscription, private or auth. So ``access_token``, ``client_secret``,
    ``Ocp-Apim-Subscription-Key`` and ``x-gateway-auth`` are credentials, and
    ``max_tokens``, ``stop_token_ids``, ``tokenizer``, ``author``, ``auth_method`` and
    a bare ``key``, as in ``prompt_cache_key``, are not."""
    words = _NOT_A_WORD.split(_CAMEL_HUMP.sub(r"\1_\2", name).lower())
    if _SECRET_WORDS.intersection(words) or words[-1] in _LAST_WORDS:
        return True
    return len(words) > 1 and words[-1] == _KEY and words[-2] in _KEY_KINDS


def masked(text: str) -> str:
    """``text`` with each known key replaced by ``REDACTED``."""
    return _replaced(text, _known_keys())


def masked_params(params: Any) -> Any:
    """``params``, a request's parameters as JSON, with each credential in it replaced
    by ``REDACTED``, at any depth, and all else as it was: every string under a
    credential's name (:func:`is_credential_name`; its numbers and booleans are kept),
    a bare ``key`` in ``extra_query`` among them; the password, or a user without one,
    and each credential in the query of an ``api_base``, ``base_url`` or
    ``azure_endpoint``; and each known key in any other string, as :func:`masked` does,
    the keys read once."""
    return _redacted(params, _known_keys())


def masked_settings(params: Any, setting: str = "") -> Iterator[str]:
    """The setting, dotted (``extra_headers.Authorization``), of each value in
    ``params`` that :func:`masked_params` masked, in whole or in part."""
    if isinstance(params, str) and _MASK in params:
        yield setting
    elif isinstance(params, Mapping):
        for name, value in params.items():
            yield from masked_settings(value, f"{setting}.{name}".lstrip("."))
    elif isinstance(params, (list, tuple)):
        for value in params:
            yield from masked_settings(value, setting)


def credentials_in(params: Any, name: str = "", within: str = "") -> Iterator[str]:
    """Each string a request sends as a credential in ``params``, its parameters, at
    any depth: each one whose name is a credential's (:func:`is_credential_name`), a
    bare ``key`` in ``extra_query`` among them, and the password, or a user without
    one, and each credential in the query of an ``api_base``, ``base_url`` or
    ``azure_endpoint``."""
    if isinstance(params, str):
        if _names_a_credential(name, within):
            yield params
        elif name in _URLS:
            yield from _url_secrets(params)
    elif isinstance(params, Mapping):
        for key, value in params.items():
            yield from credentials_in(value, str(key), name)
    elif isinstance(params, (list, tuple)):
        for value in params:
            yield from credentials_in(value, name, within)


def _names_a_credential(name: str, within: str) -> bool:
    return is_credential_name(name) or (name == _KEY and within == _QUERY)


def _redacted(value: Any, known: list[str], name: str = "", within: str = "") -> Any:
    if _names_a_credential(name, within):
        return _strings_masked(value)
    if isinstance(value, dict):
        return {
            key: _redacted(item, known, str(key), name) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redacted(item, known, name, within) for item in value]
    if isinstance(value, str):
        return _replaced(_url_masked(value) if name in _URLS else value, known)
    return value


def _strings_masked(value: Any) -> Any:
    """``value`` with every string in it, at any depth, ``REDACTED``."""
    if isinstance(value, str):
        return _MASK
    if isinstance(value, dict):
        return {key: _strings_masked(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strings_masked(item) for item in value]
    return value


def _url_masked(url: str) -> str:
    """``url`` with its password, or a user without one, and each credential in its
    query ``REDACTED``, or as it is when it holds none."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return _MASK
    userinfo, at, host = parts.netloc.rpartition("@")
    pairs = parse_qsl(parts.query, keep_blank_values=True)
    query = [(k, _MASK if _names_a_credential(k, _QUERY) else v) for k, v in pairs]
    if not at and query == pairs:
        return url
    if at:
        user, colon, _ = userinfo.partition(":")
        host = f"{user}:{_MASK}@{host}" if colon else f"{_MASK}@{host}"
    kept = parts.query if query == pairs else urlencode(query)
    return urlunsplit((parts.scheme, host, parts.path, kept, parts.fragment))


def _url_secrets(url: str) -> list[str]:
    try:
        parts = urlsplit(url)
    except ValueError:
        return []
    user, colon, password = parts.netloc.rpartition("@")[0].partition(":")
    pairs = parse_qsl(parts.query)
    keys = [v for k, v in pairs if _names_a_credential(k, _QUERY)]
    return [secret for secret in [password if colon else user, *keys] if secret]


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
