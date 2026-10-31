"""The local server between r3con's app and r3con.

It listens on 127.0.0.1 only and serves the page at ``/``, its stylesheet and script
at ``/app.css`` and ``/app.js``, and a JSON API under ``/api/``. A request must name
this server in ``Host``, and in ``Origin`` when it sends one (403 otherwise). Every
request but those for the page and its two files must carry the launch's token as
``Authorization: Bearer <token>`` (401 otherwise); the page gets the token from the
opened URL's fragment, ``#token=...``.

- ``GET /api/keys``: each provider's key, as ``null`` when it has none, ``{source:
  "here", hint}`` when it is stored here, or ``{source: "environment", variable}`` when
  litellm finds it in that variable; and ``other``, the servers stored here, in the
  order they were added, each ``{endpoint, key}``, its key a hint or ``null``. A hint
  is a key's last ``KEY_HINT_CHARS`` characters, ``""`` for a key shorter than
  ``MIN_HINTED_KEY_CHARS``.
  ``POST`` stores a provider's key (``{provider, key}``), or a server, by its URL
  (``{provider: "other", endpoint, key?}``), replacing the key of one stored at that
  URL; ``DELETE`` forgets one (``{provider}``, or ``{provider: "other", endpoint}``);
  both answer as ``GET`` does. A URL is kept without its trailing slashes, and one
  that names no host, or holds a user or password, is refused (422): the page keeps a
  server's URL with the models it saves.
- A run or ask names its model and, as ``endpoint``, the server it runs on, if any.
  With no server, OpenAI's, Anthropic's and Gemini's models get their provider's key
  stored here, and any other gets litellm's own setup from the environment. On a
  server, a model of a self-hosted provider in ``SELF_HOSTED``, or a name litellm
  can't place, is sent there with the server's key, or ``KEYLESS_SERVER_KEY`` for a
  keyless server: given none, litellm would send it one from the environment, such
  as ``OLLAMA_API_KEY``. An ``endpoint`` that isn't a stored server, or a model
  litellm places with a provider that isn't self-hosted, is refused (422), so that no
  provider's key, stored here or in the environment, ever reaches a server.
- Keys stay in this process's memory, and the server never logs or writes one, and
  answers no part of one but its hint. Each goes to r3con as a run's ``api_key``,
  which r3con registers before sending it and masks in every event's ``error``, in
  what it writes to the logs dir and in what it logs (:mod:`r3con.masking`), the
  environment's keys too; the server's own log line of a failure goes through the same
  masking.
- ``POST /api/documents/text?name=<file name>``, with a PDF's or a Word file's bytes as
  the body, chosen by the name's ``.pdf`` or ``.docx`` (422 for any other), answers
  ``{text}``: its text as ``r3con.read_documents`` extracts it, which is ``""`` for a
  scan, a file with no text or one that can't be read, as the server's log then says.
  The page sends each PDF and Word file it is given here, and reads other files itself.
- ``POST /api/documents/zip?name=<file name>&limit=<n>``, with a zip's bytes as the
  body, answers ``{files: [{name, size, text}], skipped, unread}``. Its documents are
  the entries ``r3con.read_documents``' folder walk would read, judged by their paths
  below the one top folder the zip's files share, if they do, ``__MACOSX/`` aside: by
  suffix, and with no Office lock file and nothing hidden, in a hidden folder or under
  ``__MACOSX/``. They are read in the walk's order, as that walk reads a file, until
  ``limit`` of them have text; each is named by that path, and one with no text comes
  back with ``""`` and doesn't count.
  ``skipped`` counts the files left out by their suffix, nested zips among them,
  which are never opened; ``unread``, the documents not read. A zip that is corrupt,
  encrypted, of more than ``MAX_ZIP_ENTRIES`` entries, or whose documents read would
  unzip to more than ``MAX_ZIP_BYTES``, is refused (422).
- ``POST /api/runs`` (``{question, model, endpoint?, documents: [text, ...]}``) starts
  ``r3con.run`` and answers ``{run_id}``; ``GET /api/runs/{run_id}/events`` streams its
  events as ``text/event-stream``, from the first, until ``run_finished`` or
  ``run_failed``.
- ``POST /api/runs/{run_id}/asks`` (``{question, model, endpoint?}``) starts
  ``r3con.ask`` over a run that has ended and answers ``{ask_id}``;
  ``GET /api/asks/{ask_id}/events`` streams its events until ``ask_finished`` or
  ``ask_failed``.
- ``POST /api/runs/{run_id}/cancel`` and ``POST /api/asks/{ask_id}/cancel`` cancel that
  run or ask through r3con's ``cancel`` and answer 202 at once. From then on it sends no
  new model call; a call in flight finishes, and the stream then ends with its last
  event: the ``run_failed`` or ``ask_failed`` r3con emits, naming the stage it stopped
  in, or ``run_finished`` or ``ask_finished`` when the call in flight was its last. A
  run or ask that has ended is left as it was.

The server stops in one place, its shutdown, whether ``serve``'s ``stop``, Ctrl-C or
SIGTERM asks it to. As the shutdown begins, it refuses a new run or ask (503) and
cancels every one still going, before it waits for open connections such as the page's
event stream. Once they have closed, it waits up to ``WORKERS_EXIT_SECONDS`` for those
runs and asks to end, so each finishes its call in flight and writes its run folder,
with ``run_failed`` in its ``events.jsonl``; a page whose stream has already closed does
not see it. A second Ctrl-C ends that wait at once, and ``r3con app`` quits there and
then, leaving those runs and asks as they are.

A failure answers ``{"error": "..."}``, except a body over the size limit, which gets
Starlette's plain-text 413. A run or ask that r3con refuses before it starts ends its
stream with a failure event whose ``stage`` is ``null``.
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import io
import json
import lzma
import os
import secrets
import signal
import socket
import threading
import time
import webbrowser
import zipfile
import zlib
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar
from urllib.parse import urlsplit

import litellm
import platformdirs
import uvicorn
from litellm.exceptions import BadRequestError
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    StringConstraints,
    TypeAdapter,
    ValidationError,
)
from pydantic_core import PydanticCustomError
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from r3con import r3con
from r3con.logging_setup import configure_logging, get_logger
from r3con.masking import one_line
from r3con.runs import EventRecorder
from r3con.runtime.llm import quiet_litellm

Provider = Literal["openai", "anthropic", "gemini"]
Kind = Literal["run", "ask"]

HOST = "127.0.0.1"
# Media types named, not guessed: Python 3.11, or Windows' registry, can guess otherwise.
PAGE_FILES = {
    "/": ("index.html", "text/html"),
    "/app.css": ("app.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
}
# Revalidated on every load, so that a new page never runs an old version's script.
PAGE_HEADERS = {
    "Content-Security-Policy": "frame-ancestors 'none'",
    "Cache-Control": "no-cache",
}
LOGS_DIR = platformdirs.user_data_path("r3con") / "runs"
TOKEN_BYTES = 32
ID_BYTES = 8
MAX_BODY_BYTES = 64 * 1024 * 1024
# A dropped zip is outside input: these bound what unpacking one can cost.
MAX_ZIP_ENTRIES = 1_000
MAX_ZIP_MEGABYTES = 100
MAX_ZIP_BYTES = MAX_ZIP_MEGABYTES * 1024 * 1024
ZIP_ENCRYPTED_FLAG = 0x1
# What zipfile raises, opening a zip or reading an entry, for a zip it can't read.
UNREADABLE_ZIP_ERRORS = (
    zipfile.BadZipFile,
    zlib.error,
    lzma.LZMAError,
    OSError,
    EOFError,
    ValueError,
    NotImplementedError,
)
UNREADABLE_ZIP = "it's corrupt, or not a zip"
ENCRYPTED_ZIP = "it's encrypted"
TOO_MANY_ENTRIES = f"it holds more than {MAX_ZIP_ENTRIES:,} entries"
TOO_MANY_BYTES = f"its documents unzip to more than {MAX_ZIP_MEGABYTES} MB"
SHUTDOWN_GRACE_SECONDS = 1
WORKERS_EXIT_SECONDS = 60
WORKER_POLL_SECONDS = 0.1
FORCED_EXIT_CODE = 128 + signal.SIGINT
KEY_VARIABLES: dict[Provider, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
# A key shows its last few characters only when they are a small part of it.
KEY_HINT_CHARS = 4
MIN_HINTED_KEY_CHARS = 3 * KEY_HINT_CHARS
SELF_HOSTED = frozenset(
    {"hosted_vllm", "ollama", "ollama_chat", "lm_studio", "llamafile", "custom_openai"}
)
# vLLM's placeholder for "no key"; non-empty, since r3con drops an empty api_key.
KEYLESS_SERVER_KEY = SecretStr("EMPTY")
UNKNOWN_SERVER = "This model's server isn't in settings: add it there first."
CREDENTIALS_IN_URL = "must not hold a user or password: put the key in its key field"
NO_HOST_IN_URL = "must name a host"
NOT_SELF_HOSTED = (
    "isn't a self-hosted model, so it can't run on a server: pick other model… for it."
)
ENDINGS = frozenset({"run_finished", "run_failed", "ask_finished", "ask_failed"})
FORBID_EXTRA = ConfigDict(extra="forbid")

logger = get_logger("app")
Parsed = TypeVar("Parsed")
Result = TypeVar("Result")


def _not_blank(text: str) -> str:
    if not text.strip():
        raise PydanticCustomError("blank", "must not be blank")
    return text


def _stripped_secret(secret: SecretStr) -> SecretStr:
    return SecretStr(_not_blank(secret.get_secret_value()).strip())


def _server_url(endpoint: str) -> str:
    if "@" in urlsplit(endpoint).netloc:
        raise PydanticCustomError("credentials", CREDENTIALS_IN_URL)
    stripped = endpoint.rstrip("/")
    if not urlsplit(stripped).hostname:
        raise PydanticCustomError("host", NO_HOST_IN_URL)
    return stripped


Text = Annotated[str, AfterValidator(_not_blank)]
Secret = Annotated[SecretStr, AfterValidator(_stripped_secret)]
Endpoint = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^https?://\S+$"),
    AfterValidator(_server_url),
]


class RunRequest(BaseModel):
    model_config = FORBID_EXTRA
    question: Text
    model: Text
    endpoint: Endpoint | None = None
    documents: list[Text] = Field(min_length=1)


class DocumentQuery(BaseModel):
    model_config = FORBID_EXTRA
    name: Text


class ZipQuery(BaseModel):
    model_config = FORBID_EXTRA
    name: Text
    limit: int = Field(MAX_ZIP_ENTRIES, ge=1)


class AskRequest(BaseModel):
    model_config = FORBID_EXTRA
    question: Text
    model: Text
    endpoint: Endpoint | None = None


class ProviderKey(BaseModel):
    model_config = FORBID_EXTRA
    provider: Provider
    key: Secret


class ServerKey(BaseModel):
    model_config = FORBID_EXTRA
    provider: Literal["other"]
    endpoint: Endpoint
    key: Secret | None = None


class ForgetKey(BaseModel):
    model_config = FORBID_EXTRA
    provider: Provider


class ForgetServer(BaseModel):
    model_config = FORBID_EXTRA
    provider: Literal["other"]
    endpoint: Endpoint


KEY_BODY: TypeAdapter[ProviderKey | ServerKey] = TypeAdapter(
    Annotated[ProviderKey | ServerKey, Field(discriminator="provider")]
)
FORGET_BODY: TypeAdapter[ForgetKey | ForgetServer] = TypeAdapter(
    Annotated[ForgetKey | ForgetServer, Field(discriminator="provider")]
)


@dataclass
class Job:
    """One run's or ask's events so far, and the event that cancels it. Only the event
    loop's thread touches ``events`` and ``wake``; its worker reads ``cancel``."""

    events: list[dict[str, Any]] = field(default_factory=list)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    cancel: threading.Event = field(default_factory=threading.Event)


@dataclass
class ServerState:
    logs_dir: Path
    completion: Callable[..., Any] | None
    keys: dict[Provider, SecretStr] = field(default_factory=dict)
    servers: dict[str, SecretStr | None] = field(default_factory=dict)
    jobs: dict[Kind, dict[str, Job]] = field(
        default_factory=lambda: {"run": {}, "ask": {}}
    )
    workers: list[threading.Thread] = field(default_factory=list)
    stopping: bool = False


def _state(request: Request) -> ServerState:
    return request.app.state.server


def _valid(parse: Callable[[Any], Parsed], given: Any) -> Parsed:
    try:
        return parse(given)
    except ValidationError as error:
        raise HTTPException(422, _explain(error)) from None


async def _read(request: Request, parse: Callable[[bytes], Parsed]) -> Parsed:
    return _valid(parse, await request.body())


def _explain(error: ValidationError) -> str:
    """Each field pydantic refused and why, never the value it was given."""
    return "; ".join(
        f"{'.'.join(str(part) for part in e['loc']) or 'body'}: {e['msg']}"
        for e in error.errors(include_input=False, include_url=False)
    )


def _hint(key: SecretStr) -> str:
    secret = key.get_secret_value()
    return secret[-KEY_HINT_CHARS:] if len(secret) >= MIN_HINTED_KEY_CHARS else ""


def _provider_status(state: ServerState, provider: Provider) -> dict[str, str] | None:
    stored, variable = state.keys.get(provider), KEY_VARIABLES[provider]
    if stored is not None:
        return {"source": "here", "hint": _hint(stored)}
    if os.environ.get(variable):
        return {"source": "environment", "variable": variable}
    return None


def _key_status(state: ServerState) -> dict[str, Any]:
    status: dict[str, Any] = {
        provider: _provider_status(state, provider) for provider in KEY_VARIABLES
    }
    status["other"] = [
        {"endpoint": endpoint, "key": None if key is None else _hint(key)}
        for endpoint, key in state.servers.items()
    ]
    return status


def _provider(model: str) -> str | None:
    """``model``'s provider as litellm places it, or ``None`` if it can't."""
    try:
        with quiet_litellm():
            return litellm.get_llm_provider(model)[1]
    except BadRequestError:
        return None


def _model_options(
    state: ServerState, model: str, endpoint: str | None
) -> dict[str, Any]:
    """What r3con is called with for ``model`` on the server at ``endpoint``, if any,
    as the module docstring says. Anything not stored here is passed as ``None``, so
    that litellm reads the environment."""
    provider = _provider(model)
    if endpoint is None:
        key = state.keys.get(provider) if provider in KEY_VARIABLES else None
        base = None
    elif endpoint not in state.servers:
        raise HTTPException(422, UNKNOWN_SERVER)
    elif provider is None or provider in SELF_HOSTED:
        stored = state.servers[endpoint]
        key = KEYLESS_SERVER_KEY if stored is None else stored
        base = endpoint
    else:
        raise HTTPException(422, f"{model} {NOT_SELF_HOSTED}")
    return {
        "model": model,
        "api_key": None if key is None else key.get_secret_value(),
        "api_base": base,
        "completion": state.completion,
    }


def _ended(job: Job) -> bool:
    return bool(job.events) and job.events[-1]["type"] in ENDINGS


def _add(job: Job, event: dict[str, Any]) -> None:
    if _ended(job):
        return
    job.events.append(event)
    job.wake.set()
    job.wake = asyncio.Event()


def _relay(loop: asyncio.AbstractEventLoop, job: Job, event: dict[str, Any]) -> None:
    """Hand ``event`` to ``job`` on the server's loop; once the loop has closed, as it
    has when the server stops mid-run, drop it."""
    try:
        loop.call_soon_threadsafe(_add, job, event)
    except RuntimeError:
        if not loop.is_closed():
            raise


def _work(
    job: Job,
    loop: asyncio.AbstractEventLoop,
    call: Callable[..., Any],
    failed: str,
) -> None:
    on_event = partial(_relay, loop, job)
    try:
        call(on_event=on_event, cancel=job.cancel)
    except Exception as error:  # noqa: BLE001 — logged below, as one masked line
        # No traceback: r3con writes it to the failed stage's error.txt.
        line = one_line(error)
        # A cancel was asked for, by the user or by the server stopping: no error.
        log = logger.info if isinstance(error, r3con.Cancelled) else logger.error
        log("%s: %s", failed, line)
        EventRecorder(on_event, None)(failed, stage=None, error=line)


def _start(state: ServerState, kind: Kind, call: Callable[..., Any]) -> str:
    """Run ``call`` on a thread of its own, its events landing in a new job of
    ``kind``; returns the job's id. Refused once the server is stopping."""
    if state.stopping:
        raise HTTPException(503, "The server is stopping.")
    job_id = secrets.token_hex(ID_BYTES)
    state.jobs[kind][job_id] = job = Job()
    args = (job, asyncio.get_running_loop(), call, f"{kind}_failed")
    worker = threading.Thread(
        target=_work, args=args, name=f"r3con-app-{job_id}", daemon=True
    )
    state.workers.append(worker)
    worker.start()
    return job_id


async def _stream(job: Job) -> AsyncIterator[str]:
    sent = 0
    while True:
        # Taken before draining, so that an event added after the drain still wakes us.
        wake = job.wake
        while sent < len(job.events):
            yield f"data: {json.dumps(job.events[sent])}\n\n"
            sent += 1
        if _ended(job):
            return
        await wake.wait()


def _job(jobs: dict[str, Job], job_id: str, kind: Kind) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"There is no {kind} {job_id}.")
    return job


async def page_file(request: Request, *, name: str, media_type: str) -> Response:
    return FileResponse(
        Path(__file__).with_name(name), media_type=media_type, headers=PAGE_HEADERS
    )


async def keys(request: Request) -> Response:
    return JSONResponse(_key_status(_state(request)))


async def save_key(request: Request) -> Response:
    state = _state(request)
    body = await _read(request, KEY_BODY.validate_json)
    if isinstance(body, ServerKey):
        state.servers[body.endpoint] = body.key
    else:
        state.keys[body.provider] = body.key
    return JSONResponse(_key_status(state))


async def forget_key(request: Request) -> Response:
    state = _state(request)
    body = await _read(request, FORGET_BODY.validate_json)
    if isinstance(body, ForgetServer):
        state.servers.pop(body.endpoint, None)
    else:
        state.keys.pop(body.provider, None)
    return JSONResponse(_key_status(state))


async def document_text(request: Request) -> Response:
    query = _valid(DocumentQuery.model_validate, dict(request.query_params))
    extract = r3con._extractor(query.name)
    if extract is None:
        suffixes = ", ".join(r3con._EXTRACTORS)
        raise HTTPException(422, f"name: must end in one of {suffixes}")
    file = io.BytesIO(await request.body())
    # Extracting is CPU-bound: on the event loop, it would hold up every other request.
    text = await run_in_threadpool(extract, query.name, file)
    return JSONResponse({"text": text})


def _below_shared_folder(paths: list[list[str]]) -> list[list[str]]:
    """``paths``, each as its parts, less the one folder at the top that all but those
    under ``__MACOSX`` share, if they do."""
    tops = {parts[0] for parts in paths if parts[0] != r3con._MACOS_METADATA_FOLDER}
    if len(tops) != 1 or any(len(parts) == 1 for parts in paths):
        return paths
    [top] = tops
    return [parts[1:] if parts[0] == top else parts for parts in paths]


def _refused_if_unreadable(name: str, call: Callable[[], Result]) -> Result:
    """``call()``; a 422, logged, if zipfile can't read the zip called ``name``."""
    try:
        return call()
    except UNREADABLE_ZIP_ERRORS as error:
        logger.warning("%s: not a readable zip (%s)", name, type(error).__name__)
        raise HTTPException(422, UNREADABLE_ZIP) from None


def _unzipped(name: str, body: bytes, limit: int) -> dict[str, Any]:
    """What ``POST /api/documents/zip`` answers for the zip ``body``, called ``name``.
    Only zipfile's own errors make it unreadable: an extractor gives ``""`` for a
    document it can't read."""
    archive = _refused_if_unreadable(name, lambda: zipfile.ZipFile(io.BytesIO(body)))
    with archive:
        entries = archive.infolist()
        if len(entries) > MAX_ZIP_ENTRIES:
            raise HTTPException(422, TOO_MANY_ENTRIES)
        files = [entry for entry in entries if not entry.is_dir()]
        paths = _below_shared_folder([entry.filename.split("/") for entry in files])
        kept = [
            (parts, entry)
            for parts, entry in zip(paths, files, strict=True)
            if not r3con._left_out_unsaid(parts)
        ]
        documents = sorted(
            ((parts, entry) for parts, entry in kept if r3con._walk_reads(parts[-1])),
            key=lambda document: document[0],
        )
        unzipped = _read_until(archive, name, documents, limit)
    return {
        "files": unzipped,
        "skipped": len(kept) - len(documents),
        "unread": len(documents) - len(unzipped),
    }


def _read_until(
    archive: zipfile.ZipFile,
    name: str,
    documents: list[tuple[list[str], zipfile.ZipInfo]],
    limit: int,
) -> list[dict[str, Any]]:
    """``documents``, each its entry and the parts it is named by, read in order until
    ``limit`` of them have text; each is checked before it is read, and those with no
    text come back too, for the page to refuse."""
    unzipped: list[dict[str, Any]] = []
    with_text = unzipped_bytes = 0
    for parts, entry in documents:
        if with_text == limit:
            break
        if entry.flag_bits & ZIP_ENCRYPTED_FLAG:
            raise HTTPException(422, ENCRYPTED_ZIP)
        unzipped_bytes += entry.file_size
        if unzipped_bytes > MAX_ZIP_BYTES:
            raise HTTPException(422, TOO_MANY_BYTES)
        # zipfile stops each entry at the size it claims, and fails its CRC if it lied.
        data = _refused_if_unreadable(name, partial(archive.read, entry))
        text = r3con._read_stream(f"{name}/{entry.filename}", io.BytesIO(data))
        with_text += bool(text.strip())
        unzipped.append({"name": "/".join(parts), "size": len(data), "text": text})
    return unzipped


async def document_zip(request: Request) -> Response:
    query = _valid(ZipQuery.model_validate, dict(request.query_params))
    body = await request.body()
    # As in document_text: on the event loop, it would hold up every other request.
    unzipped = await run_in_threadpool(_unzipped, query.name, body, query.limit)
    return JSONResponse(unzipped)


async def start_run(request: Request) -> Response:
    state = _state(request)
    body = await _read(request, RunRequest.model_validate_json)
    run = partial(
        r3con.run,
        body.question,
        body.documents,
        logs_dir=state.logs_dir,
        **_model_options(state, body.model, body.endpoint),
    )
    return JSONResponse({"run_id": _start(state, "run", run)}, status_code=202)


async def start_ask(request: Request) -> Response:
    state = _state(request)
    run = _job(state.jobs["run"], request.path_params["run_id"], "run")
    body = await _read(request, AskRequest.model_validate_json)
    if not _ended(run):
        raise HTTPException(409, "This run is still running; ask once it has ended.")
    started = run.events[0]
    if started["type"] != "run_started":
        raise HTTPException(409, "This run never started, so it has no folder to ask.")
    ask = partial(
        r3con.ask,
        started["run_dir"],
        body.question,
        **_model_options(state, body.model, body.endpoint),
    )
    return JSONResponse({"ask_id": _start(state, "ask", ask)}, status_code=202)


async def events(request: Request, *, kind: Kind) -> Response:
    job = _job(_state(request).jobs[kind], request.path_params["job_id"], kind)
    return StreamingResponse(_stream(job), media_type="text/event-stream")


async def cancel(request: Request, *, kind: Kind) -> Response:
    _job(_state(request).jobs[kind], request.path_params["job_id"], kind).cancel.set()
    return Response(status_code=202)


def _cancel_jobs(state: ServerState) -> None:
    for jobs in state.jobs.values():
        for job in jobs.values():
            job.cancel.set()


async def _error(request: Request, error: Exception) -> Response:
    # Starlette types every handler as taking any Exception; this one is HTTPException's.
    assert isinstance(error, HTTPException)
    return JSONResponse(
        {"error": error.detail}, error.status_code, headers=error.headers
    )


def _refusal(
    scope: Scope, *, hosts: set[str], origins: set[str], authorization: bytes
) -> Response | None:
    """Why this server will not answer the request in ``scope``, if it won't."""
    headers = Headers(scope=scope)
    if headers.get("host") not in hosts:
        return JSONResponse(
            {"error": "This server answers only to its own address."}, 403
        )
    origin = headers.get("origin")
    if origin is not None and origin not in origins:
        return JSONResponse({"error": "This server answers only its own page."}, 403)
    if scope["path"] in PAGE_FILES:
        return None
    given = headers.get("authorization", "").encode()
    if not hmac.compare_digest(given, authorization):
        return JSONResponse(
            {"error": "Send this launch's token as Authorization: Bearer <token>."},
            401,
            headers={"WWW-Authenticate": "Bearer"},
        )
    return None


def guard(app: ASGIApp, *, token: str, port: int) -> ASGIApp:
    """``app``, answering only requests that name this server and carry ``token``."""
    hosts = {f"{HOST}:{port}", f"localhost:{port}"}
    refuse = partial(
        _refusal,
        hosts=hosts,
        origins={f"http://{host}" for host in hosts},
        authorization=f"Bearer {token}".encode(),
    )

    async def guarded(scope: Scope, receive: Receive, send: Send) -> None:
        refusal = refuse(scope) if scope["type"] == "http" else None
        await (app if refusal is None else refusal)(scope, receive, send)

    return guarded


def create_app(
    *,
    token: str,
    port: int,
    state: ServerState,
    max_body_bytes: int = MAX_BODY_BYTES,
) -> ASGIApp:
    """The server's ASGI app over ``state``, for requests to 127.0.0.1 or localhost on
    ``port`` that carry ``token``. Run folders go in ``state.logs_dir``;
    ``state.completion`` is passed to r3con as its own ``completion`` is."""
    app = Starlette(
        routes=[
            *(
                Route(path, partial(page_file, name=name, media_type=media_type))
                for path, (name, media_type) in PAGE_FILES.items()
            ),
            Route("/api/keys", keys, methods=["GET"]),
            Route("/api/keys", save_key, methods=["POST"]),
            Route("/api/keys", forget_key, methods=["DELETE"]),
            Route("/api/documents/text", document_text, methods=["POST"]),
            Route("/api/documents/zip", document_zip, methods=["POST"]),
            Route("/api/runs", start_run, methods=["POST"]),
            Route("/api/runs/{job_id}/events", partial(events, kind="run")),
            Route(
                "/api/runs/{job_id}/cancel",
                partial(cancel, kind="run"),
                methods=["POST"],
            ),
            Route("/api/runs/{run_id}/asks", start_ask, methods=["POST"]),
            Route("/api/asks/{job_id}/events", partial(events, kind="ask")),
            Route(
                "/api/asks/{job_id}/cancel",
                partial(cancel, kind="ask"),
                methods=["POST"],
            ),
        ],
        exception_handlers={HTTPException: _error},
        max_body_size=max_body_bytes,
    )
    app.state.server = state
    return guard(app, token=token, port=port)


class _Server(uvicorn.Server):
    """uvicorn's server, which stops as the module docstring says: its shutdown is the
    one place for it, and runs inside uvicorn's signal handling."""

    def __init__(self, config: uvicorn.Config, state: ServerState) -> None:
        super().__init__(config)
        self.app_state = state

    async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
        state = self.app_state
        # One step on the loop, where every request is handled: no job starts after.
        state.stopping = True
        _cancel_jobs(state)
        await super().shutdown(sockets)
        workers = [worker for worker in state.workers if worker.is_alive()]
        if workers:
            print(
                f"r3con app: waiting up to {WORKERS_EXIT_SECONDS} s for {len(workers)} "
                "run(s) or ask(s) to finish their calls in flight; Ctrl-C again to "
                "quit now",
                flush=True,
            )
        deadline = time.monotonic() + WORKERS_EXIT_SECONDS
        while (
            any(worker.is_alive() for worker in workers)
            and not self.force_exit
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(WORKER_POLL_SECONDS)


def listen(port: int) -> socket.socket:
    """A socket listening on 127.0.0.1 at ``port``, 0 for a free one."""
    listener = socket.create_server((HOST, port))
    # asyncio's selector loop sets TCP_NODELAY on accepted sockets only for IPPROTO_TCP.
    return socket.socket(
        listener.family, listener.type, socket.IPPROTO_TCP, fileno=listener.detach()
    )


def _server(
    listener: socket.socket, logs_dir: Path, completion: Callable[..., Any] | None
) -> tuple[_Server, str]:
    """A server for ``listener``, and the URL that opens its page."""
    port = listener.getsockname()[1]
    token = secrets.token_urlsafe(TOKEN_BYTES)
    state = ServerState(logs_dir=logs_dir, completion=completion)
    app = create_app(token=token, port=port, state=state)
    config = uvicorn.Config(app, timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS)
    return _Server(config, state), f"http://{HOST}:{port}/#token={token}"


def serve(
    *,
    port: int = 0,
    logs_dir: Path = LOGS_DIR,
    completion: Callable[..., Any] | None = None,
) -> tuple[str, Callable[[], None]]:
    """Start the server on a thread of its own. Returns the URL that opens the page,
    with this launch's token, and a function that stops the server, returning once its
    runs and asks have ended, at most ``WORKERS_EXIT_SECONDS`` after its connections
    have closed."""
    listener = listen(port)
    server, url = _server(listener, logs_dir, completion)
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()

    def stop() -> None:
        server.should_exit = True
        thread.join()

    return url, stop


def launch(listener: socket.socket, *, logs_dir: Path, browser: bool) -> None:
    """Serve the app on ``listener`` until stopped, as ``r3con app`` does: print the
    URL that opens the page, with this launch's token, and where its run folders go,
    and with ``browser``, open the page."""
    configure_logging()
    server, url = _server(listener, logs_dir, None)
    print(f"r3con app: {url}", flush=True)
    print(f"r3con app: runs go to {logs_dir}", flush=True)
    if browser:
        webbrowser.open(url)
    # uvicorn shuts down on Ctrl-C, then raises it again, as its own run() expects.
    with contextlib.suppress(KeyboardInterrupt):
        server.run(sockets=[listener])
    if server.force_exit:
        # Quit now: at exit, Python would join r3con's document threads (a
        # ThreadPoolExecutor's), and so wait for their calls in flight.
        os._exit(FORCED_EXIT_CODE)
