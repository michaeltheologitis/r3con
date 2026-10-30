"""The local server between r3con's app and r3con.

It listens on 127.0.0.1 only and serves the page at ``/``, its stylesheet and script
at ``/app.css`` and ``/app.js``, and a JSON API under ``/api/``. A request must name
this server in ``Host``, and in ``Origin`` when it sends one (403 otherwise). Every
request but those for the page and its two files must carry the launch's token as
``Authorization: Bearer <token>`` (401 otherwise); the page gets the token from the
opened URL's fragment, ``#token=...``.

- ``GET /api/keys``: which providers have a key, stored here or in the environment.
  ``POST`` stores one (``{provider, key}``, or ``{provider: "other", endpoint, key?}``),
  ``DELETE`` forgets one (``{provider}``); both answer as ``GET`` does. OpenAI's,
  Anthropic's and Gemini's go to their own models; Other's endpoint and key go to the
  self-hosted providers in ``SELF_HOSTED`` and to names litellm can't place, and any
  other provider gets litellm's own setup from the environment.
- Keys stay in this process's memory, and the server never sends, logs or writes one.
  Each goes to r3con as a run's ``api_key``, which r3con registers before sending it
  and masks in every event's ``error``, in what it writes to the logs dir and in what
  it logs (:mod:`r3con.masking`), the environment's keys too; the server's own log
  line of a failure goes through the same masking.
- ``POST /api/documents/text?name=<file name>``, with a PDF's or a Word file's bytes as
  the body, chosen by the name's ``.pdf`` or ``.docx`` (422 for any other), answers
  ``{text}``: its text as ``r3con.read_documents`` extracts it, which is ``""`` for a
  scan, a file with no text or one that can't be read, as the server's log then says.
  The page sends each PDF and Word file it is given here, and reads other files itself.
- ``POST /api/documents/zip?name=<file name>&limit=<n>``, with a zip's bytes as the
  body, answers ``{files: [{name, size, text}], skipped, unread}``. Its documents are
  the entries ``r3con.read_documents``' folder walk would read, by suffix and with no
  Office lock file, less, by the app's own rule, anything hidden or under
  ``__MACOSX/``. The first ``limit`` of them, in the walk's order, are read as that
  walk reads a file, each named by its path in the zip less a top folder they all
  share, its text ``""`` if it has none. ``skipped`` counts the files left out by
  their suffix, nested zips among them, which are never opened; ``unread``, the
  documents past ``limit``. A zip that is corrupt, encrypted, of more than
  ``MAX_ZIP_ENTRIES`` entries, or whose documents read would unzip to more than
  ``MAX_ZIP_BYTES``, is refused (422).
- ``POST /api/runs`` (``{question, model, documents: [text, ...]}``) starts
  ``r3con.run`` and answers ``{run_id}``; ``GET /api/runs/{run_id}/events`` streams its
  events as ``text/event-stream``, from the first, until ``run_finished`` or
  ``run_failed``.
- ``POST /api/runs/{run_id}/asks`` (``{question, model}``) starts ``r3con.ask`` over a
  run that has ended and answers ``{ask_id}``; ``GET /api/asks/{ask_id}/events``
  streams its events until ``ask_finished`` or ``ask_failed``.
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
# Where macOS's Finder puts each file's metadata in a zip it makes.
MACOS_METADATA_FOLDER = "__MACOSX"
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
SELF_HOSTED = frozenset(
    {"hosted_vllm", "ollama", "ollama_chat", "lm_studio", "llamafile", "custom_openai"}
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


Text = Annotated[str, AfterValidator(_not_blank)]
Secret = Annotated[SecretStr, AfterValidator(_stripped_secret)]


class RunRequest(BaseModel):
    model_config = FORBID_EXTRA
    question: Text
    model: Text
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


class ProviderKey(BaseModel):
    model_config = FORBID_EXTRA
    provider: Provider
    key: Secret


class OtherEndpoint(BaseModel):
    model_config = FORBID_EXTRA
    provider: Literal["other"]
    endpoint: Annotated[
        str, StringConstraints(strip_whitespace=True, pattern=r"^https?://\S+$")
    ]
    key: Secret | None = None


class ForgetKey(BaseModel):
    model_config = FORBID_EXTRA
    provider: Provider | Literal["other"]


KEY_BODY: TypeAdapter[ProviderKey | OtherEndpoint] = TypeAdapter(
    Annotated[ProviderKey | OtherEndpoint, Field(discriminator="provider")]
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
    other: OtherEndpoint | None = None
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


def _key_status(state: ServerState) -> dict[str, Any]:
    status: dict[str, Any] = {
        provider: provider in state.keys or bool(os.environ.get(variable))
        for provider, variable in KEY_VARIABLES.items()
    }
    other = state.other
    status["other"] = (
        None
        if other is None
        else {"endpoint": other.endpoint, "key": other.key is not None}
    )
    return status


def _provider(model: str) -> str | None:
    """``model``'s provider as litellm places it, or ``None`` if it can't."""
    try:
        with quiet_litellm():
            return litellm.get_llm_provider(model)[1]
    except BadRequestError:
        return None


def _model_options(state: ServerState, model: str) -> dict[str, Any]:
    """What r3con is called with for ``model``: the key stored for its provider; for a
    self-hosted provider, or a name litellm can't place, the Other endpoint and its key.
    Anything not stored here is passed as ``None``, so that litellm reads the
    environment."""
    provider, other = _provider(model), state.other
    if provider in KEY_VARIABLES:
        key, base = state.keys.get(provider), None
    elif other is not None and (provider is None or provider in SELF_HOSTED):
        key, base = other.key, other.endpoint
    else:
        key, base = None, None
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
    if isinstance(body, OtherEndpoint):
        state.other = body
    else:
        state.keys[body.provider] = body.key
    return JSONResponse(_key_status(state))


async def forget_key(request: Request) -> Response:
    state = _state(request)
    body = await _read(request, ForgetKey.model_validate_json)
    if body.provider == "other":
        state.other = None
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


def _left_out_unsaid(entry: zipfile.ZipInfo) -> bool:
    """Whether the app leaves ``entry`` out without counting it: Office's lock file, as
    ``r3con run`` does, and, by the app's own rule, which ``r3con run`` doesn't follow,
    a hidden file, or one under a hidden folder or ``__MACOSX``."""
    parts = entry.filename.split("/")
    return (
        parts[-1].startswith(r3con._LOCK_FILE_PREFIX)
        or MACOS_METADATA_FOLDER in parts
        or any(part.startswith(".") for part in parts)
    )


def _below_shared_folder(paths: list[str]) -> list[str]:
    """``paths``, less the one folder at the top that they all share, if they do."""
    tops = {path.split("/", 1)[0] for path in paths}
    if len(tops) != 1 or not all("/" in path for path in paths):
        return paths
    return [path.split("/", 1)[1] for path in paths]


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
        files = [e for e in entries if not e.is_dir() and not _left_out_unsaid(e)]
        documents = sorted(
            (e for e in files if r3con._walk_reads(e.filename.rsplit("/", 1)[-1])),
            key=lambda entry: entry.filename.split("/"),
        )
        read = documents[:limit]
        if any(entry.flag_bits & ZIP_ENCRYPTED_FLAG for entry in read):
            raise HTTPException(422, ENCRYPTED_ZIP)
        if sum(entry.file_size for entry in read) > MAX_ZIP_BYTES:
            raise HTTPException(422, TOO_MANY_BYTES)
        # zipfile stops each entry at the size it claims, and fails its CRC if it lied.
        contents = _refused_if_unreadable(name, lambda: list(map(archive.read, read)))
    names = _below_shared_folder([entry.filename for entry in documents])
    unzipped = [
        {
            "name": shown,
            "size": len(data),
            "text": r3con._read_stream(f"{name}/{entry.filename}", io.BytesIO(data)),
        }
        for entry, shown, data in zip(read, names, contents, strict=False)
    ]
    return {
        "files": unzipped,
        "skipped": len(files) - len(documents),
        "unread": len(documents) - len(read),
    }


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
        **_model_options(state, body.model),
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
        **_model_options(state, body.model),
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
