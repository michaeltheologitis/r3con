"""The local server between r3con's app and r3con.

It listens on 127.0.0.1 only and serves the page at ``/`` and a JSON API under
``/api/``. A request must name this server in ``Host``, and in ``Origin`` when it sends
one (403 otherwise). Every request but the page's must carry the launch's token as
``Authorization: Bearer <token>`` (401 otherwise); the page gets the token from the
opened URL's fragment, ``#token=...``.

- ``GET /api/keys``: which providers have a key, stored here or in the environment.
  ``POST`` stores one (``{provider, key}``, or ``{provider: "other", endpoint, key?}``),
  ``DELETE`` forgets one (``{provider}``); both answer as ``GET`` does. Keys stay in
  this process's memory: the server never sends, logs or writes one.
- ``POST /api/runs`` (``{question, model, documents: [{name, text}]}``) starts
  ``r3con.run`` and answers ``{run_id}``; ``GET /api/runs/{run_id}/events`` streams its
  events as ``text/event-stream``, from the first, until ``run_finished`` or
  ``run_failed``.
- ``POST /api/runs/{run_id}/asks`` (``{question, model}``) starts ``r3con.ask`` over a
  finished run and answers ``{ask_id}``; ``GET /api/asks/{ask_id}/events`` streams its
  events until ``ask_finished`` or ``ask_failed``.

A failure answers ``{"error": "..."}``. A run or ask that r3con refuses before it starts
ends its stream with a failure event whose ``stage`` is ``null``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import datetime
import hmac
import json
import logging
import os
import secrets
import socket
import threading
import webbrowser
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

import platformdirs
import uvicorn
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    TypeAdapter,
    ValidationError,
)
from pydantic_core import PydanticCustomError
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from r3con import r3con

HOST = "127.0.0.1"
# Where the page is in a source checkout; a bundled app points this elsewhere.
PAGE = Path(__file__).resolve().parents[2] / "web" / "index.html"
DATA_DIR = Path(platformdirs.user_data_dir("r3con"))
RUNS_FOLDER = "runs"
TOKEN_BYTES = 32
ID_BYTES = 8
MAX_BODY_BYTES = 64 * 1024 * 1024
SHUTDOWN_GRACE_SECONDS = 1
KEY_VARIABLES = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
ENDINGS = frozenset({"run_finished", "run_failed", "ask_finished", "ask_failed"})
PUBLIC_PATHS = frozenset({"/"})
FORBID_EXTRA = ConfigDict(extra="forbid")

logger = logging.getLogger(__name__)
Parsed = TypeVar("Parsed")


def _not_blank(text: str) -> str:
    if not text.strip():
        raise PydanticCustomError("blank", "must not be blank")
    return text


def _secret_not_blank(secret: SecretStr) -> SecretStr:
    _not_blank(secret.get_secret_value())
    return secret


Text = Annotated[str, AfterValidator(_not_blank)]
Secret = Annotated[SecretStr, AfterValidator(_secret_not_blank)]


class Document(BaseModel):
    model_config = FORBID_EXTRA
    name: Text
    text: Text


class RunRequest(BaseModel):
    model_config = FORBID_EXTRA
    question: Text
    model: Text
    documents: list[Document] = Field(min_length=1)


class AskRequest(BaseModel):
    model_config = FORBID_EXTRA
    question: Text
    model: Text


class ProviderKey(BaseModel):
    model_config = FORBID_EXTRA
    provider: Literal["openai", "anthropic", "gemini"]
    key: Secret


class OtherEndpoint(BaseModel):
    model_config = FORBID_EXTRA
    provider: Literal["other"]
    endpoint: Annotated[str, Field(pattern=r"^https?://\S+$")]
    key: Secret | None = None


class ForgetKey(BaseModel):
    model_config = FORBID_EXTRA
    provider: Literal["openai", "anthropic", "gemini", "other"]


KEY_BODY: TypeAdapter[ProviderKey | OtherEndpoint] = TypeAdapter(
    Annotated[ProviderKey | OtherEndpoint, Field(discriminator="provider")]
)


@dataclass
class Job:
    """One run's or ask's events so far. Only the event loop's thread touches it."""

    events: list[dict[str, Any]] = field(default_factory=list)
    wake: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class ServerState:
    data_dir: Path
    completion: Callable[..., Any] | None
    keys: dict[str, SecretStr] = field(default_factory=dict)
    other: OtherEndpoint | None = None
    runs: dict[str, Job] = field(default_factory=dict)
    asks: dict[str, Job] = field(default_factory=dict)


def _state(request: Request) -> ServerState:
    return request.app.state.server


async def _read(request: Request, parse: Callable[[bytes], Parsed]) -> Parsed:
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY_BYTES:
            raise HTTPException(
                413, f"The request body is over {MAX_BODY_BYTES:,} bytes."
            )
    try:
        return parse(bytes(body))
    except ValidationError as error:
        raise HTTPException(422, _explain(error)) from None


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


def _credentials(state: ServerState, model: str) -> tuple[str | None, str | None]:
    """The ``api_key`` and ``api_base`` for ``model``, by its litellm provider prefix:
    ``None`` for a key not stored here, so that litellm reads the environment."""
    provider = model.partition("/")[0]
    if provider in KEY_VARIABLES:
        key = state.keys.get(provider)
        return (None if key is None else key.get_secret_value()), None
    other = state.other
    if other is None:
        return None, None
    key = None if other.key is None else other.key.get_secret_value()
    return key, other.endpoint


def _add(job: Job, event: dict[str, Any]) -> None:
    job.events.append(event)
    job.wake.set()
    job.wake = asyncio.Event()


def _ended(job: Job) -> bool:
    return bool(job.events) and job.events[-1]["type"] in ENDINGS


def _add_failure(job: Job, failed: str, error: Exception) -> None:
    if _ended(job):
        return
    line = " ".join(f"{type(error).__name__}: {error}".split())
    now = datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds")
    _add(job, {"type": failed, "time": now, "stage": None, "error": line})


def _work(
    job: Job, loop: asyncio.AbstractEventLoop, call: Callable[..., Any], failed: str
) -> None:
    try:
        call(on_event=lambda event: loop.call_soon_threadsafe(_add, job, event))
    except Exception as error:
        logger.exception("r3con raised in a worker thread")
        loop.call_soon_threadsafe(_add_failure, job, failed, error)


def _start(call: Callable[..., Any], failed: str) -> Job:
    """Run ``call`` on a thread of its own, its events landing in the returned job."""
    job = Job()
    loop = asyncio.get_running_loop()
    worker = threading.Thread(target=_work, args=(job, loop, call, failed), daemon=True)
    worker.start()
    return job


async def _stream(job: Job) -> AsyncIterator[str]:
    sent = 0
    while True:
        wake = job.wake
        while sent < len(job.events):
            yield f"data: {json.dumps(job.events[sent])}\n\n"
            sent += 1
        if _ended(job):
            return
        await wake.wait()


def _job(jobs: dict[str, Job], job_id: str, kind: str) -> Job:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"There is no {kind} {job_id}.")
    return job


async def page(request: Request) -> Response:
    return FileResponse(
        PAGE, headers={"Content-Security-Policy": "frame-ancestors 'none'"}
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


async def start_run(request: Request) -> Response:
    state = _state(request)
    body = await _read(request, RunRequest.model_validate_json)
    api_key, api_base = _credentials(state, body.model)
    run = partial(
        r3con.run,
        body.question,
        [document.text for document in body.documents],
        model=body.model,
        api_key=api_key,
        api_base=api_base,
        completion=state.completion,
        logs_dir=state.data_dir / RUNS_FOLDER,
    )
    run_id = secrets.token_hex(ID_BYTES)
    state.runs[run_id] = _start(run, "run_failed")
    return JSONResponse({"run_id": run_id}, status_code=202)


async def run_events(request: Request) -> Response:
    run = _job(_state(request).runs, request.path_params["run_id"], "run")
    return StreamingResponse(_stream(run), media_type="text/event-stream")


async def start_ask(request: Request) -> Response:
    state = _state(request)
    run = _job(state.runs, request.path_params["run_id"], "run")
    body = await _read(request, AskRequest.model_validate_json)
    last = run.events[-1] if run.events else {}
    if last.get("type") != "run_finished":
        raise HTTPException(409, "An ask needs a finished run, and this one is not.")
    api_key, api_base = _credentials(state, body.model)
    ask = partial(
        r3con.ask,
        last["run_dir"],
        body.question,
        model=body.model,
        api_key=api_key,
        api_base=api_base,
        completion=state.completion,
    )
    ask_id = secrets.token_hex(ID_BYTES)
    state.asks[ask_id] = _start(ask, "ask_failed")
    return JSONResponse({"ask_id": ask_id}, status_code=202)


async def ask_events(request: Request) -> Response:
    ask = _job(_state(request).asks, request.path_params["ask_id"], "ask")
    return StreamingResponse(_stream(ask), media_type="text/event-stream")


async def _error(request: Request, error: Exception) -> Response:
    if not isinstance(error, HTTPException):
        raise error
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
    if scope["path"] in PUBLIC_PATHS:
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
    data_dir: Path = DATA_DIR,
    completion: Callable[..., Any] | None = None,
) -> ASGIApp:
    """The server's ASGI app, for requests to 127.0.0.1 or localhost on ``port`` that
    carry ``token``. Runs go under ``data_dir``'s ``runs/``; ``completion`` is passed to
    r3con as its own ``completion`` is."""
    app = Starlette(
        routes=[
            Route("/", page),
            Route("/api/keys", keys, methods=["GET"]),
            Route("/api/keys", save_key, methods=["POST"]),
            Route("/api/keys", forget_key, methods=["DELETE"]),
            Route("/api/runs", start_run, methods=["POST"]),
            Route("/api/runs/{run_id}/events", run_events),
            Route("/api/runs/{run_id}/asks", start_ask, methods=["POST"]),
            Route("/api/asks/{ask_id}/events", ask_events),
        ],
        exception_handlers={HTTPException: _error},
    )
    app.state.server = ServerState(data_dir=data_dir, completion=completion)
    return guard(app, token=token, port=port)


def _server(
    port: int, data_dir: Path, completion: Callable[..., Any] | None
) -> tuple[uvicorn.Server, socket.socket, str]:
    """A server for a socket already listening on ``port`` (0 for a free one), and the
    URL that opens its page."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, port))
    listener.listen()
    port = listener.getsockname()[1]
    token = secrets.token_urlsafe(TOKEN_BYTES)
    app = create_app(token=token, port=port, data_dir=data_dir, completion=completion)
    config = uvicorn.Config(app, timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS)
    return uvicorn.Server(config), listener, f"http://{HOST}:{port}/#token={token}"


def serve(
    *,
    port: int = 0,
    data_dir: Path = DATA_DIR,
    completion: Callable[..., Any] | None = None,
) -> tuple[str, Callable[[], None]]:
    """Start the server on a thread of its own. Returns the URL that opens the page,
    with this launch's token, and a function that stops the server."""
    server, listener, url = _server(port, data_dir, completion)
    thread = threading.Thread(
        target=server.run, kwargs={"sockets": [listener]}, daemon=True
    )
    thread.start()

    def stop() -> None:
        server.should_exit = True
        thread.join()

    return url, stop


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="r3con-app", description="Serve r3con's app on 127.0.0.1."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="the port to listen on (default: a free one)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="where runs are written, under runs/ (default: %(default)s)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="print the URL, but don't open it"
    )
    args = parser.parse_args(argv)
    server, listener, url = _server(args.port, args.data_dir, None)
    print(f"r3con app: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    # uvicorn shuts down on Ctrl-C, then raises it again, as its own run() expects.
    with contextlib.suppress(KeyboardInterrupt):
        server.run(sockets=[listener])
