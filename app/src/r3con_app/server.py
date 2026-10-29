"""The local server between r3con's app and r3con.

It listens on 127.0.0.1 only and serves the page at ``/`` and a JSON API under
``/api/``. A request must name this server in ``Host``, and in ``Origin`` when it sends
one (403 otherwise). Every request but the page's must carry the launch's token as
``Authorization: Bearer <token>`` (401 otherwise); the page gets the token from the
opened URL's fragment, ``#token=...``.

- ``GET /api/keys``: which providers have a key, stored here or in the environment.
  ``POST`` stores one (``{provider, key}``, or ``{provider: "other", endpoint, key?}``),
  ``DELETE`` forgets one (``{provider}``); both answer as ``GET`` does. Keys stay in
  this process's memory: the server never sends, logs or writes one, and masks any it
  finds in an error before passing the error on.
- ``POST /api/runs`` (``{question, model, documents: [text, ...]}``) starts
  ``r3con.run`` and answers ``{run_id}``; ``GET /api/runs/{run_id}/events`` streams its
  events as ``text/event-stream``, from the first, until ``run_finished`` or
  ``run_failed``.
- ``POST /api/runs/{run_id}/asks`` (``{question, model}``) starts ``r3con.ask`` over a
  run that has started and answers ``{ask_id}``; ``GET /api/asks/{ask_id}/events``
  streams its events until ``ask_finished`` or ``ask_failed``.

A failure answers ``{"error": "..."}``, except a body over the size limit, which gets
Starlette's plain-text 413. A run or ask that r3con refuses before it starts ends its
stream with a failure event whose ``stage`` is ``null``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import json
import os
import secrets
import socket
import threading
import webbrowser
from collections.abc import AsyncIterator, Callable, Iterable
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
from r3con.logging_setup import configure_logging, get_logger
from r3con.pipeline import _one_line
from r3con.runs import EventRecorder
from r3con.runtime.llm import quiet_litellm

Provider = Literal["openai", "anthropic", "gemini"]
Kind = Literal["run", "ask"]

HOST = "127.0.0.1"
PAGE = Path(__file__).with_name("index.html")
LOGS_DIR = platformdirs.user_data_path("r3con") / "runs"
TOKEN_BYTES = 32
ID_BYTES = 8
MAX_BODY_BYTES = 64 * 1024 * 1024
SHUTDOWN_GRACE_SECONDS = 1
KEY_VARIABLES: dict[Provider, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "gemini": "GEMINI_API_KEY",
}
KEY_MASK = "[key]"
ENDINGS = frozenset({"run_finished", "run_failed", "ask_finished", "ask_failed"})
PUBLIC_PATHS = frozenset({"/"})
FORBID_EXTRA = ConfigDict(extra="forbid")

logger = get_logger("app")
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


class RunRequest(BaseModel):
    model_config = FORBID_EXTRA
    question: Text
    model: Text
    documents: list[Text] = Field(min_length=1)


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
    endpoint: Annotated[str, Field(pattern=r"^https?://\S+$")]
    key: Secret | None = None


class ForgetKey(BaseModel):
    model_config = FORBID_EXTRA
    provider: Provider | Literal["other"]


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
    logs_dir: Path
    completion: Callable[..., Any] | None
    keys: dict[Provider, SecretStr] = field(default_factory=dict)
    other: OtherEndpoint | None = None
    jobs: dict[Kind, dict[str, Job]] = field(
        default_factory=lambda: {"run": {}, "ask": {}}
    )


def _state(request: Request) -> ServerState:
    return request.app.state.server


async def _read(request: Request, parse: Callable[[bytes], Parsed]) -> Parsed:
    try:
        return parse(await request.body())
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


def _held_keys(state: ServerState) -> tuple[str, ...]:
    """Every key this server could send, stored here or in its environment, longest
    first, so that a key inside another is masked whole."""
    stored = list(state.keys.values())
    if state.other is not None and state.other.key is not None:
        stored.append(state.other.key)
    environment = (os.environ.get(variable, "") for variable in KEY_VARIABLES.values())
    keys = {key.get_secret_value() for key in stored}
    keys.update(key for key in environment if key.strip())
    return tuple(sorted(keys, key=len, reverse=True))


def _masked(text: str, keys: Iterable[str]) -> str:
    for key in keys:
        text = text.replace(key, KEY_MASK)
    return text


def _provider(model: str) -> str:
    """``model``'s provider as litellm places it, or ``"other"`` if it can't."""
    try:
        with quiet_litellm():
            return litellm.get_llm_provider(model)[1]
    except BadRequestError:
        return "other"


def _model_options(state: ServerState, model: str) -> dict[str, Any]:
    """What r3con is called with for ``model``: the key stored for its provider, or for
    any other provider the Other endpoint and its key. A key not stored here is passed
    as ``None``, so that litellm reads the environment."""
    provider, other = _provider(model), state.other
    if provider in KEY_VARIABLES:
        key, base = state.keys.get(provider), None
    elif other is not None:
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


def _relay(
    loop: asyncio.AbstractEventLoop,
    job: Job,
    keys: tuple[str, ...],
    event: dict[str, Any],
) -> None:
    """Hand ``event`` to ``job`` on the server's loop, with no key in its error; once
    the loop has closed, as it has when the server stops mid-run, drop it."""
    if "error" in event:
        event["error"] = _masked(event["error"], keys)
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
    keys: tuple[str, ...],
) -> None:
    on_event = partial(_relay, loop, job, keys)
    try:
        call(on_event=on_event)
    except Exception as error:  # noqa: BLE001 — logged below, as one masked line
        # No traceback: r3con writes it to the failed stage's error.txt.
        line = _masked(_one_line(error), keys)
        logger.error("%s: %s", failed, line)
        EventRecorder(on_event, None)(failed, stage=None, error=line)


def _start(state: ServerState, kind: Kind, call: Callable[..., Any]) -> str:
    """Run ``call`` on a thread of its own, its events landing in a new job of
    ``kind``; returns the job's id."""
    job_id = secrets.token_hex(ID_BYTES)
    state.jobs[kind][job_id] = job = Job()
    args = (job, asyncio.get_running_loop(), call, f"{kind}_failed", _held_keys(state))
    worker = threading.Thread(
        target=_work, args=args, name=f"r3con-app-{job_id}", daemon=True
    )
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
    started = run.events[0] if run.events else {}
    if started.get("type") != "run_started":
        raise HTTPException(
            409, "This run has not started, so it has no folder to ask over."
        )
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
    logs_dir: Path = LOGS_DIR,
    completion: Callable[..., Any] | None = None,
    max_body_bytes: int = MAX_BODY_BYTES,
) -> ASGIApp:
    """The server's ASGI app, for requests to 127.0.0.1 or localhost on ``port`` that
    carry ``token``. Run folders go in ``logs_dir``; ``completion`` is passed to r3con
    as its own ``completion`` is."""
    app = Starlette(
        routes=[
            Route("/", page),
            Route("/api/keys", keys, methods=["GET"]),
            Route("/api/keys", save_key, methods=["POST"]),
            Route("/api/keys", forget_key, methods=["DELETE"]),
            Route("/api/runs", start_run, methods=["POST"]),
            Route("/api/runs/{job_id}/events", partial(events, kind="run")),
            Route("/api/runs/{run_id}/asks", start_ask, methods=["POST"]),
            Route("/api/asks/{job_id}/events", partial(events, kind="ask")),
        ],
        exception_handlers={HTTPException: _error},
        max_body_size=max_body_bytes,
    )
    app.state.server = ServerState(logs_dir=logs_dir, completion=completion)
    return guard(app, token=token, port=port)


def _server(
    port: int, logs_dir: Path, completion: Callable[..., Any] | None
) -> tuple[uvicorn.Server, socket.socket, str]:
    """A server for a socket already listening on ``port`` (0 for a free one), and the
    URL that opens its page."""
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, port))
    listener.listen()
    port = listener.getsockname()[1]
    token = secrets.token_urlsafe(TOKEN_BYTES)
    app = create_app(token=token, port=port, logs_dir=logs_dir, completion=completion)
    config = uvicorn.Config(app, timeout_graceful_shutdown=SHUTDOWN_GRACE_SECONDS)
    return uvicorn.Server(config), listener, f"http://{HOST}:{port}/#token={token}"


def serve(
    *,
    port: int = 0,
    logs_dir: Path = LOGS_DIR,
    completion: Callable[..., Any] | None = None,
) -> tuple[str, Callable[[], None]]:
    """Start the server on a thread of its own. Returns the URL that opens the page,
    with this launch's token, and a function that stops the server."""
    server, listener, url = _server(port, logs_dir, completion)
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
        "--logs-dir",
        type=Path,
        default=LOGS_DIR,
        help="where run folders go (default: %(default)s)",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="print the URL, but don't open it"
    )
    args = parser.parse_args(argv)
    configure_logging()
    server, listener, url = _server(args.port, args.logs_dir, None)
    print(f"r3con app: {url}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    # uvicorn shuts down on Ctrl-C, then raises it again, as its own run() expects.
    with contextlib.suppress(KeyboardInterrupt):
        server.run(sockets=[listener])
