"""Serve the app with a scripted model, for ``check.mjs``: from the repo root,
``python -m tests.app.fake_server LOGS_DIR`` prints the URL that opens the page, with
its token, then serves until killed.

The model answers every stage as one would, citing real ids, so a run draws its whole
panel: each summary is its document's first line that is not ``QUIET`` (none if all
are), quoting that line, and each parse its first line as a memo's title, its other
lines nested under it, or nothing for a document holding ``UNPARSED``; each round
after the first cites the first summary it is shown, each parse its own document's
note, and the answer the first two documents' summaries and parses. It refuses a
document over ``PART_CHARS`` as too long, so that r3con reads it in parts.
A question holding ``FAILS`` makes every call raise, and one holding ``HANGS`` makes
every call wait until the server is killed. One holding ``STALLS`` makes every call
wait until ``check.mjs`` writes ``RELEASE`` into LOGS_DIR, one holding ``BREAKS``
makes every parsing call raise until then, and one holding ``OFF_SCHEMA`` makes
every reply with a schema miss it, so r3con sends again.
The server keeps the keys it is given in a credential store of its own, in memory,
never the system's. With ``--save-dialog`` after LOGS_DIR, it saves a tab as the Mac
app does, through a Save dialog, faked: it saves in LOGS_DIR's ``saved``, under the
name it suggests, cancels for a name holding ``CANCELS_SAVE``, and picks a folder
that isn't there for one holding ``NO_ROOM``.
"""

import os

# Before litellm is imported: otherwise it fetches its price map from GitHub.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import contextlib
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

import keyring
import litellm
from litellm.exceptions import ContextWindowExceededError

from r3con.app import serve
from tests.fakes import (
    PARSE_HEADING,
    SCHEMA_REPLY,
    FakeKeyring,
    mock_completion,
    stage_of,
    unscripted,
)

litellm.suppress_debug_info = True

FAILS = "(the model is down)"
HANGS = "(the model hangs)"
STALLS = "(the model stalls)"
BREAKS = "(the parser breaks)"
RELEASE = "release"
RELEASE_POLL_SECONDS = 0.05
OFF_SCHEMA = "(the model answers off schema)"
EMPTY_REPLY = "{}"
QUIET = "(nothing relevant here)"
UNPARSED = "(nothing to parse here)"
PART_CHARS = 2000
READS_A_DOCUMENT = frozenset({"relevance", "parsing"})
ANSWER = "The check's scripted model answers this."
CITED_PER_VIEW = 2
SAVE_DIALOG = "--save-dialog"
SAVED = "saved"
CANCELS_SAVE = "cancel-the-save"
NO_ROOM = "no-room-to-save"
never = threading.Event()


def citable(schema: Any, name: str) -> list[str]:
    """The ids the reply's ``name`` list may hold: its items' ``enum``, wherever the
    reply's JSON schema puts it."""
    if isinstance(schema, list):
        return next(
            (ids for item in schema if (ids := citable(item, name))),
            [],
        )
    if not isinstance(schema, dict):
        return []
    ids = schema.get(name, {}).get("items", {}).get("enum")
    return ids or citable(list(schema.values()), name)


def first_line(text: str) -> str:
    return next((line for line in text.splitlines() if line.strip()), "")


def summary_of(text: str) -> str:
    lines = text.splitlines()
    return next((line for line in lines if line.strip() and QUIET not in line), "")


def content(request: dict[str, Any]) -> str:
    """The reply to ``request``, by the stage that sent it."""
    user = request["messages"][-1]["content"]
    stage = stage_of(request)
    if stage == "schema":
        return SCHEMA_REPLY
    # Coding reasoning, which the checks never run.
    if "response_format" not in request:
        raise unscripted(request)
    schema = request["response_format"]["json_schema"]["schema"]
    if stage == "relevance":
        summary = summary_of(user)
        reply = {
            "summary": summary,
            "quotes": [summary] if summary else [],
            "provenance_ids": citable(schema, "provenance_ids")[:1],
        }
        return json.dumps({field: reply[field] for field in schema["properties"]})
    if stage == "parsing":
        heading = PARSE_HEADING.match(user)
        own = [heading[1]] if heading else []
        text = PARSE_HEADING.sub("", user)
        if UNPARSED in text:
            return json.dumps({"parse": {"memos": []}, "provenance_ids": []})
        lines = [{"text": line} for line in text.splitlines() if line.strip()][1:]
        parse = {"memos": [{"title": first_line(text), "lines": lines}]}
        return json.dumps({"parse": parse, "provenance_ids": own})
    provenance = {
        view: citable(schema, view)[:CITED_PER_VIEW]
        for view in ("summary_ids", "parse_ids")
    }
    return json.dumps({"answer": ANSWER, "provenance_ids": provenance})


def scripted_model(**request: Any) -> Any:
    sent = "\n".join(message["content"] for message in request["messages"])
    if FAILS in sent:
        raise RuntimeError("the check's model is down")
    if HANGS in sent:
        never.wait()
    released = Path(sys.argv[1], RELEASE)
    while STALLS in sent and not released.exists():
        time.sleep(RELEASE_POLL_SECONDS)
    if BREAKS in sent and stage_of(request) == "parsing" and not released.exists():
        raise RuntimeError("the check's parser is down")
    part = request["messages"][-1]["content"]
    if stage_of(request) in READS_A_DOCUMENT and len(part) > PART_CHARS:
        raise ContextWindowExceededError(
            message=f"{len(part)} characters, over the scripted model's {PART_CHARS}",
            model=request["model"],
            llm_provider="openai",
        )
    if OFF_SCHEMA in sent and request.get("response_format"):
        return mock_completion(request, EMPTY_REPLY)
    return mock_completion(request, content(request))


def choose_save_path(name: str) -> Path | None:
    if CANCELS_SAVE in name:
        return None
    saved = Path(sys.argv[1], SAVED)
    saved.mkdir(exist_ok=True)
    return (saved / NO_ROOM if NO_ROOM in name else saved) / name


def main() -> None:
    keyring.set_keyring(FakeKeyring())
    dialog = choose_save_path if SAVE_DIALOG in sys.argv[2:] else None
    url, _ = serve(
        logs_dir=Path(sys.argv[1]),
        completion=scripted_model,
        choose_save_path=dialog,
    )
    print(url, flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        never.wait()


if __name__ == "__main__":
    main()
