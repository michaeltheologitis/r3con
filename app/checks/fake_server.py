"""Serve the app with a scripted model, for ``check.mjs``: ``fake_server.py LOGS_DIR``
prints the URL that opens the page, with its token, then serves until killed.

The model answers every stage as one would, citing real ids, so a run draws its whole
panel: each summary is its document's first line that is not ``QUIET`` (none if all
are), and each parse its first line; round 2 cites the first summary it is shown, each
parse its own document's note, and the answer the first two documents' summaries and
parses. It refuses a document over ``PART_CHARS`` as too long, so that r3con reads it
in parts. A question holding ``FAILS`` makes every call raise, and one holding
``HANGS`` makes every call wait until the server is killed.
"""

import os

# Before litellm is imported: otherwise it fetches its price map from GitHub.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import contextlib
import json
import re
import sys
import threading
from pathlib import Path
from typing import Any

import litellm
from litellm.exceptions import ContextWindowExceededError
from r3con_app import serve

litellm.suppress_debug_info = True

FAILS = "(the model is down)"
HANGS = "(the model hangs)"
QUIET = "(nothing relevant here)"
PART_CHARS = 2000
# The schema stage asks for no response format, as round 1 of relevance does; its user
# message is the only one that starts with this.
SCHEMA_REQUEST_START = "Input:"
READS_A_DOCUMENT = frozenset({"round 1", "RelevanceSummary", "ParseReply"})
ANSWER = "The check's scripted model answers this."
CITED_PER_VIEW = 2
USAGE = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
SCHEMA_REPLY = """Thought: one row per memo.
<schema>
class Memo(BaseModel):
    title: str


class Parse(BaseModel):
    memos: list[Memo]
</schema>"""
PARSE_HEADING = re.compile(r"\ADocument ID: (\w+)\n\n")
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


def reply_name(request: dict[str, Any]) -> str:
    """The reply ``request`` asks for: its response format's name, or without one, the
    schema or a first-round summary."""
    response_format = request.get("response_format")
    if response_format is not None:
        return response_format["json_schema"]["name"]
    user = request["messages"][-1]["content"]
    return "schema" if user.startswith(SCHEMA_REQUEST_START) else "round 1"


def content(request: dict[str, Any]) -> str:
    """The reply to ``request``, by the stage that sent it."""
    user = request["messages"][-1]["content"]
    name = reply_name(request)
    if name == "schema":
        return SCHEMA_REPLY
    if name == "round 1":
        return summary_of(user)
    schema = request["response_format"]["json_schema"]["schema"]
    if name == "RelevanceSummary":
        cited = citable(schema, "provenance_ids")[:1]
        return json.dumps({"summary": summary_of(user), "provenance_ids": cited})
    if name == "ParseReply":
        heading = PARSE_HEADING.match(user)
        own = [heading[1]] if heading else []
        title = first_line(PARSE_HEADING.sub("", user))
        parse = {"memos": [{"title": title}]}
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
    part = request["messages"][-1]["content"]
    if reply_name(request) in READS_A_DOCUMENT and len(part) > PART_CHARS:
        raise ContextWindowExceededError(
            message=f"{len(part)} characters, over the scripted model's {PART_CHARS}",
            model=request["model"],
            llm_provider="openai",
        )
    choice = {
        "index": 0,
        "finish_reason": "stop",
        "message": {"content": content(request)},
    }
    return litellm.completion(
        **request, mock_response={"choices": [choice], "usage": USAGE}
    )


def main() -> None:
    url, _ = serve(logs_dir=Path(sys.argv[1]), completion=scripted_model)
    print(url, flush=True)
    with contextlib.suppress(KeyboardInterrupt):
        never.wait()


if __name__ == "__main__":
    main()
