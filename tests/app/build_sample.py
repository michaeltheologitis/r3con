"""Rebuild the app's recorded sample from its run folder: from the repo root,
``uv run python -m tests.app.build_sample`` rewrites app.js's ``SAMPLE`` and
``RECORDED_ASKS`` blocks with what it builds, and ``test_build_sample.py`` fails while
they differ.

The run folder, ``tests/app/sample/20261105T173546Z_bb4f0c37/``, is a real run, copied
unedited from the ``logs/`` r3con wrote it to:

- The run: ``r3con run QUESTION examples/memos`` with r3con 0.6.4 (main at 20964ea), its
  default config and no overrides: model ``openai/gpt-6-luna``, two relevance rounds,
  direct reasoning, empty ``params``, and that config's prompt pins (relevance v2 and
  v3, schema v1, parsing v3, direct v3). QUESTION: "Which maintenance contractor's sites
  logged the most equipment incidents in Q3 in total, and how many?"
- Its ``asks/``: ``r3con.ask(run_folder, question, model=model)``, in this order: "Which
  sites does Halloran Services maintain?" on ``openai/gpt-6-luna``, the same on
  ``openai/gpt-6.1-terra``, then "How many incidents did Merrow Facilities Group's sites
  log?" on ``openai/gpt-6-luna``.

Building calls no model: each document's name, text and size come from
``examples/memos``, each answer's provenance is ``r3con.trace``'s for its folder, and
every other value is read from the folder as it is. Each quote names the snippet it is
from, so the app can match it to the trail. Every document in this run was read whole,
and this handles only that. A sample with a document read in parts would need part
handling: joining its parts' summaries as the app does, counting its ``<id>.<k>``
snippets as the document, and fitting its several calls into the page's one call per
node.

To record a new sample, run and ask as above, put the new run folder where this one is,
keeping its name (``r3con.trace`` finds an ask's run by it), set ``RUN_FOLDER``, and run
this.
"""

import os

# Before litellm is imported: otherwise it fetches its price map from GitHub.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

import json
import re
from pathlib import Path
from typing import Any

import r3con

REPO = Path(__file__).resolve().parents[2]
RUN_FOLDER = Path(__file__).resolve().parent / "sample" / "20261105T173546Z_bb4f0c37"
MEMOS = REPO / "examples" / "memos"
APP_JS = REPO / "src" / "r3con" / "app" / "app.js"

ROUNDS_DRAWN = 2
# What the page reads of each quote: its text, and where it is in its document, in code points.
QUOTE_FIELDS = ("text", "start", "end")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def provenance(folder: Path, cited: dict[str, list[str]]) -> dict[str, Any]:
    """The provenance of the answer saved in ``folder``, as r3con.trace gives it."""
    traced = r3con.trace(folder)
    citations = {"summary_ids": traced["summary_ids"], "parse_ids": traced["parse_ids"]}
    assert citations == cited, f"{folder.name}: trace reads the saved citations"
    return dict(traced)


def summary(snippet: dict[str, Any], ids: dict[str, int]) -> dict[str, Any]:
    """One document's summary in one round, each quote naming the snippet it is from."""
    return {
        "summary": snippet["summary"],
        "provenance": [ids[cited] for cited in snippet["provenance"]],
        "quotes": [
            {**{f: q[f] for f in QUOTE_FIELDS}, "snippet": snippet["id"]}
            for q in snippet["quotes"]
        ],
    }


def round_summaries(
    snippets: list[dict[str, Any]], ids: dict[str, int]
) -> list[dict[str, Any]]:
    """Each document's summary in one round, in document order."""
    by_document = {ids[snippet["id"]]: snippet for snippet in snippets}
    assert len(snippets) == len(by_document) == len(ids), (
        "one summary per document each round"
    )
    return [summary(by_document[doc], ids) for doc in sorted(by_document)]


def events_of(run: Path, kind: str) -> list[dict[str, Any]]:
    lines = (run / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [event for event in map(json.loads, lines) if event["type"] == kind]


def document_parses(run: Path, ids: dict[str, int]) -> list[dict[str, Any]]:
    """Each document's parse, every field, empty ones included, from its parsing_finished event."""
    finished = events_of(run, "parsing_finished")
    parses = {ids[event["doc_id"]]: event["parse"] for event in finished}
    assert len(finished) == len(parses) == len(ids), (
        "one parsing_finished event per document"
    )
    return [parses[doc] for doc in sorted(parses)]


def run_totals(run: Path) -> dict[str, Any]:
    """The run's tokens and cost, from its run_finished event."""
    finished = events_of(run, "run_finished")
    assert len(finished) == 1, "one run_finished event"
    return {"tokens": finished[0]["tokens"]["total"], "usd": finished[0]["usd"]}


def call_costs(run: Path) -> dict[str, Any]:
    """Each model call's step, tokens and cost, keyed by the page's node for it."""
    relevance = [
        (f"{c['kind'].split('-')[1]}-{c['doc']}", c)
        for c in load(run / "relevance/calls.json")
    ]
    parsing = [
        (f"parsed-{c['doc']}", c) for c in load(run / "structuring/parsing/calls.json")
    ]
    schema = [("schema", c) for c in load(run / "structuring/schema/calls.json")]
    answer = [("answer", c) for c in load(run / "reasoning/calls.json")]
    keyed = [
        (key, {"step": c["step"], "tokens": c["tokens"]["total"], "usd": c["usd"]})
        for key, c in relevance + parsing + schema + answer
    ]
    calls = dict(keyed)
    assert len(calls) == len(keyed), "one call per node"
    return calls


def build_sample(run: Path, memos: Path) -> dict[str, Any]:
    ids = load(run / "ids.json")
    manifest = load(run / "manifest.json")
    relevance = load(run / "relevance/result.json")
    schema = load(run / "structuring/schema/result.json")
    parsing = load(run / "structuring/parsing/result.json")
    reasoning = load(run / "reasoning/result.json")

    files = sorted(memos.glob("*.txt"))
    docs = [
        {
            "name": p.name,
            "text": p.read_text(encoding="utf-8"),
            "size": p.stat().st_size,
        }
        for p in files
    ]
    assert len(docs) == len(ids) == manifest["n_docs"], "one memo per document id"
    assert relevance["n_rounds"] == ROUNDS_DRAWN, (
        "the page draws exactly two relevance rounds"
    )
    first, second = (round_summaries(r["snippets"], ids) for r in relevance["rounds"])
    parses = document_parses(run, ids)
    return {
        "question": manifest["task"],
        "model": manifest["config"]["model"],
        "version": manifest["r3con_version"],
        "runFolder": manifest["run_folder"],
        "created": manifest["created"],
        "docs": docs,
        "ids": sorted(ids, key=ids.__getitem__),
        "r1": first,
        "r2": second,
        "schema": {"code": schema["schema_code"], "thought": schema["thought"]},
        "parsed": [
            {"parse": parse, "provenance": [ids[c] for c in cited]}
            for parse, cited in zip(parses, parsing["provenance"], strict=True)
        ],
        "answer": {
            "text": reasoning["answer"],
            "provenance": provenance(run, reasoning["provenance"]),
        },
        "calls": call_costs(run),
        "totals": run_totals(run),
    }


def ask_entry(folder: Path) -> dict[str, Any]:
    manifest = load(folder / "manifest.json")
    result = load(folder / "reasoning" / "result.json")
    return {
        "question": manifest["task"],
        "model": manifest["litellm_model"],
        "folder": folder.name,
        "created": manifest["created"],
        "text": result["answer"],
        "provenance": provenance(folder, result["provenance"]),
        "totals": {
            "tokens": result["totals"]["tokens"]["total"],
            "usd": result["totals"]["usd"],
        },
    }


def js_block(name: str, value: Any) -> str:
    """``value`` as app.js's ``const`` named ``name``."""
    blob = json.dumps(value, ensure_ascii=False, indent=1)
    return f"  const {name} = {blob};\n"


def built_blocks() -> dict[str, str]:
    """app.js's SAMPLE and RECORDED_ASKS blocks, built from RUN_FOLDER; its asks oldest first."""
    asks = [ask_entry(folder) for folder in sorted((RUN_FOLDER / "asks").iterdir())]
    return {
        "SAMPLE": js_block("SAMPLE", build_sample(RUN_FOLDER, MEMOS)),
        "RECORDED_ASKS": js_block("RECORDED_ASKS", asks),
    }


def app_block(app: str, name: str) -> str:
    """The block app.js holds for the ``const`` named ``name``: its first line, the indented
    lines after it, and the ``};`` or ``];`` line that closes it."""
    found = re.findall(
        rf"^  const {name} = [{{\[]\n(?: .*\n)*[}}\]];\n", app, re.MULTILINE
    )
    assert len(found) == 1, f"app.js holds one {name} block"
    return found[0]


def main() -> None:
    app = APP_JS.read_text(encoding="utf-8")
    for name, block in built_blocks().items():
        app = app.replace(app_block(app, name), block)
    APP_JS.write_text(app, encoding="utf-8")


if __name__ == "__main__":
    main()
