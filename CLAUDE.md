# r3con

Context representation for large-scale reasoning: for a question whose evidence is scattered across many documents, r3con builds a task-specific representation of them (relevance, then structure) and reasons over it with any model LiteLLM supports. It is published on PyPI as `r3context`; the import package is `r3con`.

Read `../coding-skills/GUIDE.md` and follow it; in the main session a hook has already printed it.
This project's Notion page (todos, changelog, notes): https://app.notion.com/p/40362fb22237814d9e8ccbaf2d812c8c

## Goals and non-goals

- Answer one question over many documents in one call: `r3con.run(question, documents)` or `r3con run`.
- No model call reads more than one document's text, or a part of it when the request would not fit the model's window; the other documents reach it as relevance notes and, in reasoning, as parsed records.
- Show the work: by default a run writes every stage's output and model calls to `./logs/<run>/`, because the answer alone rarely says whether to trust it.
- Not a provider layer: models, connections and credentials are LiteLLM's, and r3con adds no model registry or credential variable of its own.
- Not the paper's evaluation: benchmarks, baselines and evaluation code live in [r3con-evaluation](https://github.com/michaeltheologitis/r3con-evaluation); `evals/` is only a small regression check.

## Checks

- `uv run ruff check`, `uv run ruff format --check` and `uv run pyright`: lint, format and types.
- `npx --yes --package typescript@7.0.2 tsc -p tsconfig.json`: the app script's types, from its JSDoc.
- `uv run pytest`: the unit tests, with sockets disabled and live tests deselected, so nothing reaches the network. CI also runs them on Python 3.11 to 3.14, at the lowest allowed dependency versions, and against the built wheel.
- `uv run pytest -m "live and not experiment"`: an end-to-end run against OpenAI, billed to `OPENAI_API_KEY` and skipped without it. The unit tests fake the model, so run it when a change affects what the model is sent or how it is called. A manual CI dispatch runs it by default.
- `uv run pytest -m experiment`: the experiments, measured live runs that cost far more. Run them only when Michael asks (in CI, a dispatch with `experiment` on); plain `-m live` includes them.
- `NODE_PATH=$(npm root -g) node tests/app/check.mjs`: the app's page checks, in headless Chromium through Playwright, which they find via `NODE_PATH` (elsewhere, run `npm i -g playwright && npx playwright install chromium` once). Their live area serves the page with `tests/app/fake_server.py`, a scripted model behind the real server, through `uv`. Run them for any change to `src/r3con/app/`.
- `uv run python -m evals`: the eval, r3con on 75 pinned Loong and CorpusQA tasks, judged by Luna, about $3 and three minutes a run on `OPENAI_API_KEY` ([`evals/README.md`](evals/README.md)). Run it for a change that could move answers a lot, and when Michael asks; the line it writes to `evals/results.jsonl` goes in that change's PR.

## Conventions

- `src/r3con/runtime/python_executor.py` is a modified copy of smolagents' `local_python_executor.py` (v1.25.0, Apache-2.0), and ruff, pyright and the guide's style rules skip it. Keep changes to it minimal, and list each one in its header and in `NOTICE`.
- A prompt version (`src/r3con/prompts/<stage>/vN.yaml`) is never edited once merged, and never reused after a revert: take the next number not listed by `git log --format= --name-only -- src/r3con/prompts/<stage>/`, and bump its pin in `src/r3con/configs/default.yaml`.
- Versions: release as patch versions (0.5.1, 0.5.2, …) and move slowly toward 1.0.0; bump the minor only when Michael asks.
