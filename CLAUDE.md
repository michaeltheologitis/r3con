# r3con

Context representation for large-scale reasoning: for a question whose evidence is scattered across many documents, r3con builds a task-specific representation of them (relevance, then structure) and reasons over it with any model LiteLLM supports. It is published on PyPI as `r3context`; the import package is `r3con`.

Read `../coding-skills/GUIDE.md` and follow it; in the main session a hook has already printed it.
This project's Notion page (todos, changelog, notes): https://app.notion.com/p/40362fb22237814d9e8ccbaf2d812c8c

## Goals and non-goals

- Answer one question over many documents in one call, `r3con.run(question, documents)` or `r3con run`, with any LiteLLM model.
- Each model call reads at most one document, or one part of a document too long for the model's window; other documents reach it only through their relevance notes.
- Show the work: by default a run writes every stage's output and model calls to `./logs/<run>/`, because the answer alone rarely says whether to trust it.
- Not a provider layer: models, connections and credentials are LiteLLM's, and r3con adds no model registry or credential variable of its own.
- Not the paper's evaluation: benchmarks, baselines and evaluation code live in [r3con-evaluation](https://github.com/michaeltheologitis/r3con-evaluation).

## Checks

- `uv run ruff check`, `uv run ruff format --check` and `uv run pyright`: lint, format and types.
- `uv run pytest`: the unit tests, with sockets disabled and live tests deselected, so nothing reaches the network. CI also runs them on Python 3.11 to 3.14, at the lowest allowed dependency versions (`uv run --resolution lowest-direct pytest`), and against the built wheel.
- `uv run pytest -m "live and not experiment"`: end-to-end runs against OpenAI, billed to `OPENAI_API_KEY` and skipped without it. CI runs them on a manual dispatch.
- `uv run pytest -m experiment`: the experiments, measured live runs that cost far more; CI runs them only when a dispatch turns `experiment` on. Plain `-m live` includes them.

## Conventions

- `src/r3con/runtime/python_executor.py` is a modified copy of smolagents (Apache-2.0), and ruff, pyright and the guide's style rules skip it. Keep changes to it minimal, and list each one in its header and in `NOTICE`.
- A published prompt (`src/r3con/prompts/<stage>/vN.yaml`) is never edited in place: add the next version and bump its pin in `src/r3con/configs/default.yaml`.
