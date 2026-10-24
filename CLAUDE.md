# r3con

Context representation for large-scale reasoning: for a question whose evidence is scattered across many documents, r3con builds a task-specific representation of them (relevance, then structure) and reasons over it with any model LiteLLM supports. It is published on PyPI as `r3context`; the import package is `r3con`.

Read `../coding-skills/GUIDE.md` and follow it; in the main session a hook has already printed it.
This project's Notion page (todos, changelog, notes): https://app.notion.com/p/40362fb22237814d9e8ccbaf2d812c8c

## Checks

- `uv run ruff check`, `uv run ruff format --check` and `uv run pyright`: lint, format and types, as CI runs them.
- `uv run pytest`: the unit tests. Sockets are disabled, so nothing reaches the network.
- `uv run pytest -m live`: the live tier against a real provider, billed to `OPENAI_API_KEY`. Run it when a change affects what the model is sent or how it is called. CI runs it on a manual dispatch with `live` on.

## Conventions

- `src/r3con/runtime/python_executor.py` is vendored from smolagents (Apache-2.0, see `NOTICE`), and ruff and pyright skip it. Keep changes to it minimal, and say so in the PR.
- Logging is the standard library's `logging` through `r3con.logging_setup.get_logger`, not structlog.
- A release is a version bump in `pyproject.toml`, then a `v*` tag, which runs `release.yml` and publishes to PyPI. Ask Michael before tagging.
