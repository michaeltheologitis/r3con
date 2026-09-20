"""Stage 2 — structuring: give the surfaced relevance a form, then fill it.

- :mod:`r3con.stages.structuring.schema` — propose a per-task Pydantic schema from
  the task and the relevant context.
- :mod:`r3con.stages.structuring.parsing` — parse each document, whole, into
  instances of that schema; merge them into one ``Parse``.
"""
