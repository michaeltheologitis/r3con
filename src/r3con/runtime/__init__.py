"""Execution machinery shared by the stages.

- :mod:`r3con.runtime.llm` — LiteLLM wrapper + structured-output schema enforcement.
- :mod:`r3con.runtime.codeact` — the generic multi-turn code-execution loop.
- :mod:`r3con.runtime.python_executor` — vendored sandboxed in-process Python
  interpreter used by the reasoning loop (smolagents, Apache-2.0 — see NOTICE).
"""
