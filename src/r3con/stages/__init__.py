"""Pipeline stages — the three moves the method makes.

- :mod:`r3con.stages.relevance` — stage 1, surfacing relevance: the per-document
  relevance states, built over synchronous cross-document rounds.
- :mod:`r3con.stages.structuring` — stage 2, structuring: propose the schema, then
  parse every document into it.
- :mod:`r3con.stages.reasoning` — stage 3, reasoning: answer over the structured
  parse and the corpus-wide relevance state.
"""
