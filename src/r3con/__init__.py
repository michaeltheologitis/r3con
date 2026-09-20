"""r3con — just-in-time reasoning over a collection of documents.

Ask a question whose evidence is spread across many documents; the apparatus for
answering it is built *per question*, at the moment you ask:

    >>> from r3con import r3con
    >>> result = r3con.run("Which supplier missed the most delivery windows?", docs)
    >>> print(result)                 # the answer
    >>> result.relevance              # what each document contributed
    >>> result.struct_data            # the structured parse behind it

Three moves, in order:

1. **surfacing relevance** — every document is read against the question and written up
   as a **relevance snippet**; then re-read in light of the *other* documents' snippets.
   Together they are the **corpus-wide relevance snippets**. Relevance is not a property a
   document has; it is a relation between the document, the question, and the rest of
   the collection, so it cannot be settled from a document in isolation.
2. **structuring** — a Pydantic schema is proposed for exactly what this question needs,
   then every document is parsed, whole, into instances of it.
3. **reasoning** — the merged parse is loaded into a sandboxed Python runtime and the
   agent reasons there, over the structured records and the relevance snippets together.

Nothing is chunked, and the whole collection is never placed in a single prompt: each
document is read on its own, and information crosses document boundaries through the
relevance snippets.
"""

from r3con import r3con  # noqa: F401 — `from r3con import r3con` namespace
from r3con.r3con import read_documents, run
from r3con.config import RunConfig, load_config
from r3con.pipeline import Answer, run_pipeline
from r3con.prompts import load_prompt
from r3con.runs import StageRun, TaskLogger
from r3con.settings import settings
from r3con.stages.reasoning import reason
from r3con.stages.relevance import (
    CorpusRelevanceSnippets,
    relevance_snippet,
    render_relevance,
    surface_relevance,
)
from r3con.stages.structuring.parsing import (
    ParseResult,
    SchemaError,
    check_schema,
    parse_documents,
    parse_one_document,
)
from r3con.stages.structuring.schema import ProposalAttempt, ProposalResult, propose_schema
from r3con.runtime.codeact import CodeActResult, CodeActTurn
from r3con.runtime.llm import litellm_chat_completion, litellm_chat_completion_full

try:  # the installed wheel's version is the truth; the literal is the checkout fallback
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("r3context")  # the DISTRIBUTION name, not the module
except Exception:  # noqa: BLE001 — running from a source checkout
    __version__ = "0.1.0"

__all__ = [
    # the entry point
    "r3con",
    "run",
    "Answer",
    "read_documents",
    # the pipeline, for callers who want the per-stage artifacts or a prebuilt config
    "run_pipeline",
    # run identity
    "RunConfig",
    "load_config",
    "load_prompt",
    "settings",
    # stage 1 — surfacing relevance
    "CorpusRelevanceSnippets",
    "surface_relevance",
    "relevance_snippet",
    "render_relevance",
    # stage 2 — structuring
    "ProposalResult",
    "ProposalAttempt",
    "propose_schema",
    "ParseResult",
    "SchemaError",
    "check_schema",
    "parse_documents",
    "parse_one_document",
    # stage 3 — reasoning
    "reason",
    "CodeActResult",
    "CodeActTurn",
    # artifacts
    "TaskLogger",
    "StageRun",
    # transport
    "litellm_chat_completion",
    "litellm_chat_completion_full",
]
