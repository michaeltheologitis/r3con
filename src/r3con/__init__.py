"""r3con — just-in-time reasoning over a collection of documents.

Ask a question whose evidence is spread across many documents; the apparatus for
answering it is built *per question*, at the moment you ask:

    >>> from r3con import r3con
    >>> result = r3con.run("Which supplier missed the most delivery windows?", docs)
    >>> print(result)                 # the answer
    >>> result.relevant_context              # what each document contributed
    >>> result.structured_context            # the structured parse behind it

Three moves, in order:

1. **surfacing relevance** — every document is read against the question and written up
   as a **relevance snippet**; then re-read in light of the *other* documents' snippets.
   Together they are the **relevant context**. Relevance is not a property a
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

from importlib.metadata import PackageNotFoundError, version

from r3con import r3con, settings
from r3con.config import RunConfig, load_config
from r3con.pipeline import Answer, run_pipeline
from r3con.prompts import load_prompt
from r3con.r3con import read_documents, run
from r3con.runs import StageRun, TaskLogger
from r3con.runtime.codeact import CodeActResult, CodeActTurn
from r3con.runtime.llm import litellm_chat_completion, litellm_chat_completion_full
from r3con.stages.reasoning import reason
from r3con.stages.relevance import (
    RelevantContext,
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
from r3con.stages.structuring.schema import (
    ProposalAttempt,
    ProposalResult,
    propose_schema,
)

try:
    __version__ = version("r3context")  # the distribution's name, not the module's
except PackageNotFoundError:  # a source tree that was never installed
    __version__ = "0+unknown"

__all__ = [
    "Answer",
    "CodeActResult",
    "CodeActTurn",
    "ParseResult",
    "ProposalAttempt",
    "ProposalResult",
    "RelevantContext",
    "RunConfig",
    "SchemaError",
    "StageRun",
    "TaskLogger",
    "check_schema",
    "litellm_chat_completion",
    "litellm_chat_completion_full",
    "load_config",
    "load_prompt",
    "parse_documents",
    "parse_one_document",
    "propose_schema",
    "r3con",
    "read_documents",
    "reason",
    "relevance_snippet",
    "render_relevance",
    "run",
    "run_pipeline",
    "settings",
    "surface_relevance",
]
