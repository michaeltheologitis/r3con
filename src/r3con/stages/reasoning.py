"""Stage 3 — reasoning over the structured parse.

:func:`reason` runs the multi-turn CodeAct loop: the merged parse is bound as the Python
variable ``parse`` in a sandboxed interpreter, the ``reasoning`` prompt is rendered with
the schema and a view of the records, and the agent writes and runs code until it commits
with ``final_answer(x)``.

It reasons over the parse *together with* the corpus-wide relevance snippets from stage 1 —
the two are co-equal views, and the parse alone is lossy. It never sees the source
documents.

The loop itself lives in :mod:`r3con.runtime.codeact`; this module holds only the
reasoning-specific glue — prompt rendering, the ``parse`` binding, and the sample-record
helper that feeds the prompt when the parse is too large to embed whole.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from r3con.prompts import load_prompt
from r3con.runs import StageRun
from r3con.runtime.codeact import (
    DEFAULT_EXEC_TIMEOUT_S,
    CodeActResult,
    run_codeact,
)
from r3con.settings import settings
from r3con.stages.relevance import render_relevance


class _LazyStr:
    """A value that defers an expensive string render until something ``str()``s it.

    Jinja only stringifies a template variable it actually references, so passing
    one of these as a prompt kwarg means the work runs only for prompt versions
    that use that variable. Used for the full-parse JSON dump (``parse_json``),
    which the CodeAct prompt versions in use never reference.
    """

    __slots__ = ("_render",)

    def __init__(self, render: Callable[[], str]) -> None:
        self._render = render

    def __str__(self) -> str:
        return self._render()


def _sample_record_per_field(parse_dict: Any) -> str:
    """Render one EXAMPLE record per top-level list field — *not* the whole parse.

    The agent has the full parse bound as the Python variable ``parse`` in its
    sandbox executor; dumping the whole JSON into the prompt context (a) blows the
    context up on record floods — parses running to hundreds or thousands of rows —
    and (b) defeats the print-and-reflect loop by giving the model the answer
    in-context.

    This helper renders one sample record per list field (the first record),
    in JSON for unambiguous shape — enough for the model to write code that
    indexes correctly. For non-list top-level fields the value renders
    directly. Empty lists are noted as such.
    """
    if not isinstance(parse_dict, dict):
        return f"(parse is not a JSON object — type={type(parse_dict).__name__})"
    if not parse_dict:
        return "(parse is empty — no top-level fields)"
    # ensure_ascii=False so non-ASCII values (e.g. CJK) reach the model as readable text
    # in the prompt, not \uXXXX escapes (which also degrade non-ASCII-task quality).
    lines: list[str] = []
    for key, value in parse_dict.items():
        if isinstance(value, list):
            if not value:
                lines.append(f"`{key}` (empty list — no example to show)")
            else:
                example = json.dumps(value[0], indent=2, default=repr, ensure_ascii=False)
                lines.append(f"`{key}[0]` (1 of {len(value)} record(s)):\n```json\n{example}\n```")
        elif isinstance(value, dict):
            example = json.dumps(value, indent=2, default=repr, ensure_ascii=False)
            lines.append(f"`{key}` (dict):\n```json\n{example}\n```")
        else:
            example = json.dumps(value, default=repr, ensure_ascii=False)
            lines.append(f"`{key}` (scalar): {example}")
    return "\n\n".join(lines)


def tag_source_documents(parse_dict: Any, source_docs: dict[str, list[int]] | None) -> Any:
    """Stamp each list-field record with the 1-based document it was parsed from
    (``"document": N``), from the parsing step's ``source_docs`` provenance.

    The pipeline computes this mapping at merge time (each record aligns 1:1 with
    ``source_docs[field]`` by construction) but otherwise drops it at the reasoning
    boundary, leaving the model unable to say *which* document a fact came from. The label
    matches :func:`r3con.stages.relevance.render_relevance`'s "Document N", so the parse
    and the corpus-wide relevance snippets share **one** document-id space and the agent can
    cross-reference a record against the note its document contributed. That identity rests
    on both views being built over the documents in the same order — reorder the collection
    between stages and the ids stop meaning the same thing. No-op without ``source_docs`` or
    for a non-dict parse. Mutates and returns ``parse_dict`` (``document`` first, so it reads
    first)."""
    if not source_docs or not isinstance(parse_dict, dict):
        return parse_dict
    for field, idxs in source_docs.items():
        recs = parse_dict.get(field)
        if not isinstance(recs, list):
            continue
        for k in range(min(len(recs), len(idxs))):
            if isinstance(recs[k], dict) and "document" not in recs[k]:
                recs[k] = {"document": idxs[k] + 1, **recs[k]}
    return parse_dict


@functools.lru_cache(maxsize=1)
def _parse_token_encoding():  # noqa: ANN202 — returns a tiktoken Encoding, or None
    """tiktoken encoding for the parse-size guard, cached. Returns ``None`` if tiktoken
    is unavailable — it downloads its vocabulary from the network on first use, and a
    size guard must never be what sinks a run that has already paid for stages 1 and 2."""
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # noqa: BLE001 — no tiktoken, or no network on first use
        return None


def _count_tokens(text: str) -> int:
    """Approximate token count of ``text``. It is only a size guard, so the exact
    tokenizer doesn't matter; without tiktoken, fall back to the standard ~4-chars-per-
    token estimate. ``disallowed_special=()`` so arbitrary text never errors."""
    encoding = _parse_token_encoding()
    if encoding is None:
        return len(text) // 4
    return len(encoding.encode(text, disallowed_special=()))


def _render_parse_for_codeact(parse_dict: Any) -> str:
    """The ``parse`` view embedded in the codeact system prompt: the WHOLE parse as JSON when it
    fits (the common case), else a prominent "this is only a sample" note + one sample record
    per field. Either way the full parse is also bound as the ``parse`` variable for the agent to
    compute over, so the sample path costs reach, not access. The cap is
    ``settings.REASONING_PARSE_MAX_TOKS`` (tiktoken token count; a flood guard for parses that run
    to thousands of records)."""
    full = json.dumps(parse_dict, indent=2, ensure_ascii=False, default=repr)
    if _count_tokens(full) <= settings.REASONING_PARSE_MAX_TOKS:
        return full
    n = sum(len(v) for v in parse_dict.values() if isinstance(v, list)) if isinstance(parse_dict, dict) else 0
    note = (
        "**IMPORTANT — the block below is only a SAMPLE of the parse, NOT the full data.** "
        f"The full parse is large ({n} record(s)), so only ONE example record per field is shown "
        "here, to convey the shape. The COMPLETE parse is bound as the `parse` variable — do not "
        "assume these samples are all the records; read the rest with `print(...)` before answering."
    )
    return note + "\n\n" + _sample_record_per_field(parse_dict)


def reason(
    *,
    task: str,
    schema_code: str,
    parsed: BaseModel | dict[str, Any],
    model: str,
    prompt_version: str,
    relevance_snippets: list[str] | None = None,
    source_docs: dict[str, list[int]] | None = None,
    max_turns: int = settings.REASONING_MAX_TURNS,
    timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    additional_authorized_imports: list[str] | None = None,
    run: StageRun | None = None,
    **llm_kwargs: Any,
) -> CodeActResult:
    """Answer ``task`` over ``parsed`` with the multi-turn CodeAct loop.

    The LLM never sees the long source text — only the task, the schema
    source (so it knows the parse's shape), a view of the parse itself (the whole
    thing when it fits, else one sample record per top-level field), and the
    **corpus-wide relevance snippets** from stage 1.
    The full parse is bound as the Python variable ``parse`` in the sandbox; the
    agent inspects it via ``print(...)`` across turns and commits via
    ``final_answer(x)``.

    The system prompt carries everything immutable across the loop's turns; the
    user message is the bare task. Loop mechanics live in
    :func:`r3con.runtime.codeact.run_codeact`.
    """
    parse_dict = parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed
    parse_dict = tag_source_documents(parse_dict, source_docs)

    relevance_block = render_relevance(relevance_snippets)
    # The codeact prompt variables — Jinja renders only what the active template references:
    #   parse_block  : the whole parse (or samples + a note if huge) — what current prompts use
    #   samples_block: one sample record per field — kept for older prompt versions
    #   parse_json   : the full parse dump, lazy so the huge case is computed only on demand
    parse_block = _render_parse_for_codeact(parse_dict)
    samples_block = _sample_record_per_field(parse_dict)
    parse_json = _LazyStr(lambda: json.dumps(parse_dict, indent=2, ensure_ascii=False))
    system_prompt = load_prompt(
        "reasoning",
        version=prompt_version,
        task=task,
        schema_code=schema_code,
        relevance=relevance_block,
        parse_json=parse_json,
        samples_block=samples_block,
        parse_block=parse_block,
    )

    return run_codeact(
        system_prompt=system_prompt,
        user_message=f"Input:\n<task>\n{task}\n</task>",
        model=model,
        variables={"parse": parse_dict},
        max_turns=max_turns,
        timeout_s=timeout_s,
        additional_authorized_imports=additional_authorized_imports,
        run=run,
        **llm_kwargs,
    )
