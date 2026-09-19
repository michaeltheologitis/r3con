"""Stage 2, step 2 — parsing: read each document whole and fill instances of the
per-task Pydantic schema.

Every document is assumed to fit in the model's context, so there is **no
chunking**. Parsing runs **one LLM call per document**, all documents in parallel
(bounded by :func:`r3con.settings.active_doc_workers`):

- The **system prompt** carries the task, the proposed schema source, and the
  corpus-wide relevance state (every document's note, not just this one's) — the
  cross-document context that lets a single document be read in light of what the
  rest of the corpus says.
- The **user message** is the document being parsed, whole.

Each document yields one populated ``Parse``; the per-document parses merge into
one (list fields concatenated in document order). Every merged record is tagged
with its **source-document index** (``ParseResult.source_docs``), which the
reasoning stage stamps onto each record as ``"document": N`` (the id-injection) so
the model can name the document a fact came from.

This module also owns :func:`check_schema`, stage 2's schema validator. It lives
here, with the code that *uses* a schema rather than with the code that proposes
one, because validity is defined by this module's needs: a schema is usable only
if it execs, exposes a ``Parse`` class, survives the structured-output backend
:func:`parse_one_document` drives it through, and declares every top-level field
as ``list[...]`` so the merge below cannot silently drop a document's records.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any, cast, get_origin

from pydantic import BaseModel, Field, ValidationError

from r3con.logging_setup import get_logger
from r3con.parallel import parallel_map
from r3con.prompts import load_prompt
from r3con.runs import StageRun
from r3con.runtime.llm import litellm_chat_completion
from r3con.settings import active_doc_workers, settings
from r3con.stages.relevance import render_relevance

_log = get_logger("structuring.parsing")


# Names pre-seeded into the namespace a proposed schema is exec'd in, so a schema
# that forgets an import (most commonly `from pydantic import BaseModel`) still
# loads instead of dying on NameError. An explicit import in the code just
# re-binds the same object, so seeding is harmless. Purely internal leniency —
# the schema-proposal prompt is unchanged. (Modern schemas use `str | None` /
# `list[...]` builtins and may need none of these; the seed covers the older-style
# names.)
_SCHEMA_EXEC_GLOBALS: dict[str, Any] = {
    "BaseModel": BaseModel,
    "Field": Field,
    "Optional": typing.Optional,
    "Union": typing.Union,
    "Any": typing.Any,
    "Literal": typing.Literal,
    "List": typing.List,
    "Dict": typing.Dict,
    "Tuple": typing.Tuple,
    "Set": typing.Set,
    "Annotated": typing.Annotated,
}


class SchemaError(ValueError):
    """Raised when a proposed schema is unusable (won't exec, missing `Parse`, etc.)."""


def check_schema(schema_code: str) -> type[BaseModel]:
    """Execute ``schema_code`` and return its ``Parse`` class.

    Intended to be called inside the schema-proposal loop
    (:mod:`r3con.stages.structuring.schema`): on failure, the raised error
    message is fed back to the LLM as feedback for the next attempt.

    The exec namespace is pre-seeded with common pydantic/typing names
    (:data:`_SCHEMA_EXEC_GLOBALS`) so a schema that forgets an import — most
    commonly ``from pydantic import BaseModel`` — still loads rather than dying
    on ``NameError``; this is purely an internal leniency and does not change
    the schema-proposal prompt.

    Raises:
        SchemaError: if the code fails to execute, does not define a ``Parse``
            class, ``Parse`` is not a ``pydantic.BaseModel`` subclass, its JSON
            schema cannot be generated, it contains untyped object fields
            (``dict``, ``list[dict]``, ``dict[str, X]``) that are incompatible
            with OpenAI strict structured-output mode, or any top-level ``Parse``
            field is not a ``list[...]`` (each document is parsed separately and
            the per-document parses are merged by list-concatenation, so a non-list
            top-level field would silently collapse to a single document).
    """
    namespace: dict[str, Any] = dict(_SCHEMA_EXEC_GLOBALS)
    try:
        exec(schema_code, namespace)
    except Exception as e:
        raise SchemaError(f"Schema code failed to execute: {e!r}") from e

    parse_cls = namespace.get("Parse")
    if parse_cls is None:
        raise SchemaError("Schema code did not define a class named `Parse`.")

    if not (isinstance(parse_cls, type) and issubclass(parse_cls, BaseModel)):
        raise SchemaError(f"`Parse` must be a pydantic.BaseModel subclass, got {parse_cls!r}.")

    try:
        # NOTE: classes defined via exec() into a bare dict can't resolve their nested
        # types by module lookup; rebuild with the exec namespace so Pydantic can find them.
        parse_cls.model_rebuild(_types_namespace=namespace)
        json_schema = parse_cls.model_json_schema()
    except Exception as e:
        raise SchemaError(f"`Parse.model_json_schema()` failed: {e!r}") from e

    untyped = _find_untyped_objects(json_schema)
    if untyped:
        locations = ", ".join(untyped)
        raise SchemaError(
            f"Schema contains untyped object field(s) at: {locations}. "
            f"Pydantic types `dict`, `dict[str, ...]`, or `list[dict]` emit JSON-schema "
            f"objects without a fixed `properties` block, which is incompatible with the "
            f"structured-output backend (OpenAI strict mode rejects any object that does "
            f"not declare its properties). Define a dedicated Pydantic BaseModel subclass "
            f"with named, typed fields for each object you want to capture."
        )

    non_list = [n for n, fld in parse_cls.model_fields.items() if get_origin(fld.annotation) is not list]
    if non_list:
        raise SchemaError(
            f"Top-level `Parse` field(s) not declared as `list[...]`: {', '.join(non_list)}. "
            "Every field of `Parse` must be a `list[...]`. Each document is extracted "
            "separately and the per-document `Parse` objects are merged by concatenating "
            "their list fields — so a non-list top-level field (a scalar, or a single nested "
            "object) keeps only ONE document's value and silently drops the rest. Even when "
            'the unit of the answer is the document itself (e.g. "map/classify each '
            'document"), model it as a list of per-document records, not a single object or '
            "one field per document. For example:\n"
            "    class DocRecord(BaseModel):\n"
            '        document_title: str | None = Field(description="...")\n'
            "        # ...other fields...\n"
            "    class Parse(BaseModel):\n"
            "        documents: list[DocRecord]"
        )

    return parse_cls


def _find_untyped_objects(schema: Any, path: str = "$") -> list[str]:
    """Find JSON-schema paths whose object nodes have no `properties` block.

    Such nodes come from Pydantic fields like ``dict``, ``dict[str, X]``, or
    ``list[dict]`` — they accept arbitrary keys and cannot be made compatible
    with OpenAI strict mode (which requires every object to declare its
    properties explicitly). The fix is structural: the proposed schema must
    declare a named Pydantic BaseModel subclass instead.
    """
    found: list[str] = []

    def walk(node: Any, p: str) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" not in node:
                found.append(p)
            for key, value in node.items():
                walk(value, f"{p}.{key}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{p}[{i}]")

    walk(schema, path)
    return found


def _parsing_retry_prompt(document: str, error: str) -> str:
    """Build the next user prompt after a Pydantic validation failure.

    The model's prior response was malformed enough that Pydantic couldn't
    coerce it into the requested schema. Feed the document back along with the
    validator's diagnostic so the next attempt can correct the specific
    field(s) that broke — same shape as the schema-proposal retry feedback.
    """
    return (
        f"{document}\n\n"
        "---\n\n"
        "Your previous response failed schema validation:\n\n"
        f"{error}\n\n"
        "Re-extract the same document, fixing only the offending field(s) so "
        "the output validates against the schema. Output the corrected JSON."
    )


def parse_one_document(
    *,
    document: str,
    schema_code: str,
    parse_cls: type[BaseModel],
    task: str,
    prompt_version: str,
    relevance_states: list[str] | None = None,
    model: str,
    max_attempts: int = settings.PARSING_MAX_ATTEMPTS,
    run: StageRun | None = None,
    kind: str = "llm_call",
    **llm_kwargs: Any,
) -> BaseModel:
    """Parse one whole ``document`` into a populated ``Parse``.

    The system prompt carries the task, the schema source, and the corpus-wide
    relevance state (``relevance_states`` — every document's note, the
    cross-document context); the document itself is the user message. One LLM
    call with ``schema=parse_cls``.

    On Pydantic ``ValidationError`` (schema mismatch in the model's structured
    output), retries up to ``max_attempts`` times — each retry feeds the
    validator's error back to the model so it can fix the specific field that
    broke (same pattern as the schema-proposal loop). Default comes from
    ``settings.PARSING_MAX_ATTEMPTS``. Raises ``SchemaError`` if every attempt
    fails. Non-validation exceptions bubble up immediately.
    """
    if max_attempts < 1:
        raise ValueError(
            f"max_attempts must be >= 1, got {max_attempts}. "
            f"Override via settings.PARSING_MAX_ATTEMPTS (current default: "
            f"{settings.PARSING_MAX_ATTEMPTS})."
        )

    system_prompt = load_prompt(
        "structuring/parsing",
        version=prompt_version,
        task=task,
        schema_code=schema_code,
        relevance=render_relevance(relevance_states),
    )
    user_prompt = document
    last_error: str = ""
    for attempt_idx in range(max_attempts):
        try:
            return cast(
                BaseModel,
                litellm_chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    schema=parse_cls,
                    run=run,
                    kind=kind if attempt_idx == 0 else f"{kind}-retry-{attempt_idx}",
                    **llm_kwargs,
                ),
            )
        except ValidationError as e:
            last_error = f"{type(e).__name__}: {e}"
            user_prompt = _parsing_retry_prompt(document, last_error)

    raise SchemaError(
        f"parse_one_document exhausted {max_attempts} attempts. "
        f"Last validation error: {last_error}"
    )


@dataclass
class ParseResult:
    """Output of :func:`parse_documents`: the merged parse + per-record source-doc tags.

    - ``parse`` — the single merged ``Parse`` (list fields concatenated across
      documents in document order; non-list fields take the first non-None value).
    - ``source_docs`` maps each top-level *list* field of ``parse`` to a list of
      source-document indices aligned 1:1 (and in order) with that field's merged
      records — so ``source_docs[f][i]`` is the document ``getattr(parse, f)[i]``
      was parsed from. Built during the merge; the reasoning stage threads it in
      and stamps each record with ``"document": N`` (1-based — the id-injection,
      see ``r3con.stages.reasoning.tag_source_documents``). Without it the
      merged parse is origin-blind: records from every document sit in one flat
      list and the model cannot say which document a fact came from.
    - ``doc_ids`` is the optional human-readable label per document index (e.g. a
      filename/title); ``None`` means use the bare integer index.
    """

    parse: BaseModel
    source_docs: dict[str, list[int]] = field(default_factory=dict)
    doc_ids: list[str] | None = None

    def doc_label(self, doc_idx: int) -> str:
        """Human-readable label for a doc index (its ``doc_ids`` entry, else the int)."""
        if self.doc_ids is not None and 0 <= doc_idx < len(self.doc_ids):
            return self.doc_ids[doc_idx]
        return str(doc_idx)


def parse_documents(
    *,
    documents: list[str],
    schema_code: str,
    parse_cls: type[BaseModel],
    task: str,
    prompt_version: str,
    relevance_states: list[str] | None = None,
    doc_ids: list[str] | None = None,
    model: str,
    run: StageRun | None = None,
    workers: int | None = None,
    **llm_kwargs: Any,
) -> ParseResult:
    """Parse a corpus of documents into one merged ``Parse``.

    Each document is fed **whole** (no chunking) through one
    :func:`parse_one_document` call; the documents are processed **in parallel**
    (bounded by ``workers`` / :func:`r3con.settings.active_doc_workers`). Every
    call's system prompt carries the same corpus-wide relevance state, so each
    document is read with the cross-document context even though the calls are
    independent.

    The per-document parses merge into one (list fields concatenated in document
    order), and each merged record is tagged with its source-document index in
    :attr:`ParseResult.source_docs`. Returns a :class:`ParseResult`.
    """
    if not documents:
        return ParseResult(
            parse=_empty_parse(parse_cls), source_docs=_empty_source_docs(parse_cls), doc_ids=doc_ids
        )

    max_workers = workers if workers is not None else active_doc_workers()
    _log.info("parsing %d doc(s) (≤%d parallel, no chunking)", len(documents), max_workers)

    def one(i: int, doc: str) -> BaseModel:
        _log.info("parse doc %d/%d", i + 1, len(documents))
        return parse_one_document(
            document=doc,
            schema_code=schema_code,
            parse_cls=parse_cls,
            task=task,
            prompt_version=prompt_version,
            relevance_states=relevance_states,
            model=model,
            run=run,
            # ``kind`` encodes the doc so the per-call cost ledger (calls.json)
            # ties each call back to its source document.
            kind=f"parse-d{i}",
            **llm_kwargs,
        )

    per_doc = parallel_map(one, documents, max_workers=max_workers)
    parse, source_docs = _merge_with_source_docs(list(enumerate(per_doc)), parse_cls)
    counts = {f: len(v) for f, v in source_docs.items()}
    _log.info("parsing complete · records per list field: %s", counts or "(no list fields)")
    return ParseResult(parse=parse, source_docs=source_docs, doc_ids=doc_ids)


def _empty_parse(parse_cls: type[BaseModel]) -> BaseModel:
    """Build an empty ``Parse`` (list fields → [], everything else → None)."""
    merged: dict[str, Any] = {}
    for name, fld in parse_cls.model_fields.items():
        merged[name] = [] if get_origin(fld.annotation) is list else None
    return parse_cls.model_validate(merged)


def _empty_source_docs(parse_cls: type[BaseModel]) -> dict[str, list[int]]:
    return {
        name: []
        for name, fld in parse_cls.model_fields.items()
        if get_origin(fld.annotation) is list
    }


def _merge_with_source_docs(
    per_doc: list[tuple[int, BaseModel]],
    parse_cls: type[BaseModel],
) -> tuple[BaseModel, dict[str, list[int]]]:
    """Concatenate list-valued fields across per-document Parse instances and, in
    lockstep, tag each merged record with its source-document index.

    List fields are concatenated in document order; non-list fields take the first
    non-None value across documents. ``source_docs[field]`` is aligned 1:1 with the
    merged list *by construction* — for each document we append its index once per
    record it contributed, in the same order the records are appended. Only list
    fields get source tags (scalars have no per-record origin).
    """
    parsed_objs = [p for _, p in per_doc]
    if not parsed_objs:
        return _empty_parse(parse_cls), _empty_source_docs(parse_cls)

    merged: dict[str, Any] = {}
    source_docs: dict[str, list[int]] = {}
    for field_name in parse_cls.model_fields:
        values = [getattr(p, field_name) for p in parsed_objs]
        if isinstance(values[0], list):
            merged_list: list[Any] = []
            src_list: list[int] = []
            for doc_idx, p in per_doc:
                records = getattr(p, field_name)
                merged_list.extend(records)
                src_list.extend([doc_idx] * len(records))
            merged[field_name] = merged_list
            source_docs[field_name] = src_list
        else:
            merged[field_name] = next((v for v in values if v is not None), values[0])
    return parse_cls.model_validate(merged), source_docs
