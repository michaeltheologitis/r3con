"""Stage 2, step 2 — parsing: read each document whole and fill instances of the
per-task Pydantic schema.

Every document is assumed to fit in the model's context, so there is **no
chunking**. Parsing runs **one LLM call per document**, all documents in parallel
(bounded by :func:`r3con.settings.active_doc_workers`):

- The **system prompt** carries the task, the proposed schema source, and the
  relevant context (every document's note, not just this one's) — the
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
if it runs in the restricted interpreter, exposes a ``Parse`` class, survives the
structured-output backend :func:`parse_one_document` drives it through, and declares
every top-level field as ``list[...]`` so the merge below cannot silently drop a
document's records.
"""

from __future__ import annotations

import ast
import typing
from dataclasses import dataclass, field
from typing import Any, cast, get_origin

from pydantic import BaseModel, Field, ValidationError

from r3con import settings
from r3con.logging_setup import get_logger
from r3con.parallel import parallel_map
from r3con.prompts import load_prompt
from r3con.runs import StageRun
from r3con.runtime.llm import litellm_chat_completion
from r3con.runtime.python_executor import (
    ExecutionTimeoutError,
    InterpreterError,
    LocalPythonExecutor,
)
from r3con.settings import active_doc_workers
from r3con.stages.relevance import render_relevance

_log = get_logger("structuring.parsing")


_SCHEMA_IMPORTS = ("pydantic", "typing", "datetime", "enum", "decimal")
# Bound before a schema runs, so one that forgets its pydantic import still loads.
_SCHEMA_GLOBALS: dict[str, Any] = {
    "BaseModel": BaseModel,
    "Field": Field,
    "Optional": typing.Optional,
    "Union": typing.Union,
    "Any": typing.Any,
    "Literal": typing.Literal,
    "List": list,
    "Dict": dict,
    "Tuple": tuple,
    "Set": set,
    "Annotated": typing.Annotated,
}


class SchemaError(ValueError):
    """Raised when a proposed schema is unusable (won't run, missing `Parse`, etc.)."""


def check_schema(schema_code: str) -> type[BaseModel]:
    """Run ``schema_code`` in the restricted interpreter and return its ``Parse`` class.

    The code runs in a fresh :class:`~r3con.runtime.python_executor.LocalPythonExecutor`
    that may import only from pydantic, typing, datetime, enum and decimal (plus the
    interpreter's base modules), with the common pydantic and typing names already
    bound. A decorator is refused before anything runs: the interpreter would build the
    method without it, so a validator would silently never run. The error message is
    what the schema-proposal loop feeds back to the model.

    Raises:
        SchemaError: if the code uses a decorator, fails to execute in the interpreter,
            does not define a ``Parse`` class, ``Parse`` is not a ``pydantic.BaseModel``
            subclass, its JSON schema cannot be generated, it contains untyped object
            fields (``dict``, ``list[dict]``, ``dict[str, X]``) that are incompatible
            with OpenAI strict structured-output mode, or any top-level ``Parse`` field
            is not a ``list[...]`` (each document is parsed separately and the
            per-document parses are merged by list-concatenation, so a non-list
            top-level field would silently collapse to a single document).
    """
    decorators = _decorators(schema_code)
    if decorators:
        raise SchemaError(
            "Schema code uses decorators, which r3con does not run: "
            f"{', '.join(decorators)}. Express each constraint as a field type or a "
            "Field(...) argument instead, for example Field(ge=0) or Literal[...]."
        )
    executor = LocalPythonExecutor(additional_authorized_imports=list(_SCHEMA_IMPORTS))
    executor.send_variables(dict(_SCHEMA_GLOBALS))
    try:
        executor(schema_code)
    except (InterpreterError, ExecutionTimeoutError) as e:
        raise SchemaError(f"Schema code failed to execute: {e!r}") from e
    namespace = executor.state

    parse_cls = namespace.get("Parse")
    if parse_cls is None:
        raise SchemaError("Schema code did not define a class named `Parse`.")

    if not (isinstance(parse_cls, type) and issubclass(parse_cls, BaseModel)):
        raise SchemaError(
            f"`Parse` must be a pydantic.BaseModel subclass, got {parse_cls!r}."
        )

    try:
        # Classes the interpreter builds can't resolve their nested types by module
        # lookup; rebuild with its namespace so Pydantic can find them.
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

    non_list = [
        n
        for n, fld in parse_cls.model_fields.items()
        if get_origin(fld.annotation) is not list
    ]
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


def _decorators(schema_code: str) -> list[str]:
    """Every decorator in ``schema_code``, as ``@<decorator> on <name>``; ``[]`` for
    code that does not parse, whose syntax error the interpreter reports."""
    try:
        tree = ast.parse(schema_code)
    except SyntaxError:
        return []
    return [
        f"@{ast.unparse(decorator)} on {node.name}"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for decorator in node.decorator_list
    ]


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
    relevance_snippets: list[str] | None = None,
    model: str,
    max_attempts: int | None = None,
    run: StageRun | None = None,
    kind: str = "llm_call",
    **llm_kwargs: Any,
) -> BaseModel:
    """Parse one whole ``document`` into a populated ``Parse``.

    The system prompt carries the task, the schema source, and the relevant
    context (``relevance_snippets`` — every document's note, the
    cross-document context); the document itself is the user message. One LLM
    call with ``schema=parse_cls``.

    On Pydantic ``ValidationError`` (schema mismatch in the model's structured
    output), retries up to ``max_attempts`` times — each retry feeds the
    validator's error back to the model so it can fix the specific field that
    broke (same pattern as the schema-proposal loop); ``None`` reads
    ``settings.PARSING_MAX_ATTEMPTS`` when the call runs. Raises ``SchemaError`` if
    every attempt fails. Non-validation exceptions bubble up immediately.
    """
    if max_attempts is None:
        max_attempts = settings.PARSING_MAX_ATTEMPTS
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
        relevance=render_relevance(relevance_snippets),
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

    - ``parse`` — the single merged ``Parse`` (its list fields concatenated across
      documents in document order).
    - ``source_docs`` maps each top-level *list* field of ``parse`` to a list of
      source-document indices aligned 1:1 (and in order) with that field's merged
      records — so ``source_docs[f][i]`` is the document ``getattr(parse, f)[i]``
      was parsed from. Built during the merge; the reasoning stage threads it in
      and stamps each record with ``"document": N`` (1-based — the id-injection,
      see ``r3con.stages.reasoning.tag_source_documents``). Without it the
      merged parse is origin-blind: records from every document sit in one flat
      list and the model cannot say which document a fact came from.
    """

    parse: BaseModel
    source_docs: dict[str, list[int]] = field(default_factory=dict)


def parse_documents(
    *,
    documents: list[str],
    schema_code: str,
    parse_cls: type[BaseModel],
    task: str,
    prompt_version: str,
    relevance_snippets: list[str] | None = None,
    model: str,
    max_attempts: int | None = None,
    run: StageRun | None = None,
    workers: int | None = None,
    **llm_kwargs: Any,
) -> ParseResult:
    """Parse a corpus of documents into one merged ``Parse``.

    Each document is fed **whole** (no chunking) through one
    :func:`parse_one_document` call; the documents are processed **in parallel**
    (bounded by ``workers`` / :func:`r3con.settings.active_doc_workers`). Every
    call's system prompt carries the same relevant context, so each
    document is read with the cross-document context even though the calls are
    independent.

    The per-document parses merge into one (list fields concatenated in document
    order), and each merged record is tagged with its source-document index in
    :attr:`ParseResult.source_docs`. ``max_attempts`` bounds each document's retries
    (see :func:`parse_one_document`). Returns a :class:`ParseResult`.
    """
    if not documents:
        return ParseResult(
            parse=_empty_parse(parse_cls), source_docs=_empty_source_docs(parse_cls)
        )

    max_workers = workers if workers is not None else active_doc_workers()
    _log.info(
        "parsing %d doc(s) (≤%d parallel, no chunking)", len(documents), max_workers
    )

    def one(i: int, doc: str) -> BaseModel:
        _log.info("parse doc %d/%d", i + 1, len(documents))
        return parse_one_document(
            document=doc,
            schema_code=schema_code,
            parse_cls=parse_cls,
            task=task,
            prompt_version=prompt_version,
            relevance_snippets=relevance_snippets,
            model=model,
            max_attempts=max_attempts,
            run=run,
            # ``kind`` encodes the doc so the per-call cost ledger (calls.json)
            # ties each call back to its source document.
            kind=f"parse-d{i}",
            **llm_kwargs,
        )

    per_doc = parallel_map(one, documents, max_workers=max_workers)
    parse, source_docs = _merge_with_source_docs(list(enumerate(per_doc)), parse_cls)
    counts = {f: len(v) for f, v in source_docs.items()}
    _log.info(
        "parsing complete · records per list field: %s", counts or "(no list fields)"
    )
    return ParseResult(parse=parse, source_docs=source_docs)


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
    """Concatenate each field's records across the ``(document index, parse)`` pairs, in
    pair order, and tag each merged record with its pair's document index.

    ``source_docs[field]`` is aligned 1:1 with the merged list *by construction*: each
    pair contributes its index once per record, in the order its records are appended.
    Every field of ``Parse`` is a list (:func:`check_schema` refuses any other).
    """
    fields = parse_cls.model_fields
    merged = {
        name: [r for _, p in per_doc for r in getattr(p, name)] for name in fields
    }
    sources = {
        name: [i for i, p in per_doc for _ in getattr(p, name)] for name in fields
    }
    return parse_cls.model_validate(merged), sources
