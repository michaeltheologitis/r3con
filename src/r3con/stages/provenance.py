"""Replies that cite the notes their call was shown.

Relevance's later rounds and parsing each read a document, or one part of one, beside
notes shown by id, and ask for a structured reply: a payload (the ``summary``, the
``parse``) and its ``provenance_ids``, the ids of the shown notes it used. Both build
it with :func:`citing_schema` and read it with :func:`read_citing`. The ids reach the
JSON schema as an ``enum`` (:func:`list_of_ids`), but the reply is read as plain
strings and checked by :func:`cited`, which keeps only the ids the call showed.
Reasoning's answer cites two lists of ids, one per view, each built and checked by
those two functions.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Annotated, Any, NamedTuple

from pydantic import BaseModel, Field, create_model

from r3con.logging_setup import get_logger
from r3con.runtime.llm import send_until_valid

_log = get_logger("provenance")


class CitingRequest(NamedTuple):
    """What a call sends beside the text it reads: the system prompt, the reply's
    schema (``None`` for plain text), and the ids of the notes the reply may cite."""

    system_prompt: str
    schema: type[BaseModel] | None
    citable: Sequence[str]


def list_of_ids(shown_ids: Sequence[str]) -> Any:
    """The type of a reply's list of the ids a call shows. The shown ids reach only the
    JSON schema, as an ``enum``, or, when there are none, as any string, since an empty
    enum is refused; the reply itself is read as plain strings and checked by
    :func:`cited`."""
    if not shown_ids:
        return list[str]
    return list[Annotated[str, Field(json_schema_extra={"enum": list(shown_ids)})]]


def citing_schema(name: str, citable: Sequence[str], **payload: Any) -> type[BaseModel]:
    """A reply named ``name``: each ``payload`` field, of the type given, then its
    ``provenance_ids``, a :func:`list_of_ids` of ``citable``. A reply without
    ``provenance_ids`` cites nothing, though the strict schema sent requires it."""
    fields: dict[str, Any] = {field: (type_, ...) for field, type_ in payload.items()}
    # A default_factory stays out of the schema; strict mode refuses a default.
    return create_model(
        name,
        **fields,
        provenance_ids=(list_of_ids(citable), Field(default_factory=list)),
    )


def cited(provenance: Sequence[str], shown: Collection[str], kind: str) -> list[str]:
    """The ids a reply cites, in order and without repeats, less those the call did not
    show, which are dropped with a warning."""
    unshown = [id_ for id_ in provenance if id_ not in shown]
    if unshown:
        _log.warning(
            "%s: dropped provenance id(s) the call did not show: %s", kind, unshown
        )
    return list(dict.fromkeys(id_ for id_ in provenance if id_ in shown))


def read_citing(
    request: CitingRequest,
    text: str,
    *,
    rewrite: str,
    kind: str,
    max_attempts: int,
    **llm_call: Any,
) -> tuple[Any, list[str]]:
    """``text`` read under ``request`` by :func:`~r3con.runtime.llm.send_until_valid`,
    which sends it again with ``rewrite`` when a reply does not fit: the reply's
    payload, its one field besides ``provenance_ids``, and the citable ids it cites
    (:func:`cited`); without a schema, the reply's text, citing nothing."""
    reply = send_until_valid(
        request.system_prompt,
        text,
        schema=request.schema,
        rewrite=rewrite,
        kind=kind,
        max_attempts=max_attempts,
        **llm_call,
    )
    if isinstance(reply, str):
        return reply, []
    fields = dict(reply)
    provenance = fields.pop("provenance_ids")
    [payload] = fields.values()
    return payload, cited(provenance, request.citable, kind)
