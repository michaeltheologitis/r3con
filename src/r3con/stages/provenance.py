"""Replies that cite the notes their call was shown."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Annotated, Any, NamedTuple

from pydantic import BaseModel, Field, create_model

from r3con.logging_setup import get_logger
from r3con.runtime.llm import send_until_valid

_log = get_logger("provenance")


class CitingRequest(NamedTuple):
    """What a call sends beside the text it reads: the system prompt, the reply's
    schema (:func:`citing_schema`, or ``None`` for plain text), and the ids of the
    notes the reply may cite."""

    system_prompt: str
    schema: type[BaseModel] | None
    citable: Sequence[str]


def ids_field(citable: Sequence[str]) -> tuple[Any, Any]:
    """The field, for :func:`pydantic.create_model`, of a reply's list of the ids it
    cites from ``citable``. The ids reach only the JSON schema, as an ``enum``, or,
    when there are none, as any string, since an empty enum is refused; the reply
    itself is read as plain strings and checked by :func:`cited`. A reply without the
    list cites nothing, though the strict schema sent requires it."""
    ids: Any = list[str]
    if citable:
        ids = list[Annotated[str, Field(json_schema_extra={"enum": list(citable)})]]
    # A default_factory stays out of the schema; strict mode refuses a default.
    return ids, Field(default_factory=list)


def citing_schema(
    name: str, citable: Sequence[str], field: str, type_: Any
) -> type[BaseModel]:
    """A reply named ``name``: its payload ``field``, of ``type_``, then its
    ``provenance_ids`` (:func:`ids_field`), the order :func:`read_citing` reads."""
    payload: dict[str, Any] = {field: (type_, ...)}
    return create_model(name, **payload, provenance_ids=ids_field(citable))


def cited(provenance: Sequence[str], shown: Collection[str], kind: str) -> list[str]:
    """The ids a reply cites, in order and without repeats, less those the call did not
    show, which are dropped with a warning."""
    shown_ids = frozenset(shown)
    unshown = [id_ for id_ in provenance if id_ not in shown_ids]
    if unshown:
        _log.warning(
            "%s: dropped provenance id(s) the call did not show: %s", kind, unshown
        )
    return list(dict.fromkeys(id_ for id_ in provenance if id_ in shown_ids))


def read_citing(
    request: CitingRequest,
    text: str,
    *,
    rewrite: str,
    kind: str,
    max_attempts: int,
    **llm_call: Any,
) -> tuple[object, list[str]]:
    """``text`` read under ``request`` by :func:`~r3con.runtime.llm.send_until_valid`,
    which sends it again with ``rewrite`` when a reply does not fit: the reply's
    payload and the citable ids it cites (:func:`cited`); for a request without a
    schema, the reply's text, citing nothing."""
    reply = send_until_valid(
        request.system_prompt,
        text,
        schema=request.schema,
        rewrite=rewrite,
        kind=kind,
        max_attempts=max_attempts,
        **llm_call,
    )
    if request.schema is None:
        return reply, []
    payload, provenance = (
        getattr(reply, field) for field in request.schema.model_fields
    )
    return payload, cited(provenance, request.citable, kind)
