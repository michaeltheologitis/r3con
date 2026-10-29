"""Stage 3 — reasoning over the structured parse.

:func:`reason` answers on one of two paths, the run config's ``reasoning``:

- **direct**: one completion. The ``reasoning/direct`` prompt carries the summaries and
  the whole parse, the user message is the task, and the reply is structured: the
  ``answer`` and its ``provenance_ids``, ``{"summary_ids": [...], "parse_ids": [...]}``.
- **coding**: the multi-turn CodeAct loop. The merged parse is bound as the Python
  variable ``parse`` in a sandboxed interpreter, the ``reasoning/coding`` prompt is
  rendered with a view of the records, and the agent writes and runs code until it
  commits with ``final_answer(answer, provenance_ids={...})``, the same mapping.

Either way the prompt names each document by its id for the run: a summary is headed
``### Document ID: <id>`` (``<id>.<k>`` for part ``k`` of a document read in parts), and
each record says ``"document": "<id>"``. The answer cites the two views separately: the
ids of the summaries it used, and the ids of the documents whose records it used, each
from among those the prompt shows (:func:`_cited`).

A direct run whose prompt does not fit the model's window, because ``Splits`` estimates
it over the line or the provider refuses it as too long, takes the coding path instead,
starting from one sample record per field. So does one whose every reply fails its
schema, starting from whichever view of the parse the coding path picks.

Either way it reasons over the parse *together with* the relevant context from stage 1 —
the two are co-equal views, and the parse alone is lossy. It never sees the source
documents.

The loop itself lives in :mod:`r3con.runtime.codeact`; this module holds only the
reasoning-specific glue — prompt rendering, the choice of path, the ``parse`` binding,
and the sample-record helper that feeds the coding prompt when the parse is too large to
embed whole.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, cast

from litellm.exceptions import ContextWindowExceededError
from pydantic import BaseModel, Field, ValidationError, create_model, field_validator

from r3con import settings
from r3con.config import PromptStage, ReasoningPath
from r3con.logging_setup import get_logger
from r3con.masking import masked
from r3con.prompts import load_prompt
from r3con.runs import StageRun
from r3con.runtime.codeact import (
    DEFAULT_EXEC_TIMEOUT_S,
    PROVENANCE_IDS,
    CodeActTurn,
    run_codeact,
)
from r3con.runtime.llm import count_tokens, schema_as_sent, send_until_valid
from r3con.splitting import Splits
from r3con.stages.provenance import cited, ids_field
from r3con.stages.relevance import Snippet, citable_ids, render_relevance
from r3con.stages.structuring.parsing import DOCUMENT_KEY

_log = get_logger("reasoning")

_REWRITE = "Answer the same task again"
# The answer's two views, as its provenance names them.
_VIEWS = ("summary_ids", "parse_ids")


class _DirectReply(BaseModel):
    answer: str
    provenance_ids: BaseModel

    @field_validator("answer")
    @classmethod
    def _not_empty(cls, answer: str) -> str:
        if not answer.strip():
            raise ValueError("the answer is empty")
        return answer


def cites_nothing() -> dict[str, list[str]]:
    """Provenance that cites nothing: no id under either view."""
    return {view: [] for view in _VIEWS}


def _documents_with_records(parse_dict: Any, document_ids: Sequence[str]) -> list[str]:
    """The ids in ``document_ids`` that a record of ``parse_dict`` names as its
    ``document``, in order: the documents an answer can cite for their records."""
    if not isinstance(parse_dict, dict):
        return []
    named = {
        record[DOCUMENT_KEY]
        for records in parse_dict.values()
        if isinstance(records, list)
        for record in records
        if isinstance(record, dict) and isinstance(record.get(DOCUMENT_KEY), str)
    }
    return [id_ for id_ in document_ids if id_ in named]


def _provenance_ids_model(citable: Mapping[str, Sequence[str]]) -> type[BaseModel]:
    """The ``provenance_ids`` an answer gives, on either path: under each of
    ``citable``'s keys, a list of its ids (:func:`~r3con.stages.provenance.ids_field`).
    A missing list cites nothing, and another key is ignored."""
    lists: dict[str, Any] = {key: ids_field(ids) for key, ids in citable.items()}
    return create_model("ProvenanceIds", **lists)


def _reply_schema(citable: Mapping[str, Sequence[str]]) -> type[_DirectReply]:
    """The direct reply's schema: the ``answer``, and its ``provenance_ids``
    (:func:`_provenance_ids_model`). A reply without them cites nothing."""
    provenance_ids = _provenance_ids_model(citable)
    return create_model(
        "ReasoningAnswer",
        __base__=_DirectReply,
        provenance_ids=(provenance_ids, Field(default_factory=provenance_ids)),
    )


def _cited(
    given: Mapping[str, Sequence[str]],
    citable: Mapping[str, Sequence[str]],
    kind: str,
) -> dict[str, list[str]]:
    """``given``'s ids under each of ``citable``'s keys, less those it does not hold,
    by :func:`~r3con.stages.provenance.cited`, which warns as ``<kind> <key>``."""
    return {
        key: cited(given[key], ids, f"{kind} {key}") for key, ids in citable.items()
    }


def _committed_ids(
    arguments: Mapping[str, Any], citable: Mapping[str, Sequence[str]]
) -> dict[str, list[str]]:
    """What ``final_answer``'s ``provenance_ids`` cites, read as
    :func:`_provenance_ids_model` and checked by :func:`_cited`: nothing when it was
    not given, and nothing, with a warning, when it is not a mapping of lists of ids.
    Another argument is ignored with a warning."""
    others = {
        name: value for name, value in arguments.items() if name != PROVENANCE_IDS
    }
    if others:
        _log.warning(
            "coding: ignored final_answer argument(s) other than provenance_ids: %r",
            others,
        )
    given = arguments.get(PROVENANCE_IDS)
    try:
        ids = _provenance_ids_model(citable).model_validate(given or {})
    except ValidationError:
        _log.warning(
            "coding: provenance_ids must map %s to lists of ids; got %r, which cites "
            "nothing",
            " and ".join(_VIEWS),
            given,
        )
        return cites_nothing()
    return _cited(ids.model_dump(), citable, "coding")


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
                example = json.dumps(
                    value[0], indent=2, default=repr, ensure_ascii=False
                )
                lines.append(
                    f"`{key}[0]` (1 of {len(value)} record(s)):\n```json\n{example}\n```"
                )
        elif isinstance(value, dict):
            example = json.dumps(value, indent=2, default=repr, ensure_ascii=False)
            lines.append(f"`{key}` (dict):\n```json\n{example}\n```")
        else:
            example = json.dumps(value, default=repr, ensure_ascii=False)
            lines.append(f"`{key}` (scalar): {example}")
    return "\n\n".join(lines)


def tag_source_documents(
    parse_dict: Any,
    source_docs: dict[str, list[int]] | None,
    document_ids: Sequence[str],
) -> Any:
    """Stamp each list-field record with the id of the document it was parsed from
    (``"document": "<id>"``), from the parsing step's ``source_docs``, so the model can
    say which document a fact came from and match it against that document's summary,
    headed by the same id. No-op without ``source_docs`` or for a non-dict parse.
    Mutates and returns ``parse_dict`` (``document`` first, so it reads first).
    A record's own ``document`` value is kept; ``check_schema`` warns of such schemas."""
    if not source_docs or not isinstance(parse_dict, dict):
        return parse_dict
    for name, idxs in source_docs.items():
        recs = parse_dict.get(name)
        if not isinstance(recs, list):
            continue
        for k in range(min(len(recs), len(idxs))):
            if isinstance(recs[k], dict) and DOCUMENT_KEY not in recs[k]:
                recs[k] = {DOCUMENT_KEY: document_ids[idxs[k]], **recs[k]}
    return parse_dict


def _sample_view(parse_dict: Any) -> str:
    """The parse view shown instead of the whole parse: a prominent "this is only a
    sample" note, then one sample record per field. The full parse is bound as the
    ``parse`` variable either way, so the sample view costs reach, not access."""
    n = (
        sum(len(v) for v in parse_dict.values() if isinstance(v, list))
        if isinstance(parse_dict, dict)
        else 0
    )
    note = (
        "**IMPORTANT — the block below is only a SAMPLE of the parse, NOT the full data.** "
        f"The full parse is large ({n} record(s)), so only ONE example record per field is shown "
        "here, to convey the shape. The COMPLETE parse is bound as the `parse` variable — do not "
        "assume these samples are all the records; read the rest with `print(...)` before answering."
    )
    return note + "\n\n" + _sample_record_per_field(parse_dict)


@dataclass
class ReasoningResult:
    """What reasoning produced.

    - ``answer``: the answer text.
    - ``path``: the path that produced it, ``"direct"`` or ``"coding"``.
    - ``provenance``: the ids the answer used, by view: under ``"summary_ids"``, those
      of the summaries it used (``<id>.<k>`` for a part of a document read in parts),
      and under ``"parse_ids"``, those of the documents whose records it used.
    - ``turns`` and ``terminated_by``: on the coding path, the agent's turns and how its
      loop ended, as in :class:`~r3con.runtime.codeact.CodeActResult`; on the direct
      path, no turns and ``"reply"``.
    - ``fallback``: why a direct run took the coding path, else ``None``:
      ``{"cause": "estimate", "estimate": tokens, "line": tokens}``,
      ``{"cause": "refusal", "error": message}``, keys masked
      (:func:`r3con.masking.masked`), or, when no reply fit its schema,
      ``{"cause": "invalid", "error": message}``.
    """

    answer: str
    path: ReasoningPath
    provenance: dict[str, list[str]] = field(default_factory=cites_nothing)
    turns: Sequence[CodeActTurn] = ()
    terminated_by: str = "reply"
    fallback: dict[str, Any] | None = None


def _system_prompt(
    *,
    stage: str,
    version: str,
    task: str,
    schema_code: str,
    relevance_snippets: Sequence[Snippet] | None,
    document_ids: Sequence[str],
    parse_json: str,
    samples_block: str,
    parse_block: str,
) -> str:
    """The system prompt of ``stage`` (``reasoning/direct`` or ``reasoning/coding``) at
    ``version``, from the variables a reasoning prompt version may use, a user's overlay
    included; Jinja renders only the ones its template references:

    - ``task``, ``schema_code``, ``relevance``: the question, the schema source, the
      summaries, each headed by its document's id;
    - ``parse_block``: the whole parse, or samples and a note;
    - ``samples_block``: one sample record per field;
    - ``parse_json``: the whole parse as JSON;
    - ``read_in_parts``: whether a summary is one part of a document read in parts.
    """
    return load_prompt(
        stage,
        version=version,
        task=task,
        schema_code=schema_code,
        relevance=render_relevance(relevance_snippets, document_ids),
        parse_json=parse_json,
        samples_block=samples_block,
        parse_block=parse_block,
        read_in_parts=any(isinstance(s, list) for s in relevance_snippets or ()),
    )


def _reason_with_code(
    *,
    render: Callable[..., str],
    parse_dict: Any,
    parse_json: str,
    user_message: str,
    model: str,
    splits: Splits,
    citable: Mapping[str, Sequence[str]],
    whole: bool = True,
    **codeact_kwargs: Any,
) -> ReasoningResult:
    """The coding path: the CodeAct loop over the parse, under the system prompt
    ``render(parse_block=...)``, its answer citing from among ``citable``'s ids.

    With ``whole``, the view of the parse is the whole parse when its first turn fits,
    else one sample record per top-level field: when the parse is over
    ``settings.REASONING_PARSE_MAX_TOKS``, when ``splits`` estimates the first turn with
    it over the line, or when the provider refuses that turn as too long, which sends it
    once more with the samples. Without ``whole``, it is the samples from the start.
    """
    parse_tokens = count_tokens(parse_json) if whole else None
    whole = parse_tokens is not None and (
        parse_tokens <= settings.REASONING_PARSE_MAX_TOKS
    )
    system_prompt = render(
        parse_block=parse_json if whole else _sample_view(parse_dict)
    )
    if whole and (estimate := splits.over_line(system_prompt, user_message)):
        _log.info(
            "the whole parse (about %s tokens) would put reasoning's first turn at "
            "about %s tokens, over the %s-token line; showing one sample record per "
            "field",
            f"{parse_tokens:,}",
            f"{estimate:,}",
            f"{splits.line:,}",
        )
        whole = False
        system_prompt = render(parse_block=_sample_view(parse_dict))

    def with_samples(refusal: ContextWindowExceededError) -> str:
        """The first turn's prompt with samples, once, after a refusal of it with
        the whole parse."""
        nonlocal whole
        if not whole:
            raise refusal
        _log.warning(
            "reasoning's first turn was refused as too long with the whole parse "
            "(about %s tokens); sending it again with one sample record per field",
            f"{parse_tokens:,}",
        )
        whole = False
        return render(parse_block=_sample_view(parse_dict))

    result = run_codeact(
        system_prompt=system_prompt,
        user_message=user_message,
        model=model,
        variables={"parse": parse_dict},
        on_first_turn_too_long=with_samples,
        **codeact_kwargs,
    )
    return ReasoningResult(
        answer=result.answer,
        path="coding",
        provenance=_committed_ids(result.arguments, citable),
        turns=result.turns,
        terminated_by=result.terminated_by,
    )


def _in_brief(error: ValidationError) -> str:
    """``error`` on one line, without the reply it quotes."""
    return "; ".join(
        f"{'.'.join(map(str, e['loc'])) or 'reply'}: {e['msg']}" for e in error.errors()
    )


def _reason_directly(
    *,
    render: Callable[..., str],
    parse_json: str,
    user_message: str,
    model: str,
    splits: Splits,
    citable: Mapping[str, Sequence[str]],
    max_attempts: int,
    run: StageRun | None,
    **llm_kwargs: Any,
) -> ReasoningResult | dict[str, Any]:
    """The direct path's answer, citing from among ``citable``'s ids, or, when its
    prompt does not fit or no reply fits its schema in ``max_attempts`` sends, why not:
    the ``fallback`` of a :class:`ReasoningResult`."""
    system_prompt = render(parse_block=parse_json)
    schema = _reply_schema(citable)
    schema_json = json.dumps(schema_as_sent(schema))
    if estimate := splits.over_line(system_prompt, user_message, schema_json):
        _log.info(
            "reasoning's direct prompt would take about %s tokens, over the %s-token "
            "line; reasoning with code instead",
            f"{estimate:,}",
            f"{splits.line:,}",
        )
        return {"cause": "estimate", "estimate": estimate, "line": splits.line}
    try:
        reply = send_until_valid(
            system_prompt,
            user_message,
            schema=schema,
            rewrite=_REWRITE,
            kind="direct",
            max_attempts=max_attempts,
            model=model,
            run=run,
            **llm_kwargs,
        )
    except ContextWindowExceededError as refusal:
        _log.warning(
            "reasoning's direct prompt was refused as too long; reasoning with code "
            "instead"
        )
        return {"cause": "refusal", "error": masked(str(refusal))}
    except ValidationError as invalid:
        _log.warning(
            "reasoning's direct reply did not fit its schema in %d send(s); reasoning "
            "with code instead",
            max_attempts,
        )
        return {"cause": "invalid", "error": _in_brief(invalid)}
    reply = cast(_DirectReply, reply)
    return ReasoningResult(
        answer=reply.answer.strip(),
        path="direct",
        provenance=_cited(reply.provenance_ids.model_dump(), citable, "direct"),
    )


def reason(
    *,
    task: str,
    schema_code: str,
    parsed: BaseModel | dict[str, Any],
    model: str,
    prompts: Mapping[PromptStage, str],
    path: ReasoningPath,
    document_ids: Sequence[str],
    relevance_snippets: Sequence[Snippet] | None = None,
    source_docs: dict[str, list[int]] | None = None,
    max_attempts: int | None = None,
    max_turns: int | None = None,
    timeout_s: float | None = DEFAULT_EXEC_TIMEOUT_S,
    additional_authorized_imports: list[str] | None = None,
    run: StageRun | None = None,
    splits: Splits | None = None,
    **llm_kwargs: Any,
) -> ReasoningResult:
    """Answer ``task`` over ``parsed`` and the relevant context, on ``path``.

    ``prompts`` and ``path`` are a run config's ``prompts`` and ``reasoning``.
    ``document_ids`` is each document's id for the run
    (:func:`r3con.stages.relevance.draw_document_ids`), aligned with
    ``relevance_snippets`` and the indices in ``source_docs``. ``max_attempts``
    (``None`` reads ``settings.REASONING_MAX_ATTEMPTS``) caps the direct request's
    sends; ``max_turns``, ``timeout_s`` and ``additional_authorized_imports`` apply to
    the coding path. ``splits`` (``None`` builds one for ``model``) says whether a
    prompt fits the model's window.

    Raises:
        ValueError: if ``relevance_snippets`` is not one snippet per document id,
            ``source_docs`` names a document ``document_ids`` has no id for, or
            ``max_attempts`` is below 1.
    """
    uncovered = sorted(
        {i for idxs in (source_docs or {}).values() for i in idxs}
        - set(range(len(document_ids)))
    )
    if uncovered:
        raise ValueError(
            f"source_docs refers to document index(es) {uncovered}, but document_ids "
            f"names {len(document_ids)} document(s): give one id per document."
        )
    max_attempts = settings.check_cap(
        "reasoning_max_attempts",
        settings.REASONING_MAX_ATTEMPTS if max_attempts is None else max_attempts,
        name="max_attempts",
    )
    parse_dict = (
        parsed.model_dump(mode="json") if isinstance(parsed, BaseModel) else parsed
    )
    parse_dict = tag_source_documents(parse_dict, source_docs, document_ids)
    parse_json = json.dumps(parse_dict, indent=2, ensure_ascii=False, default=repr)
    citable: dict[str, list[str]] = dict(
        zip(
            _VIEWS,
            [
                citable_ids(relevance_snippets, document_ids),
                _documents_with_records(parse_dict, document_ids),
            ],
            strict=True,
        )
    )
    splits = splits if splits is not None else Splits([], model=model)
    render = functools.partial(
        _system_prompt,
        task=task,
        schema_code=schema_code,
        relevance_snippets=relevance_snippets,
        document_ids=document_ids,
        parse_json=parse_json,
        samples_block=_sample_record_per_field(parse_dict),
    )
    user_message = f"Input:\n<task>\n{task}\n</task>"
    with_code = functools.partial(
        _reason_with_code,
        render=functools.partial(
            render, stage="reasoning/coding", version=prompts["reasoning/coding"]
        ),
        parse_dict=parse_dict,
        parse_json=parse_json,
        user_message=user_message,
        model=model,
        splits=splits,
        citable=citable,
        max_turns=max_turns,
        timeout_s=timeout_s,
        additional_authorized_imports=additional_authorized_imports,
        run=run,
        **llm_kwargs,
    )
    if path == "coding":
        return with_code()
    outcome = _reason_directly(
        render=functools.partial(
            render, stage="reasoning/direct", version=prompts["reasoning/direct"]
        ),
        parse_json=parse_json,
        user_message=user_message,
        model=model,
        splits=splits,
        citable=citable,
        max_attempts=max_attempts,
        run=run,
        **llm_kwargs,
    )
    if isinstance(outcome, ReasoningResult):
        return outcome
    # A prompt that did not fit would not fit with the whole parse either.
    return replace(with_code(whole=outcome["cause"] == "invalid"), fallback=outcome)
