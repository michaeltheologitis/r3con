"""The provenance of an answer: the graph of what cites what in its run, the rules a
walk over it may follow, and the walk, whose result is the answer's trail.

Each node of the graph is a string:

- ``answer``, the answer;
- ``summary:<round>:<id>``, the relevance snippet ``<id>`` of round ``<round>``, from
  1: its document's id, or ``<id>.<k>`` for part ``k`` of a document read in parts in
  that round (:mod:`r3con.stages.relevance`);
- ``parse:<id>``, the parse of the document ``<id>``;
- ``document:<id>``, the document ``<id>``.

Each edge is ``[from, to, kind]``, of three kinds:

- ``cites``: from the answer to each summary of the relevant context
  (:attr:`~r3con.stages.relevance.RelevantContext.context_round`) and each parse it
  cites, from a parse to each summary of the relevant context it cites, and from a
  summary of round k to each summary of round k-1 it cites. A summary of the relevant
  context is its round's, or, for one skipped there, the earlier round's kept in its
  place (:attr:`~r3con.stages.relevance.RelevantContext.note_rounds`), so a citation
  of it leads there, never to the skipped one;
- ``summarizes``: from every summary to its document, carrying the summary's quotes
  found there;
- ``parses``: from every parse to its document.

What a call was shown but did not cite is no edge.

A rule (:data:`RULES`) names the kinds of edge a walk follows. :func:`walk` goes
breadth-first from the answer along them, and returns the trail: what it reached and
the edges it followed. :func:`build_provenance` gives an answer's whole provenance:
its citations, the trail under a rule, the documents the trail reaches, and the quotes
the trail's ``summarizes`` edges carry. A run builds it under :data:`DEFAULT_RULE`
(:attr:`r3con.pipeline.Answer.provenance`); :func:`r3con.trace` builds it again from a
saved run's folder, under any rule.

An answer's provenance adds its ``claims`` (:class:`AnswerProvenance`), which a last
model call finds after reasoning (:mod:`r3con.stages.claims`). A claim names its quotes
by their place, the same under any rule, so it finds them among the ``quotes`` of a
trail under any rule.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from typing import TypedDict

from r3con.quotes import TIERS, Match, Quote
from r3con.stages.relevance import RelevantContext

ANSWER = "answer"
CITES = "cites"
SUMMARIZES = "summarizes"
PARSES = "parses"

DEFAULT_RULE = "through-summaries"
RULES: dict[str, frozenset[str]] = {
    # A document is reached only through a summary of it.
    DEFAULT_RULE: frozenset({CITES, SUMMARIZES}),
    # A cited parse also reaches its document, whole.
    "parses-light-documents": frozenset({CITES, SUMMARIZES, PARSES}),
}
"""Each rule by name: the kinds of edge a walk under it follows."""

Edge = list[str]
"""``[from, to, kind]``, in memory as in JSON."""


class Citations(TypedDict):
    """What an answer cites: the ids of the relevance snippets it used (``<id>.<k>`` for
    a part of a document read in parts), and those of the documents whose records it
    used."""

    summary_ids: list[str]
    parse_ids: list[str]


class NamedQuote(TypedDict):
    """A quote behind an answer, ``{document, name, start, end, text, document_text,
    match, score}``: the id and the name of the document it is in; where it is there
    (``documents[i][start:end]`` for the document whose id is ``document``); ``text``,
    the model's copy of it, and ``document_text``, the document's own words, that
    slice, which differ from the copy for every match but an ``exact`` one; and
    ``match``, the tier that found it, with the ``score`` of a ``fuzzy`` one
    (:class:`r3con.quotes.Quote`). A run and a resume always give ``document_text``.
    It is ``None`` only when a run folder written before quotes kept it is read without
    its documents: by :func:`r3con.trace` (:func:`r3con.pipeline.read_answer`) or
    :func:`r3con.ask`.
    """

    document: str
    name: str
    start: int
    end: int
    text: str
    document_text: str | None
    match: Match
    score: float | None


class NamedDocument(TypedDict):
    """A document behind an answer, ``{document, name}``: its id and its name."""

    document: str
    name: str


class Trail(TypedDict):
    """What :func:`walk` reached from the answer under ``rule``: ``nodes``, in the order
    it reached them, the answer first, and ``edges``, those it followed, in the order
    it followed them."""

    rule: str
    nodes: list[str]
    edges: list[Edge]


class Provenance(Citations):
    """An answer's provenance: its citations, ``{summary_ids, parse_ids}``, the
    ``documents`` behind it, those its trail reaches, in document order, the ``quotes``
    behind it, those its trail's ``summarizes`` edges carry, and its ``trail``."""

    documents: list[NamedDocument]
    quotes: list[NamedQuote]
    trail: Trail


class QuotePlace(TypedDict):
    """Which quote behind an answer a claim rests on, ``{document, start, end}``: its
    document's id and where it is there, as its entry of ``quotes`` gives them."""

    document: str
    start: int
    end: int


class Claim(TypedDict):
    """A claim the answer makes, ``{text, start, end, documents, quotes}``: one of its
    sentences, list items or table rows, ``text``, which is ``answer[start:end]``; the
    ids of the documents whose evidence supports it, in document order; and the quotes
    that support it, by place, in the order of ``quotes``, none at the document
    evidence level."""

    text: str
    start: int
    end: int
    documents: list[str]
    quotes: list[QuotePlace]


class AnswerProvenance(Provenance):
    """An answer's provenance with its ``claims``, in the order of the answer."""

    claims: list[Claim]


@dataclass(frozen=True)
class Graph:
    """The provenance graph of one answer's run (:func:`build_graph`).

    ``edges`` come from the answer back: the answer's, then each parse's, in document
    order, then each summary's, from the last round to the first, each round's in
    document order. ``quotes`` holds what each summary's ``summarizes`` edge carries,
    by the summary's node, in the same order, and ``documents`` each document's id, in
    document order.
    """

    edges: list[Edge]
    quotes: dict[str, list[NamedQuote]]
    documents: list[str]


def build_provenance(
    citations: Citations,
    *,
    relevance: RelevantContext,
    parsing_provenance: Sequence[Sequence[str]],
    names: Mapping[str, str],
    rule: str = DEFAULT_RULE,
) -> Provenance:
    """The provenance of an answer that cites ``citations``, its trail walked under
    ``rule`` over its graph (:func:`build_graph`).

    Its documents are those the trail reaches, in document order, each with its name.
    Its quotes are those the trail's ``summarizes`` edges carry, each place once, as
    the summary that matched it best copied it: by tier, ``exact``, ``normalized``,
    ``elided``, then ``fuzzy``, a fuzzy one by its score, and among equals the first in
    the graph's order, so the latest round's; in document order, then by ``start``.

    Raises:
        ValueError: if no rule is named ``rule``.
    """
    graph = build_graph(
        citations,
        relevance=relevance,
        parsing_provenance=parsing_provenance,
        names=names,
    )
    trail = walk(graph, rule)
    return Provenance(
        summary_ids=citations["summary_ids"],
        parse_ids=citations["parse_ids"],
        documents=_documents(trail, names),
        quotes=_quotes(graph, trail),
        trail=trail,
    )


def build_graph(
    citations: Citations,
    *,
    relevance: RelevantContext,
    parsing_provenance: Sequence[Sequence[str]],
    names: Mapping[str, str],
) -> Graph:
    """The provenance graph of an answer that cites the summaries of the relevant
    context and the parses ``citations`` lists, over its run's ``relevance``, each
    document's parse citing the summaries of the relevant context its
    ``parsing_provenance`` lists, ``names`` giving each document's name by its id, in
    document order."""
    rounds = relevance.note_rounds

    def note(id_: str) -> str:
        return _summary(rounds.get(id_, relevance.context_round), id_)

    edges = [[ANSWER, note(id_), CITES] for id_ in citations["summary_ids"]]
    edges += [[ANSWER, _parse(id_), CITES] for id_ in citations["parse_ids"]]
    for document, cited in zip(names, parsing_provenance, strict=True):
        edges += [[_parse(document), note(id_), CITES] for id_ in cited]
        edges.append([_parse(document), _document(document), PARSES])
    quotes: dict[str, list[NamedQuote]] = {}
    for round_ in range(len(relevance.records), 0, -1):
        for record in chain.from_iterable(relevance.records[round_ - 1]):
            summary = _summary(round_, record.id)
            document = record.id.partition(".")[0]
            edges += [
                [summary, _summary(round_ - 1, id_), CITES] for id_ in record.provenance
            ]
            edges.append([summary, _document(document), SUMMARIZES])
            quotes[summary] = _found(record.quotes, document, names[document])
    return Graph(edges=edges, quotes=quotes, documents=list(names))


def check_rule(rule: str) -> None:
    """Refuse a rule that is not one of :data:`RULES`.

    Raises:
        ValueError: if no rule is named ``rule``.
    """
    if rule not in RULES:
        known = ", ".join(repr(name) for name in RULES)
        raise ValueError(
            f"no provenance rule is named {rule!r}; the rules are {known}."
        )


def walk(graph: Graph, rule: str = DEFAULT_RULE) -> Trail:
    """The trail of ``graph``'s answer under ``rule``: breadth-first from the answer,
    along only the edges of the kinds the rule names (:data:`RULES`), every edge
    leaving a node it reaches followed, each node reached once.

    Raises:
        ValueError: if no rule is named ``rule``.
    """
    check_rule(rule)
    leaving: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        source, _, kind = edge
        if kind in RULES[rule]:
            leaving.setdefault(source, []).append(edge)
    reached = dict.fromkeys([ANSWER])
    followed: list[Edge] = []
    queue = deque(reached)
    while queue:
        out = leaving.get(queue.popleft(), [])
        followed += out
        new = dict.fromkeys(to for _, to, _ in out if to not in reached)
        reached |= new
        queue.extend(new)
    return Trail(rule=rule, nodes=list(reached), edges=followed)


def _summary(round_: int, id_: str) -> str:
    return f"summary:{round_}:{id_}"


def _parse(document: str) -> str:
    return f"parse:{document}"


def _document(document: str) -> str:
    return f"document:{document}"


def _found(quotes: Iterable[Quote], document: str, name: str) -> list[NamedQuote]:
    """The ``quotes`` found in ``document``, named ``name``."""
    return [
        NamedQuote(
            document=document,
            name=name,
            start=start,
            end=end,
            text=quote["text"],
            # A run folder written before quotes kept document_text has none.
            document_text=quote.get("document_text"),
            match=match,
            score=quote["score"],
        )
        for quote in quotes
        if (start := quote["start"]) is not None
        and (end := quote["end"]) is not None
        and (match := quote["match"]) is not None
    ]


def _documents(trail: Trail, names: Mapping[str, str]) -> list[NamedDocument]:
    """The documents ``trail`` reaches, ``names`` giving each one's name by its id, in
    document order."""
    reached = set(trail["nodes"])
    return [
        NamedDocument(document=id_, name=name)
        for id_, name in names.items()
        if _document(id_) in reached
    ]


def _rank(quote: NamedQuote) -> tuple[int, float]:
    """How well ``quote`` was matched, the best lowest: its tier, in the order
    :func:`r3con.quotes.locate_quotes` tries them, then a fuzzy one's score."""
    return TIERS.index(quote["match"]), -(quote["score"] or 0)


def _quotes(graph: Graph, trail: Trail) -> list[NamedQuote]:
    """The quotes the ``summarizes`` edges ``trail`` followed carry, as
    :func:`build_provenance` gives them."""
    summaries = {source for source, _, kind in trail["edges"] if kind == SUMMARIZES}
    carried = chain.from_iterable(
        quotes for summary, quotes in graph.quotes.items() if summary in summaries
    )
    by_place: dict[tuple[str, int, int], NamedQuote] = {}
    for quote in sorted(carried, key=_rank):
        by_place.setdefault((quote["document"], quote["start"], quote["end"]), quote)
    order = {id_: i for i, id_ in enumerate(graph.documents)}
    return sorted(by_place.values(), key=lambda q: (order[q["document"]], q["start"]))
