"""The provenance of an answer: the graph of what cites what in its run, the rules a
walk over it may follow, and the walk, whose result is the answer's trail.

Each node of the graph is a string:

- ``answer``, the answer;
- ``summary:<round>:<id>``, the relevance snippet ``<id>`` of round ``<round>``, from
  1: its document's id, or ``<id>.<k>`` for part ``k`` of a document read in parts in
  that round (:mod:`r3con.stages.relevance`);
- ``parse:<id>``, the parse of the document ``<id>``;
- ``document:<id>``, the document ``<id>``.

Each edge is ``(from, to, kind)``, of three kinds:

- ``cites``: from the answer to each final-round summary and each parse it cites, from
  a parse to each final-round summary it cites, and from a summary of round k to each
  summary of round k-1 it cites;
- ``summarizes``: from every summary to its document, carrying the summary's quotes
  found there;
- ``parses``: from every parse to its document.

What a call was shown but did not cite is no edge.

A rule (:data:`RULES`) names the kinds of edge a walk follows. :func:`walk` goes
breadth-first from the answer along them, and returns the trail: what it reached, the
edges it followed and the quotes they carry. A run walks its answer's graph under
:data:`DEFAULT_RULE` (:attr:`r3con.pipeline.Answer.provenance`); :func:`trail` walks a
saved run's again, under any rule, from its folder alone.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from r3con.quotes import Quote
from r3con.stages.relevance import RelevantContext

if TYPE_CHECKING:
    from r3con.pipeline import Answer

ANSWER = "answer"
CITES = "cites"
SUMMARIZES = "summarizes"
PARSES = "parses"

RULES: dict[str, frozenset[str]] = {
    # A document is reached only through a summary of it.
    "through-summaries": frozenset({CITES, SUMMARIZES}),
    # A cited parse reaches its document too, as the app's page lights it.
    "parses-light-documents": frozenset({CITES, SUMMARIZES, PARSES}),
}
"""Each rule by name: the kinds of edge a walk under it follows."""
DEFAULT_RULE = "through-summaries"

Edge = tuple[str, str, str]
"""``(from, to, kind)``."""


class NamedQuote(TypedDict):
    """A quote behind an answer, ``{document, start, end, text, name}``: the id of the
    document it is in, where it is there (``documents[i][start:end]`` for the document
    whose id is ``document``), the model's copy of it, which for a ``fuzzy`` match
    differs slightly from that slice, and the document's name."""

    document: str
    start: int
    end: int
    text: str
    name: str


class Trail(TypedDict):
    """What :func:`walk` reached from the answer under ``rule``, as JSON: ``nodes``, in
    the order it reached them, the answer first; ``edges``, each ``[from, to, kind]``,
    those it followed, in the order it followed them; and ``quotes``, those the
    ``summarizes`` edges it followed carry, each place once, in document order, then by
    ``start``."""

    rule: str
    nodes: list[str]
    edges: list[list[str]]
    quotes: list[NamedQuote]


@dataclass(frozen=True)
class Graph:
    """The provenance graph of one answer's run (:func:`build_graph`).

    ``edges`` come from the answer back: the answer's, then each parse's, in document
    order, then each summary's, from the last round to the first, each round's in
    document order. ``quotes`` holds what each summary's ``summarizes`` edge carries,
    by the summary's node, in the same order, and ``names`` each document's name by its
    id, in document order.
    """

    edges: list[Edge]
    quotes: dict[str, list[NamedQuote]]
    names: dict[str, str]


def build_graph(
    *,
    summary_ids: Sequence[str],
    parse_ids: Sequence[str],
    relevance: RelevantContext,
    parsing_provenance: Sequence[Sequence[str]],
    names: Mapping[str, str],
) -> Graph:
    """The provenance graph of an answer that cites the final-round summaries
    ``summary_ids`` and the parses of the documents ``parse_ids``, over its run's
    ``relevance``, each document's parse citing the final-round summaries its
    ``parsing_provenance`` lists, ``names`` giving each document's name by its id, in
    document order."""
    last = len(relevance.records)
    edges = [(ANSWER, _summary(last, id_), CITES) for id_ in summary_ids]
    edges += [(ANSWER, _parse(id_), CITES) for id_ in parse_ids]
    for document, cited in zip(names, parsing_provenance, strict=True):
        edges += [(_parse(document), _summary(last, id_), CITES) for id_ in cited]
        edges.append((_parse(document), _document(document), PARSES))
    quotes: dict[str, list[NamedQuote]] = {}
    for round_ in range(last, 0, -1):
        for record in chain.from_iterable(relevance.records[round_ - 1]):
            summary = _summary(round_, record.id)
            document = record.id.partition(".")[0]
            edges += [
                (summary, _summary(round_ - 1, id_), CITES) for id_ in record.provenance
            ]
            edges.append((summary, _document(document), SUMMARIZES))
            quotes[summary] = _found(record.quotes, document, names[document])
    return Graph(edges=edges, quotes=quotes, names=dict(names))


def answer_alone() -> Trail:
    """The trail of an answer that cites nothing, under the default rule: the answer
    alone."""
    return Trail(rule=DEFAULT_RULE, nodes=[ANSWER], edges=[], quotes=[])


def walk(graph: Graph, rule: str = DEFAULT_RULE) -> Trail:
    """The trail of ``graph``'s answer under ``rule``: breadth-first from the answer,
    along only the edges of the kinds the rule names (:data:`RULES`), every edge
    leaving a node it reaches followed, each node reached once.

    Raises:
        ValueError: if no rule is named ``rule``.
    """
    if rule not in RULES:
        known = ", ".join(repr(name) for name in RULES)
        raise ValueError(
            f"no provenance rule is named {rule!r}; the rules are {known}."
        )
    leaving: dict[str, list[Edge]] = {}
    for edge in graph.edges:
        if edge[2] in RULES[rule]:
            leaving.setdefault(edge[0], []).append(edge)
    reached = dict.fromkeys([ANSWER])
    followed: list[Edge] = []
    queue = deque(reached)
    while queue:
        out = leaving.get(queue.popleft(), [])
        followed += out
        new = dict.fromkeys(to for _, to, _ in out if to not in reached)
        reached |= new
        queue.extend(new)
    return Trail(
        rule=rule,
        nodes=list(reached),
        edges=[list(edge) for edge in followed],
        quotes=_quotes(graph, followed),
    )


def trail(source: Answer | str | Path, rule: str = DEFAULT_RULE) -> Trail:
    """The trail of the answer ``source`` saved, walked under ``rule`` over its run's
    graph, rebuilt from the run's saved files alone, with no model call.

    ``source`` is an :class:`~r3con.pipeline.Answer`, whose ``run_dir`` is read, or the
    path of a run folder or of an ask's folder (:func:`r3con.ask`): an ask's answer is
    its own, and its summaries and parses those of the run it asked over. Under
    :data:`DEFAULT_RULE` it is the ``trail`` of the answer's saved ``provenance``.

    Raises:
        FileNotFoundError: if ``source`` is not a run folder or an ask's, if its run
            never finished parsing, if it holds no answer, as when its reasoning
            failed, or if a prompt version its run pins has no file.
        ValueError: if ``source`` is an Answer without a ``run_dir``, if a file of its
            run is not as this r3con writes it, as :func:`r3con.ask` says, or if no
            rule is named ``rule``.
    """
    # Imported here, as the pipeline imports this module.
    from r3con.pipeline import read_representation

    folder = Path(source) if isinstance(source, (str, Path)) else source.run_dir
    if folder is None:
        raise ValueError(
            "trail needs the run folder, and this Answer has none: its run was made "
            "with save_artifacts=False."
        )
    built = read_representation(folder)
    answered = folder / "reasoning" / "result.json"
    if not answered.is_file():
        raise FileNotFoundError(
            f"{folder} holds no answer: reasoning/result.json is missing."
        )
    cited = json.loads(answered.read_text(encoding="utf-8"))["provenance"]
    graph = build_graph(
        summary_ids=cited["summary_ids"],
        parse_ids=cited["parse_ids"],
        relevance=built.relevance,
        parsing_provenance=built.parsing_provenance,
        names=built.names,
    )
    return walk(graph, rule)


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
            document=document, start=start, end=end, text=quote["text"], name=name
        )
        for quote in quotes
        if (start := quote["start"]) is not None and (end := quote["end"]) is not None
    ]


def _quotes(graph: Graph, followed: Iterable[Edge]) -> list[NamedQuote]:
    """The quotes the ``summarizes`` edges among ``followed`` carry, each place once,
    as the first summary quoting it in ``graph``'s order copied it, so the latest
    round's copy, in document order, then by ``start``."""
    summaries = {source for source, _, kind in followed if kind == SUMMARIZES}
    carried = chain.from_iterable(
        quotes for summary, quotes in graph.quotes.items() if summary in summaries
    )
    by_place: dict[tuple[str, int, int], NamedQuote] = {}
    for quote in carried:
        by_place.setdefault((quote["document"], quote["start"], quote["end"]), quote)
    order = {id_: i for i, id_ in enumerate(graph.names)}
    return sorted(by_place.values(), key=lambda q: (order[q["document"]], q["start"]))
