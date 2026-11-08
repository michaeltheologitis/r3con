import json
import logging

import litellm
import pytest

from r3con import ask, run
from r3con.provenance import answer_alone, build_graph, trail, walk
from r3con.runtime.llm import schema_as_sent
from r3con.stages.provenance import CitingRequest, citing_schema, read_citing
from r3con.stages.relevance import RelevantContext, SnippetRecord
from tests.fakes import citable_lists, is_later_round

MODEL = "openai/gpt-6-luna"
SHOWN = ["a1", "b2", "c3"]
NAMES = {"a": "a.txt", "b": "b.txt", "c": "c.txt"}
TWO_LINE_MEMOS = {
    f"q3/{name}.txt": f"{name.title()} memo\nThe {name.title()} memo logs two sites."
    for name in ["halloran", "merrow", "calder"]
}


def read(request, completion):
    return read_citing(
        request,
        "The document.",
        rewrite="Read it again",
        kind="test",
        max_attempts=1,
        model=MODEL,
        completion=completion,
    )


def test_the_payload_comes_first_in_order_then_the_ids_it_cites():
    reply = citing_schema("Reply", SHOWN, note=(str, ...), page=(int, ...))
    schema = schema_as_sent(reply)
    assert list(schema["properties"]) == ["note", "page", "provenance_ids"]
    assert schema["properties"]["provenance_ids"]["items"]["enum"] == SHOWN


def test_unshown_and_repeated_ids_are_dropped_and_the_rest_kept_in_order(llm, caplog):
    request = CitingRequest(
        "Prompt.", citing_schema("Reply", SHOWN, note=(str, ...)), SHOWN
    )
    reply = {"note": "A note.", "provenance_ids": ["c3", "zz", "a1", "c3", "zz"]}
    with caplog.at_level(logging.WARNING, logger="r3con.provenance"):
        read_reply, provenance = read(request, llm.replies(reply))
    assert (read_reply.note, provenance) == ("A note.", ["c3", "a1"])
    assert caplog.messages == [
        "test: dropped provenance id(s) the call did not show: ['zz', 'zz']"
    ]


def test_a_reply_with_nothing_to_cite_has_no_ids_and_cites_nothing(llm):
    request = CitingRequest(
        "Prompt.", citing_schema("Reply", None, note=(str, ...)), []
    )
    read_reply, provenance = read(request, llm.replies({"note": "A note."}))
    assert (read_reply.note, provenance) == ("A note.", [])
    sent = llm.requests[0]["response_format"]["json_schema"]["schema"]
    assert list(sent["properties"]) == ["note"]


def snippet(id_: str, *quotes: dict, cites: tuple[str, ...] = ()) -> SnippetRecord:
    return SnippetRecord(
        id=id_, summary=f"Notes on {id_}.", quotes=list(quotes), provenance=list(cites)
    )


def at(start: int, end: int, text: str = "A sentence.") -> dict:
    """A quote found in its document from ``start`` to ``end``."""
    return {"text": text, "start": start, "end": end, "match": "exact", "score": None}


def quoted(document: str, start: int, end: int, text: str = "A sentence.") -> dict:
    """A quote behind an answer, as the trail gives it."""
    name = NAMES[document]
    return {
        "document": document,
        "name": name,
        "start": start,
        "end": end,
        "text": text,
    }


def graph_of(rounds, *, summary_ids=(), parse_ids=(), parses=None):
    """The graph of an answer citing ``summary_ids`` and ``parse_ids`` over the
    documents a, b and c, with ``rounds``, each a round's snippets, and each document's
    parse citing the final-round snippets ``parses`` lists for it, none by default."""
    relevance = RelevantContext(
        records=[
            [[s for s in round_ if s.id.partition(".")[0] == id_] for id_ in NAMES]
            for round_ in rounds
        ]
    )
    return build_graph(
        summary_ids=summary_ids,
        parse_ids=parse_ids,
        relevance=relevance,
        parsing_provenance=parses or [[] for _ in NAMES],
        names=NAMES,
    )


def quotes_behind(rounds, cited=()) -> list:
    """The quotes on the trail from the final-round snippets ``cited`` names."""
    return walk(graph_of(rounds, summary_ids=cited))["quotes"]


def places(quotes: list) -> list[tuple]:
    return [(q["document"], q["start"], q["end"]) for q in quotes]


# Two rounds; the answer cites b's final summary, which cites a's first, and c's parse,
# which cites nothing.
CITED_ROUNDS = [
    [snippet("a", at(0, 5)), snippet("b"), snippet("c")],
    [snippet("a"), snippet("b", at(10, 15), cites=("a",)), snippet("c")],
]


def test_the_graph_names_each_part_of_a_split_document_by_its_round():
    first = [snippet("a"), snippet("b.1", at(0, 5)), snippet("b.2"), snippet("c")]
    second = [
        snippet("a", cites=("b.2",)),
        snippet("b.1"),
        snippet("b.2"),
        snippet("b.3", cites=("a",)),
        snippet("c"),
    ]
    graph = graph_of(
        [first, second], summary_ids=["b.3"], parse_ids=["c"], parses=[[], [], ["a"]]
    )
    assert graph.edges == [
        ("answer", "summary:2:b.3", "cites"),
        ("answer", "parse:c", "cites"),
        ("parse:a", "document:a", "parses"),
        ("parse:b", "document:b", "parses"),
        ("parse:c", "summary:2:a", "cites"),
        ("parse:c", "document:c", "parses"),
        ("summary:2:a", "summary:1:b.2", "cites"),
        ("summary:2:a", "document:a", "summarizes"),
        ("summary:2:b.1", "document:b", "summarizes"),
        ("summary:2:b.2", "document:b", "summarizes"),
        ("summary:2:b.3", "summary:1:a", "cites"),
        ("summary:2:b.3", "document:b", "summarizes"),
        ("summary:2:c", "document:c", "summarizes"),
        ("summary:1:a", "document:a", "summarizes"),
        ("summary:1:b.1", "document:b", "summarizes"),
        ("summary:1:b.2", "document:b", "summarizes"),
        ("summary:1:c", "document:c", "summarizes"),
    ]
    assert graph.quotes["summary:1:b.1"] == [quoted("b", 0, 5)]


def test_through_summaries_reaches_a_document_only_through_a_summary_of_it():
    assert walk(graph_of(CITED_ROUNDS, summary_ids=["b"], parse_ids=["c"])) == {
        "rule": "through-summaries",
        "nodes": [
            "answer",
            "summary:2:b",
            "parse:c",
            "summary:1:a",
            "document:b",
            "document:a",
        ],
        "edges": [
            ["answer", "summary:2:b", "cites"],
            ["answer", "parse:c", "cites"],
            ["summary:2:b", "summary:1:a", "cites"],
            ["summary:2:b", "document:b", "summarizes"],
            ["summary:1:a", "document:a", "summarizes"],
        ],
        "quotes": [quoted("a", 0, 5), quoted("b", 10, 15)],
    }


@pytest.mark.parametrize(
    ("rule", "documents"),
    [
        ("through-summaries", ["document:b", "document:a"]),
        ("parses-light-documents", ["document:b", "document:c", "document:a"]),
    ],
)
def test_a_cited_parse_that_cites_nothing_lights_its_document_only_if_parses_do(
    rule, documents
):
    graph = graph_of(CITED_ROUNDS, summary_ids=["b"], parse_ids=["c"])
    walked = walk(graph, rule)
    assert walked["rule"] == rule
    assert [node for node in walked["nodes"] if node.startswith("document:")] == (
        documents
    )
    assert walked["quotes"] == walk(graph)["quotes"]


def test_a_node_reached_twice_is_listed_once_and_both_edges_to_it_followed():
    graph = graph_of(
        [CITED_ROUNDS[0]], summary_ids=["a"], parse_ids=["a"], parses=[["a"], [], []]
    )
    walked = walk(graph)
    assert walked["nodes"] == ["answer", "summary:1:a", "parse:a", "document:a"]
    assert walked["edges"] == [
        ["answer", "summary:1:a", "cites"],
        ["answer", "parse:a", "cites"],
        ["summary:1:a", "document:a", "summarizes"],
        ["parse:a", "summary:1:a", "cites"],
    ]


def test_an_unknown_rule_is_refused_naming_the_known_ones():
    with pytest.raises(ValueError) as refused:
        walk(graph_of(CITED_ROUNDS), "through-parses")
    assert str(refused.value) == (
        "no provenance rule is named 'through-parses'; the rules are "
        "'through-summaries', 'parses-light-documents'."
    )


def test_the_trail_quotes_the_final_round_summaries_cited():
    final = [
        snippet("a", at(0, 9, "A said x.")),
        snippet("b", at(3, 9)),
        snippet("c", at(4, 20, "C, it said, saw y.")),
    ]
    assert quotes_behind([final], cited=["a", "c"]) == [
        quoted("a", 0, 9, "A said x."),
        quoted("c", 4, 20, "C, it said, saw y."),
    ]


def test_the_trail_quotes_the_earlier_summaries_its_summaries_cite_round_by_round():
    first = [snippet("a", at(0, 5)), snippet("b", at(10, 15)), snippet("c", at(20, 25))]
    second = [
        snippet("a", at(30, 35)),
        snippet("b", at(40, 45), cites=("c",)),
        snippet("c", at(50, 55)),
    ]
    third = [snippet("a", at(60, 65), cites=("b",)), snippet("b"), snippet("c")]
    quotes = quotes_behind([first, second, third], cited=["a"])
    assert places(quotes) == [("a", 60, 65), ("b", 40, 45), ("c", 20, 25)]


def test_the_trail_quotes_the_summaries_a_cited_parse_cites_and_theirs():
    first = [snippet("a", at(0, 5)), snippet("b", at(10, 15)), snippet("c")]
    second = [snippet("a", at(30, 35), cites=("b",)), snippet("b"), snippet("c")]
    graph = graph_of([first, second], parse_ids=["c", "b"], parses=[[], ["a"], ["a"]])
    assert places(walk(graph)["quotes"]) == [("a", 30, 35), ("b", 10, 15)]


def test_a_part_is_cited_by_its_part_id_and_quoted_under_its_documents_id():
    first = [
        snippet("a", at(0, 5)),
        snippet("b.1", at(10, 15)),
        snippet("b.2", at(5_000, 5_005)),
        snippet("c", at(20, 25)),
    ]
    second = [
        snippet("a", at(30, 35), cites=("b.2",)),
        snippet("b.1", at(40, 45)),
        snippet("b.2", at(5_040, 5_045)),
        snippet("c", at(50, 55)),
    ]
    quotes = quotes_behind([first, second], cited=["a", "b.1"])
    assert places(quotes) == [("a", 30, 35), ("b", 40, 45), ("b", 5_000, 5_005)]


def test_a_quote_not_found_in_its_document_is_left_out():
    def not_found(text: str) -> dict:
        return {"text": text, "start": None, "end": None, "match": None, "score": None}

    final = [
        snippet("a", not_found("Not in a."), at(0, 5)),
        snippet("b", not_found("Not in b.")),
        snippet("c"),
    ]
    assert places(quotes_behind([final], cited=["a", "b"])) == [("a", 0, 5)]


def test_a_place_quoted_more_than_once_on_the_trail_is_given_once_as_first_met():
    first = [
        snippet("a", at(0, 9, "As round 1 copied it.")),
        snippet("b"),
        snippet("c"),
    ]
    second = [
        snippet("a", at(0, 9, "As round 2 copied it.")),
        snippet("b", at(0, 9), cites=("a",)),
        snippet("c", cites=("a",)),
    ]
    assert quotes_behind([first, second], cited=["a", "b", "c"]) == [
        quoted("a", 0, 9, "As round 2 copied it."),
        quoted("b", 0, 9),
    ]


def test_quotes_come_in_document_order_then_by_where_they_start():
    final = [
        snippet("a", at(50, 60), at(0, 10)),
        snippet("b", at(5, 8)),
        snippet("c", at(1, 2)),
    ]
    quotes = quotes_behind([final], cited=["c", "b", "a"])
    assert places(quotes) == [("a", 0, 10), ("a", 50, 60), ("b", 5, 8), ("c", 1, 2)]


@pytest.mark.parametrize("n_rounds", [0, 2])
def test_a_trail_from_no_snippet_has_no_quotes(n_rounds):
    assert quotes_behind(CITED_ROUNDS[:n_rounds]) == []


def quoting_a_line_a_round(shown_ids):
    """A relevance reply quoting its memo's first line in round 1 and its second
    later, then citing the first summary shown."""

    def reply(request):
        first, second = request["messages"][1]["content"].splitlines()
        if not is_later_round(request):
            return {"summary": f"Notes on {first}.", "quotes": [first]}
        cites = shown_ids(request)[:1]
        return {
            "summary": f"Notes on {first}.",
            "quotes": [second],
            "provenance_ids": cites,
        }

    return reply


def citing(summaries: slice, parses: slice):
    """A direct reasoning reply citing the ``summaries`` and ``parses`` of those its
    schema lists."""

    def reply(request) -> dict:
        lists = citable_lists(request)
        return {
            "answer": "Halloran memo",
            "provenance_ids": {
                "summary_ids": lists["summary_ids"]["items"]["enum"][summaries],
                "parse_ids": lists["parse_ids"]["items"]["enum"][parses],
            },
        }

    return reply


@pytest.fixture
def cited_run(answering_llm, shown_ids, tmp_path):
    """A run over the three two-line memos: each summary quotes its memo's first line
    in round 1 and its second in round 2, citing the first summary shown, each parse
    cites nothing, and the answer cites the first summary and the last parse;
    ``answering_llm`` answering so from then on."""
    answering_llm.answers(
        relevance=quoting_a_line_a_round(shown_ids),
        reasoning=citing(slice(1), slice(-1, None)),
    )
    return run("Who?", TWO_LINE_MEMOS, completion=answering_llm, logs_dir=tmp_path)


def saved_trail(folder) -> dict:
    result = json.loads((folder / "reasoning" / "result.json").read_text())
    return result["provenance"]["trail"]


def test_the_trail_of_a_run_folder_is_the_trail_its_answer_saved(cited_run):
    a, b, c = cited_run.document_ids
    memos = list(TWO_LINE_MEMOS.values())

    def line(i: int, k: int, document: str) -> dict:
        text = memos[i].splitlines()[k]
        start = memos[i].index(text)
        name = list(TWO_LINE_MEMOS)[i]
        end = start + len(text)
        return {
            "document": document,
            "name": name,
            "start": start,
            "end": end,
            "text": text,
        }

    # a's summary cites b's round 1; c's parse cites nothing, so it ends there.
    assert cited_run.provenance["trail"] == {
        "rule": "through-summaries",
        "nodes": [
            "answer",
            f"summary:2:{a}",
            f"parse:{c}",
            f"summary:1:{b}",
            f"document:{a}",
            f"document:{b}",
        ],
        "edges": [
            ["answer", f"summary:2:{a}", "cites"],
            ["answer", f"parse:{c}", "cites"],
            [f"summary:2:{a}", f"summary:1:{b}", "cites"],
            [f"summary:2:{a}", f"document:{a}", "summarizes"],
            [f"summary:1:{b}", f"document:{b}", "summarizes"],
        ],
        "quotes": [line(0, 1, a), line(1, 0, b)],
    }
    assert cited_run.provenance["quotes"] == cited_run.provenance["trail"]["quotes"]
    assert trail(cited_run.run_dir) == saved_trail(cited_run.run_dir)
    assert trail(cited_run) == cited_run.provenance["trail"]
    lit = trail(str(cited_run.run_dir), "parses-light-documents")
    assert [f"parse:{c}", f"document:{c}", "parses"] in lit["edges"]


def test_the_trail_of_an_asks_folder_walks_the_asks_answer_over_the_runs_records(
    answering_llm, cited_run
):
    answering_llm.answers(reasoning=citing(slice(-1, None), slice(0)))
    asked = ask(cited_run, "Who else?", completion=answering_llm)
    a, _, c = cited_run.document_ids
    walked = trail(asked.run_dir)
    assert walked == saved_trail(asked.run_dir) == asked.provenance["trail"]
    # c's final summary cites a's round 1, as the run wrote it.
    assert walked["nodes"] == [
        "answer",
        f"summary:2:{c}",
        f"summary:1:{a}",
        f"document:{c}",
        f"document:{a}",
    ]


def test_a_run_folder_whose_snippets_have_no_quotes_has_a_trail_without_quotes(
    cited_run,
):
    """As r3context 0.6.4 wrote them."""
    path = cited_run.run_dir / "relevance" / "result.json"
    result = json.loads(path.read_text())
    for round_ in result["rounds"]:
        for snippet_ in round_["snippets"]:
            del snippet_["quotes"]
    path.write_text(json.dumps(result))
    walked = trail(cited_run.run_dir)
    assert walked == {**saved_trail(cited_run.run_dir), "quotes": []}


def test_a_folder_that_is_not_a_run_folder_has_no_trail(tmp_path):
    with pytest.raises(FileNotFoundError) as refused:
        trail(tmp_path / "missing")
    assert str(refused.value) == (
        f"{tmp_path / 'missing'} is not a run folder: it has no manifest.json."
    )


def test_a_run_whose_reasoning_failed_has_no_trail(answering_llm, tmp_path):
    bad_key = litellm.AuthenticationError(
        message="bad key", llm_provider="openai", model="gpt-6-luna"
    )
    answering_llm.answers(reasoning=bad_key)
    with pytest.raises(litellm.AuthenticationError):
        run("Who?", TWO_LINE_MEMOS, completion=answering_llm, logs_dir=tmp_path)
    [run_dir] = tmp_path.iterdir()
    with pytest.raises(FileNotFoundError) as refused:
        trail(run_dir)
    assert str(refused.value) == (
        f"{run_dir} holds no answer: reasoning/result.json is missing."
    )


def test_an_answer_without_a_run_folder_has_no_trail_to_read(answering_llm):
    result = run("Who?", TWO_LINE_MEMOS, completion=answering_llm, save_artifacts=False)
    with pytest.raises(ValueError, match="this Answer has none"):
        trail(result)


def test_an_answer_that_cites_nothing_has_the_answer_alone_on_its_trail():
    assert walk(graph_of(CITED_ROUNDS)) == answer_alone()
    assert answer_alone() == {
        "rule": "through-summaries",
        "nodes": ["answer"],
        "edges": [],
        "quotes": [],
    }
