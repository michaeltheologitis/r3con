import json
import logging
import re
import textwrap
import time

import litellm
import pytest
from pydantic import ValidationError

from r3con.runs import StageRun, TaskLogger
from r3con.splitting import Splits
from r3con.stages.structuring.parsing import (
    SchemaError,
    check_schema,
    parse_documents,
    parse_one_document,
)

MODEL = "openai/gpt-6-luna"
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
# 7,868 characters in 40 paragraphs, P0 to P39: refused whole under a 10,000-character
# limit beside the parse prompt, read in halves that start at P0 and P20.
LONG = "\n\n".join(f"P{i}. " + "word " * 38 for i in range(40))
SCHEMA = """from pydantic import BaseModel


class Row(BaseModel):
    who: str


class Parse(BaseModel):
    rows: list[Row]
"""
Parse = check_schema(SCHEMA)
DOCS = [
    "Alpha memo about whales.",
    "Beta memo about ships.",
    "Gamma memo about harbors.",
]
NOTES = ["Doc A is about whales.", "Doc B is about ships."]
IDS = ["a1a1a1", "b2b2b2", "c3c3c3"]
RELEVANCE_HEADING = "## Task-conditioned document summaries"
NOTHING_PARSED = {"rows": []}
ID_HEADING = re.compile(r"^### Document ID: (\S+)$", re.MULTILINE)
RICH_SCHEMA = """
from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Kind(Enum):
    SITE = "site"
    CONTRACTOR = "contractor"


class Note(BaseModel):
    text: str


class Tag(BaseModel):
    label: str


class Item(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    kind: Kind
    status: Literal["open", "closed"]
    logged: date
    cost: Decimal = Field(ge=0)
    count: Annotated[int, Field(ge=0)]
    name: str = Field(alias="itemName")
    parent: "Item | None" = None
    extras: list[Note | Tag]

    def label(self) -> str:
        return f"{self.name} ({self.kind.value})"


class Parse(BaseModel):
    items: list[Item]
"""
# Cut down from two eval tasks' schemas: Loong 9673e4be and CorpusQA financial_zh_20.
COSTS = """from decimal import Decimal

class Cost(BaseModel):
    amount: Decimal | None

class Parse(BaseModel):
    costs: list[Cost]
"""
LOOKAROUNDS = ("(?=", "(?!", "(?<=", "(?<!")
ITEM = {
    "kind": "site",
    "status": "open",
    "logged": "2026-09-30",
    "cost": "12.50",
    "count": 3,
    "itemName": "Pump",
    "extras": [{"text": "leaks"}, {"label": "urgent"}],
}


def parsed(request) -> str:
    """The document, or part, a parse request sends below its ``Document ID`` line."""
    return request["messages"][1]["content"].split("\n\n", 1)[1]


def rows_named_after_the_document(request) -> dict:
    """Two rows per document, each naming it, so a merge's order shows."""
    document = parsed(request).splitlines()[0]
    # Later documents answer first, so the merge cannot rely on arrival order.
    time.sleep(0.02 * (len(DOCS) - DOCS.index(document)))
    return {"rows": [{"who": f"{document}#a"}, {"who": f"{document}#b"}]}


def a_row_named_by_the_first_word(request) -> dict:
    """One row per document or part, naming its first word."""
    first_word = re.match(r"\w+", parsed(request)).group()
    return {"rows": [{"who": first_word}]}


def citing(by_first_word: dict[str, list[str]]):
    """A parse reply that cites the ids ``by_first_word`` gives the document's, or
    the part's, first word."""

    def reply(request) -> dict:
        first_word = re.match(r"\w+", parsed(request)).group()
        return {"parse": {"rows": []}, "provenance": by_first_word[first_word]}

    return reply


def shown_ids(request) -> list[str]:
    """The ids heading the notes a parse request shows, without its examples'."""
    system = request["messages"][0]["content"].split("## Your job", 1)[0]
    return ID_HEADING.findall(system)


def reply_schema(request) -> dict:
    return request["response_format"]["json_schema"]["schema"]


def reply_schema_name(request) -> str:
    return request["response_format"]["json_schema"]["name"]


def refuses_a_lookaround(reply: dict):
    """A provider in OpenAI's strict mode: it refuses a schema with a regex lookaround,
    as OpenAI refused those two schemas, and otherwise answers ``reply``."""

    def answer(request) -> dict:
        sent = json.dumps(request["response_format"]["json_schema"]["schema"])
        if any(lookaround in sent for lookaround in LOOKAROUNDS):
            raise litellm.BadRequestError(
                message="Invalid JSON schema: regex lookaround is not supported.",
                model=MODEL,
                llm_provider="openai",
            )
        return reply

    return answer


def parse_one(llm, prompt_version="v2", document_ids=(), **kwargs):
    """The memo parsed as the document IDS[0] names."""
    return parse_one_document(
        document="memo",
        document_id=IDS[0],
        schema_code=SCHEMA,
        parse_cls=Parse,
        task="Who?",
        prompt_version=prompt_version,
        document_ids=document_ids,
        model=MODEL,
        completion=llm,
        **kwargs,
    )


def parse_all(llm, documents, model=MODEL, **kwargs):
    """``documents`` parsed, each named by its id in IDS."""
    return parse_documents(
        documents=documents,
        schema_code=SCHEMA,
        parse_cls=Parse,
        task="Who is mentioned?",
        prompt_version="v2",
        document_ids=IDS[: len(documents)],
        model=model,
        completion=llm,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("code", "record"),
    [
        (
            """
            from pydantic import BaseModel, Field

            class Parse(BaseModel):
                answers: list[str] = Field(description="the answers")
            """,
            {"answers": ["42"]},
        ),
        (
            """
            from pydantic import BaseModel, Field

            class DocumentMove(BaseModel):
                time: int = Field(description="Sentence number")
                new_location: str

            class Parse(BaseModel):
                document_moves: list[DocumentMove]
            """,
            {"document_moves": [{"time": 1, "new_location": "kitchen"}]},
        ),
        (
            """
            from typing import Literal
            from pydantic import BaseModel

            class CakeMove(BaseModel):
                time: int
                actor: Literal["Maya", "Carlos"]

            class Parse(BaseModel):
                cake_moves: list[CakeMove]
            """,
            {"cake_moves": [{"time": 1, "actor": "Maya"}]},
        ),
        (
            """
            class Item(BaseModel):
                name: str = Field(description="a name")

            class Parse(BaseModel):
                items: list[Item]
            """,
            {"items": [{"name": "a"}]},
        ),
        (
            """
            class Item(BaseModel):
                kind: Literal["a", "b"]
                notes: Optional[str] = None
                tags: List[str] = Field(default_factory=list)

            class Parse(BaseModel):
                items: list[Item]
            """,
            {"items": [{"kind": "a", "notes": None, "tags": []}]},
        ),
        (
            """
            from pydantic import BaseModel, Field
            from typing import Optional

            class Item(BaseModel):
                note: Optional[str] = Field(default=None, description="n")

            class Parse(BaseModel):
                items: list[Item]
            """,
            {"items": [{"note": "n"}]},
        ),
    ],
)
def test_a_usable_schema_returns_its_parse_class(code, record):
    parse_cls = check_schema(textwrap.dedent(code))
    assert parse_cls.__name__ == "Parse"
    assert parse_cls.model_validate(record).model_dump() == record


@pytest.mark.parametrize(
    ("code", "error"),
    [
        ("class Parse(BaseModel):\n    x: SomeUndefinedTypeXYZ", "failed"),
        ("class Parse(:", "SyntaxError"),
        ("import this_module_does_not_exist_xyz", "failed to execute"),
        ("class Foo(BaseModel):\n    x: int", "did not define a class named `Parse`"),
        ("class Parse: pass", "must be a pydantic.BaseModel subclass"),
        ("Parse = 42", "must be a pydantic.BaseModel subclass"),
        (
            """
            class Weird:
                pass

            class Parse(BaseModel):
                model_config = {"arbitrary_types_allowed": True}
                x: Weird
            """,
            r"`Parse.model_json_schema\(\)` failed",
        ),
        ("class Parse(BaseModel):\n    parse: list[dict]", "untyped object.*BaseModel"),
        ("class Parse(BaseModel):\n    bag: dict", "untyped object"),
        ("class Parse(BaseModel):\n    bag: dict[str, Any]", "untyped object"),
        ("class Parse(BaseModel):\n    bag: dict[str, str]", "untyped object"),
        (
            "class Parse(BaseModel):\n    answer: str",
            r"not declared as `list\[...\]`: answer",
        ),
        (
            """
            class DocInfo(BaseModel):
                title: str

            class Parse(BaseModel):
                document: DocInfo
            """,
            r"not declared as `list\[...\]`: document",
        ),
    ],
)
def test_an_unusable_schema_is_refused_with_what_to_fix(code, error):
    with pytest.raises(SchemaError, match=error):
        check_schema(textwrap.dedent(code))


@pytest.mark.parametrize(
    "reach",
    [
        lambda marker: f"import os\nos.system('touch {marker}')\n",
        lambda marker: f"open('{marker}', 'w').close()\n",
        lambda marker: f"__import__('os').system('touch {marker}')\n",
    ],
    ids=["import-os", "open", "dunder-import"],
)
def test_a_schema_cannot_reach_outside_the_interpreter(tmp_path, reach):
    marker = tmp_path / "marker"
    with pytest.raises(SchemaError, match="failed to execute"):
        check_schema(reach(marker) + SCHEMA)
    assert not marker.exists()


@pytest.mark.parametrize(
    ("method", "listed"),
    [
        (
            """
            @field_validator("who")
            @classmethod
            def named(cls, value):
                return value
            """,
            ["@field_validator('who') on named", "@classmethod on named"],
        ),
        (
            """
            @model_validator(mode="after")
            def whole(self):
                return self
            """,
            ["@model_validator(mode='after') on whole"],
        ),
        (
            """
            @computed_field
            @property
            def shout(self) -> str:
                return self.who.upper()
            """,
            ["@computed_field on shout", "@property on shout"],
        ),
    ],
    ids=["field_validator", "model_validator", "computed_field"],
)
def test_a_decorated_method_is_refused_with_what_to_do_instead(method, listed):
    code = (
        "from pydantic import computed_field, field_validator, model_validator\n\n"
        "class Row(BaseModel):\n    who: str\n"
        + textwrap.indent(textwrap.dedent(method), "    ")
        + "\nclass Parse(BaseModel):\n    rows: list[Row]\n"
    )
    with pytest.raises(SchemaError, match="decorators, which r3con does not run") as no:
        check_schema(code)
    assert all(entry in str(no.value) for entry in listed)
    assert "Field(ge=0)" in str(no.value)


def test_a_schema_with_plain_methods_and_rich_types_is_accepted():
    parse_cls = check_schema(RICH_SCHEMA)
    child = {**ITEM, "itemName": "Valve", "parent": {**ITEM, "extras": []}}
    [item] = parse_cls.model_validate({"items": [child]}).items
    assert item.label() == "Valve (site)"
    assert item.parent.label() == "Pump (site)"
    assert [type(extra).__name__ for extra in item.extras] == ["Note", "Tag"]
    with pytest.raises(ValidationError, match="cost"):
        parse_cls.model_validate({"items": [{**ITEM, "cost": "-1"}]})


def test_a_decimal_field_is_parsed_though_openai_refuses_a_lookaround(llm):
    reply = {"costs": [{"amount": "1520.75"}]}
    parse, _ = parse_one_document(
        document="memo",
        schema_code=COSTS,
        parse_cls=check_schema(COSTS),
        task="What did it cost?",
        prompt_version="v2",
        document_id=IDS[0],
        document_ids=[],
        model=MODEL,
        completion=llm.answers(parsing=refuses_a_lookaround(reply)),
    )
    assert parse.model_dump(mode="json") == reply


def test_a_malformed_reply_is_retried_with_the_validator_error(llm):
    parse, _ = parse_one(
        llm.replies({"rows": [{"who": 7}]}, {"rows": [{"who": "Halloran"}]})
    )
    assert parse.rows[0].who == "Halloran"
    assert "validation error" in llm.requests[1]["messages"][1]["content"]


def test_a_document_that_parses_first_time_is_sent_once_as_it_is(llm):
    result = parse_one(llm.replies({"rows": [{"who": "Halloran"}]}))
    assert result == (Parse(rows=[{"who": "Halloran"}]), [])
    sent = [request["messages"][1]["content"] for request in llm.requests]
    assert sent == [f"Document ID: {IDS[0]}\n\nmemo"]
    assert reply_schema_name(llm.requests[0]) == "ParseReply"


def test_a_document_that_never_validates_stops_after_max_attempts(llm):
    llm.answers(parsing={"rows": "not-a-list"})
    with pytest.raises(ValidationError, match="rows"):
        parse_one(llm, max_attempts=3)
    assert len(llm.requests) == 3
    resent = llm.requests[2]["messages"][1]["content"]
    assert resent.startswith(f"Document ID: {IDS[0]}\n\nmemo\n\n---\n\n")


def test_max_attempts_below_one_is_refused_before_any_request(llm):
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        parse_one(llm, max_attempts=0)
    assert llm.requests == []


def test_a_provider_error_is_not_retried(llm):
    outage = litellm.APIConnectionError(
        "API outage", llm_provider="openai", model=MODEL
    )
    with pytest.raises(litellm.APIConnectionError, match="API outage"):
        parse_one(llm.replies(outage))
    assert len(llm.requests) == 1


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_ids_and_the_reply_schema_do_not_depend_on_the_prompt_version(llm, version):
    reply = {"parse": {"rows": [{"who": "Halloran"}]}, "provenance": [IDS[1]]}
    result = parse_one(
        llm.replies(reply), version, relevance_snippets=NOTES, document_ids=IDS[:2]
    )
    assert result == (Parse(rows=[{"who": "Halloran"}]), [IDS[1]])
    [request] = llm.requests
    assert reply_schema_name(request) == "ParseReply"
    assert shown_ids(request) == IDS[:2]
    system = request["messages"][0]["content"]
    assert f"### Document ID: {IDS[0]}\nDoc A is about whales." in system
    assert ("## Output" in system) is (version == "v2")


def test_the_reply_asks_for_the_parse_and_the_ids_of_exactly_the_notes_shown(llm):
    notes = ["Doc A is about whales.", ["Part one of B.", "Part two of B."]]
    parse_one(
        llm.replies(NOTHING_PARSED), relevance_snippets=notes, document_ids=IDS[:2]
    )
    [request] = llm.requests
    schema = reply_schema(request)
    assert schema["required"] == ["parse", "provenance"]
    assert schema["properties"]["parse"] == {"$ref": "#/$defs/Parse"}
    provenance = schema["properties"]["provenance"]
    assert provenance["items"]["enum"] == shown_ids(request)
    assert shown_ids(request) == [IDS[0], f"{IDS[1]}.1", f"{IDS[1]}.2"]
    assert "default" not in provenance


@pytest.mark.parametrize(
    "snippets",
    [None, [], ["", " ", ""]],
    ids=["relevance-off", "no-rounds", "every-note-empty"],
)
def test_with_no_note_to_cite_provenance_is_any_list_and_what_it_holds_is_dropped(
    llm, snippets, caplog
):
    llm.answers(parsing={"parse": {"rows": []}, "provenance": ["0f0f0f"]})
    with caplog.at_level(logging.WARNING, logger="r3con"):
        result = parse_all(llm, DOCS, relevance_snippets=snippets, workers=3)
    assert result.provenance == [[], [], []]
    assert "did not show: ['0f0f0f']" in caplog.text
    assert len(llm.requests) == len(DOCS)
    for request in llm.requests:
        assert reply_schema_name(request) == "ParseReply"
        schema = reply_schema(request)
        assert schema["required"] == ["parse", "provenance"]
        assert schema["properties"]["provenance"]["items"] == {"type": "string"}
        assert "## Output" in request["messages"][0]["content"]


def test_document_ids_that_are_not_one_per_document_are_refused_before_any_request(
    llm,
):
    with pytest.raises(ValueError, match=r"got 2 for 3 document\(s\)"):
        parse_documents(
            documents=DOCS,
            schema_code=SCHEMA,
            parse_cls=Parse,
            task="Who is mentioned?",
            prompt_version="v2",
            document_ids=IDS[:2],
            model=MODEL,
            completion=llm,
        )
    assert llm.requests == []


def test_an_empty_notes_id_is_shown_but_cannot_be_cited(llm, caplog):
    reply = {"parse": {"rows": []}, "provenance": [IDS[1]]}
    with caplog.at_level(logging.WARNING, logger="r3con"):
        _, provenance = parse_one(
            llm.replies(reply),
            relevance_snippets=["Doc A is about whales.", ""],
            document_ids=IDS[:2],
        )
    assert provenance == []
    [request] = llm.requests
    assert shown_ids(request) == IDS[:2]
    no_summary = f"### Document ID: {IDS[1]}\n(no relevant summary for this task)"
    assert no_summary in request["messages"][0]["content"]
    assert reply_schema(request)["properties"]["provenance"]["items"]["enum"] == [
        IDS[0]
    ]
    assert f"did not show: ['{IDS[1]}']" in caplog.text


def test_each_documents_provenance_is_recorded_aligned_with_the_documents(llm):
    notes = [f"Notes on {document}" for document in DOCS]
    cites = {"Alpha": [IDS[1]], "Beta": [], "Gamma": [IDS[2], IDS[0], IDS[2]]}
    result = parse_all(
        llm.answers(parsing=citing(cites)), DOCS, relevance_snippets=notes, workers=3
    )
    assert result.provenance == [[IDS[1]], [], [IDS[2], IDS[0]]]


def test_a_cited_id_the_call_did_not_show_is_dropped_with_a_warning(llm, caplog):
    reply = {"parse": {"rows": []}, "provenance": ["0f0f0f", IDS[0]]}
    with caplog.at_level(logging.WARNING, logger="r3con"):
        _, provenance = parse_one(
            llm.replies(reply),
            relevance_snippets=NOTES,
            document_ids=IDS[:2],
            kind="parse-d1",
        )
    assert provenance == [IDS[0]]
    assert (
        "parse-d1: dropped provenance id(s) the call did not show: ['0f0f0f']"
        in caplog.text
    )


def test_a_document_read_in_parts_combines_its_parts_ids_in_order_without_repeats(
    llm,
):
    notes = [["Notes on P0.", "Notes on P20."], "Notes on Beta."]
    cites = {
        "P0": [IDS[1], f"{IDS[0]}.1"],
        "P20": [f"{IDS[0]}.1", f"{IDS[0]}.2"],
        "Beta": [],
    }
    llm.refuses_over(10_000).answers(parsing=citing(cites))
    result = parse_all(
        llm, [LONG, DOCS[1]], model=QWEN, relevance_snippets=notes, workers=2
    )
    assert result.provenance == [[IDS[1], f"{IDS[0]}.1", f"{IDS[0]}.2"], []]
    assert len(llm.requests_for("parsing")) == 4


def test_each_document_is_parsed_whole_in_a_request_of_its_own_under_its_id(llm):
    llm.answers(parsing=rows_named_after_the_document)
    parse_all(llm, DOCS, workers=8)
    sent = [
        request["messages"][1]["content"] for request in llm.requests_for("parsing")
    ]
    headed = [f"Document ID: {id_}\n\n{doc}" for id_, doc in zip(IDS, DOCS)]
    assert sorted(sent) == sorted(headed)


@pytest.mark.parametrize(("snippets", "shown"), [(NOTES, True), (None, False)])
def test_every_parsing_prompt_carries_the_task_and_every_documents_note(
    llm, snippets, shown
):
    llm.answers(parsing=rows_named_after_the_document)
    parse_all(llm, DOCS[:2], relevance_snippets=snippets, workers=2)
    for request in llm.requests:
        system = request["messages"][0]["content"]
        assert "Who is mentioned?" in system
        assert (RELEVANCE_HEADING in system) is shown
        assert all((note in system) is shown for note in NOTES)


def test_the_merge_keeps_document_order_and_each_records_source(llm):
    llm.answers(parsing=rows_named_after_the_document)
    result = parse_all(llm, DOCS, workers=8)
    assert [row.who for row in result.parse.rows] == [
        f"{document}#{part}" for document in DOCS for part in "ab"
    ]
    assert result.source_docs == {"rows": [0, 0, 1, 1, 2, 2]}


def test_no_documents_parse_to_empty_lists_without_a_request(llm):
    result = parse_all(llm, [])
    assert result.parse == Parse(rows=[])
    assert result.source_docs == {"rows": []}
    assert result.provenance == []
    assert llm.requests == []


def test_one_documents_failure_is_raised_from_the_whole_parse(llm):
    def fails_on_beta(request):
        if parsed(request) == DOCS[1]:
            raise RuntimeError("parse blew up")
        return rows_named_after_the_document(request)

    with pytest.raises(RuntimeError, match="parse blew up"):
        parse_all(llm.answers(parsing=fails_on_beta), DOCS, workers=4)


def test_progress_is_logged_per_document_at_info(llm, caplog):
    caplog.set_level(logging.INFO, logger="r3con")
    parse_all(llm.answers(parsing=rows_named_after_the_document), DOCS[:2], workers=1)
    assert "parse doc 1/2" in caplog.text
    assert "parse doc 2/2" in caplog.text


def test_a_refused_document_is_parsed_in_parts_and_every_record_keeps_its_index(llm):
    llm.refuses_over(10_000).answers(parsing=a_row_named_by_the_first_word)
    result = parse_all(llm, [DOCS[0], LONG, DOCS[2]], model=QWEN, workers=3)
    assert [row.who for row in result.parse.rows] == ["Alpha", "P0", "P20", "Gamma"]
    assert result.source_docs == {"rows": [0, 1, 1, 2]}


def test_parsing_starts_from_the_parts_it_is_given_each_under_its_documents_id(
    llm, tmp_path
):
    def refuses_the_whole(part, kind):
        if part == LONG:
            raise litellm.ContextWindowExceededError(
                message="too long", model=QWEN, llm_provider="hosted_vllm"
            )
        return "a note"

    splits = Splits([LONG], model=QWEN)
    splits.read_in_parts(0, call="relevance-r1", rest="", send=refuses_the_whole)
    run = StageRun(stage="parsing", task_logger=TaskLogger("run", root=tmp_path))
    llm.answers(parsing=a_row_named_by_the_first_word)
    parse_all(llm, [LONG], model=QWEN, splits=splits, run=run)
    assert [step.kind for step in run.steps] == ["parse-d0c0", "parse-d0c1"]
    sent = [request["messages"][1]["content"] for request in llm.requests]
    assert sent == [f"Document ID: {IDS[0]}\n\n{part}" for part in splits.parts(0)]
