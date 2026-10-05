import json
import logging
import textwrap
import time

import litellm
import pytest

from r3con.stages.structuring.parsing import (
    SchemaError,
    check_schema,
    parse_documents,
    parse_one_document,
)

MODEL = "openai/gpt-6-luna"
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
RELEVANCE_HEADING = "## Task-conditioned document summaries"


def rows_named_after_the_document(request) -> str:
    """Two rows per document, each naming it, so a merge's order shows."""
    document = request["messages"][1]["content"].splitlines()[0]
    # Later documents answer first, so the merge cannot rely on arrival order.
    time.sleep(0.02 * (len(DOCS) - DOCS.index(document)))
    return json.dumps({"rows": [{"who": f"{document}#a"}, {"who": f"{document}#b"}]})


def parse_one(llm, **kwargs):
    return parse_one_document(
        document="memo",
        schema_code=SCHEMA,
        parse_cls=Parse,
        task="Who?",
        prompt_version="v1",
        model=MODEL,
        completion=llm,
        **kwargs,
    )


def parse_all(llm, documents, **kwargs):
    return parse_documents(
        documents=documents,
        schema_code=SCHEMA,
        parse_cls=Parse,
        task="Who is mentioned?",
        prompt_version="v1",
        model=MODEL,
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
        ("class Parse(:", "failed to execute: SyntaxError"),
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


def test_a_malformed_reply_is_retried_with_the_validator_error(llm):
    parse = parse_one(
        llm.replies('{"rows": [{"who": 7}]}', '{"rows": [{"who": "Halloran"}]}')
    )
    assert parse.rows[0].who == "Halloran"
    assert "validation error" in llm.requests[1]["messages"][1]["content"]


def test_a_document_that_parses_first_time_is_sent_once_as_it_is(llm):
    parse = parse_one(llm.replies('{"rows": [{"who": "Halloran"}]}'))
    assert parse == Parse(rows=[{"who": "Halloran"}])
    assert [request["messages"][1]["content"] for request in llm.requests] == ["memo"]
    assert llm.requests[0]["response_format"]["json_schema"]["name"] == "Parse"


def test_a_document_that_never_validates_stops_after_max_attempts(llm):
    llm.answers(parsing='{"rows": "not-a-list"}')
    with pytest.raises(SchemaError, match="exhausted 3 attempts.*ValidationError"):
        parse_one(llm, max_attempts=3)
    assert len(llm.requests) == 3
    assert llm.requests[2]["messages"][1]["content"].startswith("memo\n\n---\n\n")


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


def test_each_document_is_parsed_whole_in_a_request_of_its_own(llm):
    llm.answers(parsing=rows_named_after_the_document)
    parse_all(llm, DOCS, workers=8)
    sent = [
        request["messages"][1]["content"] for request in llm.requests_for("parsing")
    ]
    assert sorted(sent) == sorted(DOCS)


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


def test_doc_ids_label_the_documents(llm):
    llm.answers(parsing=rows_named_after_the_document)
    result = parse_all(llm, DOCS[:2], doc_ids=["fileA", "fileB"])
    assert result.doc_ids == ["fileA", "fileB"]
    assert [result.doc_label(i) for i in (0, 1, 9)] == ["fileA", "fileB", "9"]


def test_no_documents_parse_to_empty_lists_without_a_request(llm):
    result = parse_all(llm, [])
    assert result.parse == Parse(rows=[])
    assert result.source_docs == {"rows": []}
    assert llm.requests == []


def test_one_documents_failure_is_raised_from_the_whole_parse(llm):
    def fails_on_beta(request):
        if request["messages"][1]["content"] == DOCS[1]:
            raise RuntimeError("parse blew up")
        return rows_named_after_the_document(request)

    with pytest.raises(RuntimeError, match="parse blew up"):
        parse_all(llm.answers(parsing=fails_on_beta), DOCS, workers=4)


def test_progress_is_logged_per_document_at_info(llm, caplog):
    caplog.set_level(logging.INFO, logger="r3con")
    parse_all(llm.answers(parsing=rows_named_after_the_document), DOCS[:2], workers=1)
    assert "parse doc 1/2" in caplog.text
    assert "parse doc 2/2" in caplog.text
