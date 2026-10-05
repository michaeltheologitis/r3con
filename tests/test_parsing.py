import json
import logging
import re
import textwrap
import time

import litellm
import pytest
from pydantic import ValidationError

from r3con.notes import Budget, NotesTooLong
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
# 3,920 characters in 20 paragraphs, P0 to P19: refused whole under a 5,500-character
# limit beside the parse prompt, read in halves that start at P0 and P10.
LONG = "\n\n".join(f"P{i}. " + "word " * 38 for i in range(20))
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
# Twenty 60-word notes, about 1,220 tokens: beside them a parse request is about 2,160
# tokens and a 300-token document's about 2,460, against the 2,295-token line of a
# 2,700-token window.
LONG_NOTES = [" ".join(["word"] * 60)] * 20
SHORTER_THAN_THE_NOTES = LONG[:1_500]
RELEVANCE_HEADING = "## Task-conditioned document summaries"
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
ITEM = {
    "kind": "site",
    "status": "open",
    "logged": "2026-09-30",
    "cost": "12.50",
    "count": 3,
    "itemName": "Pump",
    "extras": [{"text": "leaks"}, {"label": "urgent"}],
}


def rows_named_after_the_document(request) -> str:
    """Two rows per document, each naming it, so a merge's order shows."""
    document = request["messages"][1]["content"].splitlines()[0]
    # Later documents answer first, so the merge cannot rely on arrival order.
    time.sleep(0.02 * (len(DOCS) - DOCS.index(document)))
    return json.dumps({"rows": [{"who": f"{document}#a"}, {"who": f"{document}#b"}]})


def a_row_named_by_the_first_word(request) -> str:
    """One row per document or part, naming its first word."""
    first_word = re.match(r"\w+", request["messages"][1]["content"]).group()
    return json.dumps({"rows": [{"who": first_word}]})


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


def parse_all(llm, documents, model=MODEL, **kwargs):
    return parse_documents(
        documents=documents,
        schema_code=SCHEMA,
        parse_cls=Parse,
        task="Who is mentioned?",
        prompt_version="v1",
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


def test_a_refused_document_is_parsed_in_parts_and_every_record_keeps_its_index(llm):
    llm.refuses_over(5_500).answers(parsing=a_row_named_by_the_first_word)
    result = parse_all(llm, [DOCS[0], LONG, DOCS[2]], model=QWEN, workers=3)
    assert [row.who for row in result.parse.rows] == ["Alpha", "P0", "P10", "Gamma"]
    assert result.source_docs == {"rows": [0, 1, 1, 2]}


def test_parsing_starts_from_the_parts_it_is_given(llm, tmp_path):
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
    assert sent == splits.parts(0)


def test_parsing_measures_every_document_before_sending_any(llm, window):
    model = window(2_700)
    documents = [*DOCS[:2], SHORTER_THAN_THE_NOTES]
    budget = Budget(Splits(documents, model=model))
    with pytest.raises(NotesTooLong) as handed:
        parse_all(
            llm.answers(parsing=a_row_named_by_the_first_word),
            documents,
            model=model,
            relevance_snippets=LONG_NOTES,
            splits=budget.splits,
            budget=budget,
            workers=1,
        )
    assert (handed.value.call, handed.value.cause) == ("parse-d2", "estimate")
    assert llm.requests == []


def test_a_refused_parse_whose_notes_are_bigger_hands_them_over(llm):
    llm.refuses_over(8_000).answers(parsing=a_row_named_by_the_first_word)
    documents = [SHORTER_THAN_THE_NOTES]
    budget = Budget(Splits(documents, model=QWEN))
    with pytest.raises(NotesTooLong) as handed:
        parse_all(
            llm,
            documents,
            model=QWEN,
            relevance_snippets=LONG_NOTES,
            splits=budget.splits,
            budget=budget,
        )
    assert (handed.value.call, handed.value.cause) == ("parse-d0", "refusal")
    assert isinstance(handed.value.__cause__, litellm.ContextWindowExceededError)
    assert budget.splits.parts(0) == documents


def test_without_a_budget_a_document_shorter_than_its_notes_is_cut_as_r2_does(
    llm, window
):
    model = window(2_700)
    splits = Splits([SHORTER_THAN_THE_NOTES], model=model)
    parse_all(
        llm.answers(parsing=a_row_named_by_the_first_word),
        [SHORTER_THAN_THE_NOTES],
        model=model,
        relevance_snippets=LONG_NOTES,
        splits=splits,
    )
    assert len(splits.parts(0)) > 1
    assert [r["messages"][1]["content"] for r in llm.requests] == splits.parts(0)
