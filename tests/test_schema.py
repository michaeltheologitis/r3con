import litellm
import pytest

from r3con.notes import Budget, NotesTooLong
from r3con.splitting import Splits
from r3con.stages.structuring.parsing import SchemaError
from r3con.stages.structuring.schema import propose_schema

MODEL = "openai/gpt-6-luna"
SCHEMA = """from pydantic import BaseModel, Field

class Move(BaseModel):
    time: int = Field(description="t")
    new_location: str = Field(description="loc")

class Parse(BaseModel):
    moves: list[Move]"""
OTHER_SCHEMA = """from pydantic import BaseModel

class Other(BaseModel):
    x: int

class Parse(BaseModel):
    others: list[Other]"""
NO_PARSE = "from pydantic import BaseModel\n\nclass Foo(BaseModel):\n    x: int"
THOUGHT = "One row per occurrence; inference counts the list."
QWEN = "hosted_vllm/Qwen/Qwen3.5-35B-A3B"
# Twenty 60-word notes put the schema call at about 2,920 tokens, over the 2,550-token
# line of a 3,000-token window; ten put it at about 2,200, under it.
NOTES = [" ".join(["word"] * 60)] * 20


def propose(llm, **kwargs):
    kwargs.setdefault("model", MODEL)
    return propose_schema(
        task="Where is the cake?",
        prompt_version="v1",
        completion=llm,
        **kwargs,
    )


def budget_on(model: str) -> Budget:
    return Budget(Splits([], model=model))


def too_long() -> litellm.ContextWindowExceededError:
    return litellm.ContextWindowExceededError(
        message="This model's maximum context length is 3000 tokens.",
        model=QWEN,
        llm_provider="hosted_vllm",
    )


def user_prompts(llm) -> list[str]:
    return [request["messages"][1]["content"] for request in llm.requests]


def test_a_valid_first_schema_is_accepted_after_one_request(llm):
    result = propose(llm.replies(SCHEMA))
    assert result.schema_code == SCHEMA
    assert result.parse_cls.__name__ == "Parse"
    assert [attempt.error for attempt in result.attempts] == [None]
    assert user_prompts(llm) == ["Input:\n<task>\nWhere is the cake?\n</task>\nOutput:"]


@pytest.mark.parametrize(
    ("bad", "error"),
    [
        (NO_PARSE, "did not define a class named `Parse`"),
        ("class Parse(:", "SyntaxError"),
    ],
)
def test_a_rejected_schema_is_sent_back_with_the_validators_error(llm, bad, error):
    result = propose(llm.replies(bad, SCHEMA), max_attempts=3)
    assert result.schema_code == SCHEMA
    assert error in result.attempts[0].error
    assert result.attempts[1].error is None
    retry = user_prompts(llm)[1]
    assert f"# Previous attempt (rejected by validator)\n{bad}" in retry
    assert error in retry


def test_proposing_stops_with_the_last_error_once_attempts_run_out(llm):
    llm.replies(NO_PARSE, NO_PARSE, NO_PARSE)
    with pytest.raises(SchemaError, match="exhausted 3 attempts.*did not define"):
        propose(llm, max_attempts=3)
    assert len(llm.requests) == 3


def test_a_schema_that_never_validates_is_requested_five_times(llm):
    llm.answers(schema=NO_PARSE)
    with pytest.raises(SchemaError, match="exhausted 5 attempts"):
        propose(llm)
    assert len(llm.requests) == 5


def test_max_attempts_below_one_is_refused_before_any_request(llm):
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        propose(llm, max_attempts=0)
    assert llm.requests == []


@pytest.mark.parametrize(
    ("reply", "thought"),
    [
        (f"<schema>\n{SCHEMA}\n</schema>", None),
        (f"Thought: {THOUGHT}\n\n<schema>\n{SCHEMA}\n</schema>", THOUGHT),
        (f"Thought: {THOUGHT}\n\n```python\n{SCHEMA}\n```\n", THOUGHT),
        (f"```python\n{OTHER_SCHEMA}\n```\n\n<schema>\n{SCHEMA}\n</schema>", None),
        (f"```python\n{SCHEMA}\n```", None),
        (f"```\n{SCHEMA}\n```", None),
        (SCHEMA, None),
    ],
)
def test_the_schema_is_read_from_its_tag_then_a_fence_then_the_whole_reply(
    llm, reply, thought
):
    result = propose(llm.replies(reply))
    assert result.schema_code == SCHEMA
    assert result.attempts[-1].thought == thought


@pytest.mark.parametrize(
    ("snippets", "shown"),
    [(["Doc A is about whales.", "Doc B is about ships."], True), (None, False)],
)
def test_the_relevance_notes_are_in_the_system_prompt_only_when_given(
    llm, snippets, shown
):
    propose(llm.replies(SCHEMA), relevance_snippets=snippets)
    system = llm.requests[0]["messages"][0]["content"]
    # The prompt's examples carry a similar label, so look for the block's own text.
    assert ("In order to know what the documents contain" in system) is shown
    assert ("Doc A is about whales." in system) is shown
    assert ("Doc B is about ships." in system) is shown


def test_the_callers_request_options_reach_the_request(llm):
    propose(llm.replies(SCHEMA), seed=42, api_base="http://x")
    request = llm.requests[0]
    assert request["model"] == MODEL
    assert request["seed"] == 42
    assert request["api_base"] == "http://x"


def test_a_schema_request_over_the_line_hands_over_its_notes_before_sending(
    llm, window
):
    with pytest.raises(NotesTooLong) as handed:
        propose(
            llm.replies(SCHEMA),
            relevance_snippets=NOTES,
            budget=budget_on(window(3_000)),
        )
    assert (handed.value.call, handed.value.cause) == ("schema", "estimate")
    assert handed.value.notes == NOTES
    assert llm.requests == []


@pytest.mark.parametrize("snippets", [NOTES, None], ids=["notes", "no-notes"])
def test_a_refused_schema_request_hands_over_its_notes_and_one_without_notes_is_raised(
    llm, snippets
):
    refused = too_long()
    with pytest.raises(Exception) as failure:
        propose(
            llm.replies(refused), relevance_snippets=snippets, budget=budget_on(QWEN)
        )
    if snippets is None:
        assert failure.value is refused
    else:
        assert isinstance(failure.value, NotesTooLong)
        assert (failure.value.call, failure.value.cause) == ("schema", "refusal")
        assert failure.value.__cause__ is refused
    assert len(llm.requests) == 1


def test_every_attempt_is_measured(llm, window):
    rejected = NO_PARSE + "\n# " + "padding " * 600
    with pytest.raises(NotesTooLong) as handed:
        propose(
            llm.replies(rejected, SCHEMA),
            relevance_snippets=NOTES[:10],
            max_attempts=3,
            budget=budget_on(window(3_000)),
        )
    assert len(llm.requests) == 1
    assert "# Previous attempt (rejected by validator)" not in str(llm.requests)
    assert handed.value.estimate > 2_550


def test_without_a_budget_the_schema_call_is_sent_as_today(llm, window):
    refused = too_long()
    with pytest.raises(litellm.ContextWindowExceededError) as failure:
        propose(llm.replies(refused), relevance_snippets=NOTES, model=window(3_000))
    assert failure.value is refused
    assert len(llm.requests) == 1
