"""Tests for `r3con.stages.structuring.parsing.check_schema` and
`r3con.stages.structuring.parsing.parse_one_document` retry behavior.

Run with:  uv run python tests/test_parsing.py
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ValidationError

from r3con.stages.structuring import parsing as parsing_mod
from r3con.stages.structuring.parsing import SchemaError, check_schema, parse_one_document


def test_simple_schema() -> None:
    code = """
from pydantic import BaseModel, Field

class Parse(BaseModel):
    answers: list[str] = Field(description="the answers")
"""
    cls = check_schema(code)
    assert issubclass(cls, BaseModel)
    assert cls.__name__ == "Parse"
    instance = cls(answers=["42"])
    assert instance.answers == ["42"]


def test_nested_schema_resolves_forward_types() -> None:
    """Regression: classes from exec'd code couldn't resolve nested types by module lookup."""
    code = """
from pydantic import BaseModel, Field

class DocumentMove(BaseModel):
    time: int = Field(description="Sentence number")
    new_location: str

class Parse(BaseModel):
    document_moves: list[DocumentMove]
"""
    cls = check_schema(code)
    schema = cls.model_json_schema()
    assert "document_moves" in schema["properties"]
    assert "DocumentMove" in schema.get("$defs", {})


def test_schema_with_literal() -> None:
    """Schemas commonly use `Literal[...]` for entities derivable from the question."""
    code = """
from typing import Literal
from pydantic import BaseModel, Field

class CakeMove(BaseModel):
    time: int
    actor: Literal["Maya", "Carlos"]

class Parse(BaseModel):
    cake_moves: list[CakeMove]
"""
    cls = check_schema(code)
    instance = cls(cake_moves=[{"time": 1, "actor": "Maya"}])
    assert instance.cake_moves[0].actor == "Maya"


def test_schema_without_pydantic_import_still_loads() -> None:
    """Leniency: a schema that forgets `from pydantic import BaseModel, Field`
    still loads — the exec namespace is pre-seeded with those names."""
    code = """
class Item(BaseModel):
    name: str = Field(description="a name")

class Parse(BaseModel):
    items: list[Item]
"""
    cls = check_schema(code)
    assert issubclass(cls, BaseModel)
    assert cls.__name__ == "Parse"
    assert "items" in cls.model_json_schema()["properties"]


def test_schema_without_typing_imports_still_loads() -> None:
    """Leniency: Optional / Literal / List used without importing typing."""
    code = """
class Item(BaseModel):
    kind: Literal["a", "b"]
    notes: Optional[str] = None
    tags: List[str] = Field(default_factory=list)

class Parse(BaseModel):
    items: list[Item]
"""
    cls = check_schema(code)
    assert issubclass(cls, BaseModel)
    inst = cls(items=[{"kind": "a"}])
    assert inst.items[0].kind == "a" and inst.items[0].notes is None and inst.items[0].tags == []


def test_explicit_imports_still_override_the_seed() -> None:
    """A schema that *does* import re-binds the same names — no breakage."""
    code = """
from pydantic import BaseModel, Field
from typing import Optional

class Item(BaseModel):
    note: Optional[str] = Field(default=None, description="n")

class Parse(BaseModel):
    items: list[Item]
"""
    cls = check_schema(code)
    assert issubclass(cls, BaseModel)


def test_leniency_does_not_mask_real_nameerror() -> None:
    """The seed only covers known names — a genuinely undefined name still fails."""
    code = """
class Parse(BaseModel):
    x: SomeUndefinedTypeXYZ
"""
    try:
        check_schema(code)
    except SchemaError as e:
        # Still surfaces as SchemaError (the seed doesn't cover this name) — the
        # exact message varies (Pydantic defers it to model_json_schema()).
        assert "fail" in str(e).lower() or "undefined" in str(e).lower()
    else:
        raise AssertionError("expected SchemaError for an undefined name")


def test_syntax_error() -> None:
    try:
        check_schema("class Parse(:")
    except SchemaError as e:
        assert "failed to execute" in str(e)
        assert "SyntaxError" in str(e)
    else:
        raise AssertionError("expected SchemaError")


def test_runtime_error_in_code() -> None:
    """Code that execs but raises (e.g., bad import) should surface as SchemaError."""
    try:
        check_schema("import this_module_does_not_exist_xyz")
    except SchemaError as e:
        assert "failed to execute" in str(e)
    else:
        raise AssertionError("expected SchemaError")


def test_missing_parse_class() -> None:
    code = """
from pydantic import BaseModel

class Foo(BaseModel):
    x: int
"""
    try:
        check_schema(code)
    except SchemaError as e:
        assert "did not define" in str(e)
        assert "Parse" in str(e)
    else:
        raise AssertionError("expected SchemaError")


def test_parse_not_basemodel() -> None:
    try:
        check_schema("class Parse: pass")
    except SchemaError as e:
        assert "pydantic.BaseModel" in str(e)
    else:
        raise AssertionError("expected SchemaError")


def test_parse_is_not_a_class() -> None:
    """`Parse` bound to something that isn't a class (e.g., a value) should fail cleanly."""
    try:
        check_schema("Parse = 42")
    except SchemaError as e:
        assert "pydantic.BaseModel" in str(e)
    else:
        raise AssertionError("expected SchemaError")


def test_invalid_pydantic_field_type() -> None:
    """A `Parse` class whose JSON schema can't be generated should fail at the schema step."""
    code = """
from pydantic import BaseModel

class Weird:
    pass

class Parse(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    x: Weird
"""
    try:
        check_schema(code)
    except SchemaError as e:
        assert "model_json_schema" in str(e)
    else:
        raise AssertionError("expected SchemaError")


def test_rejects_list_of_untyped_dict() -> None:
    """`list[dict]` produces an object without `properties` — OpenAI strict mode crashes
    on this at call time; check_schema should reject it upfront so the schema proposal retries.
    """
    code = """
from pydantic import BaseModel

class Parse(BaseModel):
    parse: list[dict]
"""
    try:
        check_schema(code)
    except SchemaError as e:
        assert "untyped object" in str(e)
        assert "BaseModel" in str(e)
    else:
        raise AssertionError("expected SchemaError for list[dict]")


def test_rejects_untyped_dict_field() -> None:
    code = """
from pydantic import BaseModel

class Parse(BaseModel):
    bag: dict
"""
    try:
        check_schema(code)
    except SchemaError as e:
        assert "untyped object" in str(e)
    else:
        raise AssertionError("expected SchemaError for dict field")


def test_rejects_dict_str_any() -> None:
    code = """
from typing import Any
from pydantic import BaseModel

class Parse(BaseModel):
    bag: dict[str, Any]
"""
    try:
        check_schema(code)
    except SchemaError as e:
        assert "untyped object" in str(e)
    else:
        raise AssertionError("expected SchemaError for dict[str, Any]")


def test_rejects_value_typed_dict() -> None:
    """`dict[str, str]` also has no `properties` block — strict mode rejects it too."""
    code = """
from pydantic import BaseModel

class Parse(BaseModel):
    bag: dict[str, str]
"""
    try:
        check_schema(code)
    except SchemaError as e:
        assert "untyped object" in str(e)
    else:
        raise AssertionError("expected SchemaError for dict[str, str]")


def test_rejects_scalar_top_level_field() -> None:
    """Every top-level `Parse` field must be a `list[...]` — a scalar would collapse to a
    single document at merge time. check_schema rejects it so the schema proposal retries."""
    code = """
from pydantic import BaseModel

class Parse(BaseModel):
    answer: str
"""
    try:
        check_schema(code)
    except SchemaError as e:
        assert "list[...]" in str(e) and "answer" in str(e)
    else:
        raise AssertionError("expected SchemaError for a scalar top-level field")


def test_rejects_single_object_top_level_field() -> None:
    """A single nested object at top level (the `3cc3047c` collapse shape: `Parse.document:
    <Model>`) is rejected too — each document's object would overwrite the previous at merge."""
    code = """
from pydantic import BaseModel

class DocInfo(BaseModel):
    title: str

class Parse(BaseModel):
    document: DocInfo
"""
    try:
        check_schema(code)
    except SchemaError as e:
        assert "list[...]" in str(e) and "document" in str(e)
    else:
        raise AssertionError("expected SchemaError for a single-object top-level field")


# -------------------- parse_one_document retry tests --------------------


class _Item(BaseModel):
    text: str


class _Parse(BaseModel):
    items: list[_Item]


_SCHEMA_CODE = "from pydantic import BaseModel\nclass Item(BaseModel):\n    text: str\nclass Parse(BaseModel):\n    items: list[Item]\n"


def _make_validation_error() -> ValidationError:
    """Synthesize a real ValidationError so the retry path sees the right exception type."""
    try:
        _Parse.model_validate({"items": "not-a-list"})  # wrong shape
    except ValidationError as e:
        return e
    raise RuntimeError("expected ValidationError")


@contextlib.contextmanager
def _patched_llm(responses: list[Any]) -> Iterator[list[dict[str, Any]]]:
    """Replace litellm_chat_completion with a scripted fake.

    Each call pops the next response from ``responses``. If the response is
    an exception instance, it's raised; otherwise it's returned. Yields the
    list of call-kwargs seen (in call order) for assertion.
    """
    original = parsing_mod.litellm_chat_completion
    calls: list[dict[str, Any]] = []
    queue = list(responses)

    def fake(**kwargs: Any) -> Any:
        calls.append(kwargs)
        if not queue:
            raise AssertionError("LLM called more times than the script provided")
        nxt = queue.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return nxt

    parsing_mod.litellm_chat_completion = fake  # type: ignore[assignment]
    try:
        yield calls
    finally:
        parsing_mod.litellm_chat_completion = original  # type: ignore[assignment]


def _run_extract(
    responses: list[Any],
    *,
    max_attempts: int = 3,
) -> tuple[BaseModel | None, list[dict[str, Any]], BaseException | None]:
    """Run parse_one_document with the scripted LLM. Returns (result, calls, raised_or_None)."""
    with _patched_llm(responses) as calls:
        try:
            result = parse_one_document(
                document="some text",
                schema_code=_SCHEMA_CODE,
                parse_cls=_Parse,
                task="q",
                model="m",
                prompt_version="v1",
                max_attempts=max_attempts,
            )
            return result, calls, None
        except BaseException as e:
            return None, calls, e


def test_parse_one_document_succeeds_on_first_try_no_retry() -> None:
    """When the first attempt validates, no retry happens."""
    good = _Parse(items=[_Item(text="a")])
    result, calls, raised = _run_extract([good])
    assert raised is None
    assert result is good
    assert len(calls) == 1
    assert calls[0]["kind"] == "llm_call"
    # Initial user prompt is just the document, no retry-feedback wrapper.
    assert calls[0]["user_prompt"] == "some text"


def test_parse_one_document_retries_on_validation_error_and_recovers() -> None:
    """First call raises ValidationError; second call succeeds; total 2 calls."""
    bad = _make_validation_error()
    good = _Parse(items=[_Item(text="ok")])
    result, calls, raised = _run_extract([bad, good])
    assert raised is None
    assert result is good
    assert len(calls) == 2
    # Second call's kind is a retry-tagged variant of the original.
    assert calls[1]["kind"].startswith("llm_call-retry")
    # Second call's user_prompt includes the original document + the error.
    assert "some text" in calls[1]["user_prompt"]
    assert "ValidationError" in calls[1]["user_prompt"]


def test_parse_one_document_raises_schema_error_after_max_attempts() -> None:
    """After max_attempts ValidationErrors, raises SchemaError naming the last error."""
    err = _make_validation_error()
    result, calls, raised = _run_extract([err, err, err], max_attempts=3)
    assert result is None
    assert isinstance(raised, SchemaError)
    assert "exhausted 3 attempts" in str(raised)
    assert "ValidationError" in str(raised)
    assert len(calls) == 3


def test_parse_one_document_zero_attempts_rejected() -> None:
    """max_attempts must be >= 1."""
    result, calls, raised = _run_extract([], max_attempts=0)
    assert isinstance(raised, ValueError)
    assert "max_attempts" in str(raised)
    assert len(calls) == 0


def test_parse_one_document_non_validation_error_does_not_retry() -> None:
    """An unrelated exception (e.g. RuntimeError) bubbles up immediately — no retry."""
    err = RuntimeError("API outage")
    result, calls, raised = _run_extract([err], max_attempts=3)
    assert isinstance(raised, RuntimeError)
    assert "API outage" in str(raised)
    # Only one call — we don't retry on non-ValidationError exceptions.
    assert len(calls) == 1


def test_typed_basemodel_subclass_passes() -> None:
    """A schema with a typed nested BaseModel must still pass (no regression)."""
    code = """
from pydantic import BaseModel

class Event(BaseModel):
    name: str
    when: str

class Parse(BaseModel):
    events: list[Event]
"""
    cls = check_schema(code)
    assert cls.__name__ == "Parse"


if __name__ == "__main__":
    tests = [
        test_simple_schema,
        test_nested_schema_resolves_forward_types,
        test_schema_with_literal,
        test_schema_without_pydantic_import_still_loads,
        test_schema_without_typing_imports_still_loads,
        test_explicit_imports_still_override_the_seed,
        test_leniency_does_not_mask_real_nameerror,
        test_syntax_error,
        test_runtime_error_in_code,
        test_missing_parse_class,
        test_parse_not_basemodel,
        test_parse_is_not_a_class,
        test_invalid_pydantic_field_type,
        test_rejects_list_of_untyped_dict,
        test_rejects_untyped_dict_field,
        test_rejects_dict_str_any,
        test_rejects_value_typed_dict,
        test_rejects_scalar_top_level_field,
        test_rejects_single_object_top_level_field,
        test_typed_basemodel_subclass_passes,
        test_parse_one_document_succeeds_on_first_try_no_retry,
        test_parse_one_document_retries_on_validation_error_and_recovers,
        test_parse_one_document_raises_schema_error_after_max_attempts,
        test_parse_one_document_zero_attempts_rejected,
        test_parse_one_document_non_validation_error_does_not_retry,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
