import pytest

from r3con.stages.reasoning import _sample_record_per_field, reason

MODEL = "openai/gpt-5.6-luna"
COMMIT = "Thought: commit.\n<code>\nfinal_answer('ok')\n</code>"


def reasoning_prompt(llm, **kwargs) -> str:
    """The system prompt ``reason`` sends on its first turn."""
    kwargs.setdefault("task", "?")
    kwargs.setdefault("schema_code", "class Parse(BaseModel): rows: list[Row]")
    reason(model=MODEL, prompt_version="v1", completion=llm.replies(COMMIT), **kwargs)
    return llm.requests[0]["messages"][0]["content"]


def test_the_parse_is_bound_in_the_sandbox_and_the_commit_returned(llm):
    summing = "<code>\nfinal_answer(sum(it['n'] for it in parse['items']))\n</code>"
    result = reason(
        task="sum?",
        schema_code="class Parse(BaseModel): items: list[Item]",
        parsed={"items": [{"n": 1}, {"n": 2}, {"n": 3}]},
        model=MODEL,
        prompt_version="v1",
        completion=llm.replies(summing),
    )
    assert (result.answer, result.terminated_by) == ("6", "final_answer")


def test_the_user_message_is_the_task_in_tags(llm):
    reasoning_prompt(llm, task="What did Alice say?", parsed={"rows": []})
    user = llm.requests[0]["messages"][1]["content"]
    assert user == "Input:\n<task>\nWhat did Alice say?\n</task>"


def test_every_parse_field_reaches_the_prompt_even_an_empty_one(llm):
    prompt = reasoning_prompt(
        llm, parsed={"supports": [{"ok": True}], "oppositions": []}
    )
    assert '"supports"' in prompt
    assert '"oppositions": []' in prompt


@pytest.mark.parametrize(
    ("snippets", "shown"), [(["ZZQX-note-42"], True), (None, False), ([], False)]
)
def test_the_relevance_notes_are_in_the_prompt_only_when_given(llm, snippets, shown):
    # The heading also appears in the prompt's examples, so look for the note itself.
    prompt = reasoning_prompt(
        llm, parsed={"x": [{"y": 1}]}, relevance_snippets=snippets
    )
    assert ("ZZQX-note-42" in prompt) is shown
    assert ("## Document summaries\n\nA short, per-document" in prompt) is shown


def test_each_record_is_stamped_with_its_one_based_source_document(llm):
    prompt = reasoning_prompt(
        llm, parsed={"docs": [{"a": 1}]}, source_docs={"docs": [4]}
    )
    assert '"document": 5' in prompt


def test_a_small_parse_is_shown_whole_and_the_schema_source_is_not(llm):
    schema = "class Rec(BaseModel):\n    UNIQUE_SCHEMA_MARKER: str"
    parsed = {"records": [{"name": f"name_{i}"} for i in range(8)]}
    prompt = reasoning_prompt(
        llm, schema_code=schema, parsed=parsed, source_docs={"records": list(range(8))}
    )
    assert "name_0" in prompt
    assert "name_7" in prompt
    assert "`document`" in prompt
    assert '"document": 1' in prompt
    assert "UNIQUE_SCHEMA_MARKER" not in prompt
    assert "## Schema used to extract" not in prompt
    assert "If the parse was too large" not in prompt


def test_a_huge_parse_is_shown_as_one_sample_per_field_and_says_so(llm):
    parsed = {
        "records": [
            {"name": f"name_{i}", "blob": f"item {i}: descriptive text {i * 7}"}
            for i in range(1500)
        ]
    }
    prompt = reasoning_prompt(llm, parsed=parsed)
    assert "name_0" in prompt
    assert "name_1499" not in prompt
    assert "only a SAMPLE" in prompt
    assert "`parse` variable" in prompt


def test_non_ascii_values_reach_the_prompt_as_text(llm):
    prompt = reasoning_prompt(llm, task="哪个判决结果?", parsed={"判决": ["判决结果4"]})
    assert "判决结果4" in prompt
    assert "\\u5224" not in prompt


@pytest.mark.parametrize(
    ("parse", "shown"),
    [
        ({"判决文书1_result": "判决结果4"}, '`判决文书1_result` (scalar): "判决结果4"'),
        ({"rows": [{"label": "应付账款"}]}, '"label": "应付账款"'),
        ({"meta": {"机构": "上海"}}, '"机构": "上海"'),
    ],
)
def test_the_sample_view_keeps_non_ascii_values_readable(parse, shown):
    sample = _sample_record_per_field(parse)
    assert shown in sample
    assert "\\u" not in sample
