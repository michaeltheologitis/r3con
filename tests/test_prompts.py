import json
import re

import pytest
import yaml

from r3con import settings
from r3con.config import PROMPT_STAGES, load_config
from r3con.prompts import load_prompt, prompt_search_path
from r3con.runtime.codeact import run_codeact
from r3con.stages.reasoning import reason

MODEL = "openai/gpt-6-luna"
PINNED = load_config("default").prompts

# Enough to render any stage's template; Jinja ignores the names it does not use.
CONTEXT = {
    "task": "T",
    "schema_code": "S",
    "relevance": "",
    "other_snippets": "",
    "parse_block": "{}",
    "samples_block": "",
    "parse_json": "",
}


def examples(stage: str) -> list:
    """The text of each example of the prompt the default config pins for ``stage``,
    under its name."""
    path = settings.PROMPTS_DIR / stage / f"{PINNED[stage]}.yaml"
    prompt = yaml.safe_load(path.read_text())
    return [pytest.param(e["text"], id=e["name"]) for e in prompt["examples"]]


def shown(example: str) -> tuple[list[str], dict]:
    """An example's document ids, from its summaries' headings, and its parse, each
    record of which names one of them."""
    ids = re.findall(r"^### Document ID: (\w+)$", example, re.MULTILINE)
    parse = json.loads(example.split("`parse`:\n", 1)[1].split("\n\nThought:", 1)[0])
    records = [record for field in parse.values() for record in field]
    assert {record["document"] for record in records} <= set(ids)
    return ids, parse


@pytest.fixture
def overlay(tmp_path, monkeypatch):
    """Writes ``relevance/<version>.yaml`` files into a prompt overlay."""
    monkeypatch.setenv("R3CON_PROMPTS_DIR", str(tmp_path / "overlay"))

    def write(version: str, text: str) -> None:
        path = tmp_path / "overlay" / "relevance" / f"{version}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    return write


@pytest.mark.parametrize("stage", PROMPT_STAGES)
def test_every_prompt_the_default_config_pins_ships_and_renders(stage):
    version = load_config("default").prompts[stage]
    assert (settings.PROMPTS_DIR / stage / f"{version}.yaml").is_file()
    assert load_prompt(stage, version=version, **CONTEXT).strip()


def test_the_first_round_prompt_renders_as_the_old_relevance_prompts_round_one():
    first_round = load_prompt("relevance/first-round", version="v1", task="T")
    old = load_prompt("relevance", version="v1", task="T", other_snippets="")
    assert first_round == old


@pytest.mark.parametrize(
    "context",
    [
        CONTEXT,
        {**CONTEXT, "relevance": "### Document 1\nA note.", "read_in_parts": False},
        {**CONTEXT, "relevance": "### Document 1.1\nA part.", "read_in_parts": True},
    ],
    ids=["no-summaries", "summaries", "a-part"],
)
def test_the_coding_prompt_renders_as_the_old_reasoning_v2(context):
    coding = load_prompt("reasoning/coding", version="v1", **context)
    assert coding == load_prompt("reasoning", version="v2", **context)


@pytest.mark.parametrize("example", examples("reasoning/direct"))
def test_each_direct_example_replies_in_json_its_reply_schema_takes(llm, example):
    ids, parse = shown(example)
    reply = json.loads(example.rsplit("\n", 1)[1])
    result = reason(
        task="?",
        schema_code="",
        parsed=parse,
        model=MODEL,
        prompts=PINNED,
        path="direct",
        document_ids=ids,
        completion=llm.replies(reply),
    )
    assert set(reply) == {"answer", "provenance"}
    assert (result.answer, result.provenance) == (reply["answer"], reply["provenance"])
    assert len(llm.requests) == 1


@pytest.mark.parametrize("example", examples("reasoning/coding"))
def test_each_coding_example_runs_and_commits_ids_it_shows(llm, example):
    ids, parse = shown(example)
    turns = re.findall(r"^Thought:.*?</code>", example, re.MULTILINE | re.DOTALL)
    printed = re.findall(
        r"^<observation>\n(.*?)\n</observation>$", example, re.MULTILINE | re.DOTALL
    )
    result = run_codeact(
        system_prompt="",
        user_message="",
        model=MODEL,
        variables={"parse": parse},
        completion=llm.replies(*turns),
    )
    assert (result.terminated_by, len(result.turns)) == ("final_answer", len(turns))
    assert [turn.observation for turn in result.turns[:-1]] == [
        f"<observation>\n{text}\n\n</observation>" for text in printed
    ]
    assert result.provenance and set(result.provenance) <= set(ids)


def test_the_overlay_is_searched_before_the_package(tmp_path, monkeypatch):
    monkeypatch.setenv("R3CON_PROMPTS_DIR", str(tmp_path))
    assert prompt_search_path() == [tmp_path, settings.PROMPTS_DIR]


def test_an_overlay_adds_a_version_and_the_packaged_ones_stay(overlay):
    overlay("v2", "instructions: |-\n  MY OVERLAY PROMPT {{ task }}\n")
    assert "MY OVERLAY PROMPT T" in load_prompt("relevance", version="v2", **CONTEXT)
    assert "MY OVERLAY PROMPT" not in load_prompt("relevance", version="v1", **CONTEXT)


def test_an_overlay_shadows_the_packaged_version_of_the_same_name(overlay):
    overlay("v1", "instructions: |-\n  SHADOWED\n")
    assert load_prompt("relevance", version="v1", **CONTEXT) == "SHADOWED"


def test_editing_the_file_between_calls_changes_the_rendered_prompt(overlay):
    overlay("v2", "instructions: |-\n  FIRST {{ task }}\n")
    assert load_prompt("relevance", version="v2", **CONTEXT) == "FIRST T"
    overlay("v2", "instructions: |-\n  SECOND {{ task }}\n")
    assert load_prompt("relevance", version="v2", **CONTEXT) == "SECOND T"


def test_a_missing_version_names_every_place_it_looked(tmp_path):
    with pytest.raises(
        FileNotFoundError, match="'relevance' version 'v999'"
    ) as missing:
        load_prompt("relevance", version="v999", **CONTEXT)
    assert str(tmp_path / "prompts") in str(missing.value)
    assert str(settings.PROMPTS_DIR) in str(missing.value)


def test_examples_are_appended_to_the_instructions(overlay):
    overlay(
        "v3",
        "instructions: |-\n  BODY\nexamples:\n  - name: one\n    text: |-\n      ONE\n",
    )
    assert (
        load_prompt("relevance", version="v3", **CONTEXT) == "BODY\n\nExamples:\n\nONE"
    )
