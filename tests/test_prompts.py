import pytest

from r3con import settings
from r3con.config import PROMPT_STAGES, load_config
from r3con.prompts import load_prompt, prompt_search_path

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
