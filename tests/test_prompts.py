"""`r3con.prompts` — the packaged prompts, and the overlay that keeps versioning usable
after the package is installed."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from r3con.config import PROMPT_STAGES, load_config  # noqa: E402
from r3con.prompts import load_prompt, prompt_search_path  # noqa: E402
from r3con.settings import settings  # noqa: E402

# Enough kwargs to render any stage's template; Jinja ignores the ones it doesn't use.
CTX = dict(task="T", schema_code="S", relevance="", other_states="", parsed="{}",
           parse_block="{}", samples_block="", parse_json="")


def test_every_stage_pinned_by_the_default_config_actually_ships() -> None:
    """The shipped config must not pin a prompt version that isn't in the wheel — the
    failure would land after the first LLM call, not at import."""
    cfg = load_config("default")
    for stage in PROMPT_STAGES:
        version = cfg.prompts[stage]
        assert (settings.PROMPTS_DIR / stage / f"{version}.yaml").is_file(), f"{stage}/{version}"
        assert load_prompt(stage, version=version, **CTX).strip()


def test_search_path_is_overlay_then_package() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        os.environ["R3CON_PROMPTS_DIR"] = tmp
        try:
            path = prompt_search_path()
        finally:
            del os.environ["R3CON_PROMPTS_DIR"]
    assert path[0] == Path(tmp)
    assert path[-1] == settings.PROMPTS_DIR


def test_an_overlay_version_is_found_and_the_packaged_one_still_is() -> None:
    """Adding a version means dropping one file in the overlay — you don't have to copy
    the versions you didn't change."""
    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp) / "relevance"
        stage_dir.mkdir(parents=True)
        (stage_dir / "v2.yaml").write_text('instructions: |-\n  MY OVERLAY PROMPT {{ task }}\n', encoding="utf-8")
        os.environ["R3CON_PROMPTS_DIR"] = tmp
        try:
            assert "MY OVERLAY PROMPT T" in load_prompt("relevance", version="v2", **CTX)
            # the packaged v1 is still reachable — the overlay adds, it doesn't replace
            assert "MY OVERLAY PROMPT" not in load_prompt("relevance", version="v1", **CTX)
        finally:
            del os.environ["R3CON_PROMPTS_DIR"]


def test_an_overlay_shadows_a_packaged_version_of_the_same_name() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp) / "relevance"
        stage_dir.mkdir(parents=True)
        (stage_dir / "v1.yaml").write_text("instructions: |-\n  SHADOWED\n", encoding="utf-8")
        os.environ["R3CON_PROMPTS_DIR"] = tmp
        try:
            assert load_prompt("relevance", version="v1", **CTX).strip() == "SHADOWED"
        finally:
            del os.environ["R3CON_PROMPTS_DIR"]


def test_a_missing_version_names_every_place_it_looked() -> None:
    try:
        load_prompt("relevance", version="v999", **CTX)
    except FileNotFoundError as e:
        msg = str(e)
        assert "v999" in msg and "relevance" in msg
        assert str(settings.PROMPTS_DIR) in msg, "the error must name the packaged location too"
        return
    raise AssertionError("expected FileNotFoundError")


def test_examples_are_appended_to_the_instructions() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = Path(tmp) / "relevance"
        stage_dir.mkdir(parents=True)
        (stage_dir / "v3.yaml").write_text(
            "instructions: |-\n  BODY\nexamples:\n  - name: one\n    text: |-\n      EXAMPLE-ONE\n",
            encoding="utf-8",
        )
        os.environ["R3CON_PROMPTS_DIR"] = tmp
        try:
            out = load_prompt("relevance", version="v3", **CTX)
        finally:
            del os.environ["R3CON_PROMPTS_DIR"]
    assert "BODY" in out and "EXAMPLE-ONE" in out and "Examples:" in out


if __name__ == "__main__":
    tests = [
        test_every_stage_pinned_by_the_default_config_actually_ships,
        test_search_path_is_overlay_then_package,
        test_an_overlay_version_is_found_and_the_packaged_one_still_is,
        test_an_overlay_shadows_a_packaged_version_of_the_same_name,
        test_a_missing_version_names_every_place_it_looked,
        test_examples_are_appended_to_the_instructions,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
