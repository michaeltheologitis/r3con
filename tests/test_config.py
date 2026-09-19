"""Tests for `r3con.config` — the RunConfig (experiment identity) + its loaders.

The happy-path tests run against the real ``configs/`` folder; validation / error
cases use a temp configs dir. Run with:
uv run python tests/test_config.py
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Iterator

import yaml

from r3con.config import available_configs, load_config
from r3con.settings import settings

_FULL = {
    "model": "openai/gpt-5.6-luna",
    "seed": 42,
    "relevance_rounds": 3,
    "prompts": {
        "relevance": "v1", "structuring/schema": "v1", "structuring/parsing": "v1",
        "reasoning": "v1",
    },
}


@contextlib.contextmanager
def _temp_configs(files: dict) -> Iterator[Path]:
    """Point ``settings.CONFIGS_DIR`` at a tmp dir holding ``files`` (``name -> dict``).
    Each key is a config name; each value is the config dict."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, data in files.items():
            p = root / f"{name}.yaml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(yaml.safe_dump(data))
        original = settings.CONFIGS_DIR
        settings.CONFIGS_DIR = root
        try:
            yield root
        finally:
            settings.CONFIGS_DIR = original


# ---------- available_* (real configs/) ----------



def test_load_default_config() -> None:
    c = load_config("default")
    assert c.name == "default"
    assert c.model == "openai/gpt-5.6-luna"
    assert c.seed == 42 and c.relevance_rounds == 2
    assert c.params == {}
    assert c.prompts["relevance"] == "v1" and c.prompts["reasoning"] == "v1"
    assert c.prompts["reasoning"] == "v1"
    assert c.label() == "default[model=gpt-5.6-luna,seed=42,rounds=2,prompts=(rel=v1,schema=v1,parse=v1,reason=v1)]"


# ---------- CLI overrides ----------


def test_overrides_apply_and_show_in_label() -> None:
    c = load_config("default", seed=7, model="openai/gpt-x")
    assert c.seed == 7 and c.model == "openai/gpt-x"
    assert c.overrides == {"seed": 7, "model": "openai/gpt-x"}
    # the label renders the RESOLVED identity (seed 7, bare model) — not the override mechanics
    assert c.label().startswith("default[model=gpt-x,seed=7,rounds=2,prompts=(")


def test_label_sanitizes_model_override() -> None:
    """The label shows the bare model name (provider/route prefix dropped — it's transport),
    while the live config keeps the full string for LiteLLM routing."""
    c = load_config("default", model="hosted_vllm/Qwen/Qwen3.5-35B-A3B")
    assert c.model == "hosted_vllm/Qwen/Qwen3.5-35B-A3B"  # live model unchanged (routing)
    assert c.label().startswith("default[model=Qwen3.5-35B-A3B,seed=42,rounds=2,prompts=(")


def test_none_overrides_are_ignored() -> None:
    c = load_config("default", seed=None, model=None, relevance_rounds=None)
    assert c.overrides == {} and c.seed == 42
    assert c.label().startswith("default[model=gpt-5.6-luna,seed=42,rounds=2,prompts=(")


def test_label_is_full_resolved_identity() -> None:
    """The label is built from the resolved identity (model, seed, relevance_rounds, prompts) — so
    two runs differing in ANY of these are distinct cells, even under the same config name."""
    bumped = {**_FULL, "prompts": {**_FULL["prompts"], "reasoning": "v9"}}
    with _temp_configs({"exp": _FULL, "exp2": bumped}):
        base = "exp[model=gpt-5.6-luna,seed=42,rounds=3,prompts=(rel=v1,schema=v1,parse=v1,reason=v1)]"
        assert load_config("exp").label() == base
        # seed and relevance_rounds are part of the identity (always, not just when overridden)
        assert load_config("exp", seed=7).label() == \
            "exp[model=gpt-5.6-luna,seed=7,rounds=3,prompts=(rel=v1,schema=v1,parse=v1,reason=v1)]"
        assert load_config("exp", relevance_rounds=5).label() == \
            "exp[model=gpt-5.6-luna,seed=42,rounds=5,prompts=(rel=v1,schema=v1,parse=v1,reason=v1)]"
        # bumping ONE prompt stage changes the label too
        assert load_config("exp2").label() == \
            "exp2[model=gpt-5.6-luna,seed=42,rounds=3,prompts=(rel=v1,schema=v1,parse=v1,reason=v9)]"


def test_model_dump_round_trips_for_manifest() -> None:
    d = load_config("default", seed=9).model_dump()
    assert d["model"] == "openai/gpt-5.6-luna" and d["seed"] == 9
    assert d["overrides"] == {"seed": 9} and "prompts" in d and "params" in d


# ---------- generation params (temp configs) ----------


def test_unknown_config_raises() -> None:
    try:
        load_config("does-not-exist-xyz")
    except KeyError as e:
        assert "does-not-exist-xyz" in str(e)
    else:
        raise AssertionError("expected KeyError")



def test_missing_model_raises() -> None:
    bad = {k: v for k, v in _FULL.items() if k != "model"}
    with _temp_configs({"bad": bad}):
        try:
            load_config("bad")
        except ValueError as e:
            assert "model" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_missing_prompt_stage_raises() -> None:
    with _temp_configs({"bad": {**_FULL, "prompts": {"relevance": "v1"}}}):
        try:
            load_config("bad")
        except ValueError as e:
            assert "prompt versions" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_params_are_part_of_the_identity() -> None:
    """Generation params shape the output, so two runs differing only in temperature are
    different runs — but a provider-default run carries no `params=` noise."""
    base = {"model": "openai/m", "prompts": {k: "v1" for k in
            ("relevance", "structuring/schema", "structuring/parsing", "reasoning")}}
    with _temp_configs({"plain": base, "warm": {**base, "params": {"temperature": 0.7}}}):
        assert "params=" not in load_config("plain").label()
        assert "params={temperature=0.7}" in load_config("warm").label()
        assert load_config("warm").params == {"temperature": 0.7}
        # an override is applied and recorded
        c = load_config("plain", params={"top_p": 0.8})
        assert c.params == {"top_p": 0.8} and c.overrides["params"] == {"top_p": 0.8}
        assert "params={top_p=0.8}" in c.label()


def test_available_configs_lists_what_load_config_accepts() -> None:
    with _temp_configs({"one": {"model": "m", "prompts": {k: "v1" for k in
                        ("relevance", "structuring/schema", "structuring/parsing", "reasoning")}}}):
        assert "one" in available_configs()


if __name__ == "__main__":
    tests = [
        test_params_are_part_of_the_identity,
        test_available_configs_lists_what_load_config_accepts,
        test_load_default_config,
        test_overrides_apply_and_show_in_label,
        test_label_sanitizes_model_override,
        test_none_overrides_are_ignored,
        test_label_is_full_resolved_identity,
        test_model_dump_round_trips_for_manifest,
        test_unknown_config_raises,
        test_missing_model_raises,
        test_missing_prompt_stage_raises,
    ]
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
    print(f"\nOK — {len(tests)} tests")
