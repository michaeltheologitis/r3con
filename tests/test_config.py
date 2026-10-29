import re

import pytest
from pydantic import ValidationError

from r3con import settings
from r3con.config import PROMPT_STAGES, RunConfig, available_configs, load_config

PROMPTS = dict.fromkeys(PROMPT_STAGES, "v1")
CODING = "reasoning/coding"
FULL = {
    "model": "openai/gpt-6-luna",
    "relevance_rounds": 3,
    "prompts": PROMPTS,
}
V1_PROMPTS = "prompts=(rel1=v1,rel2=v1,schema=v1,parse=v1,direct=v1,coding=v1)"
DEFAULT_PROMPTS = {
    **PROMPTS,
    "relevance/later-rounds": "v2",
    "structuring/parsing": "v3",
    "reasoning/direct": "v3",
    CODING: "v3",
}
DEFAULT_LABEL_PROMPTS = (
    "prompts=(rel1=v1,rel2=v2,schema=v1,parse=v3,direct=v3,coding=v3)"
)
# The prompts the default config pinned before reasoning had two paths.
OLD_DEFAULT_PROMPTS = {
    "relevance/first-round": "v1",
    "relevance/later-rounds": "v1",
    "structuring/schema": "v1",
    "structuring/parsing": "v1",
    "reasoning": "v2",
}


def test_the_default_config_loads_with_its_shipped_values():
    config = load_config("default")
    assert config.name == "default"
    assert config.model == "openai/gpt-6-luna"
    assert (config.relevance_rounds, config.reasoning, config.params) == (
        2,
        "direct",
        {},
    )
    assert config.prompts == DEFAULT_PROMPTS
    assert "seed" not in config.model_dump()
    assert config.label() == (
        f"default[model=gpt-6-luna,rounds=2,reasoning=direct,{DEFAULT_LABEL_PROMPTS}]"
    )


@pytest.mark.parametrize(
    ("overrides", "recorded", "label"),
    [
        (
            {"relevance_rounds": 3, "model": "openai/gpt-x"},
            {"relevance_rounds": 3, "model": "openai/gpt-x"},
            "default[model=gpt-x,rounds=3,",
        ),
        (
            {"model": "hosted_vllm/Qwen/Qwen3.5-35B-A3B"},
            {"model": "hosted_vllm/Qwen/Qwen3.5-35B-A3B"},
            "default[model=Qwen3.5-35B-A3B,rounds=2,",
        ),
        (
            {"reasoning": "coding"},
            {"reasoning": "coding"},
            "default[model=gpt-6-luna,rounds=2,reasoning=coding,",
        ),
        (
            {"relevance_rounds": 0},
            {"relevance_rounds": 0},
            "default[model=gpt-6-luna,rounds=0,",
        ),
        (
            {
                "model": None,
                "relevance_rounds": None,
                "reasoning": None,
                "params": None,
            },
            {},
            "default[model=gpt-6-luna,rounds=2,reasoning=direct,",
        ),
    ],
)
def test_overrides_are_applied_recorded_and_shown_in_the_label(
    overrides, recorded, label
):
    config = load_config("default", **overrides)
    assert config.overrides == recorded
    assert config.model_dump()["overrides"] == recorded
    assert config.label().startswith(label)
    assert config.model == recorded.get("model", "openai/gpt-6-luna")


def test_the_label_is_every_axis_of_the_resolved_identity(configs, tmp_path):
    configs(
        exp=FULL,
        exp2={**FULL, "reasoning": "coding", "prompts": {**PROMPTS, CODING: "v9"}},
    )
    v9 = tmp_path / "prompts" / CODING / "v9.yaml"
    v9.parent.mkdir(parents=True)
    v9.write_text("instructions: |-\n  V9\n")
    assert load_config("exp").label() == (
        f"exp[model=gpt-6-luna,rounds=3,reasoning=direct,{V1_PROMPTS}]"
    )
    assert load_config("exp", relevance_rounds=5).label() == (
        f"exp[model=gpt-6-luna,rounds=5,reasoning=direct,{V1_PROMPTS}]"
    )
    assert load_config("exp", params={"seed": 7}).label() == (
        f"exp[model=gpt-6-luna,rounds=3,reasoning=direct,params={{seed=7}},{V1_PROMPTS}]"
    )
    assert load_config("exp2").label() == (
        "exp2[model=gpt-6-luna,rounds=3,reasoning=coding,"
        "prompts=(rel1=v1,rel2=v1,schema=v1,parse=v1,direct=v1,coding=v9)]"
    )


def test_params_are_part_of_the_identity_only_when_set(configs):
    base = {"model": "openai/m", "prompts": PROMPTS}
    configs(plain=base, warm={**base, "params": {"temperature": 0.7}})
    assert "params=" not in load_config("plain").label()
    warm = load_config("warm")
    assert warm.params == {"temperature": 0.7}
    assert "params={temperature=0.7}" in warm.label()
    overridden = load_config("plain", params={"top_p": 0.8})
    assert overridden.params == overridden.overrides["params"] == {"top_p": 0.8}
    assert "params={top_p=0.8}" in overridden.label()


NO_MODEL = "config 'bad': `model` is empty or missing."
MISSING_PROMPTS = "config 'bad' is missing prompt versions for stage(s): "
NO_PROMPTS = f"{MISSING_PROMPTS}{list(PROMPT_STAGES)}"


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"prompts": PROMPTS}, NO_MODEL),
        ({"model": None, "prompts": PROMPTS}, NO_MODEL),
        ({"model": "openai/m"}, NO_PROMPTS),
        ({"model": "openai/m", "prompts": None}, NO_PROMPTS),
        ({"model": "openai/m", "prompts": {}}, NO_PROMPTS),
        (
            {**FULL, "prompts": {CODING: "v1"}},
            f"{MISSING_PROMPTS}{list(PROMPT_STAGES[:-1])}",
        ),
    ],
    ids=[
        "no-model",
        "empty-model",
        "no-prompts",
        "empty-prompts",
        "no-stage-pinned",
        "one-stage-pinned",
    ],
)
def test_a_config_missing_what_a_run_needs_is_refused(configs, fields, message):
    configs(bad=fields)
    with pytest.raises(ValueError) as refused:
        load_config("bad")
    assert str(refused.value) == message


def test_a_config_without_a_model_loads_when_the_model_is_overridden(configs):
    configs(any_model={"prompts": PROMPTS})
    config = load_config("any_model", model="openai/m")
    assert (config.model, config.overrides) == ("openai/m", {"model": "openai/m"})


THE_STAGES = (
    "Input should be 'relevance/first-round', 'relevance/later-rounds', "
    "'structuring/schema', 'structuring/parsing', 'reasoning/direct' or "
    "'reasoning/coding'"
)


@pytest.mark.parametrize(
    ("prompts", "refusal"),
    [
        ({**PROMPTS, "relevance": "v1"}, f"`prompts.relevance`: {THE_STAGES}"),
        ({"relevance": "v1"}, f"`prompts.relevance`: {THE_STAGES}"),
        (OLD_DEFAULT_PROMPTS, f"`prompts.reasoning`: {THE_STAGES}"),
        ("v1", "`prompts='v1'`: Input should be a valid dictionary"),
        (5, "`prompts=5`: Input should be a valid dictionary"),
    ],
    ids=[
        "unknown-beside-the-stages",
        "unknown-alone",
        "the-old-reasoning-stage",
        "a-string",
        "a-number",
    ],
)
def test_bad_prompts_in_a_config_are_refused_in_one_line(configs, prompts, refusal):
    configs(old={**FULL, "prompts": prompts})
    with pytest.raises(ValueError) as refused:
        load_config("old")
    assert str(refused.value) == f"config 'old': {refusal}."


def test_a_run_config_refuses_a_stage_r3con_does_not_have():
    with pytest.raises(ValidationError, match=re.escape(THE_STAGES)):
        RunConfig(model="openai/m", prompts={**PROMPTS, "relevance": "v1"})


NOT_AN_INTEGER = "Input should be a valid integer"
BAD_SETTINGS = [
    ("relevance_rounds", "3", NOT_AN_INTEGER),
    ("relevance_rounds", True, NOT_AN_INTEGER),
    ("relevance_rounds", -1, "Input should be greater than or equal to 0"),
    ("reasoning", "Direct", "Input should be 'direct' or 'coding'"),
    ("params", "temperature=0.7", "Input should be a valid dictionary"),
    ("params", ["temperature", 0.7], "Input should be a valid dictionary"),
    ("model", 5, "Input should be a valid string"),
    ("model", "", "String should have at least 1 character"),
]


@pytest.mark.parametrize(("field", "value", "reason"), BAD_SETTINGS)
@pytest.mark.parametrize(
    ("name", "where"),
    [("default", "override"), ("bad", "config 'bad'")],
    ids=["override", "file"],
)
def test_a_bad_setting_is_refused_in_one_line_naming_it(
    configs, name, where, field, value, reason
):
    configs(bad={**FULL, field: value})
    overrides = {field: value} if where == "override" else {}
    with pytest.raises(ValueError) as refused:
        load_config(name, **overrides)
    assert str(refused.value) == f"{where}: `{field}={value!r}`: {reason}."


def test_every_refused_setting_is_named_in_one_line_with_where_it_was_set(configs):
    configs(bad={**FULL, "reasoning": "agent"})
    with pytest.raises(ValueError) as refused:
        load_config("bad", relevance_rounds=-1)
    assert str(refused.value) == (
        "override: `relevance_rounds=-1`: Input should be greater than or equal to 0; "
        "config 'bad': `reasoning='agent'`: Input should be 'direct' or 'coding'."
    )


def test_a_long_refused_value_is_shortened():
    with pytest.raises(ValueError) as refused:
        load_config("default", params=["temperature"] * 100)
    assert str(refused.value) == (
        "override: `params=['temperature', 'temperature', 'temperature', 'temperature', "
        "'temperature', 'temperature', ...]`: Input should be a valid dictionary."
    )


def test_a_setting_left_empty_in_a_config_file_takes_its_default(configs):
    configs(empty={**FULL, "reasoning": None, "params": None})
    config = load_config("empty")
    assert (config.reasoning, config.params) == ("direct", {})


def test_a_run_config_refuses_a_reasoning_path_r3con_does_not_have():
    with pytest.raises(ValidationError, match="Input should be 'direct' or 'coding'"):
        RunConfig(model="openai/m", prompts=PROMPTS, reasoning="agent")


@pytest.mark.parametrize(("name", "value"), [("relevence_rounds", 3), ("seed", 7)])
def test_a_misspelt_override_is_refused(name, value):
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
        load_config("default", **{name: value})


@pytest.mark.parametrize(
    ("field", "message"),
    [
        (
            {"seed": 42},
            r"\['seed'\].*`seed` was removed in r3con 0\.2\.0.*params: \{seed: 42\}",
        ),
        ({"relevence_rounds": 3}, r"does not read: \['relevence_rounds'\]"),
        ({"overrides": {"model": "x"}}, r"does not read: \['overrides'\]"),
        ({"name": "other"}, r"does not read: \['name'\]"),
    ],
    ids=["seed", "misspelt", "overrides", "name"],
)
def test_a_config_field_r3con_does_not_read_is_refused(configs, field, message):
    configs(old={**FULL, **field})
    with pytest.raises(ValueError, match=message):
        load_config("old")


def test_a_run_config_refuses_a_field_it_does_not_have():
    with pytest.raises(ValidationError, match="relevence_rounds"):
        RunConfig(model="openai/m", prompts=PROMPTS, relevence_rounds=3)


def test_a_config_pinning_a_prompt_that_does_not_exist_is_refused_on_load(
    configs, tmp_path
):
    configs(exp={**FULL, "prompts": {**PROMPTS, CODING: "v9"}})
    with pytest.raises(FileNotFoundError, match=f"'{CODING}' version 'v9'") as missing:
        load_config("exp")
    assert str(tmp_path / "prompts" / CODING / "v9.yaml") in str(missing.value)
    assert str(settings.PROMPTS_DIR / CODING / "v9.yaml") in str(missing.value)


def test_an_unknown_config_is_refused_by_name():
    with pytest.raises(KeyError, match="does-not-exist-xyz"):
        load_config("does-not-exist-xyz")


def test_a_config_in_the_working_directory_is_available(configs):
    configs(one={"model": "m", "prompts": PROMPTS})
    assert {"default", "one"} <= set(available_configs())
