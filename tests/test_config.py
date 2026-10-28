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
    assert config.prompts == PROMPTS
    assert "seed" not in config.model_dump()
    assert config.label() == (
        f"default[model=gpt-6-luna,rounds=2,reasoning=direct,{V1_PROMPTS}]"
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


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"prompts": PROMPTS}, "must define a `model`"),
        ({**FULL, "prompts": {CODING: "v1"}}, "missing prompt versions"),
    ],
)
def test_a_config_missing_what_a_run_needs_is_refused(configs, fields, message):
    configs(bad=fields)
    with pytest.raises(ValueError, match=message):
        load_config("bad")


THE_STAGES = (
    "The stages are ['relevance/first-round', 'relevance/later-rounds', "
    "'structuring/schema', 'structuring/parsing', 'reasoning/direct', "
    "'reasoning/coding']."
)


@pytest.mark.parametrize(
    ("prompts", "old"),
    [
        ({**PROMPTS, "relevance": "v1"}, "relevance"),
        ({"relevance": "v1"}, "relevance"),
        (OLD_DEFAULT_PROMPTS, "reasoning"),
    ],
    ids=["beside-the-stages", "alone", "the-old-reasoning-stage"],
)
def test_a_config_pinning_a_stage_r3con_does_not_have_is_refused_with_the_stages(
    configs, prompts, old
):
    unknown = f"pins a prompt for stage(s) r3con does not have: [{old!r}]. {THE_STAGES}"
    configs(old={**FULL, "prompts": prompts})
    with pytest.raises(ValueError) as refused:
        load_config("old")
    assert type(refused.value) is ValueError
    assert str(refused.value) == f"config 'old' {unknown}"
    with pytest.raises(ValidationError, match=re.escape(f"RunConfig {unknown}")):
        RunConfig(model="openai/m", prompts=prompts)


@pytest.mark.parametrize(
    ("name", "overrides", "message"),
    [
        ("agent", {}, "config 'agent': `reasoning` must be one of"),
        ("default", {"reasoning": "Direct"}, "override: `reasoning` must be one of"),
    ],
    ids=["file", "override"],
)
def test_a_reasoning_path_r3con_does_not_have_is_refused(
    configs, name, overrides, message
):
    configs(agent={**FULL, "reasoning": "agent"})
    with pytest.raises(ValueError) as refused:
        load_config(name, **overrides)
    assert type(refused.value) is ValueError
    assert str(refused.value).startswith(f"{message} ['direct', 'coding'], got '")


def test_a_run_config_refuses_a_reasoning_path_r3con_does_not_have():
    with pytest.raises(ValidationError, match="Input should be 'direct' or 'coding'"):
        RunConfig(model="openai/m", prompts=PROMPTS, reasoning="agent")


@pytest.mark.parametrize(("name", "value"), [("relevence_rounds", 3), ("seed", 7)])
def test_a_misspelt_override_is_refused(name, value):
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{name}'"):
        load_config("default", **{name: value})


@pytest.mark.parametrize(
    ("name", "overrides"),
    [("default", {"params": "temperature=0.7"}), ("bad", {})],
    ids=["override", "file"],
)
def test_params_that_are_not_a_mapping_are_a_type_error(configs, name, overrides):
    configs(bad={**FULL, "params": "temperature=0.7"})
    with pytest.raises(TypeError, match="`params` must be a mapping.*got str"):
        load_config(name, **overrides)


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
