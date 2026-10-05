import pytest
import yaml
from pydantic import ValidationError

from r3con.config import PROMPT_STAGES, RunConfig, available_configs, load_config

PROMPTS = dict.fromkeys(PROMPT_STAGES, "v1")
FULL = {
    "model": "openai/gpt-6-luna",
    "relevance_rounds": 3,
    "prompts": PROMPTS,
}
DEFAULT_PROMPTS = "prompts=(rel=v1,schema=v1,parse=v1,reason=v1)"


@pytest.fixture
def configs(tmp_path):
    """Writes ``name -> fields`` into the working directory's config overlay."""
    root = tmp_path / "configs"
    root.mkdir()

    def write(**by_name: dict) -> None:
        for name, fields in by_name.items():
            (root / f"{name}.yaml").write_text(yaml.safe_dump(fields))

    return write


def test_the_default_config_loads_with_its_shipped_values():
    config = load_config("default")
    assert config.name == "default"
    assert config.model == "openai/gpt-6-luna"
    assert (config.relevance_rounds, config.params) == (2, {})
    assert config.prompts == PROMPTS
    assert "seed" not in config.model_dump()
    assert config.label() == f"default[model=gpt-6-luna,rounds=2,{DEFAULT_PROMPTS}]"


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
            {"model": None, "relevance_rounds": None, "params": None},
            {},
            "default[model=gpt-6-luna,rounds=2,",
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


def test_the_label_is_every_axis_of_the_resolved_identity(configs):
    configs(exp=FULL, exp2={**FULL, "prompts": {**PROMPTS, "reasoning": "v9"}})
    assert load_config("exp").label() == (
        f"exp[model=gpt-6-luna,rounds=3,{DEFAULT_PROMPTS}]"
    )
    assert load_config("exp", relevance_rounds=5).label() == (
        f"exp[model=gpt-6-luna,rounds=5,{DEFAULT_PROMPTS}]"
    )
    assert load_config("exp", params={"seed": 7}).label() == (
        f"exp[model=gpt-6-luna,rounds=3,params={{seed=7}},{DEFAULT_PROMPTS}]"
    )
    assert load_config("exp2").label() == (
        "exp2[model=gpt-6-luna,rounds=3,prompts=(rel=v1,schema=v1,parse=v1,reason=v9)]"
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
        ({**FULL, "prompts": {"relevance": "v1"}}, "missing prompt versions"),
    ],
)
def test_a_config_missing_what_a_run_needs_is_refused(configs, fields, message):
    configs(bad=fields)
    with pytest.raises(ValueError, match=message):
        load_config("bad")


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
    ],
)
def test_a_config_field_r3con_does_not_read_is_refused(configs, field, message):
    configs(old={**FULL, **field})
    with pytest.raises(ValueError, match=message):
        load_config("old")


def test_a_run_config_refuses_a_field_it_does_not_have():
    with pytest.raises(ValidationError, match="relevence_rounds"):
        RunConfig(model="openai/m", prompts=PROMPTS, relevence_rounds=3)


def test_an_unknown_config_is_refused_by_name():
    with pytest.raises(KeyError, match="does-not-exist-xyz"):
        load_config("does-not-exist-xyz")


def test_a_config_in_the_working_directory_is_available(configs):
    configs(one={"model": "m", "prompts": PROMPTS})
    assert {"default", "one"} <= set(available_configs())
