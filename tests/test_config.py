import pytest
import yaml

from r3con.config import PROMPT_STAGES, available_configs, load_config

PROMPTS = dict.fromkeys(PROMPT_STAGES, "v1")
FULL = {
    "model": "openai/gpt-6-luna",
    "seed": 42,
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
    assert (config.seed, config.relevance_rounds, config.params) == (42, 2, {})
    assert config.prompts == PROMPTS
    assert (
        config.label()
        == f"default[model=gpt-6-luna,seed=42,rounds=2,{DEFAULT_PROMPTS}]"
    )


@pytest.mark.parametrize(
    ("overrides", "recorded", "label"),
    [
        (
            {"seed": 7, "model": "openai/gpt-x"},
            {"seed": 7, "model": "openai/gpt-x"},
            "default[model=gpt-x,seed=7,rounds=2,",
        ),
        (
            {"model": "hosted_vllm/Qwen/Qwen3.5-35B-A3B"},
            {"model": "hosted_vllm/Qwen/Qwen3.5-35B-A3B"},
            "default[model=Qwen3.5-35B-A3B,seed=42,rounds=2,",
        ),
        (
            {"seed": None, "model": None, "relevance_rounds": None},
            {},
            "default[model=gpt-6-luna,seed=42,rounds=2,",
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
        f"exp[model=gpt-6-luna,seed=42,rounds=3,{DEFAULT_PROMPTS}]"
    )
    assert load_config("exp", seed=7).label() == (
        f"exp[model=gpt-6-luna,seed=7,rounds=3,{DEFAULT_PROMPTS}]"
    )
    assert load_config("exp", relevance_rounds=5).label() == (
        f"exp[model=gpt-6-luna,seed=42,rounds=5,{DEFAULT_PROMPTS}]"
    )
    assert load_config("exp2").label() == (
        "exp2[model=gpt-6-luna,seed=42,rounds=3,"
        "prompts=(rel=v1,schema=v1,parse=v1,reason=v9)]"
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
        ({**FULL, "params": "temperature=0.7"}, "`params` must be a mapping"),
    ],
)
def test_a_config_missing_what_a_run_needs_is_refused(configs, fields, message):
    configs(bad=fields)
    with pytest.raises(ValueError, match=message):
        load_config("bad")


def test_an_unknown_config_is_refused_by_name():
    with pytest.raises(KeyError, match="does-not-exist-xyz"):
        load_config("does-not-exist-xyz")


def test_a_config_in_the_working_directory_is_available(configs):
    configs(one={"model": "m", "prompts": PROMPTS})
    assert {"default", "one"} <= set(available_configs())
