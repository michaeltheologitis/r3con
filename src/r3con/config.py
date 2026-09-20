"""The run config: every knob that shapes the OUTPUT, in one place.

A :class:`RunConfig` is the **identity of a run** — the model, seed, number of relevance
rounds, the prompt version of each stage, and any generation params. It is loaded from a
bundled ``configs/<name>.yaml`` (with optional overrides) and flows through the whole
pipeline as one object, so nothing output-affecting is threaded ad-hoc. Runtime-only
knobs (parallelism, paths, retry caps, timeouts) stay in :mod:`r3con.settings` — they
don't change a correct answer.

Two runs are the same run iff their :meth:`RunConfig.label` matches, so **every**
output-shaping axis must appear in :meth:`RunConfig._identity_parts` — that function is
the one place to extend the identity.

``params`` is a plain dict handed **straight to** ``litellm.completion`` — ``temperature``,
``top_p``, ``extra_body``, anything litellm accepts. There is no r3con-specific
vocabulary for generation settings: if litellm takes it, put it there. It shapes the
output, so it is part of the run's identity.

Configs ship inside the installed package (``src/r3con/configs/``); they are read-only
package data, resolved from the package directory.

Config file shape (``configs/<name>.yaml``)::

    model: openai/gpt-5.6-luna
    seed: 42
    relevance_rounds: 2
    prompts:
      relevance: v1
      structuring/schema: v1
      structuring/parsing: v1
      reasoning: v1
    params:                      # optional; passed straight to litellm.completion
      temperature: 0.7
"""

from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import yaml
from pydantic import BaseModel, Field

from r3con.settings import settings

# The pipeline stages that carry a versioned prompt — a config must pin all of them.
PROMPT_STAGES: tuple[str, ...] = (
    "relevance",
    "structuring/schema",
    "structuring/parsing",
    "reasoning",
)

DEFAULT_CONFIG = "default"

# Compact per-stage abbreviations for the prompts tag in RunConfig.label().
_PROMPT_STAGE_ABBR: dict[str, str] = {
    "relevance": "rel",
    "structuring/schema": "schema",
    "structuring/parsing": "parse",
    "reasoning": "reason",
}


def normalize_model_name(model: str | None) -> str | None:
    """Reduce a LiteLLM model string to the bare model name — drop the provider/route
    prefix, which is *transport*, not identity::

        openai/gpt-5.6-luna             -> gpt-5.6-luna
        hosted_vllm/Qwen/Qwen3.5-35B-A3B -> Qwen3.5-35B-A3B
        gpt-5.6-luna                    -> gpt-5.6-luna   (no prefix → unchanged)
        None                            -> None

    Used wherever the model is **persisted or shown** (the run label, the manifest's
    recorded config block, the run header, transcript headers). The *live* call site
    keeps the full provider-prefixed string — LiteLLM needs it to route."""
    if not model:
        return model
    return model.split("/")[-1]


class RunConfig(BaseModel):
    """Everything that shapes the output of one run (the experiment identity)."""

    name: str = DEFAULT_CONFIG  # the config it was loaded from (the board's run label base)
    model: str
    seed: int = 42
    relevance_rounds: int = 2  # keep in sync with configs/default.yaml (the shipped default)
    prompts: dict[str, str]  # {stage: version} for every stage in PROMPT_STAGES
    params: dict[str, Any] = Field(default_factory=dict)  # extra kwargs for litellm.completion
    overrides: dict[str, Any] = Field(default_factory=dict)  # CLI overrides applied (recorded for the board)

    def _identity_parts(self) -> list[str]:
        """The output-shaping components that make up the run identity, in label order, each a
        ``key=value`` string. **This is the one place to extend the identity** — add a line here
        and the new axis flows into the board's grouping key *and* the eval's resume key. Values are
        the *resolved* config (not how they were set), so seed 42 reads the same whether it came
        from the config file or ``--seed 42``. The model is reduced to its bare name (the
        provider/route prefix is transport — see :func:`normalize_model_name`)."""
        prompts = ",".join(
            f"{_PROMPT_STAGE_ABBR.get(s, s)}={self.prompts[s]}"
            for s in PROMPT_STAGES
            if s in self.prompts
        )
        parts = [
            f"model={normalize_model_name(self.model)}",
            f"seed={self.seed}",
            f"rounds={self.relevance_rounds}",
        ]
        # Generation params are part of the identity only when set, so a provider-default
        # run carries no `params=` noise in its label.
        if self.params:
            parts.append("params={" + ",".join(f"{k}={self.params[k]}" for k in sorted(self.params)) + "}")
        parts.append(f"prompts=({prompts})")
        return parts

    def label(self) -> str:
        """Readable run label — the board's grouping key and the eval's resume key. The config
        name plus the run-identity components (:meth:`_identity_parts`), e.g.
        ``default[model=gpt-5.6-luna,seed=42,rounds=2,prompts=(rel=v1,schema=v1,parse=v1,reason=v1)]``.
        Two runs are the same run iff their labels match, so **every** output-shaping axis must
        appear in ``_identity_parts``."""
        return f"{self.name}[" + ",".join(self._identity_parts()) + "]"


def config_search_path() -> list[Path]:
    """Where run configs are looked up, in order: a user overlay first, then the copy
    that ships inside the package.

    The overlay is ``R3CON_CONFIGS_DIR`` if set, else ``./configs`` under the current
    working directory — so you can add your own run config without forking the package."""
    overlay = os.environ.get("R3CON_CONFIGS_DIR")
    return [Path(overlay) if overlay else Path.cwd() / "configs", settings.CONFIGS_DIR]


def available_configs() -> list[str]:
    """Run-config names available to :func:`load_config`, from every root on the search
    path (an overlay config shadows a packaged one of the same name)."""
    return sorted({p.stem for root in config_search_path() if root.is_dir() for p in root.glob("*.yaml")})


def resolve_config_path(name: str) -> Path | None:
    """The file a config name actually resolves to, or ``None``. See
    :func:`r3con.prompts.resolve_prompt_path` for why the winning root is recorded."""
    return _find(f"{name}.yaml")


def _check_params(params: Any, where: str) -> dict[str, Any]:
    """``params`` must be a mapping of litellm keyword arguments. Caught here with a
    readable message rather than surfacing as a TypeError from deep inside a call."""
    if params is None:
        return {}
    if not isinstance(params, Mapping):
        raise ValueError(
            f"{where}: `params` must be a mapping of litellm keyword arguments "
            f"(temperature, top_p, extra_body, …) — got {type(params).__name__}."
        )
    return dict(params)


def _find(relative: str) -> Path | None:
    """First existing ``relative`` path across the config search path, else ``None``."""
    for root in config_search_path():
        candidate = root / relative
        if candidate.is_file():
            return candidate
    return None


def load_config(name: str = DEFAULT_CONFIG, **overrides: Any) -> RunConfig:
    """Load ``configs/<name>.yaml`` into a :class:`RunConfig`, then apply any non-``None``
    overrides (``model`` / ``seed`` /
    ``relevance_rounds`` / ``params``). Every override is recorded on the config and shows
    up in its label, because each of them shapes the output.

    Raises ``KeyError`` for an unknown config name, ``ValueError`` if the config omits a
    required field or a prompt version for any stage.
    """
    path = _find(f"{name}.yaml")
    if path is None:
        raise KeyError(f"Unknown config {name!r}. Known: {available_configs()}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "model" not in raw:
        raise ValueError(f"config {name!r} must define a `model`.")

    prompts = dict(raw.get("prompts") or {})
    missing = [s for s in PROMPT_STAGES if s not in prompts]
    if missing:
        raise ValueError(f"config {name!r} is missing prompt versions for stage(s): {missing}")

    cfg = RunConfig(
        name=name,
        model=raw["model"],
        seed=int(raw.get("seed", 42)),
        relevance_rounds=int(raw.get("relevance_rounds", 2)),
        prompts=prompts,
        params=_check_params(raw.get("params"), f"config {name!r}"),
    )

    # Overrides are recorded on the config as well as applied, so the manifest shows both
    # the resolved value and the fact that it was overridden.
    applied = {
        k: v for k, v in overrides.items()
        if v is not None and k in {"model", "seed", "relevance_rounds", "params"}
    }
    if "params" in applied:
        applied["params"] = _check_params(applied["params"], "override")
    if applied:
        cfg = cfg.model_copy(update={**applied, "overrides": applied})
    return cfg
