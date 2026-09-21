"""Versioned, per-stage prompts — ``prompts/<stage>/<version>.yaml``.

Each stage reads exactly one prompt, and **which version it reads is part of the run's
identity**: the version is pinned per stage in the :class:`~r3con.config.RunConfig` and
passed explicitly to :func:`load_prompt`. There is deliberately no ambient default and no
environment variable that selects a version — a run that cannot say which prompt text
produced it is not reproducible.

Prompts ship **inside the installed package** (``src/r3con/prompts/``) as read-only package
data. A user overlay — ``R3CON_PROMPTS_DIR``, else ``./prompts`` — is searched first, so
trying a new wording is one file dropped beside your work plus a config pin, never a fork.

**A published version is immutable.** A run's manifest records the version *string*, so
editing ``v1.yaml`` in place retroactively changes what every past run claims to have run
under. To change a prompt, add ``v2.yaml`` and bump the config's pin; the old file stays on
disk untouched. The same holds while a run is in flight — :func:`load_prompt` re-reads the
YAML on every call.

Because the files travel in the wheel, ``v1`` names content only *relative to a release*,
which is why a run's manifest records ``r3con_version`` alongside the version strings.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import jinja2
import yaml

from r3con.settings import settings


def prompt_search_path() -> list[Path]:
    """Where prompts are looked up, in order: a user overlay first, then the copy that
    ships inside the package.

    The overlay is ``R3CON_PROMPTS_DIR`` if set, else ``./prompts`` under the current
    working directory. It exists so the versioning discipline survives installation: to
    try a new wording you drop ``prompts/<stage>/v2.yaml`` next to your work, pin it in a
    config, and run — without forking the package. A version that isn't in the overlay
    falls through to the packaged one, so you only carry the files you actually changed.
    """
    overlay = os.environ.get("R3CON_PROMPTS_DIR")
    return [Path(overlay) if overlay else Path.cwd() / "prompts", settings.PROMPTS_DIR]


def resolve_prompt_path(name: str, version: str) -> Path | None:
    """The file a stage's prompt actually resolves to, or ``None`` if there isn't one.

    Which root won matters: an overlay file and the packaged file of the same version are
    indistinguishable in a run label, so the run folder records the resolved path instead.
    """
    for root in prompt_search_path():
        path = root / name / f"{version}.yaml"
        if path.is_file():
            return path
    return None


def load_prompt(name: str, *, version: str, **context: Any) -> str:
    """Load and render a stage's prompt at the given ``version``.

    Prompts live at ``prompts/<name>/<version>.yaml``; ``name`` may be slash-separated
    for the nested stages (``structuring/schema`` → ``prompts/structuring/schema/<v>.yaml``). The ``version`` comes from the run's
    :class:`~r3con.config.RunConfig` (``config.prompts[name]``) — prompt versions are
    part of a run's identity, never an ambient default.

    Lookup order is :func:`prompt_search_path`: a user overlay, then the packaged copy.

    YAML shape::

        instructions: |-
          ...prose, may contain {{ jinja }} placeholders...
        examples:                 # optional
          - name: <example-name>
            text: |-
              Input: ...
              Output: ...
    """
    tried: list[Path] = []
    for root in prompt_search_path():
        path = root / name / f"{version}.yaml"
        tried.append(path)
        if path.is_file():
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            body: str = data["instructions"]
            if examples := data.get("examples"):
                body += "\n\nExamples:\n\n" + "\n\n".join(ex["text"] for ex in examples)
            return jinja2.Template(body).render(**context)
    locations = " or ".join(str(t) for t in tried)
    raise FileNotFoundError(f"No prompt for stage {name!r} version {version!r} (looked in {locations}).")
