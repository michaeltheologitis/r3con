import os

# Before litellm is imported: otherwise it fetches its price map from GitHub.
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

from pathlib import Path

import pytest

KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for variable in KEY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    # Each test starts with no key registered for masking by an earlier one's runs.
    monkeypatch.setattr("r3con.masking._registered", set())
    monkeypatch.chdir(tmp_path)
