from pathlib import Path

import pytest

KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY")


# In place of the root's: the server's tests need litellm.completion and r3con's
# resend wait as they are.
@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for variable in KEY_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    # Each test starts with no key registered for masking by an earlier one's runs.
    monkeypatch.setattr("r3con.masking._registered", set())
    monkeypatch.chdir(tmp_path)
