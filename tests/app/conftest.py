import pytest

from r3con.app.server import KEY_VARIABLES


# In place of the root's: no fake model or recorded resend waits, as the server's
# tests need the real ones.
@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, isolated_r3con: None) -> None:
    for variable in KEY_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)
