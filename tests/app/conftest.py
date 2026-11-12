import sys
from pathlib import Path
from types import ModuleType

import pytest

from r3con.app.server import KEY_VARIABLES
from tests.fakes import FakeObjC


# In place of the root's: no fake model or recorded resend waits, as the server's
# tests need the real ones.
@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch, isolated_r3con: None) -> None:
    for variable in KEY_VARIABLES.values():
        monkeypatch.delenv(variable, raising=False)


@pytest.fixture
def objc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> FakeObjC:
    """The Mac app's Objective-C runtime, through rubicon.objc, Toga's bridge to it,
    which only the Mac app has: its bundle's Frameworks folder has no Sparkle."""
    runtime = FakeObjC(tmp_path / "Frameworks")
    bridge = ModuleType("rubicon.objc")
    vars(bridge).update(ObjCClass=runtime.ObjCClass)
    monkeypatch.setitem(sys.modules, "rubicon", ModuleType("rubicon"))
    monkeypatch.setitem(sys.modules, "rubicon.objc", bridge)
    return runtime
