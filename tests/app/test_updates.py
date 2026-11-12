"""The Mac app's updates (r3con.app.updates), with Toga and the Objective-C runtime
faked."""

import base64
import logging
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
import toga
from Crypto.Signature import eddsa

from r3con.app import updates
from tests.fakes import FakeCommand, FakeCommands, FakeObjC

FRAMEWORK = "{frameworks}/Sparkle.framework"
PYPROJECT = Path(__file__).parents[2] / "pyproject.toml"


@pytest.fixture(autouse=True)
def commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(toga, "Command", FakeCommand)


def test_with_sparkle_the_app_menu_checks_for_updates_right_after_about(
    objc: FakeObjC,
):
    (objc.frameworks / updates.SPARKLE).mkdir(parents=True)
    app = SimpleNamespace(commands=FakeCommands())
    updates.start(app)  # pyright: ignore[reportArgumentType]
    (controller,) = objc.controllers
    about, command = app.commands.values()
    assert command.text == "Check for Updates…"
    placement = (command.group, command.section, command.order)
    assert placement == (about.group, about.section, about.order + 1)
    assert command.action
    command.action(command)
    assert controller.checks == 1


@pytest.mark.parametrize(
    ("sparkle", "logged"),
    [
        (None, "No Sparkle.framework in the app"),
        ({"loads": False}, f"{FRAMEWORK} didn't load"),
        (
            {"has_controller": False},
            f"{FRAMEWORK} has no SPUStandardUpdaterController",
        ),
        ({"initializes": False}, "Sparkle's updater didn't start"),
        ({"starts": False}, "Sparkle's updater didn't start"),
    ],
    ids=["no framework", "not loaded", "no controller", "nil init", "not started"],
)
def test_an_app_whose_sparkle_does_not_start_runs_without_updates_and_logs_why(
    objc: FakeObjC,
    caplog: pytest.LogCaptureFixture,
    sparkle: dict[str, bool] | None,
    logged: str,
):
    if sparkle is not None:
        (objc.frameworks / updates.SPARKLE).mkdir(parents=True)
        vars(objc).update(sparkle)
    app = SimpleNamespace(commands=FakeCommands())
    with caplog.at_level(logging.INFO, logger="r3con"):
        updates.start(app)  # pyright: ignore[reportArgumentType]
    assert list(app.commands) == [FakeCommand.ABOUT]
    message = f"{logged.format(frameworks=objc.frameworks)}: {updates.NO_UPDATES}."
    records = [(record.levelname, record.getMessage()) for record in caplog.records]
    assert records == [("WARNING", message)]


# The Mac app's own, which a run that isn't signed swaps for a throwaway one.
def test_the_mac_app_trusts_an_ed25519_public_key_for_its_updates():
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    info = config["tool"]["briefcase"]["app"]["r3con"]["macOS"]["info"]
    key = base64.b64decode(info["SUPublicEDKey"], validate=True)
    assert len(key) == 32
    eddsa.import_public_key(key)
