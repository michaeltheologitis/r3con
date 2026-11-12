"""The Mac app's updates (r3con.app.updates), with Toga and the Objective-C runtime
faked."""

import logging

import pytest
import toga

from r3con.app import updates
from tests.fakes import FakeCommand, FakeCommands, FakeObjC


class MenuApp:
    """Toga's app, as far as its menus go."""

    def __init__(self) -> None:
        self.commands = FakeCommands()


@pytest.fixture(autouse=True)
def commands(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(toga, "Command", FakeCommand)


@pytest.fixture
def logged(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    caplog.set_level(logging.INFO, logger="r3con")
    return caplog


def test_an_app_without_sparkle_runs_without_updates_and_logs_it(
    objc: FakeObjC, logged: pytest.LogCaptureFixture
):
    app = MenuApp()
    updates.start(app)  # pyright: ignore[reportArgumentType]
    assert (app.commands, objc.controllers) == ([], [])
    assert [(record.levelname, record.getMessage()) for record in logged.records] == [
        ("WARNING", "No Sparkle.framework in the app: it won't update itself.")
    ]


def test_with_sparkle_the_app_menu_checks_for_updates_right_after_about(
    objc: FakeObjC,
):
    (objc.frameworks / updates.SPARKLE).mkdir(parents=True)
    app = MenuApp()
    updates.start(app)  # pyright: ignore[reportArgumentType]
    (controller,) = objc.controllers
    assert controller.starts_updater
    (command,) = app.commands
    # Toga 0.5.7's Mac About item is the app menu's in section -1, order 0.
    placement = {"group": toga.Group.APP, "section": -1, "order": 1}
    assert (command.text, command.placement) == ("Check for Updates…", placement)
    command.action(command)
    assert controller.checks == 1


def test_a_sparkle_that_does_not_load_leaves_the_app_without_updates(
    objc: FakeObjC, logged: pytest.LogCaptureFixture
):
    sparkle = objc.frameworks / updates.SPARKLE
    sparkle.mkdir(parents=True)
    objc.loads = False
    app = MenuApp()
    updates.start(app)  # pyright: ignore[reportArgumentType]
    assert (app.commands, objc.controllers) == ([], [])
    assert [(record.levelname, record.getMessage()) for record in logged.records] == [
        ("ERROR", f"{sparkle} didn't load: the app won't update itself.")
    ]
