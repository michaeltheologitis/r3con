"""The Mac app's window (r3con.app.window), with Toga faked at the screen."""

import asyncio
import os
import runpy
import threading
import time
from collections.abc import Callable, Iterator
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx2
import pytest
import toga
from toga.constants import WindowState

from r3con.app import updates, window
from r3con.app.server import APP_KEY_SERVICE, MAC_APP_KEY_SERVICE, listen, serve
from tests.fakes import (
    GATE_TIMEOUT_SECONDS,
    FakeCommand,
    FakeCommands,
    FakeKeyring,
    FakeLLM,
    FakeNSSize,
    FakeObjC,
)

pytestmark = [pytest.mark.enable_socket, pytest.mark.allow_hosts(["127.0.0.1"])]
TIMEOUT_SECONDS = 10
TOGA_WINDOW_SIZE = (640, 480)
MAC_APP_KEY = "sk-saved-by-the-mac-app-1234"
R3CON_APP_KEY = "sk-saved-by-r3con-app-5678"
SAVED_NAME = "which-site-logged-the-most-incidents.html"
CHOSEN_NAME = "chosen.html"
SAVED_TAB = b"<!doctype html><title>a saved tab</title>"
# AppKit's NSTerminateNow: what Toga answers a Dock Quit or a logout with.
NS_TERMINATE_NOW = 1
# AppKit's values, and the style mask and least size of the window Toga makes.
TOGA_STYLE_MASK = 0b1111
TOGA_MIN_SIZE = FakeNSSize(100, 100)
NS_FULL_SIZE_CONTENT_VIEW = 1 << 15
NS_WINDOW_TITLE_VISIBLE = 0
NS_WINDOW_TITLE_HIDDEN = 1
NS_LEFT_MOUSE_DOWN = 1
NS_LEFT_MOUSE_UP = 2
NS_LEFT_MOUSE_DRAGGED = 6
# The page's least width: narrower, it stacks its chat under its panel.
PAGE_WIDE_LAYOUT_PX = 761
TOGGLE_FULL_SCREEN = "document.documentElement.classList.toggle"
HELD_RUN = {
    "question": "Which memo?",
    "model": "openai/held-model",
    "documents": ["A memo."],
}
# Past the stop's bound, and well short of the held call's answer, at GATE_TIMEOUT_SECONDS.
QUIT_BOUND_SECONDS = window.QUIT_SECONDS + 1
WINDOW_THREAD = "window"


class FakeWebView:
    """Toga's web view: its WKWebView keeps each script message handler by its name,
    in ``handlers``, unless it ``takes_handlers`` not, and it keeps the scripts it is
    asked to run, in ``scripts``."""

    takes_handlers = True

    def __init__(self, *, url: str | None = None) -> None:
        self.url = url
        self.on_webview_load: Callable[..., None] | None = None
        self.handlers: dict[str, Any] = {}
        self.scripts: list[str] = []
        contents = SimpleNamespace(addScriptMessageHandler=self._add_handler)
        configuration = SimpleNamespace(userContentController=contents)
        self._impl = SimpleNamespace(
            native=SimpleNamespace(configuration=configuration)
        )

    def _add_handler(self, handler: Any, *, name: str) -> None:
        if not self.takes_handlers:
            raise AttributeError(
                "WKUserContentController has no addScriptMessageHandler"
            )
        self.handlers[name] = handler

    def post(self, body: object, native: "FakeNSWindow") -> None:
        """Post ``body`` to the window's handler, as the page in ``native`` would."""
        message = SimpleNamespace(body=body, webView=SimpleNamespace(window=native))
        handler = self.handlers["window"]
        handler.userContentController_didReceiveScriptMessage_(None, message)

    def evaluate_javascript(self, javascript: str) -> None:
        self.scripts.append(javascript)


class FakeNSWindow:
    """AppKit's window under Toga's, its title bar shown: it keeps each drag, by its
    event, and each zoom and minimize it is asked for, in ``moves``."""

    def __init__(self) -> None:
        self.styleMask = TOGA_STYLE_MASK
        self.minSize = TOGA_MIN_SIZE
        self.titlebarAppearsTransparent = False
        self.titleVisibility = NS_WINDOW_TITLE_VISIBLE
        self.moves: list[tuple[str, object]] = []

    def performWindowDragWithEvent(self, event: object) -> None:
        self.moves.append(("drag", event))

    def performZoom(self, sender: object) -> None:
        self.moves.append(("zoom", sender))

    def performMiniaturize(self, sender: object) -> None:
        self.moves.append(("minimize", sender))


def fake_save_file_dialog(
    title: str, suggested_filename: str
) -> tuple[str, str, threading.Thread]:
    return (title, suggested_filename, threading.current_thread())


class FakeWindow:
    """Toga's window. Its Save dialog answers with ``chooses``, as the user would,
    unless it never ``answers``."""

    chooses: Path | None = None
    answers = True

    def __init__(self) -> None:
        self.size = TOGA_WINDOW_SIZE
        self._impl = SimpleNamespace(native=FakeNSWindow())
        self.state = WindowState.NORMAL
        self.on_resize: Callable[..., None] | None = None
        self.hidden = False
        self.dialogs: list[tuple[str, str, threading.Thread]] = []
        self.opened = threading.Event()

    def hide(self) -> None:
        self.hidden = True

    async def dialog(self, dialog: tuple[str, str, threading.Thread]) -> Path | None:
        self.dialogs.append(dialog)
        self.opened.set()
        if not self.answers:
            await asyncio.Event().wait()
        return self.chooses


class FakeBackend:
    """Toga 0.5.7's Mac backend for the app, as AppKit asks it on a Dock Quit or a
    logout: it lets the app end at once."""

    def cocoa_applicationShouldTerminate(self, sender: object) -> int:
        return NS_TERMINATE_NOW


class FakeApp:
    """Toga's app, the running one as ``app``. Its main loop opens a window on
    ``startup``'s web view, reads the page there as a user would, saves the key in
    ``saving``, if any, as its settings would, and with ``saving_tab``, saves a tab
    as its page would, and with ``running``, starts a run on that model and waits for
    its call. Then it quits as the app menu's Quit does, which ``on_exit`` allows or
    not, or with ``quits_mid_save``, once the tab's Save dialog is open; or with
    ``quits_from_dock``, as the Dock's Quit or a logout does, noting what that left
    done, ``ended``, by the time it returns, if it does. Its event loop runs while the
    page is read, as Toga's does."""

    app: "FakeApp"
    saving: dict[str, str] | None = None
    saving_tab = False
    running: FakeLLM | None = None
    quits_mid_save = False
    quits_from_dock = False
    exits: list[int]

    def __init__(
        self, *, startup: Callable[..., FakeWebView], on_exit: Callable[..., bool]
    ) -> None:
        self.startup = startup
        self.on_exit = on_exit
        self.main_window = FakeWindow()
        self.windows = [self.main_window]
        self.commands = FakeCommands()
        self._impl = FakeBackend()
        type(self).app = self

    def main_loop(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.main_thread = threading.current_thread()
        self.webview = self.startup(self)
        assert self.webview.url is not None, "the window loaded no page"
        self.url = self.webview.url
        reading = asyncio.create_task(asyncio.to_thread(self.read_page))
        if self.quits_mid_save:
            await asyncio.to_thread(self.main_window.opened.wait, TIMEOUT_SECONDS)
        else:
            await reading
        self.quit_at = time.monotonic()
        if self.quits_from_dock:
            self._impl.cocoa_applicationShouldTerminate(self)
            hidden = all(shown.hidden for shown in self.windows)
            self.ended = (hidden, refuses(self.url), [*self.exits])
        else:
            self.quits = self.on_exit(self)
        await reading

    def read_page(self) -> None:
        base_url, fragment = self.url.split("/#")
        token = parse_qs(fragment)["token"][0]
        auth = {"Authorization": f"Bearer {token}"}
        with httpx2.Client(
            base_url=base_url, timeout=TIMEOUT_SECONDS, trust_env=False
        ) as http:
            self.page = http.get("/")
            self.keys = http.get("/api/keys", headers=auth)
            if self.saving is not None:
                http.post("/api/keys", json=self.saving, headers=auth)
            if self.saving_tab:
                self.saved = http.post(
                    "/api/saves",
                    params={"name": SAVED_NAME},
                    headers=auth,
                    content=SAVED_TAB,
                )
            if self.running is not None:
                self.started = http.post("/api/runs", json=HELD_RUN, headers=auth)
                self.running.called.wait(TIMEOUT_SECONDS)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch, objc: FakeObjC) -> list[FakeApp]:
    """The apps the window opens, in a Mac app without Sparkle."""
    apps: list[FakeApp] = []

    class OpenedApp(FakeApp):
        def __init__(self, **handlers: Any) -> None:
            super().__init__(**handlers)
            apps.append(self)

    monkeypatch.setattr(toga, "App", OpenedApp)
    monkeypatch.setattr(toga, "WebView", FakeWebView)
    monkeypatch.setattr(toga, "SaveFileDialog", fake_save_file_dialog)
    monkeypatch.setattr(toga, "Command", FakeCommand)
    return apps


@pytest.fixture(autouse=True)
def exits(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """The codes the window ends the process with, which here it never does: a window
    left running on a thread is waited for, so that it never ends pytest's."""
    codes: list[int] = []
    monkeypatch.setattr(os, "_exit", codes.append)
    monkeypatch.setattr(FakeApp, "exits", codes, raising=False)
    yield codes
    for running in threading.enumerate():
        if running.name == WINDOW_THREAD:
            running.join()


@pytest.fixture(autouse=True)
def own_port(monkeypatch: pytest.MonkeyPatch) -> int:
    """The window's port, one free when the test starts, never the Mac app's own."""
    with listen(0) as probe:
        port = probe.getsockname()[1]
    monkeypatch.setattr(window, "PORT", port)
    return port


def refuses(url: str) -> bool:
    try:
        httpx2.get(url, timeout=TIMEOUT_SECONDS, trust_env=False)
    except httpx2.ConnectError:
        return True
    return False


def launch_of(app: FakeApp) -> dict[str, list[str]]:
    """What the window opens its page with, in its URL's fragment."""
    return parse_qs(urlsplit(app.url).fragment)


def test_the_window_opens_the_page_this_launch_serves_with_its_token(
    opened: list[FakeApp],
):
    window.main()
    (app,) = opened
    assert app.page.status_code == 200
    assert "<title>" in app.page.text
    assert app.keys.status_code == 200


def test_every_launch_serves_on_the_windows_port_so_the_page_keeps_its_settings(
    opened: list[FakeApp], own_port: int
):
    window.main()
    window.main()
    assert [urlsplit(app.url).port for app in opened] == [own_port, own_port]
    assert [app.keys.status_code for app in opened] == [200, 200]


def test_a_launch_whose_port_another_program_has_serves_on_a_free_one(
    opened: list[FakeApp], own_port: int
):
    with listen(own_port):
        window.main()
    (app,) = opened
    assert urlsplit(app.url).port != own_port
    assert (app.page.status_code, app.keys.status_code) == (200, 200)


def test_the_mac_app_keeps_its_keys_under_its_bundle_id_never_r3con_apps(
    opened: list[FakeApp],
    credential_store: FakeKeyring,
    monkeypatch: pytest.MonkeyPatch,
):
    assert MAC_APP_KEY_SERVICE == "io.github.michaeltheologitis.r3con"
    credential_store.passwords[MAC_APP_KEY_SERVICE, "openai"] = MAC_APP_KEY
    credential_store.passwords[APP_KEY_SERVICE, "gemini"] = R3CON_APP_KEY
    saving = {"provider": "anthropic", "key": MAC_APP_KEY}
    monkeypatch.setattr(FakeApp, "saving", saving)
    window.main()
    (app,) = opened
    loaded = app.keys.json()
    assert (loaded["openai"]["hint"], loaded["gemini"]) == (MAC_APP_KEY[-4:], None)
    assert credential_store.passwords == {
        (MAC_APP_KEY_SERVICE, "openai"): MAC_APP_KEY,
        (MAC_APP_KEY_SERVICE, "anthropic"): MAC_APP_KEY,
        (APP_KEY_SERVICE, "gemini"): R3CON_APP_KEY,
    }


def test_a_mac_app_with_sparkle_checks_for_updates_from_its_app_menu(
    opened: list[FakeApp], objc: FakeObjC
):
    (objc.frameworks / updates.SPARKLE).mkdir(parents=True)
    window.main()
    (app,) = opened
    assert updates.CHECK_FOR_UPDATES in app.commands


def test_the_window_opens_whatever_stops_the_apps_updates(
    opened: list[FakeApp], objc: FakeObjC, capsys: pytest.CaptureFixture[str]
):
    # A runtime without NSBundle, which updates.start does not expect.
    del objc.classes["NSBundle"]
    window.main()
    (app,) = opened
    assert app.page.status_code == 200
    # The window's log, on stderr, with the traceback.
    logged = capsys.readouterr().err
    assert f"Sparkle's updater failed: {updates.NO_UPDATES}." in logged
    assert "NameError: ObjC Class 'NSBundle' couldn't be found." in logged


def test_the_window_opens_at_its_size(opened: list[FakeApp]):
    window.main()
    (app,) = opened
    assert [shown.size for shown in app.windows] == [window.WINDOW_SIZE]


def test_the_window_hides_its_title_bar_and_opens_its_page_saying_so(
    opened: list[FakeApp],
):
    window.main()
    (app,) = opened
    native = app.main_window._impl.native
    assert native.titlebarAppearsTransparent
    assert native.titleVisibility == NS_WINDOW_TITLE_HIDDEN
    assert native.styleMask == TOGA_STYLE_MASK | NS_FULL_SIZE_CONTENT_VIEW
    assert native.minSize.width >= PAGE_WIDE_LAYOUT_PX
    assert native.minSize.height == TOGA_MIN_SIZE.height
    assert launch_of(app)["titlebar"] == ["hidden"]


def test_a_window_that_cant_hide_its_title_bar_keeps_it_and_the_page_its_top_row(
    opened: list[FakeApp],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    monkeypatch.setattr(FakeWebView, "takes_handlers", False)
    window.main()
    (app,) = opened
    native = app.main_window._impl.native
    shown = (native.titlebarAppearsTransparent, native.styleMask, native.minSize)
    assert shown == (False, TOGA_STYLE_MASK, TOGA_MIN_SIZE)
    assert launch_of(app).keys() == {"token", "save"}
    assert app.page.status_code == 200
    assert (
        "The window keeps its title bar: hiding it failed." in capsys.readouterr().err
    )


@pytest.fixture
def titled(objc: FakeObjC) -> tuple[FakeWindow, FakeWebView]:
    """A window whose title bar its web view's page has taken the place of."""
    shown, webview = FakeWindow(), FakeWebView()
    window.hide_title_bar(shown, webview)  # pyright: ignore[reportArgumentType]
    return shown, webview


@pytest.mark.parametrize(
    ("body", "event", "moves"),
    [
        ("drag", NS_LEFT_MOUSE_DOWN, ["drag"]),
        ("drag", NS_LEFT_MOUSE_DRAGGED, ["drag"]),
        ("drag", NS_LEFT_MOUSE_UP, []),
        ("drag", None, []),
        ("zoom", None, ["zoom"]),
        ("close", NS_LEFT_MOUSE_DOWN, []),
        ("", NS_LEFT_MOUSE_DOWN, []),
        ({"drag": True}, NS_LEFT_MOUSE_DOWN, []),
    ],
    ids=["press", "dragged", "release", "no event", "zoom", "other", "empty", "object"],
)
def test_the_page_drags_the_window_from_a_left_press_and_zooms_it_and_nothing_else(
    titled: tuple[FakeWindow, FakeWebView],
    objc: FakeObjC,
    body: object,
    event: int | None,
    moves: list[str],
):
    shown, webview = titled
    current = None if event is None else SimpleNamespace(type=event)
    objc.application.currentEvent = current
    native = shown._impl.native
    webview.post(body, native)
    senders = {"drag": current, "zoom": None}
    assert native.moves == [(move, senders[move]) for move in moves]


@pytest.mark.parametrize(
    ("setting", "moves"),
    [
        (None, ["zoom"]),
        ("Maximize", ["zoom"]),
        ("Minimize", ["minimize"]),
        ("None", []),
    ],
    ids=["unset", "zoom", "minimize", "do nothing"],
)
def test_a_double_click_on_the_row_does_what_the_users_title_bar_setting_says(
    titled: tuple[FakeWindow, FakeWebView],
    objc: FakeObjC,
    setting: str | None,
    moves: list[str],
):
    if setting is not None:
        objc.defaults["AppleActionOnDoubleClick"] = setting
    shown, webview = titled
    native = shown._impl.native
    webview.post("zoom", native)
    assert native.moves == [(move, None) for move in moves]


def test_a_message_the_window_cant_act_on_is_logged_never_raised_into_appkit(
    titled: tuple[FakeWindow, FakeWebView],
    objc: FakeObjC,
    caplog: pytest.LogCaptureFixture,
):
    del objc.classes["NSApplication"]
    shown, webview = titled
    webview.post("drag", shown._impl.native)
    assert [record.getMessage() for record in caplog.records] == [
        "The window couldn't do the page's 'drag'."
    ]


def test_the_page_knows_while_the_window_is_in_full_screen_and_after_a_reload(
    titled: tuple[FakeWindow, FakeWebView],
):
    shown, webview = titled
    assert shown.on_resize is not None and webview.on_webview_load is not None
    for state in (WindowState.FULLSCREEN, WindowState.NORMAL):
        shown.state = state
        shown.on_resize(shown)
    shown.state = WindowState.FULLSCREEN
    webview.on_webview_load(webview)
    assert webview.scripts == [
        f'{TOGGLE_FULL_SCREEN}("full-screen", true)',
        f'{TOGGLE_FULL_SCREEN}("full-screen", false)',
        f'{TOGGLE_FULL_SCREEN}("full-screen", true)',
    ]


def test_hiding_two_title_bars_defines_the_handlers_objective_c_class_once(
    titled: tuple[FakeWindow, FakeWebView],
):
    _, first = titled
    second = FakeWebView()
    window.hide_title_bar(FakeWindow(), second)  # pyright: ignore[reportArgumentType]
    assert type(first.handlers["window"]) is type(second.handlers["window"])


def test_a_tab_saved_in_the_window_is_written_where_its_save_dialog_chose(
    opened: list[FakeApp], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    chosen = tmp_path / CHOSEN_NAME
    monkeypatch.setattr(FakeApp, "saving_tab", True)
    monkeypatch.setattr(FakeWindow, "chooses", chosen)
    window.main()
    (app,) = opened
    assert launch_of(app)["save"] == ["dialog"]
    assert app.saved.status_code == 204
    shown = [(window.SAVE_DIALOG_TITLE, SAVED_NAME, app.main_thread)]
    assert app.main_window.dialogs == shown
    assert chosen.read_bytes() == SAVED_TAB


def test_a_save_cancelled_in_the_windows_dialog_is_answered_quietly(
    opened: list[FakeApp], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(FakeApp, "saving_tab", True)
    window.main()
    (app,) = opened
    assert (app.saved.status_code, len(app.main_window.dialogs)) == (204, 1)


def test_quitting_with_a_save_dialog_open_cancels_its_save_and_stops_the_server(
    opened: list[FakeApp], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(FakeApp, "saving_tab", True)
    monkeypatch.setattr(FakeApp, "quits_mid_save", True)
    monkeypatch.setattr(FakeWindow, "answers", False)
    running = threading.Thread(target=window.main, name=WINDOW_THREAD, daemon=True)
    running.start()
    running.join(TIMEOUT_SECONDS)
    (app,) = opened
    assert not running.is_alive()
    assert (app.quits, app.saved.status_code) == (True, 204)
    assert refuses(app.url)


def test_quitting_hides_the_window_then_stops_the_server_and_ends_the_process(
    opened: list[FakeApp], exits: list[int]
):
    window.main()
    (app,) = opened
    assert app.quits
    assert all(shown.hidden for shown in app.windows)
    assert refuses(app.url)
    assert exits == [0]


def test_a_dock_quit_or_a_logout_hides_the_window_and_stops_the_server_before_it_ends(
    opened: list[FakeApp], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(FakeApp, "quits_from_dock", True)
    window.main()
    (app,) = opened
    hidden, stopped, exited = app.ended
    assert (hidden, stopped, exited) == (True, True, [0])


def test_quitting_with_a_model_call_in_flight_ends_the_process_without_waiting_for_it(
    opened: list[FakeApp],
    exits: list[int],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    held = FakeLLM().answers(relevance="A memo.")
    held.gate.clear()
    monkeypatch.setattr(
        window, "serve", partial(serve, logs_dir=tmp_path, completion=held)
    )
    monkeypatch.setattr(FakeApp, "running", held)
    try:
        window.main()
        ended = time.monotonic()
    finally:
        held.gate.set()
    (app,) = opened
    assert (app.started.status_code, held.called.is_set()) == (202, True)
    assert ended - app.quit_at < QUIT_BOUND_SECONDS < GATE_TIMEOUT_SECONDS
    assert exits == [0]
    for worker in threading.enumerate():
        if worker.name.startswith("r3con-app-"):
            worker.join(TIMEOUT_SECONDS)


def test_python_m_r3con_opens_the_window_in_the_mac_app(
    opened: list[FakeApp], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # The distribution Briefcase writes into the Mac app, beside r3con's code.
    dist_info = tmp_path / "app" / "r3con-0.6.10.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: r3con\nVersion: 0.6.10\n"
    )
    monkeypatch.syspath_prepend(tmp_path / "app")
    runpy.run_module("r3con", run_name="__main__")
    (app,) = opened
    assert app.page.status_code == 200


def test_the_server_stops_when_toga_cannot_start(monkeypatch: pytest.MonkeyPatch):
    def no_app(**handlers: Any) -> FakeApp:
        raise RuntimeError("Toga has no backend for this platform")

    monkeypatch.setattr(toga, "App", no_app)
    before = set(threading.enumerate())
    with pytest.raises(RuntimeError):
        window.main()
    assert [thread for thread in threading.enumerate() if thread not in before] == []
