"""The Mac app's window (r3con.app.window), with Toga faked at the screen."""

import asyncio
import json
import os
import runpy
import sys
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
    FakeNSPoint,
    FakeObjC,
    fake_ns_make_rect,
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
# AppKit's values, and the style mask of the window Toga makes.
TOGA_STYLE_MASK = 0b1111
NS_FULL_SIZE_CONTENT_VIEW = 1 << 15
NS_WINDOW_TITLE_VISIBLE = 0
NS_WINDOW_TITLE_HIDDEN = 1
NS_VIEW_WIDTH_SIZABLE = 1 << 1
NS_VIEW_MAX_Y_MARGIN = 1 << 5
TOGGLE_FULL_SCREEN = "document.documentElement.classList.toggle"
# The title bar's height, and where the page says its top row's controls are.
TITLE_BAR_PX = 28
# AppKit's traffic lights, close to zoom, by their x in a title bar of TITLE_BAR_PX: 14
# points wide and 16 high, centred in it.
NS_WINDOW_BUTTONS = {0: 7, 1: 27, 2: 47}
NS_WINDOW_BUTTON_SIZE = (14, 16)
SETTINGS = (1000.0, 3.5, 1065.0, 24.5)
THEME = (1081.0, 3.5, 1186.0, 24.5)
HELD_RUN = {
    "question": "Which memo?",
    "model": "openai/held-model",
    "documents": ["A memo."],
}
# Past the stop's bound, and well short of the held call's answer, at GATE_TIMEOUT_SECONDS.
QUIT_BOUND_SECONDS = window.QUIT_SECONDS + 1
WINDOW_THREAD = "window"


class FakeWebView:
    """Toga's web view: it keeps the scripts it is asked to run, in ``scripts``, and
    answers the window's query of the page's controls with ``controls``, or, for a
    page that isn't R3Con's, None."""

    controls: tuple[tuple[float, ...], ...] | None = (SETTINGS, THEME)

    def __init__(self, *, url: str | None = None) -> None:
        self.url = url
        self.on_webview_load: Callable[..., Any] | None = None
        self.scripts: list[str] = []

    def evaluate_javascript(self, javascript: str) -> asyncio.Future[str | None]:
        self.scripts.append(javascript)
        answer: asyncio.Future[str | None] = asyncio.get_running_loop().create_future()
        query = javascript == window.CONTROLS_QUERY
        answer.set_result(json.dumps(self.controls) if query else None)
        return answer


class FakeContentView:
    """Toga's view that fills the window, flipped: each view it adds is in
    ``subviews``, and a window button's rect in it is the button's ``in_content``."""

    def __init__(self, native: "FakeNSWindow") -> None:
        self.native = native
        self.frame = SimpleNamespace(size=SimpleNamespace(width=TOGA_WINDOW_SIZE[0]))
        self.subviews: list[Any] = []

    def addSubview(self, view: Any) -> None:
        view.superview, view.window = self, self.native
        self.subviews.append(view)

    def convertRect(self, rect: object, *, fromView: Any) -> Any:
        return fromView.in_content


class FakeNSWindow:
    """AppKit's window under Toga's, its title bar shown, with its traffic lights,
    ``buttons``: it keeps each drag, by its event, and each zoom and minimize it is
    asked for, in ``moves``."""

    def __init__(self) -> None:
        self.styleMask = TOGA_STYLE_MASK
        self.titlebarAppearsTransparent = False
        self.titleVisibility = NS_WINDOW_TITLE_VISIBLE
        self.contentView = FakeContentView(self)
        top = (TITLE_BAR_PX - NS_WINDOW_BUTTON_SIZE[1]) / 2
        self.buttons = {
            kind: SimpleNamespace(
                bounds=None,
                in_content=fake_ns_make_rect(x, top, *NS_WINDOW_BUTTON_SIZE),
            )
            for kind, x in NS_WINDOW_BUTTONS.items()
        }
        self.moves: list[tuple[str, object]] = []

    def standardWindowButton(self, kind: int) -> Any:
        return self.buttons.get(kind)

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
        self.on_resize: Callable[..., Any] | None = None
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
    assert app.main_window.on_resize is app.webview.on_webview_load is not None
    assert launch_of(app)["titlebar"] == ["hidden"]


def test_a_window_that_cant_hide_its_title_bar_keeps_it_and_the_page_its_top_row(
    opened: list[FakeApp],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    # A rubicon without it, as if it had moved.
    monkeypatch.delattr(sys.modules["rubicon.objc"], "NSMakeRect")
    window.main()
    (app,) = opened
    native = app.main_window._impl.native
    assert (native.titlebarAppearsTransparent, native.styleMask) == (
        False,
        TOGA_STYLE_MASK,
    )
    assert app.main_window.on_resize is None
    assert launch_of(app).keys() == {"token", "save"}
    assert app.page.status_code == 200
    assert (
        "The window keeps its title bar: hiding it failed." in capsys.readouterr().err
    )


@pytest.fixture
def titled(objc: FakeObjC) -> tuple[FakeWindow, FakeWebView]:
    """A window whose title bar is hidden, its page loaded."""
    shown, webview = FakeWindow(), FakeWebView()
    window.hide_title_bar(shown, webview)  # pyright: ignore[reportArgumentType]
    assert webview.on_webview_load is not None
    asyncio.run(webview.on_webview_load(webview))
    return shown, webview


def band_of(shown: FakeWindow) -> Any:
    (band,) = shown._impl.native.contentView.subviews
    return band


def test_once_its_page_loads_the_window_lays_one_band_over_its_top_row(
    titled: tuple[FakeWindow, FakeWebView],
):
    shown, webview = titled
    assert webview.on_webview_load is not None
    asyncio.run(webview.on_webview_load(webview))
    band = band_of(shown)
    assert band.frame == fake_ns_make_rect(0, 0, TOGA_WINDOW_SIZE[0], TITLE_BAR_PX)
    assert band.autoresizingMask == NS_VIEW_WIDTH_SIZABLE | NS_VIEW_MAX_Y_MARGIN
    assert band.acceptsFirstMouse_(None)


def page_properties(webview: FakeWebView) -> list[str]:
    return [script for script in webview.scripts if "setProperty" in script]


def test_the_page_learns_where_the_traffic_lights_are_and_the_band_their_height(
    objc: FakeObjC,
):
    shown, webview = FakeWindow(), FakeWebView()
    # Larger and lower than in a title bar of TITLE_BAR_PX, as some macOS draws them.
    for kind, x in {0: 10, 1: 34, 2: 58}.items():
        shown._impl.native.buttons[kind].in_content = fake_ns_make_rect(x, 9, 16, 16)
    window.hide_title_bar(shown, webview)  # pyright: ignore[reportArgumentType]
    assert webview.on_webview_load is not None
    asyncio.run(webview.on_webview_load(webview))
    set_property = "document.documentElement.style.setProperty"
    centre = f'{set_property}("--titlebar-center", "17.0px")'
    inset = f'{set_property}("--titlebar-inset", "86px")'
    assert page_properties(webview) == [f"{centre};{inset}"]
    assert band_of(shown).frame == fake_ns_make_rect(0, 0, TOGA_WINDOW_SIZE[0], 34)


@pytest.mark.parametrize(
    "full_screen", [False, True], ids=["no buttons", "full screen"]
)
def test_without_the_traffic_lights_to_read_the_page_and_band_keep_their_defaults(
    objc: FakeObjC, full_screen: bool
):
    shown, webview = FakeWindow(), FakeWebView()
    if full_screen:
        shown.state = WindowState.FULLSCREEN
    else:
        shown._impl.native.buttons.clear()
    window.hide_title_bar(shown, webview)  # pyright: ignore[reportArgumentType]
    assert webview.on_webview_load is not None
    asyncio.run(webview.on_webview_load(webview))
    assert page_properties(webview) == []
    assert band_of(shown).frame.size.height == TITLE_BAR_PX


@pytest.mark.parametrize(
    ("point", "drags"),
    [
        ((200, 14), True),
        ((10, 1), True),
        ((1066, 14), True),
        ((1000, 14), False),
        ((1185, 24), False),
        ((200, TITLE_BAR_PX), False),
        ((200, 300), False),
    ],
    ids=["status", "corner", "between", "settings", "theme", "below", "page"],
)
def test_the_band_takes_presses_on_the_top_row_off_its_controls_and_no_other(
    titled: tuple[FakeWindow, FakeWebView], point: tuple[float, float], drags: bool
):
    band = band_of(titled[0])
    assert (band.hitTest_(FakeNSPoint(*point)) is band) == drags


def test_until_the_page_says_where_its_controls_are_every_press_goes_to_it(
    objc: FakeObjC,
):
    band = window._drag_band().alloc().init()
    assert band.hitTest_(FakeNSPoint(200, 14)) is None


@pytest.mark.parametrize(
    ("clicks", "setting", "moves"),
    [
        (1, "Minimize", ["drag"]),
        (2, None, ["zoom"]),
        (2, "Maximize", ["zoom"]),
        (2, "Minimize", ["minimize"]),
        (2, "None", []),
    ],
    ids=["press", "unset", "zoom", "minimize", "do nothing"],
)
def test_a_press_on_the_band_drags_the_window_and_a_double_click_does_as_set(
    titled: tuple[FakeWindow, FakeWebView],
    objc: FakeObjC,
    clicks: int,
    setting: str | None,
    moves: list[str],
):
    if setting is not None:
        objc.defaults["AppleActionOnDoubleClick"] = setting
    shown, _ = titled
    event = SimpleNamespace(clickCount=clicks)
    band_of(shown).mouseDown_(event)
    senders = {"drag": event, "zoom": None, "minimize": None}
    assert shown._impl.native.moves == [(move, senders[move]) for move in moves]


def test_a_press_the_window_cant_take_is_logged_never_raised_into_appkit(
    titled: tuple[FakeWindow, FakeWebView],
    objc: FakeObjC,
    caplog: pytest.LogCaptureFixture,
):
    del objc.classes["NSUserDefaults"]
    band_of(titled[0]).mouseDown_(SimpleNamespace(clickCount=2))
    assert [record.getMessage() for record in caplog.records] == [
        "The window couldn't take a press on its top row."
    ]


def test_the_band_learns_the_controls_again_on_each_load_and_resize(
    titled: tuple[FakeWindow, FakeWebView], monkeypatch: pytest.MonkeyPatch
):
    shown, _ = titled
    band = band_of(shown)
    assert band.controls == [SETTINGS, THEME]
    moved = ((1300.0, 3.5, 1365.0, 24.5),)
    monkeypatch.setattr(FakeWebView, "controls", moved)
    assert shown.on_resize is not None
    asyncio.run(shown.on_resize(shown))
    assert band.controls == list(moved)


def test_a_page_that_isnt_r3cons_has_no_controls_so_the_band_takes_no_press(
    titled: tuple[FakeWindow, FakeWebView],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
):
    shown, webview = titled
    monkeypatch.setattr(FakeWebView, "controls", None)
    assert webview.on_webview_load is not None
    asyncio.run(webview.on_webview_load(webview))
    band = band_of(shown)
    assert band.controls is None
    assert band.hitTest_(FakeNSPoint(200, 14)) is None
    assert caplog.records == []


def test_the_page_knows_while_the_window_is_in_full_screen_and_after_a_reload(
    titled: tuple[FakeWindow, FakeWebView],
):
    shown, webview = titled
    assert shown.on_resize is not None and webview.on_webview_load is not None
    for state in (WindowState.FULLSCREEN, WindowState.NORMAL):
        shown.state = state
        asyncio.run(shown.on_resize(shown))
    shown.state = WindowState.FULLSCREEN
    asyncio.run(webview.on_webview_load(webview))
    toggles = [script for script in webview.scripts if TOGGLE_FULL_SCREEN in script]
    assert toggles == [
        f'{TOGGLE_FULL_SCREEN}("full-screen", false)',
        f'{TOGGLE_FULL_SCREEN}("full-screen", true)',
        f'{TOGGLE_FULL_SCREEN}("full-screen", false)',
        f'{TOGGLE_FULL_SCREEN}("full-screen", true)',
    ]


def test_hiding_two_title_bars_defines_the_bands_objective_c_class_once(
    titled: tuple[FakeWindow, FakeWebView],
):
    other, webview = FakeWindow(), FakeWebView()
    window.hide_title_bar(other, webview)  # pyright: ignore[reportArgumentType]
    assert webview.on_webview_load is not None
    asyncio.run(webview.on_webview_load(webview))
    assert type(band_of(other)) is type(band_of(titled[0]))


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
