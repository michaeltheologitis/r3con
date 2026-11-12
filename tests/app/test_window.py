"""The Mac app's window (r3con.app.window), with Toga faked at the screen."""

import asyncio
import runpy
import threading
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import httpx2
import pytest
import toga

from r3con.app import window
from r3con.app.server import APP_KEY_SERVICE, MAC_APP_KEY_SERVICE
from tests.fakes import FakeKeyring

pytestmark = [pytest.mark.enable_socket, pytest.mark.allow_hosts(["127.0.0.1"])]
TIMEOUT_SECONDS = 10
TOGA_WINDOW_SIZE = (640, 480)
MAC_APP_KEY = "sk-saved-by-the-mac-app-1234"
R3CON_APP_KEY = "sk-saved-by-r3con-app-5678"
SAVED_NAME = "which-site-logged-the-most-incidents.html"
SAVED_TAB = b"<!doctype html><title>a saved tab</title>"


class FakeWebView:
    def __init__(self, *, url: str) -> None:
        self.url = url


class FakeSaveFileDialog:
    def __init__(self, title: str, suggested_filename: str) -> None:
        self.shown = (title, suggested_filename, threading.current_thread())


class FakeWindow:
    """Toga's window. Its Save dialog answers with ``chooses``, as the user would."""

    chooses: Path | None = None

    def __init__(self) -> None:
        self.size = TOGA_WINDOW_SIZE
        self.hidden = False
        self.dialogs: list[tuple[str, str, threading.Thread]] = []

    def hide(self) -> None:
        self.hidden = True

    async def dialog(self, dialog: FakeSaveFileDialog) -> Path | None:
        self.dialogs.append(dialog.shown)
        return self.chooses


class FakeApp:
    """Toga's app, the running one as ``app``. Its main loop opens a window on
    ``startup``'s web view, reads the page there as a user would, saves the key in
    ``saving``, if any, as its settings would, and with ``saving_tab``, saves a tab
    as its page would, then quits as the app menu's Quit does, which ``on_exit``
    allows or not. Its event loop runs while the page is read, as Toga's does."""

    app: "FakeApp"
    saving: dict[str, str] | None = None
    saving_tab = False

    def __init__(
        self, *, startup: Callable[..., FakeWebView], on_exit: Callable[..., bool]
    ) -> None:
        self.startup = startup
        self.on_exit = on_exit
        self.main_window = FakeWindow()
        self.windows = [self.main_window]
        type(self).app = self

    def main_loop(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.main_thread = threading.current_thread()
        self.url = self.startup(self).url
        await asyncio.to_thread(self.read_page)
        self.quits = self.on_exit(self)

    def read_page(self) -> None:
        base_url, fragment = self.url.split("/#")
        token = fragment.removeprefix("token=").split("&")[0]
        auth = {"Authorization": f"Bearer {token}"}
        with httpx2.Client(
            base_url=base_url, timeout=TIMEOUT_SECONDS, trust_env=False
        ) as http:
            self.page = http.get("/")
            self.keys = http.get("/api/keys", headers=auth)
            if self.saving is not None:
                http.post("/api/keys", json=self.saving, headers=auth)
            if self.saving_tab:
                params = {"name": SAVED_NAME}
                post = partial(http.post, "/api/saves", params=params, headers=auth)
                self.saved = post(content=SAVED_TAB)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[FakeApp]:
    """The apps the window opens."""
    apps: list[FakeApp] = []

    class OpenedApp(FakeApp):
        def __init__(self, **handlers: Any) -> None:
            super().__init__(**handlers)
            apps.append(self)

    monkeypatch.setattr(toga, "App", OpenedApp)
    monkeypatch.setattr(toga, "WebView", FakeWebView)
    monkeypatch.setattr(toga, "SaveFileDialog", FakeSaveFileDialog)
    return apps


def refuses(url: str) -> bool:
    try:
        httpx2.get(url, timeout=TIMEOUT_SECONDS, trust_env=False)
    except httpx2.ConnectError:
        return True
    return False


def test_the_window_opens_the_page_this_launch_serves_with_its_token(
    opened: list[FakeApp],
):
    window.main()
    (app,) = opened
    assert app.page.status_code == 200
    assert "<title>" in app.page.text
    assert app.keys.status_code == 200


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


def test_the_window_opens_at_its_size(opened: list[FakeApp]):
    window.main()
    (app,) = opened
    assert [shown.size for shown in app.windows] == [window.WINDOW_SIZE]


def test_a_tab_saved_in_the_window_is_written_where_its_save_dialog_chose(
    opened: list[FakeApp], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    chosen = tmp_path / "chosen.html"
    monkeypatch.setattr(FakeApp, "saving_tab", True)
    monkeypatch.setattr(FakeWindow, "chooses", chosen)
    window.main()
    (app,) = opened
    assert app.url.endswith("&save=dialog")
    assert app.saved.status_code == 204
    assert app.main_window.dialogs == [("Save this tab", SAVED_NAME, app.main_thread)]
    assert chosen.read_bytes() == SAVED_TAB


def test_a_save_cancelled_in_the_windows_dialog_is_answered_quietly(
    opened: list[FakeApp], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(FakeApp, "saving_tab", True)
    window.main()
    (app,) = opened
    assert (app.saved.status_code, len(app.main_window.dialogs)) == (204, 1)


def test_quitting_hides_the_window_then_stops_the_server(opened: list[FakeApp]):
    window.main()
    (app,) = opened
    assert app.quits
    assert all(shown.hidden for shown in app.windows)
    assert refuses(app.url)


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
