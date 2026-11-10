"""The Mac app's window (r3con.app.window), with Toga faked at the screen."""

from collections.abc import Callable
from typing import Any

import httpx2
import pytest
import toga

from r3con.app import window

pytestmark = [pytest.mark.enable_socket, pytest.mark.allow_hosts(["127.0.0.1"])]
TIMEOUT_SECONDS = 10


class FakeWebView:
    def __init__(self, *, url: str) -> None:
        self.url = url


class FakeWindow:
    def __init__(self) -> None:
        self.hidden = False

    def hide(self) -> None:
        self.hidden = True


class FakeApp:
    """Toga's app. Its main loop opens a window on ``startup``'s web view, reads the
    page there as a user would, then quits as the app menu's Quit does, which
    ``on_exit`` allows or not."""

    def __init__(
        self, *, startup: Callable[..., FakeWebView], on_exit: Callable[..., bool]
    ) -> None:
        self.startup = startup
        self.on_exit = on_exit
        self.windows = [FakeWindow()]

    def main_loop(self) -> None:
        self.url = self.startup(self).url
        base_url, token = self.url.split("/#token=")
        with httpx2.Client(base_url=base_url, timeout=TIMEOUT_SECONDS) as http:
            self.page = http.get("/")
            self.keys = http.get(
                "/api/keys", headers={"Authorization": f"Bearer {token}"}
            )
        self.quits = self.on_exit(self)


@pytest.fixture
def opened(monkeypatch: pytest.MonkeyPatch) -> list[FakeApp]:
    """The apps the window opens."""
    apps: list[FakeApp] = []

    def app(**handlers: Any) -> FakeApp:
        apps.append(FakeApp(**handlers))
        return apps[-1]

    monkeypatch.setattr(toga, "App", app)
    monkeypatch.setattr(toga, "WebView", FakeWebView)
    return apps


def refuses(url: str) -> bool:
    try:
        httpx2.get(url, timeout=TIMEOUT_SECONDS)
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


def test_quitting_hides_the_window_then_stops_the_server(opened: list[FakeApp]):
    window.main()
    (app,) = opened
    assert app.quits
    assert all(shown.hidden for shown in app.windows)
    assert refuses(app.url)


def test_the_server_stops_when_the_window_fails(
    opened: list[FakeApp], monkeypatch: pytest.MonkeyPatch
):
    given: list[str] = []

    def broken(*, url: str) -> FakeWebView:
        given.append(url)
        raise RuntimeError("no screen to show it on")

    monkeypatch.setattr(toga, "WebView", broken)
    with pytest.raises(RuntimeError):
        window.main()
    assert refuses(*given)
