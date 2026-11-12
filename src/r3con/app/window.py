"""The app's page in a window of its own: the Mac app, which Briefcase starts as
``python -m r3con``.

It serves the app as ``r3con app`` does, on a free port with a new token, but with its
keys under its own service in the Keychain, ``MAC_APP_KEY_SERVICE``, and shows the
page in a Toga window of ``WINDOW_SIZE``. Closing the window, or Quit in the app's
menu, hides it, then stops the server as ``serve``'s stop does, before the app exits;
if Toga can't start the app, the server stops before the error goes on. Its web view
has no downloads, so a tab is saved through a Save dialog on the window. Toga is the
Mac app's own: neither r3con nor its ``app`` extra installs it.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError, Future
from functools import partial
from pathlib import Path
from typing import Any, cast

import toga

from r3con.app.server import MAC_APP_KEY_SERVICE, serve
from r3con.logging_setup import configure_logging

WINDOW_SIZE = (1280, 820)
SAVE_DIALOG_TITLE = "Save this tab"


def choose_save_path(open_dialogs: set[Future[Path | None]], name: str) -> Path | None:
    """Where the user saves the file ``name`` suggests, in a Save dialog on the app's
    window, or ``None`` if they cancel, or the app quits first, which cancels each of
    ``open_dialogs``. The server asks off the app's main thread, and waits for the
    dialog there."""
    # Toga types both as maybe missing: both are there once the page can save.
    app = cast("toga.App", toga.App.app)
    window = cast("toga.Window", app.main_window)
    dialog = asyncio.run_coroutine_threadsafe(_save_dialog(window, name), app.loop)
    open_dialogs.add(dialog)
    try:
        return dialog.result()
    except CancelledError:
        return None
    finally:
        open_dialogs.discard(dialog)


async def _save_dialog(window: toga.Window, name: str) -> Path | None:
    # Made here, on the main thread, where AppKit makes its panels.
    return await window.dialog(toga.SaveFileDialog(SAVE_DIALOG_TITLE, name))


def main() -> None:
    """Serve the app and show its page in a window until the app quits."""
    configure_logging()
    open_dialogs: set[Future[Path | None]] = set()
    choose = partial(choose_save_path, open_dialogs)
    url, stop = serve(key_service=MAC_APP_KEY_SERVICE, choose_save_path=choose)

    def page(app: toga.App, **kwargs: Any) -> toga.Widget:
        for window in app.windows:
            window.size = WINDOW_SIZE
        return toga.WebView(url=url)

    def hide(app: toga.App, **kwargs: Any) -> bool:
        # At once: the stop waits for runs still going to end their calls in flight.
        for window in app.windows:
            window.hide()
        # Its save would wait for the dialog forever, and the server's stop for the save.
        for dialog in [*open_dialogs]:
            dialog.cancel()
        return True

    try:
        toga.App(startup=page, on_exit=hide).main_loop()
    finally:
        stop()
