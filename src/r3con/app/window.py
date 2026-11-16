"""The app's page in a window of its own: the Mac app, which Briefcase starts as
``python -m r3con``.

It serves the app as ``r3con app`` does, with a new token, but on ``PORT`` at every
launch, so that the page keeps its settings, which it stores by its origin, port and
all; when ``PORT`` can't be had, as when another program has it, on a free port, which
the log then names. The window opens on the port its server holds, so such a program
can't pose as R3Con's server: it costs that launch the settings stored on ``PORT``.
Its keys are under its own service in the Keychain, ``MAC_APP_KEY_SERVICE``, and it
shows the page in a Toga window of ``WINDOW_SIZE``. Closing the window, Quit in the
app's menu or in the Dock, or logging out hides it, then stops the server as
``serve``'s stop does, for up to ``QUIT_SECONDS``, and ends the process: a model call
still in flight is cut off, and its run folder left without ``run_failed``. If Toga
can't start the app, the server stops before the error goes on. Its web view has no
downloads, so a tab is saved through a Save dialog on the window. The window shows no
title bar but its traffic lights, over the left of the page's tab row, which the page,
opened with ``TITLE_BAR_HIDDEN``, makes the window's top row: it drags the window by
posting ``DRAG`` to its ``WINDOW_MESSAGES`` handler and zooms it by ``ZOOM``, and has
``FULL_SCREEN`` on its root while the window is in full screen, where the lights are
hidden. It updates itself through Sparkle (:mod:`r3con.app.updates`), and opens
without updates whatever stops them. Toga is the Mac app's own: neither r3con nor its
``app`` extra installs it.
"""

from __future__ import annotations

import asyncio
import json
import os
from concurrent.futures import CancelledError, Future
from functools import partial
from pathlib import Path
from typing import Any, NoReturn, cast

import toga

from r3con.app import updates
from r3con.app.server import MAC_APP_KEY_SERVICE, QUIT_SECONDS, serve
from r3con.logging_setup import configure_logging, get_logger

WINDOW_SIZE = (1280, 820)
# Below macOS's ephemeral ports, 49152 and up; a new one would lose the page's settings.
PORT = 47313
SAVE_DIALOG_TITLE = "Save this tab"
TITLE_BAR_HIDDEN = "titlebar=hidden"
# AppKit's NSWindowStyleMaskFullSizeContentView and NSWindowTitleHidden.
FULL_SIZE_CONTENT_VIEW = 1 << 15
TITLE_HIDDEN = 1
# The page's window.webkit.messageHandlers.window, and what it posts there.
WINDOW_MESSAGES = "window"
DRAG = "drag"
ZOOM = "zoom"
FULL_SCREEN = "full-screen"
ENTERED_FULL_SCREEN = "NSWindowDidEnterFullScreenNotification"
EXITED_FULL_SCREEN = "NSWindowDidExitFullScreenNotification"

logger = get_logger("app")


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


def hide_title_bar(window: toga.Window, webview: toga.WebView) -> None:
    """Hide ``window``'s title bar but its traffic lights, which then sit over the top
    of ``webview``, and let its page drag and zoom the window, and know when the
    window is in full screen. Call it once, in the app's startup."""
    # Toga's bridge to Objective-C, which only the Mac app has.
    from rubicon.objc import (  # pyright: ignore[reportMissingImports]
        SEL,
        NSObject,
        ObjCClass,
        objc_method,
    )

    native = window._impl.native
    native.titlebarAppearsTransparent = True
    native.titleVisibility = TITLE_HIDDEN
    native.styleMask |= FULL_SIZE_CONTENT_VIEW
    application = ObjCClass("NSApplication").sharedApplication

    class R3ConWindowMessages(NSObject):
        @objc_method
        def userContentController_didReceiveScriptMessage_(
            self, controller, message
        ) -> None:
            _move_window(native, application.currentEvent, str(message.body))

        @objc_method
        def fullScreenChanged_(self, notification) -> None:
            full = str(notification.name) == ENTERED_FULL_SCREEN
            webview.evaluate_javascript(
                "document.documentElement.classList.toggle("
                f"{json.dumps(FULL_SCREEN)}, {json.dumps(full)})"
            )

    messages = R3ConWindowMessages.alloc().init()
    contents = webview._impl.native.configuration.userContentController
    contents.addScriptMessageHandler(messages, name=WINDOW_MESSAGES)
    center = ObjCClass("NSNotificationCenter").defaultCenter
    for name in (ENTERED_FULL_SCREEN, EXITED_FULL_SCREEN):
        center.addObserver(
            messages, selector=SEL("fullScreenChanged:"), name=name, object=native
        )


def _move_window(native: Any, event: Any, message: str) -> None:
    """Drag the NSWindow ``native`` from the mouse-down ``event`` on ``DRAG``, or zoom
    it on ``ZOOM``, as its title bar would; ignore any other ``message``."""
    if message == DRAG:
        native.performWindowDragWithEvent(event)
    elif message == ZOOM:
        native.performZoom(None)


def main() -> None:
    """Serve the app and show its page in a window until the app quits, then end the
    process."""
    configure_logging()
    open_dialogs: set[Future[Path | None]] = set()
    choose = partial(choose_save_path, open_dialogs)
    serving = partial(
        serve,
        key_service=MAC_APP_KEY_SERVICE,
        choose_save_path=choose,
        stop_within=QUIT_SECONDS,
    )
    try:
        url, stop = serving(port=PORT)
    except OSError as error:
        logger.warning("Port %d can't be had (%s): serving on a free one.", PORT, error)
        url, stop = serving()

    def page(app: toga.App, **kwargs: Any) -> toga.Widget:
        for window in app.windows:
            window.size = WINDOW_SIZE
        # Updates are optional: whatever stops them, the window still opens.
        try:
            updates.start(app)
        except Exception:
            logger.exception("Sparkle's updater failed: %s.", updates.NO_UPDATES)
        webview = toga.WebView(url=f"{url}&{TITLE_BAR_HIDDEN}")
        hide_title_bar(cast("toga.Window", app.main_window), webview)
        return webview

    def hide(app: toga.App, **kwargs: Any) -> bool:
        # At once: the stop takes up to QUIT_SECONDS.
        for window in app.windows:
            window.hide()
        # Its save would wait for the dialog forever, and the server's stop for the save.
        for dialog in [*open_dialogs]:
            dialog.cancel()
        return True

    def end(app: toga.App, sender: Any) -> NoReturn:
        # AppKit asks on a Dock Quit or a logout; Toga lets it end the app unstopped.
        hide(app)
        stop()
        os._exit(0)

    try:
        app = toga.App(startup=page, on_exit=hide)
        # Toga 0.5.7's private hook, which uv.lock pins: a newer Toga may move it.
        app._impl.cocoa_applicationShouldTerminate = partial(end, app)
        app.main_loop()
    finally:
        stop()
    # Python's exit would join r3con's document threads, so wait for their calls.
    os._exit(0)
