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
downloads, so a tab is saved through a Save dialog on the window. Its title bar is
hidden, all but its traffic lights, which sit in the page's top row
(``hide_title_bar``). It updates itself through Sparkle (:mod:`r3con.app.updates`),
and opens without updates whatever stops them. Toga is the Mac app's own: neither
r3con nor its ``app`` extra installs it.
"""

# No `from __future__ import annotations`: rubicon needs hitTest_'s real annotations.
import asyncio
import json
import os
from concurrent.futures import CancelledError, Future
from functools import cache, partial
from pathlib import Path
from typing import Any, NoReturn, cast

import toga
from toga.constants import WindowState

from r3con.app import updates
from r3con.app.server import MAC_APP_KEY_SERVICE, QUIT_SECONDS, SAVE_DIALOG, serve
from r3con.logging_setup import configure_logging, get_logger

WINDOW_SIZE = (1280, 820)
# Below macOS's ephemeral ports, 49152 and up; a new one would lose the page's settings.
PORT = 47313
SAVE_DIALOG_TITLE = "Save this tab"
TITLE_BAR_HIDDEN = "titlebar=hidden"
# The default title bar's height on macOS 11 and later, until its buttons are read.
TITLE_BAR_HEIGHT = 28
# AppKit's NSWindowCloseButton, NSWindowMiniaturizeButton and NSWindowZoomButton.
CLOSE_BUTTON = 0
MINIATURIZE_BUTTON = 1
ZOOM_BUTTON = 2
# Between the zoom button and the page's brand.
TRAFFIC_LIGHTS_GAP = 12
# AppKit's NSWindowStyleMaskFullSizeContentView and NSWindowTitleHidden.
FULL_SIZE_CONTENT_VIEW = 1 << 15
TITLE_HIDDEN = 1
# AppKit's NSViewWidthSizable and NSViewMaxYMargin: in a flipped view, the top's width.
WIDTH_SIZABLE = 1 << 1
MAX_Y_MARGIN = 1 << 5
DOUBLE_CLICK = 2
# The user's choice of what double-clicking a title bar does, and two that don't zoom.
DOUBLE_CLICK_ACTION = "AppleActionOnDoubleClick"
MINIMIZE = "Minimize"
NO_ACTION = "None"
FULL_SCREEN = "full-screen"
# JSON either way: null from a page that isn't R3Con's, as about:blank before it.
CONTROLS_QUERY = (
    "typeof r3conTitleBarControls === 'function' ? r3conTitleBarControls() : 'null'"
)

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
    """Hide ``window``'s title bar, the bar itself last, and let the page's top row drag
    the window but where the page says it has a control."""
    from rubicon.objc import NSMakeRect  # pyright: ignore[reportMissingImports]

    native = window._impl.native
    band = _drag_band().alloc().init()
    band.autoresizingMask = WIDTH_SIZABLE | MAX_Y_MARGIN
    height = TITLE_BAR_HEIGHT

    async def refresh(widget: object, **kwargs: Any) -> None:
        nonlocal height
        full = window.state == WindowState.FULLSCREEN
        lights = None if full else _traffic_lights(native)
        if lights is not None:
            centre, inset = lights
            height = 2 * centre
            webview.evaluate_javascript(_root_properties(centre, inset))
        band.frame = NSMakeRect(0, 0, native.contentView.frame.size.width, height)
        # Once Toga has put the web view in the window: over it.
        if band.superview is None:
            native.contentView.addSubview(band)
        webview.evaluate_javascript(
            "document.documentElement.classList.toggle("
            f"{json.dumps(FULL_SCREEN)}, {json.dumps(full)})"
        )
        rects = json.loads(await webview.evaluate_javascript(CONTROLS_QUERY))
        band.controls = None if rects is None else [tuple(map(float, r)) for r in rects]

    window.on_resize = refresh
    # Toga runs a coroutine handler as a task, which its type leaves out.
    webview.on_webview_load = refresh  # pyright: ignore[reportAttributeAccessIssue]
    native.titlebarAppearsTransparent = True
    native.titleVisibility = TITLE_HIDDEN
    native.styleMask |= FULL_SIZE_CONTENT_VIEW


@cache
def _drag_band() -> Any:
    """The class of the view over the top row, made once: Objective-C keeps classes
    by name."""
    from rubicon.objc import (  # pyright: ignore[reportMissingImports]
        NSPoint,
        ObjCClass,
        ObjCInstance,
        objc_method,
        objc_property,
    )

    class R3ConDragBand(ObjCClass("NSView")):
        controls = objc_property(object)

        @objc_method
        def acceptsFirstMouse_(self, event) -> bool:
            return True

        @objc_method
        def hitTest_(self, point: NSPoint) -> ObjCInstance:
            height = self.frame.size.height
            return self if _in_drag_area(self.controls, height, point) else None

        @objc_method
        def mouseDown_(self, event) -> None:
            _press(self.window, event)

    return R3ConDragBand


def _traffic_lights(native: Any) -> tuple[float, float] | None:
    """The NSWindow ``native``'s traffic lights' centre line and the inset past them,
    in its content view's top-left points, or None where AppKit gives no buttons."""
    content = native.contentView
    kinds = (CLOSE_BUTTON, MINIATURIZE_BUTTON, ZOOM_BUTTON)
    buttons = [native.standardWindowButton(kind) for kind in kinds]
    if any(button is None for button in buttons):
        return None
    frames = [content.convertRect(button.bounds, fromView=button) for button in buttons]
    zoom = frames[-1]
    right = max(frame.origin.x + frame.size.width for frame in frames)
    return zoom.origin.y + zoom.size.height / 2, right + TRAFFIC_LIGHTS_GAP


def _root_properties(centre: float, inset: float) -> str:
    """The script that hands the page the traffic lights' ``centre`` and ``inset``."""
    values = {"--titlebar-center": centre, "--titlebar-inset": inset}
    return ";".join(
        "document.documentElement.style.setProperty("
        f"{json.dumps(name)}, {json.dumps(f'{px}px')})"
        for name, px in values.items()
    )


def _in_drag_area(
    controls: list[tuple[float, ...]] | None, height: float, point: Any
) -> bool:
    """Whether the page's ``point`` is on its top row, ``height`` tall, off its
    ``controls``' rects; with none known yet, nowhere is."""
    if controls is None or not 0 <= point.y < height:
        return False
    return not any(
        left <= point.x < right and top <= point.y < bottom
        for left, top, right, bottom in controls
    )


def _press(native: Any, event: Any) -> None:
    """Drag the NSWindow ``native`` from the press ``event``, or on a double click do
    what the user's title bar does; log, never raise into AppKit."""
    from rubicon.objc import ObjCClass  # pyright: ignore[reportMissingImports]

    try:
        if event.clickCount != DOUBLE_CLICK:
            native.performWindowDragWithEvent(event)
            return
        defaults = ObjCClass("NSUserDefaults").standardUserDefaults
        _double_click(native, defaults.stringForKey(DOUBLE_CLICK_ACTION))
    except Exception:
        logger.exception("The window couldn't take a press on its top row.")


def _double_click(native: Any, action: Any) -> None:
    chosen = "" if action is None else str(action)
    if chosen == MINIMIZE:
        native.performMiniaturize(None)
    elif chosen != NO_ACTION:
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
        webview = toga.WebView()
        flags = [SAVE_DIALOG]
        try:
            hide_title_bar(cast("toga.Window", app.main_window), webview)
            flags.append(TITLE_BAR_HIDDEN)
        except Exception:
            logger.exception("The window keeps its title bar: hiding it failed.")
        webview.url = "&".join([url, *flags])
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
