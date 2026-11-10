"""The app's page in a window of its own: the Mac app, which Briefcase starts as
``python -m r3con``.

It serves the app as ``r3con app`` does, on a free port with a new token, and shows the
page in a Toga window. Closing the window, or Quit in the app's menu, hides it, then
stops the server as ``serve``'s stop does, before the app exits; the server stops too
if the window fails. Toga is the Mac app's own: neither r3con nor its ``app`` extra
installs it.
"""

from __future__ import annotations

from typing import Any

import toga

from r3con.app.server import serve
from r3con.logging_setup import configure_logging


def main() -> None:
    """Serve the app and show its page in a window until the app quits."""
    configure_logging()
    url, stop = serve()

    def page(app: toga.App, **kwargs: Any) -> toga.Widget:
        return toga.WebView(url=url)

    def hide(app: toga.App, **kwargs: Any) -> bool:
        # At once: the stop waits for runs still going to end their calls in flight.
        for window in app.windows:
            window.hide()
        return True

    try:
        toga.App(startup=page, on_exit=hide).main_loop()
    finally:
        stop()
