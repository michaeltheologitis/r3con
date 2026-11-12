"""The Mac app's updates, by Sparkle (https://sparkle-project.org), which
.github/workflows/mac-app.yml adds to the app as ``Contents/Frameworks/Sparkle.framework``.

``start`` loads it and starts its standard updater, which reads its settings, the feed
and the key that signs updates among them, from the app's Info.plist (pyproject.toml's
[tool.briefcase.app.r3con.macOS]). On the app's second launch Sparkle asks once whether
to check for updates automatically, and the app menu gets ``CHECK_FOR_UPDATES``. A
newer release shows its notes and an install button; Sparkle then downloads it, checks
its signature, replaces the app and relaunches it. It installs when the app's process
ends, so the window's prompt quit (:mod:`r3con.app.window`) lets it: Sparkle's installer
waits for that process to exit, and nothing in the app runs for it at quit. An app built
without it, as a local ``briefcase build`` is, runs without updates, and its log says
so.
"""

from __future__ import annotations

import toga

from r3con.logging_setup import get_logger

SPARKLE = "Sparkle.framework"
UPDATER_CONTROLLER = "SPUStandardUpdaterController"
CHECK_FOR_UPDATES = "Check for Updates…"
# Toga's About item's section in the app menu, so that the item comes right after it.
ABOUT_SECTION = -1

logger = get_logger("app")


def start(app: toga.App) -> None:
    """Start Sparkle's updater for ``app``, and add ``CHECK_FOR_UPDATES`` to its app
    menu, whose action keeps the updater for the app's life. Call it on the main
    thread, in the app's startup."""
    # Toga's bridge to Objective-C, which only the Mac app has.
    from rubicon.objc import ObjCClass  # pyright: ignore[reportMissingImports]

    bundles = ObjCClass("NSBundle")
    path = f"{bundles.mainBundle.privateFrameworksPath}/{SPARKLE}"
    sparkle = bundles.bundleWithPath(path)
    if sparkle is None:
        logger.warning("No %s in the app: it won't update itself.", SPARKLE)
        return
    if not sparkle.load():
        logger.error("%s didn't load: the app won't update itself.", path)
        return
    # Sparkle logs why if its updater can't start, and tells the user to ask us.
    controller = (
        ObjCClass(UPDATER_CONTROLLER)
        .alloc()
        .initWithStartingUpdater(True, updaterDelegate=None, userDriverDelegate=None)
    )
    app.commands.add(
        toga.Command(
            lambda command, **kwargs: controller.checkForUpdates(None),
            CHECK_FOR_UPDATES,
            group=toga.Group.APP,
            section=ABOUT_SECTION,
            order=1,
        )
    )
    logger.info("%s loaded: %s is in the app menu.", path, CHECK_FOR_UPDATES)
