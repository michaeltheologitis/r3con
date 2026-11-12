"""The Mac app's updates, by Sparkle (https://sparkle-project.org), which
.github/workflows/mac-app.yml adds to the app as ``Contents/Frameworks/Sparkle.framework``.

``start`` loads it and starts its standard updater, which reads its settings, the feed
and the key that signs updates among them, from the app's Info.plist (pyproject.toml's
[tool.briefcase.app.r3con.macOS]). On the app's second launch Sparkle asks once whether
to check for updates automatically, and the app menu gets ``CHECK_FOR_UPDATES``. A
newer release shows its notes and an install button; Sparkle then downloads it, checks
its signature, replaces the app and relaunches it. It installs when the app's process
ends, so the window's prompt quit (:mod:`r3con.app.window`) lets it: Sparkle's installer
waits for that process to exit, and nothing in the app runs for it at quit. An app
without a Sparkle that starts, as a local ``briefcase build`` is, runs without updates,
and its log says why.
"""

from __future__ import annotations

import toga

from r3con.logging_setup import get_logger

SPARKLE = "Sparkle.framework"
UPDATER_CONTROLLER = "SPUStandardUpdaterController"
CHECK_FOR_UPDATES = "Check for Updates…"
NO_UPDATES = "the app won't update itself"

logger = get_logger("app")


def start(app: toga.App) -> None:
    """Start Sparkle's updater for ``app``, and add ``CHECK_FOR_UPDATES`` to its app
    menu, right after About, with an action that keeps the updater for the app's life.
    Call it on the main thread, in the app's startup."""
    # Toga's bridge to Objective-C, which only the Mac app has.
    from rubicon.objc import ObjCClass  # pyright: ignore[reportMissingImports]

    bundles = ObjCClass("NSBundle")
    path = f"{bundles.mainBundle.privateFrameworksPath}/{SPARKLE}"
    sparkle = bundles.bundleWithPath(path)
    if sparkle is None:
        logger.warning("No %s in the app: %s.", SPARKLE, NO_UPDATES)
        return
    if not sparkle.load():
        logger.warning("%s didn't load: %s.", path, NO_UPDATES)
        return
    try:
        controllers = ObjCClass(UPDATER_CONTROLLER)
    except NameError:
        logger.warning("%s has no %s: %s.", path, UPDATER_CONTROLLER, NO_UPDATES)
        return
    # Sparkle logs why its updater can't start, and tells the user to ask us.
    controller = controllers.alloc().initWithStartingUpdater(
        True, updaterDelegate=None, userDriverDelegate=None
    )
    # Its updater can check only once it has started.
    if controller is None or not controller.updater.canCheckForUpdates:
        logger.warning("Sparkle's updater didn't start: %s.", NO_UPDATES)
        return
    about = app.commands[toga.Command.ABOUT]
    app.commands.add(
        toga.Command(
            lambda command, **kwargs: controller.checkForUpdates(None),
            CHECK_FOR_UPDATES,
            group=about.group,
            section=about.section,
            order=about.order + 1,
        )
    )
    logger.info("Sparkle's updater started: %s is in the app menu.", CHECK_FOR_UPDATES)
