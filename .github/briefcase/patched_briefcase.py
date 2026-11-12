"""Briefcase, patched to package the Mac app in .github/workflows/mac-app.yml:

    BRIEFCASE_PYTHON -I .github/briefcase/patched_briefcase.py [--keep-signature] ARGS...

runs ``briefcase ARGS``. Briefcase signs each Mach-O file, framework and app in the app,
inside out, with the app's entitlements. In Sparkle.framework, this signs only what
Sparkle's docs sign (https://sparkle-project.org/documentation/sandboxing/#code-signing),
its helpers Autoupdate and Updater.app and then the framework, which seals the rest,
and with none of the app's entitlements, which Python needs and would only weaken the
helper that installs updates. With ``--keep-signature`` it signs nothing in the app, so
that packing a notarized app keeps the ticket stapled to it. Both patch Briefcase 0.4.7,
which .github/briefcase pins.
"""

import sys
from pathlib import Path

from briefcase.__main__ import main
from briefcase.platforms.macOS import SigningIdentity, macOSSigningMixin
from briefcase.platforms.macOS.app import macOSAppPackageCommand

SPARKLE = "Sparkle.framework"
# In the framework, what Sparkle's docs sign: Briefcase signs the framework itself last.
SPARKLE_CODE = {Path("Versions/B/Autoupdate"), Path("Versions/B/Updater.app"), Path()}
KEEP_SIGNATURE = "--keep-signature"

sign_file = macOSSigningMixin.sign_file


def sign_as_sparkle_does(
    self: macOSSigningMixin,
    path: Path,
    identity: SigningIdentity,
    entitlements: Path | None = None,
) -> None:
    parts = Path(path).parts
    if SPARKLE not in parts:
        sign_file(self, path, identity=identity, entitlements=entitlements)
        return
    framework = Path(*parts[: parts.index(SPARKLE) + 1])
    if Path(path).relative_to(framework) in SPARKLE_CODE:
        sign_file(self, path, identity=identity)


macOSSigningMixin.sign_file = sign_as_sparkle_does
if KEEP_SIGNATURE in sys.argv:
    sys.argv.remove(KEEP_SIGNATURE)
    macOSAppPackageCommand.sign_app = lambda *args, **kwargs: None
sys.exit(main())
