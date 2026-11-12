"""Run Briefcase, signing Sparkle.framework as Sparkle's documentation does
(https://sparkle-project.org/documentation/sandboxing/#code-signing), for
.github/workflows/mac-app.yml:

    BRIEFCASE_PYTHON -I .github/briefcase/sign_sparkle.py package macOS app ARGS...

Briefcase signs each Mach-O file, framework and app in the app, inside out, with the
app's entitlements, and with the hardened runtime unless the identity is ad hoc. In
Sparkle.framework, this signs only what Sparkle's docs sign, its helpers Autoupdate and
Updater.app and then the framework, which seals the rest; and each with the hardened
runtime, a secure timestamp unless the identity is ad hoc, and none of the app's
entitlements, which Python needs and would only weaken the helper that installs
updates. It replaces the ``sign_file`` of Briefcase 0.4.7, which .github/briefcase pins.
"""

import sys
from pathlib import Path

from briefcase.__main__ import main
from briefcase.platforms.macOS import SigningIdentity, macOSSigningMixin

SPARKLE = "Sparkle.framework"
# In the framework, what Sparkle's docs sign: Briefcase signs the framework itself last.
SPARKLE_CODE = {Path("Versions/B/Autoupdate"), Path("Versions/B/Updater.app"), Path()}

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
    if Path(path).relative_to(framework) not in SPARKLE_CODE:
        return
    command = ["codesign", path, "--sign", identity.id, "--force"]
    command += ["--options", "runtime"]
    if not identity.is_adhoc:
        command.append("--timestamp")
    self.tools.subprocess.run(command, check=True)


macOSSigningMixin.sign_file = sign_as_sparkle_does
sys.exit(main())
