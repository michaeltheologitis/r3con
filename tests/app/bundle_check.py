"""Check a built Mac app with its own Python, for .github/workflows/mac-app.yml:

    APP/Contents/Frameworks/Python.framework/Versions/3.14/bin/python3 -I -B \\
        tests/app/bundle_check.py APP CONSTRAINTS BUILD_CONSTRAINTS

It puts the app's code and packages on the path as the app's launcher does, and checks
that the packages there but the project are exactly the ones CONSTRAINTS, the build's
`uv export` of uv.lock, pins for this Python, at its versions, and that the project was
built by the backend BUILD_CONSTRAINTS pins. Then it imports, each from inside APP,
r3con and the packages a bundle most likely lacks, and every compiled module under its
packages, so that one built for another platform fails here. It checks that keyring,
where the app keeps its keys, finds the macOS Keychain, and saves, reads back and
deletes a throwaway item there, giving up after ``KEYCHAIN_SECONDS``, since a Keychain
dialog would wait for a click no one makes. Then it serves the app as the window does:
the page, and the API with this launch's token, must answer, and once the server has
stopped, nothing must.
"""

import importlib.metadata
import secrets
import site
import sys
import tempfile
import threading
from email.parser import HeaderParser
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.error import URLError
from urllib.request import Request, urlopen

if TYPE_CHECKING:
    from keyring.backend import KeyringBackend
    from packaging.specifiers import SpecifierSet

BUNDLED = (
    "r3con",
    "litellm",
    "pydantic_core",
    "tokenizers",
    "tiktoken",
    "fastuuid",
    "toga",
    # Toga's Mac web view, which loads AppKit and WebKit.
    "toga_cocoa.widgets.webview",
    # keyring's Keychain calls, which load the Security framework.
    "keyring.backends.macOS.api",
)
TIMEOUT_SECONDS = 10
KEYCHAIN_SECONDS = 30
# Not the app's service, so that the check never touches a key the app saved.
THROWAWAY_SERVICE = "r3con-bundle-check"
# Built from the checkout, so not among uv.lock's exported pins: its backend is.
PROJECT = "r3context"
BACKEND = "uv-build"


def imported(name: str, resources: Path) -> Path:
    """The file module ``name`` is imported from, which must be in ``resources``."""
    where = Path(importlib.import_module(name).__file__ or "").resolve()
    if not where.is_relative_to(resources):
        sys.exit(f"{name} is imported from {where}, not from {resources}")
    return where


def compiled(packages: Path) -> list[str]:
    """The module of each compiled file under ``packages``."""
    return sorted(
        ".".join(path.relative_to(packages).with_name(path.name.split(".")[0]).parts)
        for path in packages.rglob("*.so")
    )


def locked(constraints: Path) -> "dict[str, SpecifierSet]":
    """The version ``constraints``, a ``uv export``, pins for each package this Python
    takes, by normalized name."""
    # The bundle's own packaging, on the path once main has added app_packages.
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    text = constraints.read_text(encoding="utf-8").replace("\\\n", "")
    lines = [line.split(" --hash=")[0] for line in text.splitlines()]
    requirements = [Requirement(line) for line in lines if line and line[0] not in " #"]
    return {
        canonicalize_name(requirement.name): requirement.specifier
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate()
    }


def pinned(packages: Path, constraints: Path) -> int:
    """How many packages ``packages`` holds besides the project, which must be the ones
    ``constraints`` pins for this Python, each at its pinned version."""
    from packaging.utils import canonicalize_name

    pins = locked(constraints)
    found = {
        canonicalize_name(distribution.name): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(packages)])
    }
    found.pop(canonicalize_name(PROJECT), None)
    problems = [
        f"{name}: uv.lock pins {pins[name]}, but it is missing"
        for name in sorted(pins.keys() - found.keys())
    ]
    for name, version in sorted(found.items()):
        pin = pins.get(name)
        if pin is None or version not in pin:
            problems.append(f"{name} {version}: uv.lock pins {pin or 'no version'}")
    if problems:
        sys.exit("\n".join(["Not as uv.lock pins it:", *problems]))
    return len(found)


def built_by(packages: Path, build_constraints: Path) -> str:
    """The backend that built the project's wheel in ``packages``, which must be at the
    version ``build_constraints`` pins."""
    found = importlib.metadata.distributions(name=PROJECT, path=[str(packages)])
    distribution = next(iter(found), None)
    if distribution is None:
        sys.exit(f"{PROJECT} is missing")
    wheel = distribution.read_text("WHEEL") or ""
    generator = HeaderParser().parsestr(wheel)["Generator"]
    pin = locked(build_constraints)[BACKEND]
    if generator.split()[-1] not in pin:
        sys.exit(
            f"{PROJECT} was built by {generator}, but uv.lock pins {BACKEND} {pin}"
        )
    return generator


def keychain() -> "KeyringBackend":
    """The store keyring finds, which must be its macOS Keychain's."""
    import keyring
    from keyring.backends import macOS

    store = keyring.get_keyring()
    if not isinstance(store, macOS.Keyring):
        sys.exit(f"keyring finds {type(store).__qualname__}, not the macOS Keychain")
    return store


def round_trip(store: "KeyringBackend") -> None:
    """Save a throwaway item in ``store``, read it back and delete it, on a thread of
    its own, waited for at most ``KEYCHAIN_SECONDS``."""
    account, password = secrets.token_hex(8), secrets.token_urlsafe(16)
    read: list[str | None] = []

    def save_read_delete() -> None:
        store.set_password(THROWAWAY_SERVICE, account, password)
        try:
            read.append(store.get_password(THROWAWAY_SERVICE, account))
        finally:
            store.delete_password(THROWAWAY_SERVICE, account)
        read.append(store.get_password(THROWAWAY_SERVICE, account))

    worker = threading.Thread(target=save_read_delete, daemon=True)
    worker.start()
    worker.join(KEYCHAIN_SECONDS)
    if worker.is_alive():
        sys.exit(f"the Keychain did not answer within {KEYCHAIN_SECONDS} s")
    if read != [password, None]:
        sys.exit("a throwaway item was not saved, read back as saved, and deleted")


def answers(url: str) -> bool:
    try:
        urlopen(url, timeout=TIMEOUT_SECONDS).close()
    except URLError:
        return False
    return True


def main(app: Path, constraints: Path, build_constraints: Path) -> None:
    resources = app / "Contents" / "Resources"
    packages = resources / "app_packages"
    sys.path.append(str(resources / "app"))
    site.addsitedir(str(packages))
    count = pinned(packages, constraints)
    print(f"{count} packages under app_packages but {PROJECT}: each as uv.lock pins it")
    generator = built_by(packages, build_constraints)
    print(f"{PROJECT}: built by {generator}, as uv.lock pins {BACKEND}")
    for name in BUNDLED:
        print(f"{name}: {imported(name, resources).relative_to(app)}")
    modules = compiled(packages)
    for name in modules:
        imported(name, resources)
    print(f"{len(modules)} compiled modules under app_packages: each imports")
    store = keychain()
    print(f"keyring finds {type(store).__module__}.{type(store).__qualname__}")
    round_trip(store)
    print(f"{THROWAWAY_SERVICE}: saved, read back and deleted in the Keychain")

    from r3con import __version__
    from r3con.app import serve
    from r3con.app.server import MAC_APP_KEY_SERVICE

    print(f"r3con {__version__}")
    url, stop = serve(
        logs_dir=Path(tempfile.mkdtemp()), key_service=MAC_APP_KEY_SERVICE
    )
    base_url, token = url.split("/#token=")
    keys = Request(f"{base_url}/api/keys", headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(f"{base_url}/", timeout=TIMEOUT_SECONDS) as answer:
            page_status, page = answer.status, answer.read().decode()
        with urlopen(keys, timeout=TIMEOUT_SECONDS) as answer:
            keys_status, keys_body = answer.status, answer.read().decode()
    finally:
        stop()
    if "<title>" not in page:
        sys.exit("GET / did not answer with the page")
    print(f"GET / {page_status}")
    print(f"GET /api/keys {keys_status} {keys_body}")
    if answers(f"{base_url}/"):
        sys.exit("the server still answers after its stop")
    print("stopped: nothing answers")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3]))
