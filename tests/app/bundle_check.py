"""Check a built Mac app with its own Python, for .github/workflows/mac-app.yml:

    APP/Contents/Frameworks/Python.framework/Versions/3.14/bin/python3 -I -B \\
        tests/app/bundle_check.py APP CONSTRAINTS BUILD_CONSTRAINTS

It puts the app's code and packages on the path as the app's launcher does, and checks
that the packages there but the project are exactly the ones CONSTRAINTS, the build's
`uv export` of uv.lock, pins for this Python, at its versions, and that the project was
built by the backend BUILD_CONSTRAINTS pins. Then it imports, each from inside APP,
r3con and the packages a bundle most likely lacks, and loads every compiled file under
its packages, importing those that are Python extensions, so that one built for another
platform fails here. It checks that keyring,
where the app keeps its keys, finds the macOS Keychain, and saves, reads back and
deletes a throwaway item there, giving up after ``KEYCHAIN_SECONDS``, since a Keychain
dialog would wait for a click no one makes. Then it serves the app as the window does:
the page, and the API with this launch's token, must answer, a PDF encrypted as signed
ones often are must read back as its text, which pypdf does only with an AES library
(here pycryptodome: cryptography is Linux-only), and once the server has stopped,
nothing must. Last, it checks the app's updates: its Info.plist must give Sparkle an
https feed, a public EdDSA key and the app's version to compare, Sparkle.framework must
load and have the updater controller the app starts, and Sparkle's code must be signed
as Sparkle's docs sign it, with the app's team and timestamp.
"""

import base64
import binascii
import ctypes
import importlib.metadata
import json
import plistlib
import re
import secrets
import site
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping
from email.parser import HeaderParser
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple
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
    # Toga's Mac dialogs, the Save panel a tab is saved through among them.
    "toga_cocoa.dialogs",
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
# What encrypted_pdf's PDF says, which the server must read back from it.
ENCRYPTED_PDF_TEXT = "Read through AES-256."
SPARKLE = Path("Contents/Frameworks/Sparkle.framework")
# Sparkle's code as its docs sign it (.github/briefcase/sign_sparkle.py), inside out.
SPARKLE_CODE = ("Versions/B/Autoupdate", "Versions/B/Updater.app", ".")
ED25519_KEY_BYTES = 32
# In `codesign --display --verbose=2`'s description of a signature.
CODE_FLAGS = re.compile(r"\bflags=0x[0-9a-f]+\((?P<flags>[^)]*)\)")
TEAM = re.compile(r"^TeamIdentifier=(?P<team>.*)$", re.MULTILINE)
TIMESTAMP = re.compile(r"^Timestamp=", re.MULTILINE)


class Signature(NamedTuple):
    flags: frozenset[str]
    team: str
    timestamped: bool


def imported(name: str, resources: Path) -> Path:
    """The file module ``name`` is imported from, which must be in ``resources``."""
    where = Path(importlib.import_module(name).__file__ or "").resolve()
    if not where.is_relative_to(resources):
        sys.exit(f"{name} is imported from {where}, not from {resources}")
    return where


def compiled(packages: Path) -> dict[str, Path]:
    """Each compiled file under ``packages``, by the module it would be."""
    return {
        ".".join(
            path.relative_to(packages).with_name(path.name.split(".")[0]).parts
        ): path
        for path in sorted(packages.rglob("*.so"))
    }


def loads(name: str, path: Path, resources: Path) -> None:
    """Load compiled file ``path``, as fails for one built for another platform, and
    import it as module ``name`` if it is a Python extension. Others, such as
    pycryptodome's, are plain libraries their package loads with ctypes."""
    library = ctypes.CDLL(str(path))
    if hasattr(library, f"PyInit_{name.rpartition('.')[2]}"):
        imported(name, resources)


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


def encrypted_pdf(text: str) -> bytes:
    """A one-page PDF that says ``text``, encrypted as signed and permission-locked PDFs
    often are: with AES-256, an empty password to open it and a random one to change
    it. pypdf writes and reads it only with cryptography or pycryptodome."""
    # The bundle's own pypdf, on the path once main has added app_packages.
    from pypdf import PdfWriter
    from pypdf.generic import ContentStream, DictionaryObject, NameObject

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    writer = PdfWriter()
    page = writer.add_blank_page(300, 200)
    page[NameObject("/Resources")] = DictionaryObject(
        {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
    )
    contents = ContentStream(None, writer)
    contents.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode())
    page.replace_contents(contents)
    writer.encrypt("", secrets.token_urlsafe(16), algorithm="AES-256")
    pdf = BytesIO()
    writer.write(pdf)
    return pdf.getvalue()


def encrypted_pdf_text(base_url: str, token: str) -> str:
    """The text the server at ``base_url`` reads from :func:`encrypted_pdf`'s PDF, sent
    as the window's page sends a PDF, which must be what it says."""
    request = Request(
        f"{base_url}/api/documents/text?name=encrypted.pdf",
        data=encrypted_pdf(ENCRYPTED_PDF_TEXT),
        headers={"Authorization": f"Bearer {token}"},
    )
    with urlopen(request, timeout=TIMEOUT_SECONDS) as answer:
        text = json.load(answer)["text"]
    if text != ENCRYPTED_PDF_TEXT:
        sys.exit(f"an AES-256 PDF reads as {text!r}, not {ENCRYPTED_PDF_TEXT!r}")
    return text


def update_feed(info: Mapping[str, object]) -> str:
    """The feed Sparkle updates the app from, by the app's Info.plist ``info``, which
    must also give it a public EdDSA key to check updates by, and the app's version as
    CFBundleVersion, which Sparkle compares with the feed's."""
    feed = str(info.get("SUFeedURL", ""))
    if not feed.startswith("https://"):
        sys.exit(f"SUFeedURL is {feed!r}, not an https URL")
    key = str(info.get("SUPublicEDKey", ""))
    try:
        size = len(base64.b64decode(key, validate=True))
    except binascii.Error:
        size = 0
    if size != ED25519_KEY_BYTES:
        sys.exit(f"SUPublicEDKey {key!r} is not a base64 Ed25519 public key")
    version, shown = info.get("CFBundleVersion"), info.get("CFBundleShortVersionString")
    if version != shown:
        sys.exit(f"CFBundleVersion is {version!r}, not the app's version, {shown!r}")
    return feed


def sparkle_loads(framework: Path) -> str:
    """Load ``framework`` as the app does, which must give it the updater controller
    the app starts; its version."""
    # The bundle's own Toga bridge and app, on the path once main has added them.
    from rubicon.objc import ObjCClass  # pyright: ignore[reportMissingImports]

    from r3con.app.updates import UPDATER_CONTROLLER

    bundle = ObjCClass("NSBundle").bundleWithPath(str(framework))
    if bundle is None or not bundle.load():
        sys.exit(f"{framework} does not load")
    try:
        ObjCClass(UPDATER_CONTROLLER)
    except NameError:
        sys.exit(f"{framework} has no {UPDATER_CONTROLLER}")
    info = plistlib.loads((framework / "Resources" / "Info.plist").read_bytes())
    return info["CFBundleShortVersionString"]


def signature(described: str) -> Signature:
    """The signature ``codesign --display --verbose=2`` describes as ``described``."""
    flags, team = CODE_FLAGS.search(described), TEAM.search(described)
    return Signature(
        frozenset(flags["flags"].split(",") if flags else ()),
        team["team"] if team else "",
        TIMESTAMP.search(described) is not None,
    )


def unlike_sparkles_docs(
    code: Signature, entitlements: str, app: Signature
) -> list[str]:
    """How a signature of Sparkle's code, ``code``, with ``entitlements`` as
    ``codesign --display --entitlements - --xml`` gives them, is not as Sparkle's docs
    sign it, with the team and timestamp of the app's, ``app``."""
    problems = []
    if "runtime" not in code.flags:
        problems.append("no hardened runtime")
    granted = plistlib.loads(entitlements.encode()) if entitlements.strip() else {}
    if granted:
        problems.append(f"entitlements {', '.join(sorted(granted))}")
    if code.team != app.team:
        problems.append(f"team {code.team}, not the app's, {app.team}")
    if code.timestamped != app.timestamped:
        problems.append("a timestamp" if code.timestamped else "no timestamp")
    return problems


def codesign(*args: str) -> subprocess.CompletedProcess[str]:
    """``codesign`` run with ``args``, which must succeed."""
    done = subprocess.run(
        ["codesign", *args], capture_output=True, text=True, check=False
    )
    if done.returncode:
        sys.exit(f"codesign {' '.join(args)} failed: {done.stderr}")
    return done


def sparkle_signed(app: Path) -> list[str]:
    """How each of Sparkle's code in ``app`` is signed, which must verify strictly and
    be as Sparkle's docs sign it, with the app's team and timestamp."""
    expected = signature(codesign("--display", "--verbose=2", str(app)).stderr)
    signed = []
    for relative in SPARKLE_CODE:
        code = app / SPARKLE / relative
        codesign("--verify", "--strict", str(code))
        found = signature(codesign("--display", "--verbose=2", str(code)).stderr)
        entitlements = codesign("--display", "--entitlements", "-", "--xml", str(code))
        problems = unlike_sparkles_docs(found, entitlements.stdout, expected)
        if problems:
            sys.exit(f"{code} is signed with {', '.join(problems)}")
        signed.append(f"{SPARKLE / relative}: {','.join(sorted(found.flags))}")
    return signed


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
    for name, path in modules.items():
        loads(name, path, resources)
    print(f"{len(modules)} compiled files under app_packages: each loads or imports")
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
        pdf_text = encrypted_pdf_text(base_url, token)
    finally:
        stop()
    if "<title>" not in page:
        sys.exit("GET / did not answer with the page")
    print(f"GET / {page_status}")
    print(f"GET /api/keys {keys_status} {keys_body}")
    print(f"POST /api/documents/text: an AES-256 PDF reads as {pdf_text!r}")
    if answers(f"{base_url}/"):
        sys.exit("the server still answers after its stop")
    print("stopped: nothing answers")
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    print(f"Sparkle checks {update_feed(info)} for a version after {__version__}")
    print(f"Sparkle {sparkle_loads(app / SPARKLE)}: loads, as the app starts it")
    for signed in sparkle_signed(app):
        print(f"{signed}: signed as Sparkle's docs sign it")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3]))
