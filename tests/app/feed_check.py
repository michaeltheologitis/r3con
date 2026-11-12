"""Check the update feed that .github/workflows/mac-app.yml has Sparkle's
generate_appcast make for the built Mac app, with the app's own Python:

    APP/Contents/Frameworks/Python.framework/Versions/3.14/bin/python3 -I -B \\
        tests/app/feed_check.py APP FEED DMG URL NOTES

As the app requires (SURequireSignedFeed), FEED must be signed by the key the app
trusts, its Info.plist's SUPublicEDKey. It must offer one update: the app at its
version, as its CFBundleVersion and CFBundleShortVersionString give it, with NOTES as
its markdown description, to download from URL as DMG, whose length it gives and which
that key signs too. The signatures are checked as Sparkle checks them, with the
Ed25519 of pycryptodome, which the app bundles for pypdf, apart from Sparkle's tools.
"""

import base64
import plistlib
import site
import sys
from collections.abc import Mapping
from pathlib import Path
from xml.etree import ElementTree

# The block Sparkle appends to a feed it signs (common_cli/Signing.swift).
SIGNATURES = b"<!-- sparkle-signatures:\n"
SPARKLE = "{http://www.andymatuschak.org/xml-namespaces/sparkle}"


def verify(key: bytes, data: bytes, signature: str, signed: str) -> None:
    """Check that base64 ``signature`` signs ``data`` with Ed25519 public ``key``."""
    # The bundle's own pycryptodome, on the path once main has added app_packages.
    from Crypto.Signature import eddsa

    try:
        verifier = eddsa.new(eddsa.import_public_key(key), "rfc8032")
        verifier.verify(data, base64.b64decode(signature, validate=True))
    except ValueError:
        sys.exit(f"{signed} is not signed by the app's key")


def signed_content(feed: bytes, key: bytes) -> bytes:
    """The content of ``feed``, which the signing block at its end must sign with
    ``key``, as Sparkle reads it (Sparkle/SPUExtractSignedFeed.m)."""
    start = feed.rfind(SIGNATURES)
    end = feed.find(b"-->", start + len(SIGNATURES))
    if start < 0 or end < 0:
        sys.exit("the feed is not signed")
    block = feed[start + len(SIGNATURES) : end].decode()
    lines = (line.partition(":") for line in block.splitlines())
    fields = {name: value.strip() for name, _, value in lines}
    content = feed[:start]
    if fields.get("length") != str(len(content)):
        sys.exit(f"the feed signs {fields.get('length')} bytes, not its {len(content)}")
    verify(key, content, fields.get("edSignature", ""), "the feed")
    return content


def check(
    info: Mapping[str, str], feed: bytes, dmg: bytes, url: str, notes: str
) -> str:
    """The update ``feed`` offers, which must be as the module says, for the app whose
    Info.plist is ``info``, its .dmg ``dmg`` at ``url``, and ``notes``."""
    key = base64.b64decode(info["SUPublicEDKey"])
    items = ElementTree.fromstring(signed_content(feed, key)).findall("channel/item")
    if len(items) != 1:
        sys.exit(f"the feed offers {len(items)} updates, not one")
    (item,) = items
    expected = {
        "sparkle:version": info["CFBundleVersion"],
        "sparkle:shortVersionString": info["CFBundleShortVersionString"],
        "description": notes,
    }
    for name, value in expected.items():
        found = (item.findtext(name.replace("sparkle:", SPARKLE)) or "").strip()
        if found != value.strip():
            sys.exit(f"the feed's {name} is {found!r}, not {value.strip()!r}")
    description = item.find("description")
    if description is None or description.get(f"{SPARKLE}format") != "markdown":
        sys.exit("the feed's notes are not markdown")
    enclosure = item.find("enclosure")
    if enclosure is None:
        sys.exit("the feed offers nothing to download")
    if (enclosure.get("url"), enclosure.get("length")) != (url, str(len(dmg))):
        sys.exit(f"the feed offers {enclosure.attrib}, not {len(dmg)} bytes at {url}")
    verify(key, dmg, enclosure.get(f"{SPARKLE}edSignature", ""), url)
    return f"{info['CFBundleShortVersionString']} at {url}"


def main(app: Path, feed: Path, dmg: Path, url: str, notes: Path) -> None:
    site.addsitedir(str(app / "Contents" / "Resources" / "app_packages"))
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    text = notes.read_text(encoding="utf-8")
    offered = check(info, feed.read_bytes(), dmg.read_bytes(), url, text)
    print(f"The feed, signed by the app's key, offers {offered}, signed by it too")


if __name__ == "__main__":
    app, feed, dmg, url, notes = sys.argv[1:]
    main(Path(app), Path(feed), Path(dmg), url, Path(notes))
