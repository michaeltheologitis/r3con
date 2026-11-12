import base64
import re

import pytest
from Crypto.PublicKey import ECC
from Crypto.Signature import eddsa

from tests.app.feed_check import check

URL = "https://github.com/michaeltheologitis/r3con/releases/download/v0.6.20/R3Con.dmg"
NOTES = "**Updates.** The Mac app updates itself.\n"
DMG = b"a disk image"
# As generate_appcast writes a feed, with SURequireSignedFeed, before it signs it.
CONTENT = """\
<?xml version="1.0" standalone="yes"?>
<!-- sparkle-sign-warning:
IMPORTANT: This file was signed by Sparkle. Any modifications to this file requires \
re-signing this file with generate_appcast or sign_update! The signed signature will be \
embedded at the end of this file.
-->
<rss xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle" version="2.0">
    <channel>
        <title>R3Con</title>
{items}
    </channel>
</rss>
"""
ITEM = """\
        <item>
            <title>0.6.20</title>
            <pubDate>Thu, 12 Nov 2026 07:00:00 +0000</pubDate>
            <sparkle:version>0.6.20</sparkle:version>
            <sparkle:shortVersionString>0.6.20</sparkle:shortVersionString>
            <sparkle:minimumSystemVersion>11.0</sparkle:minimumSystemVersion>
            <sparkle:hardwareRequirements>arm64</sparkle:hardwareRequirements>
            <description sparkle:format="markdown"><![CDATA[{notes}]]></description>
            <enclosure url="{url}" length="{length}" type="application/octet-stream" \
sparkle:edSignature="{signature}"/>
        </item>"""


def signature(key: ECC.EccKey, data: bytes) -> str:
    return base64.b64encode(eddsa.new(key, "rfc8032").sign(data)).decode()


def signed(key: ECC.EccKey, content: str) -> bytes:
    """``content`` with the signing block Sparkle appends, by ``key``."""
    data = content.encode()
    block = f"<!-- sparkle-signatures:\nedSignature: {signature(key, data)}\n"
    return data + f"{block}length: {len(data)}\n-->\n".encode()


@pytest.fixture(scope="module")
def key() -> ECC.EccKey:
    return ECC.generate(curve="Ed25519")


@pytest.fixture
def info(key: ECC.EccKey) -> dict[str, str]:
    """The app's Info.plist, as far as updates go."""
    public_key = key.public_key().export_key(format="raw")
    return {
        "SUPublicEDKey": base64.b64encode(public_key).decode(),
        "CFBundleVersion": "0.6.20",
        "CFBundleShortVersionString": "0.6.20",
    }


def item(key: ECC.EccKey, **changed: str) -> str:
    fields = {"notes": NOTES, "url": URL, "length": str(len(DMG))}
    return ITEM.format_map(fields | {"signature": signature(key, DMG)} | changed)


def test_check_names_the_update_a_feed_made_as_generate_appcast_makes_it_offers(
    key: ECC.EccKey, info: dict[str, str]
):
    feed = signed(key, CONTENT.format(items=item(key)))
    assert check(info, feed, DMG, URL, NOTES) == f"0.6.20 at {URL}"


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        ({"url": URL.replace("v0.6.20", "latest")}, "not 12 bytes at"),
        ({"length": "13"}, "not 12 bytes at"),
        ({"notes": "Other notes."}, "description is 'Other notes.'"),
        ({"signature": ""}, f"{URL} is not signed by the app's key"),
    ],
    ids=["another url", "another length", "other notes", "an unsigned .dmg"],
)
def test_check_exits_on_an_update_unlike_this_release(
    key: ECC.EccKey, info: dict[str, str], change: dict[str, str], problem: str
):
    feed = signed(key, CONTENT.format(items=item(key, **change)))
    with pytest.raises(SystemExit, match=re.escape(problem)):
        check(info, feed, DMG, URL, NOTES)


def test_check_exits_on_a_dmg_its_signature_does_not_sign(
    key: ECC.EccKey, info: dict[str, str]
):
    feed = signed(key, CONTENT.format(items=item(key)))
    with pytest.raises(SystemExit, match="is not signed by the app's key"):
        check(info, feed, b"another disk", URL, NOTES)


@pytest.mark.parametrize(
    ("changed", "problem"),
    [
        ({"CFBundleVersion": "1"}, "sparkle:version is '0.6.20', not '1'"),
        (
            {"CFBundleShortVersionString": "0.6.21"},
            "sparkle:shortVersionString is '0.6.20', not '0.6.21'",
        ),
    ],
    ids=["briefcase's build", "another version"],
)
def test_check_exits_on_a_feed_for_another_version_of_the_app(
    key: ECC.EccKey, info: dict[str, str], changed: dict[str, str], problem: str
):
    feed = signed(key, CONTENT.format(items=item(key)))
    with pytest.raises(SystemExit, match=re.escape(problem)):
        check(info | changed, feed, DMG, URL, NOTES)


def test_check_exits_on_a_feed_signed_by_a_key_the_app_does_not_trust(
    key: ECC.EccKey, info: dict[str, str]
):
    other = ECC.generate(curve="Ed25519")
    feed = signed(other, CONTENT.format(items=item(other)))
    with pytest.raises(SystemExit, match="the feed is not signed by the app's key"):
        check(info, feed, DMG, URL, NOTES)


def test_check_exits_on_a_feed_edited_after_it_was_signed(
    key: ECC.EccKey, info: dict[str, str]
):
    feed = signed(key, CONTENT.format(items=item(key)))
    edited = feed.replace(b"<title>R3Con</title>", b"<title>R3Kon</title>")
    with pytest.raises(SystemExit, match="the feed is not signed by the app's key"):
        check(info, edited, DMG, URL, NOTES)


def test_check_exits_on_an_unsigned_feed(info: dict[str, str], key: ECC.EccKey):
    feed = CONTENT.format(items=item(key)).encode()
    with pytest.raises(SystemExit, match="the feed is not signed$"):
        check(info, feed, DMG, URL, NOTES)


def test_check_exits_on_a_feed_that_offers_more_than_one_update(
    key: ECC.EccKey, info: dict[str, str]
):
    feed = signed(key, CONTENT.format(items=item(key) + "\n" + item(key)))
    with pytest.raises(SystemExit, match="the feed offers 2 updates, not one"):
        check(info, feed, DMG, URL, NOTES)


def test_check_exits_on_notes_that_are_not_markdown(
    key: ECC.EccKey, info: dict[str, str]
):
    plain = item(key).replace(' sparkle:format="markdown"', "")
    feed = signed(key, CONTENT.format(items=plain))
    with pytest.raises(SystemExit, match="the feed's notes are not markdown"):
        check(info, feed, DMG, URL, NOTES)
