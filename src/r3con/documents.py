"""Read documents off the filesystem: :func:`read_documents` turns a folder, a glob or a
list of files into the ``{name: text}`` mapping :func:`~r3con.r3con.run` takes, and the
readers here turn a PDF, a Word file or a web page into text.
"""

from __future__ import annotations

import codecs
import io
import json
import re
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from fnmatch import fnmatch
from itertools import count, takewhile
from pathlib import Path
from typing import Any, BinaryIO, Literal

import mammoth
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning
from bs4.dammit import EncodingDetector
from bs4.element import NavigableString, PageElement, PreformattedString, Tag
from markdownify import ATX, MarkdownConverter
from pypdf import PdfReader

from r3con.logging_setup import get_logger

_log = get_logger("documents")
# Office's lock file for a document someone has open: ~$report.docx.
_LOCK_FILE_PREFIX = "~$"
# Where macOS's Finder puts each file's metadata in a zip it makes.
_MACOS_METADATA_FOLDER = "__MACOSX"
_GLOB_WILDCARDS = frozenset("*?[")
_RECURSIVE_WILDCARD = "**"
# The number unique_names gives a name's second document.
_FIRST_REPEAT = 2
# How a browser reads a page that declares these (WHATWG's Encoding standard): Latin-1
# and ASCII as windows-1252, and UTF-16, which a declaration read as ASCII can't be, as
# UTF-8.
_DECLARED_READ_AS = {
    "ascii": "cp1252",
    "iso8859-1": "cp1252",
    "utf-16": "utf-8",
    "utf-16-be": "utf-8",
    "utf-16-le": "utf-8",
}
_UNTIL_FOUND = "until-found"
# schema.org's types for an article, and for a page, by how their names end.
_ARTICLE_TYPES = (
    "Article",
    "Report",
    "BlogPosting",
    "SocialMediaPosting",
    "DiscussionForumPosting",
)
_PAGE_TYPES = ("Page",)
# A URL, with a scheme or without: not an author's name.
_URL = re.compile(r"([a-z][a-z0-9+.-]*:)?//|www\.", re.IGNORECASE)
# The elements that sit within a line of text, as an image does.
_INLINE = frozenset(
    {
        "a",
        "abbr",
        "b",
        "big",
        "cite",
        "code",
        "del",
        "em",
        "font",
        "i",
        "ins",
        "kbd",
        "label",
        "mark",
        "picture",
        "q",
        "s",
        "small",
        "span",
        "strong",
        "sub",
        "sup",
        "time",
        "u",
    }
)
# What needs no space before it after an image's alt text.
_NO_SPACE_BEFORE = frozenset(" \t\r\n.,;:!?)")
# What Office writes, in any case, into alt text it made up: a guess, not the author's.
_OFFICE_ALT_TEXT_MARKERS = (
    "Description automatically generated",
    "Description generated with",
    "AI-generated content may be incorrect",
)
# Where markdownify starts and ends a code block.
_FENCE = "```"
# Why a PDF has no text: no page has any (a scan, likely), it is encrypted and pypdf
# can't open it, as with a password, or pypdf can't read it.
_PdfNoText = Literal["no_text", "password", "unreadable"]
# What an encrypted PDF's trailer names its encryption dictionary.
_PDF_ENCRYPT_KEY = b"/Encrypt"
# bs4 warns that a page starting <?xml may be XML, and that one whose text looks like a
# URL or a path isn't one; a web page is read as HTML either way.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def _read_one(path: Path) -> str:
    """One file's text, as :func:`_read_stream` reads it. A file that can't be opened is
    logged, and gives ``""``, which the caller then drops."""
    try:
        file = path.open("rb")
    except OSError as e:
        _log.warning("%s: not readable (%s); skipping it", path, type(e).__name__)
        return ""
    with file:
        return _read_stream(str(path), file)


def _read_stream(name: str, file: BinaryIO) -> str:
    """The text of a file called ``name``, from a stream of its bytes.

    A file whose suffix is in :data:`_EXTRACTORS` is extracted, and an extractor gives
    ``""`` for a file it can't read; everything else is read as UTF-8, replacing
    undecodable bytes rather than failing — one bad byte in one document should not
    sink a run.
    """
    extract = _extractor(name)
    if extract is not None:
        return extract(name, file)
    return io.TextIOWrapper(file, encoding="utf-8", errors="replace").read()


def _pdf_text(name: str, pdf: BinaryIO) -> str:
    """A PDF's text, as :func:`_pdf_read` reads it."""
    return _pdf_read(name, pdf)[0]


def _pdf_read(name: str, pdf: BinaryIO) -> tuple[str, _PdfNoText | None]:
    """A PDF's text, page by page, from a stream of its bytes, and why it has none, if
    it hasn't; ``name`` is what the warnings call it.

    A PDF with no text layer (a scan) gives ``""``, and so does one pypdf can't read,
    which the caller then drops: say which rather than letting it vanish. One that is
    encrypted is ``"password"`` whatever stopped it, a password other than an empty one
    or encryption pypdf can't undo: the fix is a copy saved unprotected either way. The
    fix for a scan is OCR, outside this package. Half a surrogate pair, which a font's
    ToUnicode map can give and no UTF-8 encoder takes, becomes U+FFFD.
    """
    try:
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
    except Exception as e:  # noqa: BLE001 — one unreadable file, not a dead run
        if _encrypted(pdf):
            _log.warning(
                "%s: password-protected (%s); skipping it", name, type(e).__name__
            )
            return "", "password"
        _log.warning(
            "%s: not readable as a PDF (%s); skipping it", name, type(e).__name__
        )
        return "", "unreadable"
    # pypdf decodes ToUnicode maps with surrogatepass, so a pair can come as two code
    # units, or half of one alone: UTF-16 joins the pairs and replaces the halves.
    text = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    if text.strip():
        return text, None
    _log.warning(
        "%s: no extractable text — likely a scan, which needs OCR first; skipping it",
        name,
    )
    return text, "no_text"


def _encrypted(pdf: BinaryIO) -> bool:
    """Whether the PDF ``pdf`` is encrypted: whether its bytes name an encryption
    dictionary, as an encrypted PDF's trailer does. They are searched, not parsed:
    pypdf's reader fails as it is made for encryption it can't undo."""
    pdf.seek(0)
    return _PDF_ENCRYPT_KEY in pdf.read()


def _docx_text(name: str, docx: BinaryIO) -> str:
    """A Word file's text as Markdown, from a stream of its bytes; ``name`` is what the
    warnings call it.

    mammoth turns its body into HTML, and markdownify that into Markdown: headings,
    lists and tables keep their shape, a table's first row is its header, and a tracked
    insertion stays while a deletion goes. The text is kept as written, its ``_`` and
    ``*`` unescaped, but for a ``|`` in a table's cell. A picture is its description,
    Word's alt text, put where it was as a web page's image is (:func:`_images_as_text`).
    A file that can't be read gives ``""``, and so does one with no text, which the
    caller then drops: say so.
    """
    try:
        # An <img> with only its alt text, and no src, so no picture's bytes are read.
        no_src = mammoth.images.img_element(lambda picture: {})
        converted = mammoth.convert_to_html(docx, convert_image=no_src)
        soup = BeautifulSoup(converted.value, "html.parser")
        _images_as_text(soup)
        text = _Markdown().convert_soup(soup)
    except Exception as e:  # noqa: BLE001 — one unreadable file, not a dead run
        _log.warning(
            "%s: not readable as a Word file (%s); skipping it", name, type(e).__name__
        )
        return ""
    for message in converted.messages:
        _log.debug("%s: %s", name, message.message)
    if not text.strip():
        _log.warning("%s: no extractable text; skipping it", name)
    return text


def _html_text(name: str, html: BinaryIO) -> str:
    """A web page's text as Markdown, from a stream of its bytes; ``name`` is what the
    warnings call it.

    It is decoded as :func:`_page_text` says, and turned into Markdown as a Word file is:
    headings, lists and tables keep their shape. Scripts, styles, ``<noscript>``,
    ``<template>``, ``<?xml?>`` and what is ``hidden`` are left out; above what is left
    go the lines of :func:`_page_header`, which say what the page is. A link keeps its
    text, not its URL, and an image its alt text, if it has one. A page nested too deep
    for Markdown is read as its text alone. A file that can't be read gives ``""``, and
    so does one with no text but its header, which the caller then drops: say so.
    """
    try:
        soup = BeautifulSoup(_page_text(html.read()), "html.parser")
        for unseen in _unseen(soup):
            unseen.extract()
        header = _page_header(soup)
        # The page's own <title> is in the header.
        for title in soup(_is_page_title):
            title.extract()
        _images_as_text(soup)
        body = _page_markdown(name, soup)
    except Exception as e:  # noqa: BLE001 — one unreadable file, not a dead run
        _log.warning(
            "%s: not readable as HTML (%s); skipping it", name, type(e).__name__
        )
        return ""
    if not body.strip():
        _log.warning("%s: no extractable text; skipping it", name)
        return ""
    return "\n\n".join(filter(None, [header, body]))


def _unseen(soup: BeautifulSoup) -> list[PageElement]:
    """What in ``soup`` a browser doesn't show."""
    # HTML reads <?…?>, CDATA and declarations as comments; markdownify keeps them.
    return [
        *soup(["noscript", "template"]),
        *soup(hidden=_hides),
        *soup(string=lambda text: isinstance(text, PreformattedString)),
    ]


def _is_page_title(tag: Tag) -> bool:
    """Whether ``tag`` is the page's own ``<title>``, not an SVG's, which labels it."""
    return tag.name == "title" and tag.find_parent("svg") is None


def _images_as_text(soup: BeautifulSoup) -> None:
    """Put each image in ``soup`` as its alt text, or as nothing if it has none or
    Office made it up (:data:`_OFFICE_ALT_TEXT_MARKERS`)."""
    for image in soup("img"):
        alt = _alt_text(image)
        if alt:
            image.replace_with(alt)
        else:
            image.extract()


def _alt_text(image: Tag) -> str:
    """An image's alt text, or ``""`` for one Office made up, set apart by a space from
    the text beside it on its line, as the image's own room sets it apart, but in
    ``<pre>`` or ``<code>``."""
    alt = _one_line(image.get("alt"))
    if not alt or _office_made_up(alt):
        return ""
    if image.find_parent(["pre", "code"]):
        return alt
    block = _block(image)
    before = _text_or_break(image.previous_elements)
    after = _text_or_break(image.next_elements)
    left = _beside(before, block) and not str(before)[-1:].isspace()
    right = _beside(after, block) and str(after)[:1] not in _NO_SPACE_BEFORE
    return f"{' ' * left}{alt}{' ' * right}"


def _office_made_up(alt: str) -> bool:
    """Whether Office made up the alt text ``alt``, by its markers in any case."""
    folded = alt.casefold()
    return any(marker.casefold() in folded for marker in _OFFICE_ALT_TEXT_MARKERS)


def _text_or_break(elements: Iterator[PageElement]) -> PageElement | None:
    """The first of ``elements`` that is text or a line break, which ends a line."""
    for element in elements:
        if isinstance(element, NavigableString):
            return element
        if isinstance(element, Tag) and element.name == "br":
            return element
    return None


def _block(element: PageElement) -> Tag | None:
    """The element that holds ``element``'s line: its nearest that isn't inline."""
    return next((tag for tag in element.parents if tag.name not in _INLINE), None)


def _beside(text: PageElement | None, block: Tag | None) -> bool:
    """Whether ``text`` is text on a line of ``block``'s."""
    return isinstance(text, NavigableString) and _block(text) is block


def _page_header(soup: BeautifulSoup) -> str:
    """The lines that say what a web page is: its title, author, publication and update
    dates and address, each that it gives, as ``Title: …``.

    Each is read from the page's JSON-LD (:func:`_json_ld_object`), else its meta tags
    (OpenGraph's, ``article:*`` and ``author``), else its ``<title>``, or for its address
    its canonical link, then ``og:url``. An author is a name, never a URL, and each of
    several ``article:author`` tags is one.
    """
    described = _json_ld_object(soup)
    metas = _metas(soup)
    title = soup.find(_is_page_title)
    canonical = soup.find("link", rel="canonical")
    lines = {
        "Title": _first(
            _one_line(described.get("name")),
            _one_line(described.get("headline")),
            *metas.get("og:title", []),
            _one_line(title.get_text()) if title else None,
        ),
        "Author": _first(
            _authors(described.get("author")),
            _authors(metas.get("article:author")),
            _authors(metas.get("author")),
        ),
        "Published": _first(
            _one_line(described.get("datePublished")),
            *metas.get("article:published_time", []),
        ),
        "Updated": _first(
            _one_line(described.get("dateModified")),
            *metas.get("article:modified_time", []),
        ),
        "Address": _first(
            _one_line(described.get("url")),
            _one_line(canonical.get("href")) if canonical else None,
            *metas.get("og:url", []),
        ),
    }
    return "\n".join(f"{label}: {value}" for label, value in lines.items() if value)


def _json_ld_object(soup: BeautifulSoup) -> dict[str, Any]:
    """The one object a web page's JSON-LD describes it as: its article, if it has just
    one, else its page, else none (``{}``), from a block's top level, a list there, or
    its ``@graph``. A page that lists several articles, each a teaser, is not one of
    them. A block that isn't JSON is skipped."""
    found: list[Any] = []
    for block in soup("script", type="application/ld+json"):
        try:
            data = json.loads(block.get_text())
        except (ValueError, RecursionError):
            continue
        for item in _listed(data):
            graph = item.get("@graph") if isinstance(item, dict) else None
            found += [item, *_listed(graph)]
    objects = [obj for obj in found if isinstance(obj, dict)]
    articles = [obj for obj in objects if _has_type(obj, _ARTICLE_TYPES)]
    pages = [obj for obj in objects if _has_type(obj, _PAGE_TYPES)]
    return articles[0] if len(articles) == 1 else next(iter(pages), {})


def _has_type(obj: dict[str, Any], kinds: tuple[str, ...]) -> bool:
    """Whether the JSON-LD object ``obj`` has a type whose name ends as one of ``kinds``."""
    return any(
        isinstance(kind, str) and kind.endswith(kinds)
        for kind in _listed(obj.get("@type"))
    )


def _metas(soup: BeautifulSoup) -> dict[str, list[str]]:
    """A web page's meta tags' contents, by their property or name in lower case, in the
    page's order."""
    metas: dict[str, list[str]] = {}
    for meta in soup("meta"):
        key = meta.get("property") or meta.get("name")
        content = _one_line(meta.get("content"))
        if isinstance(key, str) and content:
            metas.setdefault(key.lower(), []).append(content)
    return metas


def _authors(value: Any) -> str | None:
    """The names of the authors ``value`` gives, as JSON-LD or meta tags do: a name, an
    object with a name, or a list of those; a URL is not a name."""
    named = [
        _one_line(author.get("name") if isinstance(author, dict) else author)
        for author in _listed(value)
    ]
    names = [name for name in named if name and not _URL.match(name)]
    return ", ".join(names) or None


def _listed(value: Any) -> list[Any]:
    """``value``, if it is a list, else a list of it alone."""
    return value if isinstance(value, list) else [value]


def _one_line(value: Any) -> str | None:
    """``value`` as one line, if it is text that isn't blank."""
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or None


def _first(*values: str | None) -> str | None:
    """The first of ``values`` that is given."""
    return next((value for value in values if value), None)


def _hides(hidden: str | None) -> bool:
    """Whether a ``hidden`` attribute of this value hides its tag from a reader: any value
    but ``until-found``, whose text the browser's find in page reveals."""
    return hidden is not None and hidden.lower() != _UNTIL_FOUND


def _page_text(page: bytes) -> str:
    """A web page's bytes as text, decoded by its byte-order mark, else by the charset it
    declares, read as a browser reads it (:data:`_DECLARED_READ_AS`), else as UTF-8, as
    is a page that declares a charset Python doesn't know. A byte that doesn't decode
    becomes U+FFFD. Nothing is guessed: a guess turns a short page into mojibake."""
    page, marked = EncodingDetector.strip_byte_order_mark(page)
    if marked:
        return page.decode(marked, "replace")
    declared = EncodingDetector.find_declared_encoding(page, is_html=True) or "utf-8"
    try:
        codec = codecs.lookup(declared).name
        return page.decode(_DECLARED_READ_AS.get(codec, codec), "replace")
    # A name Python doesn't know or can't look up, or one of a codec that isn't for text.
    except (LookupError, ValueError):
        return page.decode("utf-8", "replace")


def _page_markdown(name: str, soup: BeautifulSoup) -> str:
    """``soup``'s Markdown, or, for a page nested deeper than markdownify can recurse,
    its text alone."""
    try:
        return _Markdown(strip=["a"]).convert_soup(soup)
    except RecursionError:
        _log.warning("%s: nested too deep for Markdown; reading its text alone", name)
        return _one_blank_line(soup.get_text("\n"))


def _one_blank_line(markdown: str) -> str:
    """``markdown`` with each run of blank or whitespace-only lines outside a fenced code
    block made one blank line."""
    lines: list[str] = []
    fenced = False
    for line in markdown.split("\n"):
        if line.lstrip().startswith(_FENCE):
            fenced = not fenced
        blank = not fenced and not line.strip()
        if blank and lines and not lines[-1]:
            continue
        lines.append("" if blank else line)
    return "\n".join(lines)


class _Markdown(MarkdownConverter):
    """r3con's Markdown, of a Word file or a web page: ATX headings, a table's first row
    its header, a ``|`` in a table's cell escaped so it can't split the cell, and no more
    than one blank line in a row outside fenced code."""

    class Options:
        heading_style = ATX
        table_infer_header = True
        # The text kept as written, so the answer's quotes match it.
        escape_asterisks = False
        escape_underscores = False
        escape_misc = False

    def convert_soup(self, soup: Any) -> str:
        return _one_blank_line(super().convert_soup(soup))

    # markdownify's stub leaves out the convert_* hooks its README says to override.
    def convert_td(self, el: Any, text: str, parent_tags: set[str]) -> str:
        cell = text.replace("|", r"\|")
        return super().convert_td(el, cell, parent_tags)  # pyright: ignore[reportAttributeAccessIssue]

    def convert_th(self, el: Any, text: str, parent_tags: set[str]) -> str:
        cell = text.replace("|", r"\|")
        return super().convert_th(el, cell, parent_tags)  # pyright: ignore[reportAttributeAccessIssue]


_EXTRACTORS: dict[str, Callable[[str, BinaryIO], str]] = {
    ".pdf": _pdf_text,
    ".docx": _docx_text,
    ".html": _html_text,
    ".htm": _html_text,
}

# Which files a folder walk picks up. A walk that swallowed binaries would produce
# garbage documents and confusing model errors, so keep it to files we can turn into
# text — the text-ish ones, read directly, plus those in _EXTRACTORS. Pass explicit paths
# for anything else.
TEXT_SUFFIXES: frozenset[str] = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".rst",
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".yaml",
        ".yml",
        ".xml",
        *_EXTRACTORS,
    }
)


def _extractor(name: str) -> Callable[[str, BinaryIO], str] | None:
    """How to extract the file called ``name``, by the suffix it ends with, so that a
    file called just ``.docx`` is a Word file; or ``None`` if it is read as text."""
    lower = name.lower()
    return next(
        (f for suffix, f in _EXTRACTORS.items() if lower.endswith(suffix)), None
    )


def _left_out_unsaid(parts: Sequence[str]) -> bool:
    """Whether a folder walk leaves out, without a word, the file at ``parts`` below the
    folder: Office's lock file, or one that is hidden or under a hidden folder or
    ``__MACOSX``."""
    return parts[-1].startswith(_LOCK_FILE_PREFIX) or any(
        part.startswith(".") or part == _MACOS_METADATA_FOLDER for part in parts
    )


def _walk_reads(name: str) -> bool:
    """Whether a folder walk reads a file called ``name`` by its suffix: one in
    :data:`TEXT_SUFFIXES`."""
    return name.lower().endswith(tuple(TEXT_SUFFIXES))


def _shell_matches(pattern: Sequence[str], parts: Sequence[str]) -> bool:
    """Whether the shell's glob matches the path ``parts`` to ``pattern``, part by part:
    as pathlib's does, but ``**`` takes no hidden name, nor does any other part that
    doesn't start with a dot."""
    if not pattern:
        return not parts
    if pattern[0] == _RECURSIVE_WILDCARD:
        return _shell_matches(pattern[1:], parts) or (
            bool(parts)
            and not parts[0].startswith(".")
            and _shell_matches(pattern, parts[1:])
        )
    return (
        bool(parts)
        and fnmatch(parts[0], pattern[0])
        and (pattern[0].startswith(".") or not parts[0].startswith("."))
        and _shell_matches(pattern[1:], parts[1:])
    )


def _resolves_inside(path: Path, root: Path) -> bool:
    """Whether ``path`` resolves inside ``root``, itself resolved; logs it if not."""
    if path.resolve().is_relative_to(root):
        return True
    _log.warning("%s: resolves outside %s; skipping it", path, root)
    return False


def _utf8(name: str) -> str:
    """``name`` as UTF-8 can write it: each byte of a file's name that is not UTF-8,
    which Python reads as half a surrogate pair, becomes U+FFFD."""
    return name.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


def _files_at(src: str | Path) -> list[tuple[str, Path]]:
    """The files the source ``src`` names, as :func:`read_documents` finds them, each
    with its name (:func:`_utf8`): its path below the folder walked, or below the glob's
    part before its first wildcard, or, for a file named directly, its file name."""
    path = Path(src)
    if path.is_dir():
        root = path.resolve()
        files = sorted(
            p
            for p in path.rglob("*")
            if not _left_out_unsaid(p.relative_to(path).parts)
            and _walk_reads(p.name)
            and p.is_file()
            and _resolves_inside(p, root)
        )
        return [(_utf8(p.relative_to(path).as_posix()), p) for p in files]
    if path.is_file():
        return [(_utf8(path.name), path)]
    # Otherwise a glob, anchored if it is absolute.
    pattern, anchor = str(src), path.anchor
    base = Path(*takewhile(_GLOB_WILDCARDS.isdisjoint, path.parts))
    root, below = base.resolve(), len(base.parts)
    files = sorted(
        p
        for p in (
            Path(anchor).glob(pattern[len(anchor) :])
            if anchor
            else Path().glob(pattern)
        )
        if not p.name.startswith(_LOCK_FILE_PREFIX)
        and _shell_matches(path.parts[below:], p.parts[below:])
        and p.is_file()
        and _resolves_inside(p, root)
    )
    return [(_utf8("/".join(p.parts[below:])), p) for p in files]


def unique_names(names: Sequence[str]) -> list[str]:
    """``names``, in order, with each one already met numbered so that no two are
    alike: the second ``notes.txt`` becomes ``notes.txt (2)``, the third ``notes.txt
    (3)``, a number that would give a name already in ``names`` skipped.

    :func:`read_documents` numbers the names of documents read from one path so, and
    r3con app the names a page sends twice.
    """
    taken = set(names)
    seen: set[str] = set()
    next_number: dict[str, int] = {}
    unique: list[str] = []
    for name in names:
        if name in seen:
            start = next_number.get(name, _FIRST_REPEAT)
            number = next(k for k in count(start) if f"{name} ({k})" not in taken)
            next_number[name] = number + 1
            name = f"{name} ({number})"
            taken.add(name)
        seen.add(name)
        unique.append(name)
    return unique


def _apart(names: Sequence[str], paths: Sequence[str]) -> list[str]:
    """``names``, each that another one also is replaced by its document's path as read
    (in ``paths``), again until none clash but those of documents read from one path,
    which are then numbered (:func:`unique_names`)."""
    names = list(names)
    while True:
        given = Counter(names)
        clashing = [
            i for i, name in enumerate(names) if given[name] > 1 and name != paths[i]
        ]
        if not clashing:
            return unique_names(names)
        for i in clashing:
            names[i] = paths[i]


def read_documents(source: str | Path | Iterable[str | Path]) -> dict[str, str]:
    """Read documents off the filesystem into the ``{name: text}`` mapping
    :func:`~r3con.r3con.run` takes.

    This is **only** a filesystem reader — it never treats its argument as document text.
    Each source may be a directory (every text file under it, recursively, sorted by
    path), a single file, or a glob such as ``"reports/*.md"``; pass several and they are
    concatenated in the order given.

    Each document is named by its path below the folder you named, subfolders included,
    with ``/`` between them (``read_documents("archive")`` gives
    ``"2021/jan/front-page.txt"``); for a glob, below its part before its first wildcard
    (``"reports/**/*.md"`` gives ``"q3/summary.md"``); and for a file named directly, by
    its file name. Two documents given the same name, from two sources, are each named
    by their path as read instead, the source as you gave it in front (``a/notes.txt``,
    ``b/notes.txt``), and so on until no name clashes but those of documents read from
    one path, as when one file is read twice: they are numbered from the second on
    (``a/notes.txt (2)``, :func:`unique_names`). A byte of a file's name that is not
    UTF-8 is named U+FFFD (``caf\ufffd.txt``).

    A directory walk picks up only the suffixes in :data:`TEXT_SUFFIXES` — text-ish
    files, PDFs, Word files (``.docx``) and web pages (``.html``, ``.htm``). An explicit
    path or a glob reads **whatever you name**, so that a ``.log``, a ``.py`` or an
    extensionless file is yours to pass; the flip side is that a glob aimed at binaries
    will happily turn them into garbage documents. A walk and a glob both skip Office's
    lock files (``~$report.docx``). A walk skips anything hidden or under ``__MACOSX``
    below the folder you named; a glob follows the shell, where a wildcard matches no
    hidden name but a name in the pattern that starts with a dot does (``"d/.*"``,
    ``"**/.env"``). Both read only files that resolve inside the folder you named, or
    the glob's part before its first wildcard: a file that a link or a ``..`` takes
    outside it is logged and skipped, so a folder you received can't point r3con at
    files elsewhere. What you name, a file or that folder, is taken as named, even if
    hidden or a link.

    PDFs are extracted page by page. A PDF with no text layer — a scan — yields nothing,
    as does one encrypted so that it won't open, as with a password, or one that can't
    be read: each is logged, saying which, and dropped; OCR is outside this package.
    A Word file or a web page is converted to Markdown, its tables included, a web page
    under the title, author, dates and address its metadata gives; one that can't be
    read, or has no text, is logged and dropped. Everything else is read as UTF-8,
    replacing undecodable bytes rather than failing: one bad byte in one document should
    not sink a run. Empty documents are dropped — an empty document contributes nothing
    and costs a relevance call. A dropped file is not in the mapping, nor counted when
    names clash.

    The mapping holds the documents in the order read, and that order is preserved end
    to end (it is what ``Document N`` refers to in the structuring prompts, and
    ``document_ids`` and ``document_names`` align with it), so a stable, sorted walk
    matters: ``list(documents.values())[i]`` is the i-th document.

    Raises:
        FileNotFoundError: if a source matches nothing.
    """
    sources = [source] if isinstance(source, (str, Path)) else list(source)
    found: list[tuple[str, Path]] = []
    for src in sources:
        files = _files_at(src)
        if not files:
            raise FileNotFoundError(f"No documents found at {src!r}.")
        found.extend(files)
    read = [(n, p, text) for n, p in found if (text := _read_one(p)).strip()]
    names = _apart([n for n, _, _ in read], [_utf8(p.as_posix()) for _, p, _ in read])
    return dict(zip(names, [text for _, _, text in read], strict=True))
