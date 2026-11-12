"""The public entry point: ask a question over a collection of documents, get an answer.

This is the whole surface most callers need::

    from r3con import r3con

    result = r3con.run("Who approved the Q3 budget increase?", [memo_a, memo_b, memo_c])
    print(result)                 # the answer text
    result.relevant_context              # what each document was found to contribute
    result.structured_context            # the structured parse the answer was computed over
    result.provenance             # the snippets, parses, quotes and trail behind it
    result.schema_code            # the schema proposed for this question

The answer cites its two views separately: the relevance snippets it used
(``result.provenance["summary_ids"]``) and the documents whose parsed records it used
(``result.provenance["parse_ids"]``). Each document's relevance snippet and its parse
also record the snippets they used (``result.relevance_provenance``,
``result.parsing_provenance``). ``result.provenance["trail"]`` is what the answer
reaches through those citations, and ``result.provenance["quotes"]`` the sentences
behind it, the quotes on that trail (:attr:`~r3con.pipeline.Answer.provenance`,
:mod:`r3con.provenance`).

(``from r3con import run`` works too — same function, imported from the package root.)

:func:`ask` reruns only reasoning over a finished run, for a new question, and
:func:`trace` builds a saved answer's provenance again, under any rule, calling no
model.

``documents`` is the documents themselves: their texts, each under its name, or a plain
list of them. To read them off disk, :func:`read_documents` turns a folder, a glob or a
list of files into ``{name: text}``, each named by its path — one job each, so nothing
has to guess which you meant. ``result.document_names`` lines up with
``result.document_ids``, and each quote in ``result.provenance["quotes"]`` names its
document.

The pipeline itself lives in :mod:`r3con.pipeline`; this module only resolves inputs
and hands back the answer. Reach for :func:`r3con.pipeline.run_pipeline` directly when
you want to pass a prebuilt :class:`~r3con.config.RunConfig` or your own
:class:`~r3con.runs.TaskLogger`.

**Provider credentials work the way litellm's do.** ``model`` is a litellm model string,
and the key comes from the provider's own environment variable (``OPENAI_API_KEY``,
``ANTHROPIC_API_KEY``, ``GEMINI_API_KEY``, …) or a ``.env`` in your working directory.
r3con invents no environment variable of its own for credentials; ``api_key`` /
``api_base`` are here for the cases litellm itself takes them explicitly, such as a
self-hosted endpoint.
"""

from __future__ import annotations

import codecs
import io
import json
import threading
import warnings
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from fnmatch import fnmatch
from itertools import count, takewhile
from pathlib import Path
from typing import Any, BinaryIO

import mammoth
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning, XMLParsedAsHTMLWarning
from bs4.dammit import EncodingDetector
from bs4.element import PreformattedString
from dotenv import find_dotenv, load_dotenv
from markdownify import ATX, MarkdownConverter
from pypdf import PdfReader

from r3con.config import (
    DEFAULT_CONFIG,
    ReasoningPath,
    RunConfig,
    check_prompts,
    load_config,
)
from r3con.logging_setup import get_logger
from r3con.pipeline import (
    Answer,
    ask_config,
    ask_pipeline,
    names_and_texts,
    read_answer,
    read_representation,
    run_pipeline,
)
from r3con.provenance import DEFAULT_RULE, Provenance
from r3con.runs import TaskLogger, new_run_folder
from r3con.runtime.llm import Cancelled
from r3con.settings import active_logs_dir, settings_snapshot

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
# The schema.org types a page's JSON-LD describes it as, by how their names end: an
# article (Article, NewsArticle, BlogPosting, Report…) before a page (WebPage, FAQPage…).
_ARTICLE_TYPES = ("Article", "Posting", "Report")
_PAGE_TYPES = ("Page",)
_URL_STARTS = ("http://", "https://")
# What, starting the text after an image's alt text, needs no space before it.
_APART_ALREADY = frozenset(" \t\r\n.,;:!?)")
# Where markdownify starts and ends a code block.
_FENCE = "```"
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
    """A PDF's text, page by page, from a stream of its bytes; ``name`` is what the
    warnings call it.

    A PDF pypdf can't read gives ``""``, and so does one with no text layer (a scan),
    which the caller then drops: say so rather than letting it vanish. The fix is OCR,
    outside this package. Half a surrogate pair, which a font's ToUnicode map can give
    and no UTF-8 encoder takes, becomes U+FFFD.
    """
    try:
        text = "\n\n".join(page.extract_text() or "" for page in PdfReader(pdf).pages)
    except Exception as e:  # noqa: BLE001 — one unreadable file, not a dead run
        _log.warning(
            "%s: not readable as a PDF (%s); skipping it", name, type(e).__name__
        )
        return ""
    # pypdf decodes ToUnicode maps with surrogatepass, so a pair can come as two code
    # units, or half of one alone: UTF-16 joins the pairs and replaces the halves.
    text = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "replace")
    if not text.strip():
        _log.warning(
            "%s: no extractable text — a scanned PDF needs OCR first; skipping it", name
        )
    return text


def _docx_text(name: str, docx: BinaryIO) -> str:
    """A Word file's text as Markdown, from a stream of its bytes; ``name`` is what the
    warnings call it.

    mammoth turns its body into HTML, and markdownify that into Markdown: headings,
    lists and tables keep their shape, a table's first row is its header, and a tracked
    insertion stays while a deletion goes. The text is kept as written, its ``_`` and
    ``*`` unescaped, but for a ``|`` in a table's cell. Images are left out. A file that
    can't be read gives ``""``, and so does one with no text, which the caller then
    drops: say so.
    """
    try:
        # An image converter that makes no element, so no image's bytes are read.
        converted = mammoth.convert_to_html(docx, convert_image=lambda image: [])
        text = _Markdown().convert(converted.value)
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
    headings, lists and tables keep their shape. Above it go the lines of
    :func:`_page_header`, which say what the page is. A link keeps its text, not its
    URL, and an image its alt text, if it has one. Scripts, styles, ``<noscript>``,
    ``<template>``, ``<?xml?>`` and what is ``hidden`` are left out. A page nested too
    deep for that is read as its text alone. A file that can't be read gives ``""``, and
    so does one with no text, which the caller then drops: say so.
    """
    try:
        soup = BeautifulSoup(_page_text(html.read()), "html.parser")
        header = _page_header(soup)
        _keep_what_is_shown(soup)
        text = "\n\n".join(filter(None, [header, _page_markdown(name, soup)]))
    except Exception as e:  # noqa: BLE001 — one unreadable file, not a dead run
        _log.warning(
            "%s: not readable as HTML (%s); skipping it", name, type(e).__name__
        )
        return ""
    if not text.strip():
        _log.warning("%s: no extractable text; skipping it", name)
    return text


def _keep_what_is_shown(soup: BeautifulSoup) -> None:
    """Leave in ``soup`` only what a browser shows, with each image as its alt text, or
    nothing if it has none."""
    # What a browser doesn't show: <noscript> while scripts run, <template>, the page's
    # <title>, which the header holds, and an SVG's, which is a tooltip; what is hidden
    # but not until found, and the <?…?>, CDATA and declarations HTML reads as
    # comments, which markdownify keeps.
    unseen = [
        *soup(["noscript", "template", "title"]),
        *soup(hidden=_hides),
        *soup(string=lambda text: isinstance(text, PreformattedString)),
    ]
    for hidden in unseen:
        hidden.extract()
    for image in soup("img"):
        alt = _text(image.get("alt"))
        if not alt:
            image.extract()
            continue
        before, after = image.find_previous(string=True), image.find_next(string=True)
        image.replace_with(_set_apart(before, alt, after))


def _page_header(soup: BeautifulSoup) -> str:
    """The lines that say what a web page is: its title, author, publication and update
    dates and address, each that it gives, as ``Title: …``.

    Each is read from the page's JSON-LD (:func:`_described`), else its meta tags
    (OpenGraph's, ``article:*`` and ``author``), else its ``<title>``, or for its address
    its canonical link, then ``og:url``. An author is a name, never a URL.
    """
    described = _described(soup)
    metas = _metas(soup)
    title = soup.find(lambda tag: tag.name == "title" and not tag.find_parent("svg"))
    canonical = soup.find("link", rel="canonical")
    lines = {
        "Title": _first(
            *(_text(obj.get(key)) for obj in described for key in ("name", "headline")),
            metas.get("og:title"),
            _text(title.get_text()) if title else None,
        ),
        "Author": _first(
            *(_authors(obj.get("author")) for obj in described),
            _authors(metas.get("article:author")),
            _authors(metas.get("author")),
        ),
        "Published": _first(
            *(_text(obj.get("datePublished")) for obj in described),
            metas.get("article:published_time"),
        ),
        "Updated": _first(
            *(_text(obj.get("dateModified")) for obj in described),
            metas.get("article:modified_time"),
        ),
        "Address": _first(
            *(_text(obj.get("url")) for obj in described),
            _text(canonical.get("href")) if canonical else None,
            metas.get("og:url"),
        ),
    }
    return "\n".join(f"{label}: {value}" for label, value in lines.items() if value)


def _described(soup: BeautifulSoup) -> list[dict[str, Any]]:
    """The objects a web page's JSON-LD describes it as: each article, then each page
    (:data:`_ARTICLE_TYPES`, :data:`_PAGE_TYPES`), in the page's order, from a block's
    top level, a list there, or its ``@graph``. A block that isn't JSON is skipped."""
    found: list[dict[str, Any]] = []
    for block in soup("script", type="application/ld+json"):
        try:
            data = json.loads(block.get_text())
        except (ValueError, RecursionError):
            continue
        for item in data if isinstance(data, list) else [data]:
            graph = item.get("@graph") if isinstance(item, dict) else None
            found += [item, *(graph if isinstance(graph, list) else [graph])]
    objects = [obj for obj in found if isinstance(obj, dict)]
    return [
        obj
        for kinds in (_ARTICLE_TYPES, _PAGE_TYPES)
        for obj in objects
        if _is(obj, kinds)
    ]


def _is(obj: dict[str, Any], kinds: tuple[str, ...]) -> bool:
    """Whether the JSON-LD object ``obj`` has a type whose name ends as one of ``kinds``."""
    types = obj.get("@type")
    return any(
        isinstance(kind, str) and kind.endswith(kinds)
        for kind in (types if isinstance(types, list) else [types])
    )


def _metas(soup: BeautifulSoup) -> dict[str, str]:
    """A web page's meta tags' contents, by their property or name, in lower case: the
    first of each."""
    metas: dict[str, str] = {}
    for meta in soup("meta"):
        key = meta.get("property") or meta.get("name")
        content = _text(meta.get("content"))
        if isinstance(key, str) and content:
            metas.setdefault(key.lower(), content)
    return metas


def _authors(value: Any) -> str | None:
    """The names of the authors ``value`` gives, as JSON-LD or a meta tag does: a name,
    an object with a name, or a list of those; a URL is not a name."""
    named = [
        _text(author.get("name") if isinstance(author, dict) else author)
        for author in (value if isinstance(value, list) else [value])
    ]
    names = [name for name in named if name and not name.startswith(_URL_STARTS)]
    return ", ".join(names) or None


def _text(value: Any) -> str | None:
    """``value`` as one line, if it is text that isn't blank."""
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or None


def _set_apart(before: Any, alt: str, after: Any) -> str:
    """An image's ``alt`` text, with a space on each side where the text read before or
    after the image doesn't already set it apart, as the image's own room does."""
    left = "" if isinstance(before, str) and before[-1:].isspace() else " "
    right = "" if isinstance(after, str) and after[:1] in _APART_ALREADY else " "
    return f"{left}{alt}{right}"


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
    """Read documents off the filesystem into the ``{name: text}`` mapping :func:`run`
    takes.

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
    is logged and then dropped; OCR is outside this package. A Word file or a web page
    is converted to Markdown, its tables included, a web page under the title, author,
    dates and address its metadata gives; one that can't be read, or has no text, is
    logged and dropped. Everything else is read as UTF-8, replacing undecodable bytes
    rather than failing: one bad byte in one document should not sink a run. Empty
    documents are dropped — an empty document contributes nothing and costs a relevance
    call. A dropped file is not in the mapping, nor counted when names clash.

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


def run(
    question: str,
    documents: Mapping[str, str] | Sequence[str],
    *,
    model: str | None = None,
    config: str | RunConfig = DEFAULT_CONFIG,
    relevance_rounds: int | None = None,
    reasoning: ReasoningPath | None = None,
    params: dict[str, Any] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    completion: Callable[..., Any] | None = None,
    logs_dir: str | Path | None = None,
    save_artifacts: bool = True,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> Answer:
    """Run the pipeline: answer ``question`` over ``documents``.

    Args:
        question: what to ask. It is the task the whole pipeline is built for — the
            schema and the reasoning are proposed *from it*, so a specific question
            gets a much better apparatus than a vague one.
        documents: the document **texts**, each under its name, as
            :func:`read_documents` gives them, or as a sequence of strings, each then
            named by its index (``"0"``, ``"1"``, …, so ``documents[int(name)]`` is
            the document); no text may be blank, so that every index the run reports
            refers to *your* documents, in their order. The names are recorded, never
            sent to the model. A document too long for
            the model's window is read in parts.
        model: a litellm model string (e.g. ``"openai/gpt-6-luna"``,
            ``"anthropic/claude-sonnet-5-5"``, ``"hosted_vllm/Qwen/Qwen3.5-35B-A3B"``).
            Defaults to the config's.
        config: a bundled config name, or a :class:`RunConfig` you built yourself.
        relevance_rounds: overrides the config's.
        reasoning: overrides the config's reasoning path: ``"direct"``, one completion
            over the relevant context and the whole parse, or ``"coding"``, an agent
            that reads the parse with code. A direct run takes the coding path when its
            prompt does not fit the model's window.
        params: extra keyword arguments passed straight to ``litellm.completion`` —
            ``temperature``, ``top_p``, ``seed``, ``extra_body``, anything it accepts.
        api_key, api_base: passed through to litellm. Usually unnecessary: litellm reads
            the provider's own environment variable, and a ``.env`` in your working
            directory is loaded for you.
        completion: the callable that makes each request, defaulting to
            ``litellm.completion``. Supply your own when you already own the connection —
            a configured ``litellm.Router``'s ``.completion`` for fallbacks or load
            balancing, or a wrapper that adds caching, logging or an internal gateway.
            It is transport, so it does not change the run's identity. r3con calls it
            once per request and leaves any resending to it.
        logs_dir: where the run folder goes. Defaults to ``./logs``
            (or ``R3CON_LOGS_DIR``).
        save_artifacts: write the per-stage artifacts — the relevance snippets, the
            proposed schema, the parse, the reasoning transcript — into a run folder.
            On by default: they are how you see what the run actually did. Set ``False``
            to run without touching the filesystem.
        on_event: called with each progress event as it happens, a copy as a dict of
            JSON values with its ``type`` (``run_started``, ``relevance_started``, …,
            ``run_finished`` or ``run_failed``), its ``time`` and its fields, listed in
            :func:`~r3con.pipeline.run_pipeline`. It is called from worker threads,
            but never twice at once, so it needs no lock of its own; the workers wait
            for it, so it should return quickly (put the event on a queue, say). If it
            raises, the error is logged and the run goes on. The run folder's
            ``events.jsonl`` records the same events, whether or not it is given.
        cancel: a ``threading.Event`` that stops the run once it is set, from any
            thread. From then on no new model call is sent, no stage starts, and a
            wait to resend a rate-limited call ends at once; the stage under way, or
            the next one if the last call of one was in flight, raises
            :class:`Cancelled` and fails as any stage does, with ``run_failed``
            naming it and, with artifacts on, its folder holding the calls it made
            and ``error.txt``, beside the earlier stages' artifacts. What is already
            under way runs its course: a call already sent (litellm cannot stop one),
            which is recorded, so a run whose last call is in flight finishes as
            usual; the resends r3con does not make itself, a supplied
            ``completion``'s (a ``litellm.Router``'s, say) and those of
            ``num_retries``, ``max_retries`` or ``retry_policy`` in ``params``; and a
            step the coding path is running in its sandbox, to its end or its
            timeout.

    Returns:
        An :class:`~r3con.pipeline.Answer`: the answer text (``.answer``, and
        ``str()`` of the result), the ids of the relevance snippets and parsed records
        it used, its trail and the quotes on it (``.provenance``: ids from
        ``.document_ids``, a part's snippet as ``<id>.<k>``, whose document is the part
        before the dot, each quote with its document's name; see
        :attr:`~r3con.pipeline.Answer.provenance`), each document's name
        (``.document_names``), the relevant context
        (``.relevant_context``), the structured parse (``.structured_context``), the
        schema proposed for this question (``.schema_code``), and where the artifacts
        were written (``.run_dir``).

    Raises:
        TypeError: if ``documents`` is not a mapping of names to strings or a sequence
            of strings (a bare string or a path is rejected rather than guessed at).
        ValueError: if ``documents`` is empty or any entry is blank (a blank is refused,
            not dropped — dropping it would renumber every index the run reports), if
            ``config`` is a prebuilt :class:`RunConfig` *and* field overrides were also
            given (which would be silently ignored), if the config pins no prompt
            version for a stage, if an override is not valid for its setting (a
            ``reasoning`` that is not a reasoning path, say), or if a setting in
            :mod:`r3con.settings` is below its floor.
        FileNotFoundError: if the config, by name or as a :class:`RunConfig`, pins a
            prompt version that has no file.
        Each of these is raised before any request and before a run folder exists, as
        is anything :func:`~r3con.config.load_config` refuses.
        Anything the underlying stages raise — most usefully
        ``litellm.ContextWindowExceededError`` when the prompt and notes sent beside a
        document leave it no room in the model's window. With
        artifacts on, an exception from a stage carries a note naming the run folder,
        where the stage left its completed calls and ``error.txt``.
        Cancelled: once ``cancel`` is set, in place of the next model call or stage.
    """
    # `run` is the batteries-included path, so it loads the caller's .env (from THEIR
    # working directory) the way the CLI does. Importing r3con does not.
    load_dotenv(find_dotenv(usecwd=True))

    overrides = {
        "model": model,
        "relevance_rounds": relevance_rounds,
        "reasoning": reasoning,
        "params": params,
    }
    given = {k: v for k, v in overrides.items() if v is not None}
    if isinstance(config, RunConfig) and given:
        raise ValueError(
            f"`config` is already a RunConfig, so {sorted(given)} would be ignored — set "
            "them on the RunConfig instead."
        )

    # Refuse what run_pipeline would refuse, the documents and a RunConfig's prompts
    # included, before a run folder exists.
    names, texts = names_and_texts(documents)
    cfg = config if isinstance(config, RunConfig) else load_config(config, **given)
    check_prompts(cfg)
    settings_snapshot()

    task_logger = None
    if save_artifacts:
        root = Path(logs_dir) if logs_dir is not None else active_logs_dir()
        task_logger = TaskLogger(new_run_folder(), root=root)

    return run_pipeline(
        task=question,
        documents=dict(zip(names, texts, strict=True)),
        config=cfg,
        task_logger=task_logger,
        api_base=api_base,
        api_key=api_key,
        completion=completion,
        on_event=on_event,
        cancel=cancel,
    )


def ask(
    source: Answer | str | Path,
    question: str,
    *,
    model: str | None = None,
    reasoning: ReasoningPath | None = None,
    params: dict[str, Any] | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
    completion: Callable[..., Any] | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    cancel: threading.Event | None = None,
) -> Answer:
    """Answer a new ``question`` over what a finished run built, running only reasoning.

    The run's relevance snippets, schema and parse are reused as its folder holds them,
    so an ask costs one reasoning step rather than a run. Reasoning is sent what the
    run's was, with ``question`` in place of the run's question. Its prompts are the
    run's pins, resolved as a run resolves them (a ``./prompts`` overlay in the working
    directory included), and the ask's manifest records the files it used. It falls
    back to the coding path as a run's does.

    Args:
        source: an :class:`Answer` from :func:`run` or :func:`ask`, whose ``run_dir``
            is read, or the path of a run folder or of an ask's folder. An ask's folder
            stands for the run it asked over, so asking over an ask asks that run again.
        question: the question to answer; it may be the run's own.
        model: a litellm model string. Defaults to the one the run called; for a run
            recorded before r3con kept that string, its bare name (``gpt-6-luna``).
        reasoning: ``"direct"`` or ``"coding"``, as in :func:`run`. Defaults to the
            run's.
        params: as in :func:`run`. Defaults to the run's when the ask runs on the run's
            model (by its name without the provider prefix), and to none on another
            model, since one provider's params (``reasoning_effort``, ``extra_body``)
            can break another's request. A credential among the run's params is saved
            as ``REDACTED`` (:mod:`r3con.masking`), so pass its real value here.
        api_key, api_base, completion: as in :func:`run`.
        on_event: as in :func:`run`, called with the ask's events: ``ask_started``,
            ``reasoning_started``, ``reasoning_finished``, then ``ask_finished`` or
            ``ask_failed`` (:func:`~r3con.pipeline.ask_pipeline`).
        cancel: as in :func:`run`: the ask fails with ``ask_failed``, its folder
            holding the calls reasoning made and ``error.txt``.

    Each ask is written to a new folder in the run folder's ``asks/``, named as a run
    folder is: a ``manifest.json`` as a run writes one, for the run's config with this
    ask's overrides (its ``task`` the question, ``ask_of`` the run folder's name),
    ``events.jsonl`` and ``reasoning/`` as a run writes them. The run's own files are
    left as they are.

    Returns:
        An :class:`Answer` whose ``answer`` and ``provenance`` come from the new
        reasoning (its trail and quotes through the run's snippets of every round and
        its parses), and whose other fields are the run's; ``run_dir`` is the ask's own
        folder, which holds its manifest, calls and transcript.

    Raises:
        FileNotFoundError: if ``source`` is neither a run folder nor an ask's folder in
            the ``asks/`` of the run it asked over, if its run never finished parsing,
            or if a prompt version it pins has no file.
        ValueError: if ``source`` is an Answer without a ``run_dir`` (its run was made
            with ``save_artifacts=False``), if a file in its folder is not as this
            r3con writes it (one an older r3con wrote, say), if it leaves a prompt
            version unpinned, if an override is not valid for its setting, if a param
            it would send holds ``REDACTED``, at any depth, naming each one, or if a
            setting in :mod:`r3con.settings` is out of range.
        Each of these is raised in one line, before any request and before the ask's
        folder exists.
        Anything reasoning raises, with a note naming the ask's folder, where it left
        its completed calls and ``error.txt``; :class:`Cancelled` among them, once
        ``cancel`` is set.
    """
    load_dotenv(find_dotenv(usecwd=True))
    built = read_representation(_folder_of(source, "ask"))
    config = ask_config(built, model=model, reasoning=reasoning, params=params)
    return ask_pipeline(
        built,
        config,
        task=question,
        api_base=api_base,
        api_key=api_key,
        completion=completion,
        on_event=on_event,
        cancel=cancel,
    )


def trace(source: Answer | str | Path, rule: str = DEFAULT_RULE) -> Provenance:
    """The provenance of the answer ``source`` saved, its trail walked under ``rule``
    (:mod:`r3con.provenance`), built from the files of its folder alone.

    It calls no model, and reads neither the run's config nor its prompts, so a run
    whose config or prompts this r3con no longer runs is traced all the same.

    Args:
        source: an :class:`Answer` from :func:`run` or :func:`ask`, whose ``run_dir``
            is read, or the path of a run folder or of an ask's folder: an ask's answer
            is its own, and its snippets and parses those of the run it asked over.
        rule: the name of a rule in :data:`r3con.provenance.RULES`. Under the default,
            the provenance is the one the answer was saved with.

    Raises:
        ValueError: if no rule is named ``rule``, before anything is read; if
            ``source`` is an Answer without a ``run_dir``; or, in one line naming the
            folder that holds it, if a file it reads is not as this r3con writes it.
        FileNotFoundError: if ``source`` is neither a run folder nor an ask's folder in
            the ``asks/`` of the run it asked over, or it holds no answer, as when its
            reasoning failed.
    """
    return read_answer(_folder_of(source, "trace"), rule)


def _folder_of(source: Answer | str | Path, reader: str) -> Path:
    """The folder ``source`` names for ``reader``: an Answer's ``run_dir``, or the path
    given.

    Raises:
        ValueError: if ``source`` is an Answer without a ``run_dir``.
    """
    if not isinstance(source, Answer):
        return Path(source)
    if source.run_dir is None:
        raise ValueError(
            f"{reader} needs the run folder, and this Answer has none: its run was "
            "made with save_artifacts=False."
        )
    return source.run_dir


__all__ = [
    "TEXT_SUFFIXES",
    "Answer",
    "Cancelled",
    "ask",
    "read_documents",
    "run",
    "trace",
    "unique_names",
]
