import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from r3con.documents import read_documents, unique_names
from tests.app.bundle_check import encrypted_pdf
from tests.documents import (
    CORRUPT_DOCX,
    LINE_BREAK,
    paragraph,
    picture,
    table,
    text_run,
    write_docx,
    write_pdf,
    write_tree,
)

# A paragraph whose second run Word tracks as inserted, and third as deleted.
TRACKED_CHANGES = (
    '<w:p><w:r><w:t xml:space="preserve">Revenue was </w:t></w:r>'
    '<w:ins w:id="1" w:author="a"><w:r><w:t>12 million</w:t></w:r></w:ins>'
    '<w:del w:id="2" w:author="a"><w:r><w:delText>9 million</w:delText></w:r></w:del>'
    '<w:r><w:t xml:space="preserve"> in Q3.</w:t></w:r></w:p>'
)
# A content control around a paragraph, and one around a run.
CONTENT_CONTROLS = (
    f"<w:sdt><w:sdtContent>{paragraph('Inside a block control.')}</w:sdtContent></w:sdt>"
    '<w:p><w:r><w:t xml:space="preserve">Name: </w:t></w:r>'
    "<w:sdt><w:sdtContent><w:r><w:t>Jane Doe</w:t></w:r></w:sdtContent></w:sdt></w:p>"
)
MERGED_HEADER = (
    '<w:tbl><w:tr><w:tc><w:tcPr><w:gridSpan w:val="3"/></w:tcPr>'
    f"{paragraph('Total')}</w:tc></w:tr><w:tr>"
    + "".join(f"<w:tc>{paragraph(text)}</w:tc>" for text in ["Site A", "Q2", "5"])
    + "</w:tr></w:tbl>"
)
NESTED_TABLE = (
    f"<w:tbl><w:tr><w:tc>{paragraph('outer')}</w:tc>"
    f"<w:tc>{table([['inner1', 'inner2']])}{paragraph('')}</w:tc></w:tr></w:tbl>"
)
PLAIN_TEXT = b"Plain words, saved under the wrong name."
# Curly quotes and a dash, which windows-1252 has and Latin-1 does not.
QUOTED = "“Incidents” — 7, at Merrow’s site."
NOT_A_WORD_FILE = "not readable as a Word file (BadZipFile)"
# What a person wrote to describe a picture in a Word file.
CHART_DESCRIPTION = "A chart of incidents by site"
Q3_ADDRESS = "https://example.com/q3"
# Deeper than markdownify can recurse: each level takes it at least one call.
NESTED_DEEPER_THAN_RECURSION = sys.getrecursionlimit()
# Deeper than Python's JSON decoder goes, on every version r3con runs on.
JSON_DEPTH = 100_000
JSON_LD_NESTED_TOO_DEEP = (
    f'<script type="application/ld+json">{"[" * JSON_DEPTH}{"]" * JSON_DEPTH}</script>'
)
# A page a script fills in: metadata, and nothing to read until the script runs.
APP_SHELL = (
    b'<title>Q3</title><link rel="canonical" href="https://example.com/q3">'
    b'<body><div id="root"></div></body>'
)
# What Word writes beside a document someone has open: their name, padded.
WORD_OWNER_FILE = b"\x07Michael" + bytes(154)
# Copies of one name that took a quadratic numbering seconds to tell apart.
MANY_COPIES = 20_000
AT_ONCE_SECONDS = 1


def write_files(texts: dict[str, str]) -> None:
    """Each of ``texts``' files, by its path from the working directory."""
    for name, text in texts.items():
        Path(name).parent.mkdir(parents=True, exist_ok=True)
        Path(name).write_text(text, encoding="utf-8")


def json_ld(data: Any) -> str:
    """``data`` as a web page's JSON-LD block."""
    return f'<script type="application/ld+json">{json.dumps(data)}</script>'


def meta(key: str, content: str) -> str:
    """A web page's meta tag for the property ``key``, as OpenGraph writes one."""
    return f'<meta property="{key}" content="{content}">'


def test_a_directory_is_read_recursively_in_path_order_skipping_binaries_and_blanks(
    tmp_path,
):
    write_tree(tmp_path)
    assert list(read_documents(tmp_path).items()) == [
        ("a.md", "alpha document"),
        ("b.txt", "beta document"),
        ("nested/c.md", "gamma document"),
    ]


def test_pdfs_are_extracted_and_one_that_cannot_be_read_is_dropped_saying_why(
    tmp_path, caplog
):
    write_pdf(tmp_path / "a_memo.pdf", "Incidents this quarter: 7.")
    write_pdf(tmp_path / "b_scanned.pdf")
    (tmp_path / "c_corrupt.pdf").write_bytes(b"%PDF-1.4 truncated garbage")
    write_pdf(tmp_path / "c_locked.pdf", "A locked memo.", password="secret")
    (tmp_path / "d_plain.txt").write_text("a plain memo", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        documents = read_documents(tmp_path)
    assert list(documents) == ["a_memo.pdf", "d_plain.txt"]
    assert "Incidents this quarter: 7." in documents["a_memo.pdf"]
    assert documents["d_plain.txt"] == "a plain memo"
    said = {
        "b_scanned.pdf": "no extractable text — likely a scan, which needs OCR first",
        "c_corrupt.pdf": "not readable as a PDF (PdfStreamError)",
        "c_locked.pdf": "password-protected (FileNotDecryptedError)",
    }
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"{tmp_path / name}: {why}; skipping it" for name, why in said.items()
    ]


# pypdf picks its AES library on import: cryptography if it imports, else pycryptodome.
READ_DOCUMENTS_WITHOUT = """
import json, sys
sys.modules.update(dict.fromkeys(sys.argv[2:]))
from r3con import read_documents
print(json.dumps(read_documents(sys.argv[1])))
"""


@pytest.mark.parametrize(
    "blocked", [[], ["cryptography"]], ids=["as_installed", "without_cryptography"]
)
def test_an_aes_256_pdf_with_an_empty_password_is_read_with_either_aes_library(
    tmp_path, blocked
):
    pdf = tmp_path / "ordinance.pdf"
    pdf.write_bytes(encrypted_pdf("Incidents this quarter: 7."))
    read = subprocess.run(
        [sys.executable, "-c", READ_DOCUMENTS_WITHOUT, str(pdf), *blocked],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(read.stdout) == {"ordinance.pdf": "Incidents this quarter: 7."}


def test_a_word_files_paragraphs_and_tables_come_out_in_order_as_markdown(tmp_path):
    docx = tmp_path / "report.docx"
    table = [["Site", "Incidents"], ["Halloran", "7"]]
    write_docx(docx, ["Incidents by site:", table, "Merrow's count is provisional."])
    lines = [
        "Incidents by site:",
        "",
        "| Site | Incidents |",
        "| --- | --- |",
        "| Halloran | 7 |",
        "",
        "Merrow's count is provisional.",
    ]
    assert read_documents(docx) == {"report.docx": "\n".join(lines)}


@pytest.mark.parametrize(
    ("xml", "text"),
    [
        (TRACKED_CHANGES, "Revenue was 12 million in Q3."),
        (CONTENT_CONTROLS, "Inside a block control.\n\nName: Jane Doe"),
        (
            MERGED_HEADER,
            "| Total | | |\n| --- | --- | --- |\n| Site A | Q2 | 5 |",
        ),
    ],
    ids=[
        "tracked-insertion-kept-deletion-dropped",
        "content-controls",
        "merged-header",
    ],
)
def test_a_word_files_text_is_what_word_shows(tmp_path, xml, text):
    docx = tmp_path / "report.docx"
    write_docx(docx, xml=xml)
    assert read_documents(docx) == {"report.docx": text}


def test_a_word_files_underscores_and_asterisks_are_kept_as_written(tmp_path):
    docx = tmp_path / "report.docx"
    write_docx(docx, ["account_id_2024 is *important* 5*3"])
    assert read_documents(docx) == {"report.docx": "account_id_2024 is *important* 5*3"}


@pytest.mark.parametrize("header", [False, True], ids=["plain-rows", "header-row"])
def test_a_pipe_in_a_word_tables_cell_is_escaped_so_each_row_keeps_its_cells(
    tmp_path, header: bool
):
    docx = tmp_path / "report.docx"
    write_docx(docx, xml=table([["Name|Alias", "Value"], ["a|b", "7"]], header=header))
    [text] = read_documents(docx).values()
    assert text == "| Name\\|Alias | Value |\n| --- | --- |\n| a\\|b | 7 |"
    cells = {len(re.split(r"(?<!\\)\|", line)) for line in text.splitlines()}
    assert cells == {len(["", "Name|Alias", "Value", ""])}


def test_a_file_called_just_docx_is_a_word_file_to_its_path_and_hidden_to_a_walk(
    tmp_path,
):
    write_docx(tmp_path / ".docx", ["Incidents this quarter: 7."])
    assert read_documents(tmp_path / ".docx") == {".docx": "Incidents this quarter: 7."}
    with pytest.raises(FileNotFoundError, match="No documents found at"):
        read_documents(tmp_path)


def test_a_table_nested_in_a_word_table_keeps_its_text(tmp_path):
    docx = tmp_path / "report.docx"
    write_docx(docx, xml=NESTED_TABLE)
    [text] = read_documents(docx).values()
    assert all(word in text for word in ["outer", "inner1", "inner2"])


@pytest.mark.parametrize(
    ("description", "text"),
    [
        (CHART_DESCRIPTION, "Before the chart. A chart of incidents by site After."),
        (
            (
                "A picture containing text, screenshot, font\n\n"
                "Description automatically generated"
            ),
            "Before the chart. After.",
        ),
        (
            "A close up of a logo\n\nDescription generated with high confidence",
            "Before the chart. After.",
        ),
        (
            "A bar chart of incidents\n\nAI-generated content may be incorrect.",
            "Before the chart. After.",
        ),
        (None, "Before the chart. After."),
    ],
    ids=[
        "written-by-a-person-kept",
        "office-generated-dropped",
        "office-generated-with-confidence-dropped",
        "ai-generated-dropped",
        "no-description-dropped",
    ],
)
def test_a_word_files_picture_is_its_description_where_it_was_unless_office_wrote_it(
    tmp_path, description: str | None, text: str
):
    docx = tmp_path / "report.docx"
    runs = [text_run("Before the chart."), picture(description), text_run(" After.")]
    write_docx(docx, xml=f"<w:p>{''.join(runs)}</w:p>")
    assert read_documents(docx) == {"report.docx": text}


def test_a_word_files_picture_gives_its_description_without_its_bytes_being_read(
    tmp_path,
):
    docx = tmp_path / "report.docx"
    pictured = f"<w:p>{picture(CHART_DESCRIPTION)}</w:p>"
    # Its bytes left out of the file, so reading them would fail.
    write_docx(docx, ["By site:"], xml=pictured, picture_bytes=False)
    assert read_documents(docx) == {"report.docx": f"By site:\n\n{CHART_DESCRIPTION}"}


def test_a_word_files_picture_between_line_breaks_takes_no_spaces(tmp_path):
    docx = tmp_path / "report.docx"
    runs = [text_run("Figure 1"), LINE_BREAK, picture("A chart"), LINE_BREAK]
    write_docx(docx, xml=f"<w:p>{''.join(runs)}{text_run('Next')}</w:p>")
    assert read_documents(docx) == {"report.docx": "Figure 1  \nA chart  \nNext"}


@pytest.mark.parametrize(
    ("name", "content", "logged"),
    [
        ("a_memo.docx", CORRUPT_DOCX, NOT_A_WORD_FILE),
        ("a_memo.docx", PLAIN_TEXT, NOT_A_WORD_FILE),
        ("a_page.html", b"<script>alert('Merrow')</script>", "no extractable text"),
        ("a_page.html", APP_SHELL, "no extractable text"),
    ],
    ids=[
        "corrupt-docx",
        "plain-text-docx",
        "html-with-only-a-script",
        "html-with-only-metadata",
    ],
)
def test_a_file_that_cannot_be_read_or_has_no_text_is_logged_and_dropped(
    tmp_path, caplog, name: str, content: bytes, logged: str
):
    unreadable = tmp_path / name
    unreadable.write_bytes(content)
    (tmp_path / "b_plain.txt").write_text("a plain memo", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        documents = read_documents(tmp_path)
    assert documents == {"b_plain.txt": "a plain memo"}
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"{unreadable}: {logged}; skipping it"
    ]


@pytest.mark.parametrize(
    ("html", "text"),
    [
        ("<h1>Incidents</h1><h2>By site</h2>", "# Incidents\n\n## By site"),
        (
            "<p>Revenue was <b>12 million</b> in Q3.</p>",
            "Revenue was **12 million** in Q3.",
        ),
        ("<ul><li>Halloran</li><li>Merrow</li></ul>", "* Halloran\n* Merrow"),
        (
            (
                "<table><tr><th>Name|Alias</th><th>Value</th></tr>"
                "<tr><td>a|b</td><td>7</td></tr></table>"
            ),
            "| Name\\|Alias | Value |\n| --- | --- |\n| a\\|b | 7 |",
        ),
        (
            '<p>See <a href="https://example.com/q3">the Q3 report</a>.</p>',
            "See the Q3 report.",
        ),
        (
            '<p>Before the chart.<img src="chart.png" alt="Incidents by site"> After.</p>',
            "Before the chart. Incidents by site After.",
        ),
        (
            '<p>A chart<img src="a.png" alt=""> and<img src="b.png" alt="  "> a map.</p>',
            "A chart and a map.",
        ),
        (
            '<p><a href="https://example.com/q3"><img alt="The Q3 report"></a>.</p>',
            "The Q3 report.",
        ),
        ('<a href="/"><img alt="Example"></a><h1>Q3</h1>', "Example\n\n# Q3"),
        (
            '<p>Q3.</p><footer><a href="/about"><img alt="About us"></a></footer>',
            "Q3.\n\nAbout us",
        ),
        ('<pre>x<img alt="+">y</pre>', "```\nx+y\n```"),
        (
            '<p>Figure 1<br><img alt="A chart"><br>Next</p>',
            "Figure 1  \nA chart  \nNext",
        ),
        (
            (
                "<p>Before the chart.<img alt='Diagram, timeline&#10;&#10;"
                "description automatically generated'> After.</p>"
            ),
            "Before the chart. After.",
        ),
        (
            '<p><a href="/q3.pdf"><svg><title>Download the Q3 report</title></svg></a></p>',
            "Download the Q3 report",
        ),
        (
            (
                "<script>var site = 'Merrow';</script><style>p { color: red }</style>"
                "<noscript>Turn on JavaScript.</noscript>"
                "<template><p>Later.</p></template><p>Incidents: 7.</p>"
            ),
            "Incidents: 7.",
        ),
        (
            '<?xml version="1.0"?><!DOCTYPE html><html><!-- a note --><p>Incidents: 7.',
            "Incidents: 7.",
        ),
        ("<nav>Home</nav>\n<main><h1>Incidents</h1></main>", "Home\n\n# Incidents"),
        (
            "<p>Incidents: 7.<br><br><br><br>Merrow: 3.</p>",
            "Incidents: 7.  \n\nMerrow: 3.",
        ),
        (
            "<pre>def a():\n    pass\n\n\ndef b():\n    pass</pre>",
            "```\ndef a():\n    pass\n\n\ndef b():\n    pass\n```",
        ),
        (
            (
                "<div><p>We use some essential cookies.</p>"
                "<p hidden>You have accepted additional cookies.</p>"
                '<p hidden="">You have rejected additional cookies.</p></div>'
                "<p>Incidents: 7.</p>"
            ),
            "We use some essential cookies.\n\nIncidents: 7.",
        ),
        ('<div hidden="until-found"><p>Incidents: 7.</p></div>', "Incidents: 7."),
        ("<p><site-name>Merrow</site-name>: <o:p>7</o:p></p>", "Merrow: 7"),
    ],
    ids=[
        "headings",
        "bold",
        "list",
        "table-with-a-pipe-in-a-cell",
        "link-keeps-its-text",
        "image-alt-text-kept",
        "empty-alt-text-dropped",
        "linked-image-alt-text-kept-url-dropped",
        "alt-text-starting-a-block-unpadded",
        "alt-text-ending-a-block-unpadded",
        "alt-text-in-code-unpadded",
        "alt-text-between-line-breaks-unpadded",
        "alt-text-office-made-up-dropped-in-any-case",
        "svg-title-kept",
        "script-style-noscript-template-dropped",
        "xml-declaration-doctype-comment-dropped",
        "nav-then-main-one-blank-line-apart",
        "line-breaks-in-a-row-one-blank-line",
        "code-keeps-its-blank-lines",
        "hidden-cookie-messages-dropped",
        "hidden-until-found-kept",
        "unknown-tags-keep-their-text",
    ],
)
def test_a_web_page_is_read_as_markdown(tmp_path, html, text):
    page = tmp_path / "page.html"
    page.write_text(html, encoding="utf-8")
    assert read_documents(page) == {"page.html": text}


@pytest.mark.parametrize(
    ("head", "header"),
    [
        (
            [
                json_ld(
                    {
                        "@type": "NewsArticle",
                        "headline": "Q3 incidents",
                        "author": "Ana Faguy",
                        "datePublished": "2026-11-11T18:42:27Z",
                        "dateModified": "2026-11-12",
                        "url": Q3_ADDRESS,
                    }
                )
            ],
            [
                "Title: Q3 incidents",
                "Author: Ana Faguy",
                "Published: 2026-11-11T18:42:27Z",
                "Updated: 2026-11-12",
                f"Address: {Q3_ADDRESS}",
            ],
        ),
        (
            [json_ld({"@type": "Article", "name": "Q3", "headline": "A summary"})],
            ["Title: Q3"],
        ),
        (
            [
                json_ld(
                    {
                        "@type": "Article",
                        "author": [
                            {"@type": "Person", "name": "Ana Faguy"},
                            "Ben Okri",
                            {"@id": "#editor"},
                            "HTTPS://example.com/staff/cal",
                            "www.facebook.com/cal",
                            {"name": "//example.com/staff/dee"},
                        ],
                    }
                )
            ],
            ["Author: Ana Faguy, Ben Okri"],
        ),
        (
            [
                json_ld(
                    {
                        "@graph": [
                            {
                                "@type": "WebPage",
                                "name": "Q3 - Example",
                                "url": Q3_ADDRESS,
                            },
                            {"@type": "Person", "name": "Ana Faguy"},
                            {"@type": "BlogPosting", "headline": "Q3 incidents"},
                        ]
                    }
                )
            ],
            ["Title: Q3 incidents"],
        ),
        (
            [
                json_ld(
                    [
                        {"@type": "WebPage", "name": "Front page", "url": Q3_ADDRESS},
                        {"@type": "NewsArticle", "headline": "A", "datePublished": "1"},
                        {"@type": "NewsArticle", "headline": "B", "datePublished": "2"},
                    ]
                )
            ],
            ["Title: Front page", f"Address: {Q3_ADDRESS}"],
        ),
        (
            [
                json_ld({"@type": "Article", "headline": "Q3"}),
                json_ld({"@type": "Article", "headline": "Q2", "datePublished": "2"}),
                "<title>Q3 incidents</title>",
            ],
            ["Title: Q3 incidents"],
        ),
        (
            [
                json_ld({"@type": "JobPosting", "name": "Engineer"}),
                "<title>Jobs</title>",
            ],
            ["Title: Jobs"],
        ),
        (
            [
                json_ld(
                    [
                        {"@type": "Person", "name": "Ana"},
                        {"@type": ["Report"], "name": "Q3"},
                    ]
                )
            ],
            ["Title: Q3"],
        ),
        (
            [
                '<script type="application/ld+json">{"@type": "Article", "name":</script>',
                json_ld("Q3"),
                json_ld({"@type": "Article", "headline": ["Q3"], "author": 7}),
                meta("og:title", "Q3 incidents"),
            ],
            ["Title: Q3 incidents"],
        ),
        (
            [
                JSON_LD_NESTED_TOO_DEEP,
                "<title>Q3</title>",
            ],
            ["Title: Q3"],
        ),
        (
            [
                f"<template>{meta('og:title', 'Q2')}</template>",
                '<noscript><meta name="author" content="Ana Faguy"></noscript>',
                f'<div hidden><link rel="canonical" href="{Q3_ADDRESS}"></div>',
                "<title>Q3</title>",
            ],
            ["Title: Q3"],
        ),
        (
            [meta("article:author", "Ana Faguy"), meta("article:author", "Ben Okri")],
            ["Author: Ana Faguy, Ben Okri"],
        ),
        (
            [
                meta("og:title", "Q3 incidents"),
                meta("article:author", "Ana Faguy"),
                meta("article:published_time", "2026-11-11"),
                meta("article:modified_time", "2026-11-12"),
                meta("og:url", Q3_ADDRESS),
            ],
            [
                "Title: Q3 incidents",
                "Author: Ana Faguy",
                "Published: 2026-11-11",
                "Updated: 2026-11-12",
                f"Address: {Q3_ADDRESS}",
            ],
        ),
        (
            [
                meta("article:author", "https://example.com/ana"),
                '<meta name="author" content="Ana Faguy">',
            ],
            ["Author: Ana Faguy"],
        ),
        (
            [
                json_ld({"@type": "Article", "name": "Q3", "datePublished": "2026-11"}),
                meta("og:title", "Q3 - Example"),
                meta("article:published_time", "2026-01"),
                "<title>Q3 | Example</title>",
            ],
            ["Title: Q3", "Published: 2026-11"],
        ),
        (
            [meta("og:title", "Q3 - Example"), "<title>Q3 | Example</title>"],
            ["Title: Q3 - Example"],
        ),
        (
            [
                "<title>Q3 incidents</title>",
                f'<link rel="canonical" href="{Q3_ADDRESS}">',
                meta("og:url", f"{Q3_ADDRESS}?ref=feed"),
            ],
            ["Title: Q3 incidents", f"Address: {Q3_ADDRESS}"],
        ),
        (
            [
                json_ld({"@type": "WebPage", "url": Q3_ADDRESS}),
                f'<link rel="canonical" href="{Q3_ADDRESS}?page=1">',
            ],
            [f"Address: {Q3_ADDRESS}"],
        ),
    ],
    ids=[
        "json-ld-article",
        "json-ld-name-over-headline",
        "json-ld-author-list-names-only",
        "json-ld-graph-every-field-from-its-one-article",
        "json-ld-teasers-the-page-not-a-story",
        "json-ld-two-articles-neither-taken",
        "json-ld-job-posting-not-an-article",
        "json-ld-list-and-type-list",
        "bad-json-ld-skipped",
        "json-ld-nested-too-deep-skipped",
        "metadata-not-shown-not-read",
        "each-article-author-meta",
        "opengraph-and-article-meta",
        "article-author-url-skipped-for-meta-author",
        "json-ld-over-meta-over-title",
        "og-title-over-title",
        "title-once-canonical-over-og-url",
        "json-ld-url-over-canonical",
    ],
)
def test_a_web_page_starts_with_what_its_metadata_says_it_is(
    tmp_path, head: list[str], header: list[str]
):
    page = tmp_path / "page.html"
    page.write_text(f"<head>{''.join(head)}</head><body><p>Incidents: 7.</p></body>")
    text = "\n".join(header) + "\n\nIncidents: 7." if header else "Incidents: 7."
    assert read_documents(page) == {"page.html": text}


@pytest.mark.parametrize(
    ("page", "text"),
    [
        (f'<meta charset="windows-1252"><p>{QUOTED}'.encode("cp1252"), QUOTED),
        (
            '<meta http-equiv="Content-Type" content="text/html; charset=windows-1252">'
            f"<p>{QUOTED}".encode("cp1252"),
            QUOTED,
        ),
        (
            f'<?xml version="1.0" encoding="windows-1252"?><html><p>{QUOTED}'.encode(
                "cp1252"
            ),
            QUOTED,
        ),
        (b'<meta charset="iso-8859-1"><p>\x93Fines\x94 \x97 \x8012.', "“Fines” — €12."),
        (b'<meta charset="us-ascii"><p>Merrow\x92s caf\xe9.', "Merrow’s café."),
        (b'<meta charset="iso-8859-15"><p>Fines: \xa47,500.', "Fines: €7,500."),
        (b'<meta charset="utf-16"><p>Caf\xc3\xa9.', "Café."),
        (
            b'<meta charset="no-such-charset"><p>Merrow\x92s caf\xe9.',
            "Merrow\ufffds caf\ufffd.",
        ),
        (
            b"<p>Caf\xc3\xa9 at Merrow\xe2\x80\x99s site, 7 \xff.",
            "Café at Merrow’s site, 7 \ufffd.",
        ),
        (b'\xef\xbb\xbf<meta charset="windows-1252"><p>Caf\xc3\xa9.', "Café."),
        ("<p>Merrow’s café.".encode("utf-16"), "Merrow’s café."),
    ],
    ids=[
        "windows-1252",
        "windows-1252-by-http-equiv",
        "windows-1252-by-xml-declaration",
        "latin-1-read-as-windows-1252",
        "ascii-read-as-windows-1252",
        "latin-9",
        "utf-16-declared-in-ascii-read-as-utf-8",
        "unknown-charset-read-as-utf-8",
        "none-declared-read-as-utf-8-a-bad-byte-replaced",
        "byte-order-mark-over-declaration",
        "utf-16-by-byte-order-mark",
    ],
)
def test_a_web_page_is_decoded_by_its_byte_order_mark_else_its_charset_else_as_utf_8(
    tmp_path, page: bytes, text: str
):
    path = tmp_path / "page.html"
    path.write_bytes(page)
    assert read_documents(path) == {"page.html": text}


def test_a_walk_reads_html_and_htm_pages_as_markdown_and_xml_as_written(tmp_path):
    (tmp_path / "a_page.htm").write_text("<h1>Incidents</h1>", encoding="utf-8")
    (tmp_path / "b_page.HTML").write_text("<p><b>7</b> in Q3.</p>", encoding="utf-8")
    (tmp_path / "c_feed.xml").write_text("<count>7</count>", encoding="utf-8")
    assert read_documents(tmp_path) == {
        "a_page.htm": "# Incidents",
        "b_page.HTML": "**7** in Q3.",
        "c_feed.xml": "<count>7</count>",
    }


def test_a_page_nested_too_deep_for_markdown_is_read_as_its_text_with_a_warning(
    tmp_path, caplog
):
    page = tmp_path / "page.html"
    page.write_text("<div>" * NESTED_DEEPER_THAN_RECURSION + "<p>Incidents: 7.</p>")
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        assert read_documents(page) == {"page.html": "Incidents: 7."}
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"{page}: nested too deep for Markdown; reading its text alone"
    ]


def test_reading_an_xhtml_page_or_one_that_reads_like_a_url_prints_no_warning(
    tmp_path,
):
    (tmp_path / "a_page.html").write_text('<?xml version="1.0"?><p>Incidents: 7.</p>')
    (tmp_path / "b_link.html").write_text("https://example.com/incidents")
    read = subprocess.run(
        [sys.executable, "-c", READ_DOCUMENTS_WITHOUT, str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(read.stdout) == {
        "a_page.html": "Incidents: 7.",
        "b_link.html": "https://example.com/incidents",
    }
    assert read.stderr == ""


def test_a_file_that_cannot_be_opened_is_logged_and_skipped(tmp_path, caplog):
    write_pdf(tmp_path / "a_memo.pdf", "Incidents this quarter: 7.")
    write_docx(tmp_path / "b_report.docx", ["Incidents by site."])
    (tmp_path / "c_notes.txt").write_text("a locked note", encoding="utf-8")
    (tmp_path / "d_plain.txt").write_text("a plain memo", encoding="utf-8")
    locked = [
        tmp_path / name for name in ["a_memo.pdf", "b_report.docx", "c_notes.txt"]
    ]
    for path in locked:
        path.chmod(0)
    if os.access(locked[0], os.R_OK):
        pytest.skip("this user, root say, reads a file whatever its mode")
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        documents = read_documents(tmp_path)
    assert documents == {"d_plain.txt": "a plain memo"}
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"{path}: not readable (PermissionError); skipping it" for path in locked
    ]


@pytest.mark.parametrize(
    "source",
    [lambda root: root, lambda root: str(root / "*.docx")],
    ids=["walk", "glob"],
)
def test_a_walk_or_a_glob_skips_office_lock_files_and_logs_a_file_with_no_text(
    tmp_path, caplog, source: Callable[[Path], str | Path]
):
    write_docx(tmp_path / "a_report.docx", ["Incidents this quarter: 7."])
    write_docx(tmp_path / "b_empty.docx")
    (tmp_path / "~$a_report.docx").write_bytes(WORD_OWNER_FILE)
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        documents = read_documents(source(tmp_path))
    assert documents == {"a_report.docx": "Incidents this quarter: 7."}
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"{tmp_path / 'b_empty.docx'}: no extractable text; skipping it"
    ]


def test_a_walk_skips_what_is_hidden_or_under_macosx_below_its_folder(tmp_path):
    write_files(
        {
            "a_memo.md": "a memo",
            ".notes.md": "a hidden note",
            ".git/description.md": "a hidden folder's file",
            "q3/.cache/b_memo.md": "a file in a hidden folder further down",
            "__MACOSX/a_memo.md": "macOS's metadata, named like a memo",
        }
    )
    assert read_documents(tmp_path) == {"a_memo.md": "a memo"}


@pytest.mark.parametrize(
    ("pattern", "documents"),
    [
        ("d/*.md", ["a memo"]),
        ("d/**/*.md", ["a memo", "a memo further down"]),
        ("d/.*", ["a hidden memo"]),
        ("**/.env.md", ["a hidden file further down"]),
        ("d/*/.env.md", ["a hidden file further down"]),
        ("d/*/../*.md", ["a memo"]),
    ],
    ids=[
        "wildcard",
        "recursive",
        "dot-wildcard",
        "recursive-to-a-dot-name",
        "wildcard-to-a-dot-name",
        "climbing-back-after-a-wildcard",
    ],
)
def test_a_globs_wildcards_match_no_hidden_name_but_a_dot_in_the_pattern_does(
    pattern: str, documents: list[str]
):
    write_files(
        {
            "d/a.md": "a memo",
            "d/.x.md": "a hidden memo",
            "d/q3/b.md": "a memo further down",
            "d/q3/.env.md": "a hidden file further down",
            "d/.git/.env.md": "a hidden folder's hidden file",
        }
    )
    assert list(read_documents(pattern).values()) == documents


def test_a_recursive_glob_reads_a_file_once_past_a_link_to_its_own_folder(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.md").write_text("a memo", encoding="utf-8")
    (tmp_path / "d" / "loop").symlink_to(".")
    assert read_documents("d/**/*.md") == {"a.md": "a memo"}


@pytest.mark.parametrize(
    ("source", "documents"),
    [
        (".env.md", ["a hidden file"]),
        (".github/*.yml", ["a hidden folder's file"]),
        (".github", ["a hidden folder's file"]),
    ],
    ids=["path", "glob-rooted-in-it", "walk-of-it"],
)
def test_a_hidden_file_or_folder_you_name_is_read(
    tmp_path, source: str, documents: list[str]
):
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "ci.yml").write_text(
        "a hidden folder's file", encoding="utf-8"
    )
    (tmp_path / ".env.md").write_text("a hidden file", encoding="utf-8")
    assert list(read_documents(source).values()) == documents


FOLDER_DOCUMENTS = ["a received memo", "a memo stored as a blob"]
HUB_DOCUMENTS = ["the first memo in the hub", "the second memo in the hub"]


def write_folder_with_links(root: Path) -> None:
    for folder in ["folder/blobs", "outside_dir", "hub/blobs", "hub/snapshot"]:
        (root / folder).mkdir(parents=True)
    texts = {
        "secret.txt": "a secret outside the folder",
        "outside_dir/b.txt": "outside the folder",
        "folder/a.txt": "a received memo",
        "folder/blobs/7f3c": "a memo stored as a blob",
        "hub/blobs/1a2b": "the first memo in the hub",
        "hub/blobs/3c4d": "the second memo in the hub",
    }
    for name, text in texts.items():
        (root / name).write_text(text, encoding="utf-8")
    links = {
        "folder/link.txt": "../secret.txt",
        "folder/linked.txt": "blobs/7f3c",
        "folder/linkdir": "../outside_dir",
        "alias": "folder",
        "hub/snapshot/first.txt": "../blobs/1a2b",
        "hub/snapshot/second.txt": "../blobs/3c4d",
    }
    for name, target in links.items():
        (root / name).symlink_to(target)


@pytest.mark.parametrize(
    ("source", "documents"),
    [
        (lambda root: "folder", FOLDER_DOCUMENTS),
        (lambda root: "folder/*.txt", FOLDER_DOCUMENTS),
        (lambda root: "folder/**/*.txt", FOLDER_DOCUMENTS),
        (lambda root: str(root / "folder" / "*.txt"), FOLDER_DOCUMENTS),
        (lambda root: "alias", FOLDER_DOCUMENTS),
        (lambda root: "hub", HUB_DOCUMENTS),
    ],
    ids=[
        "walk",
        "glob",
        "recursive-glob",
        "absolute-glob",
        "folder-named-through-a-link",
        "folder-of-links-inside-it",
    ],
)
def test_a_walk_or_a_glob_reads_every_file_inside_the_folder_and_none_outside(
    tmp_path, source: Callable[[Path], str], documents: list[str]
):
    write_folder_with_links(tmp_path)
    assert list(read_documents(source(tmp_path)).values()) == documents


def test_a_walk_logs_each_file_that_resolves_outside_the_folder(tmp_path, caplog):
    write_folder_with_links(tmp_path)
    with caplog.at_level(logging.WARNING, logger="r3con.documents"):
        documents = read_documents("folder")
    assert list(documents.values()) == FOLDER_DOCUMENTS
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"folder/link.txt: resolves outside {tmp_path / 'folder'}; skipping it"
    ]


@pytest.mark.parametrize(
    ("pattern", "skipped"),
    [
        ("folder/*/*.txt", "folder/linkdir/b.txt"),
        ("folder/*/../../secret.txt", "folder/blobs/../../secret.txt"),
    ],
    ids=["through-a-linked-folder", "climbing-out-after-a-wildcard"],
)
def test_a_glob_that_leaves_its_folder_finds_nothing_and_logs_why(
    tmp_path, caplog, pattern: str, skipped: str
):
    write_folder_with_links(tmp_path)
    with (
        caplog.at_level(logging.WARNING, logger="r3con.documents"),
        pytest.raises(FileNotFoundError, match="No documents found at"),
    ):
        read_documents(pattern)
    assert [r.getMessage() for r in caplog.records if r.name == "r3con.documents"] == [
        f"{skipped}: resolves outside {tmp_path / 'folder'}; skipping it"
    ]


def test_a_file_named_directly_is_read_even_through_a_link(tmp_path):
    write_folder_with_links(tmp_path)
    assert read_documents("folder/link.txt") == {
        "link.txt": "a secret outside the folder"
    }


def test_a_glob_is_expanded_from_the_working_directory(tmp_path):
    write_tree(tmp_path)
    assert read_documents("*.md") == {"a.md": "alpha document"}


def test_several_sources_are_read_in_the_order_given(tmp_path):
    (tmp_path / "second.txt").write_text("second", encoding="utf-8")
    (tmp_path / "first.txt").write_text("first", encoding="utf-8")
    sources = [tmp_path / "first.txt", tmp_path / "second.txt"]
    assert list(read_documents(sources).items()) == [
        ("first.txt", "first"),
        ("second.txt", "second"),
    ]


@pytest.mark.parametrize(
    "source", [Path("/definitely/not/here/at/all"), "this is document text, not a path"]
)
def test_a_source_that_matches_nothing_is_an_error_not_a_document(source):
    with pytest.raises(FileNotFoundError, match="No documents found at"):
        read_documents(source)


def test_undecodable_bytes_are_replaced_not_fatal(tmp_path):
    (tmp_path / "bad.txt").write_bytes(b"good text \xff\xfe more text")
    [document] = read_documents(tmp_path / "bad.txt").values()
    assert document.startswith("good text ")
    assert document.endswith(" more text")


@pytest.mark.parametrize(
    ("source", "names"),
    [
        ("archive", ["2021/jan/front-page.txt", "index.md"]),
        ("archive/*.md", ["index.md"]),
        ("archive/**/*.txt", ["2021/jan/front-page.txt"]),
        ("archive/2021/*/front-page.txt", ["jan/front-page.txt"]),
        ("*/index.md", ["archive/index.md"]),
        ("archive/2021/jan/front-page.txt", ["front-page.txt"]),
    ],
    ids=["walk", "glob", "recursive-glob", "glob-in-a-subfolder", "glob-here", "file"],
)
def test_a_document_is_named_by_its_path_below_its_folder_or_globs_folder(
    source: str, names: list[str]
):
    write_files(
        {
            "archive/index.md": "the archive's index",
            "archive/2021/jan/front-page.txt": "January's front page",
        }
    )
    assert list(read_documents(source)) == names


@pytest.mark.parametrize(
    "sources",
    [["a", "b"], ["a/*.txt", "b/*.txt"], ["a/notes.txt", "b/notes.txt"]],
    ids=["walks", "globs", "files"],
)
def test_a_name_two_sources_give_is_replaced_by_each_ones_path_as_read(
    sources: list[str],
):
    write_files({"a/notes.txt": "A's notes", "b/notes.txt": "B's notes"})
    assert list(read_documents(sources).items()) == [
        ("a/notes.txt", "A's notes"),
        ("b/notes.txt", "B's notes"),
    ]


def test_a_name_that_still_clashes_is_numbered_from_its_second_document():
    write_files({"a/notes.txt": "A's notes", "a/plan.txt": "A's plan"})
    assert list(read_documents(["a", "a/notes.txt", "a"]).items()) == [
        ("a/notes.txt", "A's notes"),
        ("a/plan.txt", "A's plan"),
        ("a/notes.txt (2)", "A's notes"),
        ("a/notes.txt (3)", "A's notes"),
        ("a/plan.txt (2)", "A's plan"),
    ]


def test_a_name_a_clash_renames_to_is_renamed_again_where_it_clashes_in_turn():
    write_files({"a/x.txt": "A's", "b/x.txt": "B's", "c/a/x.txt": "C's"})
    assert list(read_documents(["a", "b", "c"]).items()) == [
        ("a/x.txt", "A's"),
        ("b/x.txt", "B's"),
        ("c/a/x.txt", "C's"),
    ]


def test_many_copies_of_one_name_are_numbered_at_once():
    started = time.perf_counter()
    numbered = unique_names(["notes.txt"] * MANY_COPIES)
    assert time.perf_counter() - started < AT_ONCE_SECONDS
    assert numbered[:3] == ["notes.txt", "notes.txt (2)", "notes.txt (3)"]
    assert len(set(numbered)) == MANY_COPIES


def test_a_dropped_file_is_left_out_and_clashes_with_no_name():
    write_files({"a/notes.txt": " \n", "b/notes.txt": "B's notes"})
    assert read_documents(["a", "b"]) == {"notes.txt": "B's notes"}
