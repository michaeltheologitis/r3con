import base64
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from xml.sax.saxutils import escape

CORRUPT_DOCX = b"PK\x03\x04 truncated garbage"
# A 1x1 PNG.
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
OFFICE_RELATIONSHIPS_NS = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
)
WORD_DRAWING_NS = (
    "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
)
DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PICTURE_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
PACKAGE_RELATIONSHIPS_NS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)
CONTENT_TYPES = """<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="png" ContentType="image/png"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
PACKAGE_RELS = f"""<Relationships xmlns="{PACKAGE_RELATIONSHIPS_NS}">
<Relationship Id="rId1" Type="{OFFICE_RELATIONSHIPS_NS}/officeDocument" Target="word/document.xml"/>
</Relationships>"""
DOCUMENT_RELS = f"""<Relationships xmlns="{PACKAGE_RELATIONSHIPS_NS}">
<Relationship Id="rIdImage" Type="{OFFICE_RELATIONSHIPS_NS}/image" Target="media/image1.png"/>
</Relationships>"""
IMAGE = f"""<w:p><w:r><w:drawing><wp:inline><wp:extent cx="9525" cy="9525"/>
<wp:docPr id="1" name="Picture 1"/><a:graphic><a:graphicData uri="{PICTURE_NS}"><pic:pic>
<pic:nvPicPr><pic:cNvPr id="0" name="image1.png"/><pic:cNvPicPr/></pic:nvPicPr>
<pic:blipFill><a:blip r:embed="rIdImage"/></pic:blipFill><pic:spPr/></pic:pic>
</a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"""


def write_pdf(
    path: Path,
    text: str | None = None,
    *,
    to_unicode: dict[str, str] | None = None,
    padding: int = 0,
) -> None:
    """A minimal valid PDF (correct xref offsets and %%EOF), with or without a text
    layer: generated, so the fixture stays readable. ``to_unicode`` gives its font a
    ToUnicode CMap, from characters of ``text`` to the UTF-16BE hex each one stands
    for; ``padding`` adds a stream of that many bytes that no page uses."""
    objs = ["<</Type/Catalog/Pages 2 0 R>>", "<</Type/Pages/Kids[3 0 R]/Count 1>>"]
    if text:
        cmap = "/ToUnicode 6 0 R" if to_unicode else ""
        objs += [
            (
                "<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Contents 4 0 R"
                "/Resources<</Font<</F1 5 0 R>>>>>>"
            ),
            _stream(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET"),
            f"<</Type/Font/Subtype/Type1/BaseFont/Helvetica{cmap}>>",
        ]
        if to_unicode:
            objs.append(_stream(_cmap(to_unicode)))
    else:
        objs.append("<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]>>")
    if padding:
        objs.append(_stream(" " * padding))
    out, offsets = bytearray(b"%PDF-1.4\n"), []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj{body}endobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for o in offsets:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer<</Size {len(objs) + 1}/Root 1 0 R>>\n".encode()
    out += f"startxref\n{xref}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def _stream(data: str) -> str:
    return f"<</Length {len(data)}>>stream\n{data}\nendstream"


def _cmap(to_unicode: dict[str, str]) -> str:
    pairs = "\n".join(
        f"<{ord(char):02X}> <{code}>" for char, code in to_unicode.items()
    )
    return (
        "/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
        "1 begincodespacerange <00> <FF> endcodespacerange\n"
        f"{len(to_unicode)} beginbfchar\n{pairs}\nendbfchar\n"
        "endcmap CMapName currentdict /CMap defineresource pop end end"
    )


def paragraph(text: str) -> str:
    return f'<w:p><w:r><w:t xml:space="preserve">{escape(text)}</w:t></w:r></w:p>'


def table(rows: Sequence[Sequence[str]], *, header: bool = False) -> str:
    """A table of these rows of cells' text; ``header`` marks the first as Word's header
    row."""
    cells = ["".join(f"<w:tc>{paragraph(text)}</w:tc>" for text in row) for row in rows]
    if header:
        cells[0] = "<w:trPr><w:tblHeader/></w:trPr>" + cells[0]
    return "<w:tbl>" + "".join(f"<w:tr>{row}</w:tr>" for row in cells) + "</w:tbl>"


def write_docx(
    path: Path,
    blocks: Sequence[str | Sequence[Sequence[str]]] = (),
    *,
    xml: str = "",
    image: bool = False,
) -> None:
    """A Word file, written as the few parts a reader needs. Its body is ``blocks`` in
    order, a string a paragraph and rows of cells' text a table, then ``xml``, more of
    the body as WordprocessingML, then :data:`PNG` if ``image``."""
    body = "".join(paragraph(b) if isinstance(b, str) else table(b) for b in blocks)
    body += xml + (IMAGE if image else "")
    namespaces = (
        f'xmlns:w="{WORD_NS}" xmlns:r="{OFFICE_RELATIONSHIPS_NS}" '
        f'xmlns:wp="{WORD_DRAWING_NS}" xmlns:a="{DRAWING_NS}" xmlns:pic="{PICTURE_NS}"'
    )
    with zipfile.ZipFile(path, "w") as docx:
        docx.writestr("[Content_Types].xml", CONTENT_TYPES)
        docx.writestr("_rels/.rels", PACKAGE_RELS)
        docx.writestr(
            "word/document.xml",
            f"<w:document {namespaces}><w:body>{body}</w:body></w:document>",
        )
        if image:
            docx.writestr("word/_rels/document.xml.rels", DOCUMENT_RELS)
            docx.writestr("word/media/image1.png", PNG)


# What check.mjs writes each document with, by suffix.
WRITERS: dict[str, Callable[..., None]] = {
    ".pdf": write_pdf,
    ".docx": write_docx,
}
