from pathlib import Path


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
