from pathlib import Path


def write_pdf(path: Path, text: str | None = None) -> None:
    """A minimal valid PDF (correct xref offsets and %%EOF), with or without a text
    layer: generated, so the fixture stays readable."""
    objs = ["<</Type/Catalog/Pages 2 0 R>>", "<</Type/Pages/Kids[3 0 R]/Count 1>>"]
    if text:
        stream = f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET"
        objs += [
            (
                "<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]/Contents 4 0 R"
                "/Resources<</Font<</F1 5 0 R>>>>>>"
            ),
            f"<</Length {len(stream)}>>stream\n{stream}\nendstream",
            "<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
        ]
    else:
        objs.append("<</Type/Page/Parent 2 0 R/MediaBox[0 0 300 200]>>")
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
