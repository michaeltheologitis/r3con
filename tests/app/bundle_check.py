"""Check a built Mac app with its own Python, for .github/workflows/mac-app.yml:

    APP/Contents/Frameworks/Python.framework/Versions/3.14/bin/python3 -I \\
        tests/app/bundle_check.py APP

It puts the app's code and packages on the path as the app's launcher does, imports r3con
and the packages a bundle most likely lacks, its compiled ones and the window's toolkit,
each from inside APP, then serves the app as the window does: the page, and the API with
this launch's token, must answer, and once the server has stopped, nothing must.
"""

import importlib
import site
import sys
import tempfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

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
)
TIMEOUT_SECONDS = 10


def main(app: Path) -> None:
    resources = app / "Contents" / "Resources"
    sys.path.append(str(resources / "app"))
    site.addsitedir(str(resources / "app_packages"))
    for name in BUNDLED:
        where = Path(importlib.import_module(name).__file__ or "").resolve()
        if not where.is_relative_to(resources):
            sys.exit(f"{name} is imported from {where}, not from {app}")
        print(f"{name}: {where.relative_to(app)}")

    from r3con import __version__
    from r3con.app import serve

    print(f"r3con {__version__}")
    with tempfile.TemporaryDirectory() as logs_dir:
        url, stop = serve(logs_dir=Path(logs_dir))
        base_url, token = url.split("/#token=")
        keys = Request(
            f"{base_url}/api/keys", headers={"Authorization": f"Bearer {token}"}
        )
        try:
            with urlopen(f"{base_url}/", timeout=TIMEOUT_SECONDS) as page:
                if "<title>" not in page.read().decode():
                    sys.exit("GET / did not answer with the page")
                print(f"GET / {page.status}")
            with urlopen(keys, timeout=TIMEOUT_SECONDS) as answer:
                print(f"GET /api/keys {answer.status} {answer.read().decode()}")
        finally:
            stop()
    try:
        urlopen(f"{base_url}/", timeout=TIMEOUT_SECONDS)
    except URLError:
        print("stopped: nothing answers")
    else:
        sys.exit("the server still answers after its stop")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
