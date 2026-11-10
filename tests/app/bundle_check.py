"""Check a built Mac app with its own Python, for .github/workflows/mac-app.yml:

    APP/Contents/Frameworks/Python.framework/Versions/3.14/bin/python3 -I \\
        tests/app/bundle_check.py APP CONSTRAINTS

It puts the app's code and packages on the path as the app's launcher does, and checks
that each package there but the project is at the version CONSTRAINTS, the build's
`uv export` of uv.lock, pins for this Python. Then it imports, each from inside APP,
r3con and the packages a bundle most likely lacks, and every compiled module under its
packages, so that one built for another platform fails here. Then it serves the app as
the window does: the page, and the API with this launch's token, must answer, and once
the server has stopped, nothing must.
"""

import importlib.metadata
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
# Installed from the checkout, so not among uv.lock's exported pins.
PROJECT = "r3context"


def imported(name: str, resources: Path) -> Path:
    """The file module ``name`` is imported from, which must be in ``resources``."""
    where = Path(importlib.import_module(name).__file__ or "").resolve()
    if not where.is_relative_to(resources):
        sys.exit(f"{name} is imported from {where}, not from {resources}")
    return where


def compiled(packages: Path) -> list[str]:
    """The module of each compiled file under ``packages``."""
    return sorted(
        ".".join(path.relative_to(packages).with_name(path.name.split(".")[0]).parts)
        for path in packages.rglob("*.so")
    )


def pinned(packages: Path, constraints: Path) -> int:
    """How many packages ``packages`` holds besides the project, each of which must be
    at the version ``constraints`` pins for this Python."""
    # The bundle's own packaging, on the path once main has added app_packages.
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name

    lines = constraints.read_text(encoding="utf-8").splitlines()
    requirements = [Requirement(line) for line in lines if line and line[0] not in " #"]
    pins = {
        canonicalize_name(requirement.name): requirement.specifier
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate()
    }
    found = {
        canonicalize_name(distribution.name): distribution.version
        for distribution in importlib.metadata.distributions(path=[str(packages)])
    }
    found.pop(canonicalize_name(PROJECT), None)
    problems = []
    for name, version in sorted(found.items()):
        pin = pins.get(name)
        if pin is None or version not in pin:
            problems.append(f"{name} {version}: uv.lock pins {pin or 'no version'}")
    if problems:
        sys.exit("\n".join(["Not at uv.lock's version:", *problems]))
    return len(found)


def answers(url: str) -> bool:
    try:
        urlopen(url, timeout=TIMEOUT_SECONDS).close()
    except URLError:
        return False
    return True


def main(app: Path, constraints: Path) -> None:
    resources = app / "Contents" / "Resources"
    packages = resources / "app_packages"
    sys.path.append(str(resources / "app"))
    site.addsitedir(str(packages))
    count = pinned(packages, constraints)
    print(f"{count} packages under app_packages but {PROJECT}: each as uv.lock pins it")
    for name in BUNDLED:
        print(f"{name}: {imported(name, resources).relative_to(app)}")
    modules = compiled(packages)
    for name in modules:
        imported(name, resources)
    print(f"{len(modules)} compiled modules under app_packages: each imports")

    from r3con import __version__
    from r3con.app import serve

    print(f"r3con {__version__}")
    url, stop = serve(logs_dir=Path(tempfile.mkdtemp()))
    base_url, token = url.split("/#token=")
    keys = Request(f"{base_url}/api/keys", headers={"Authorization": f"Bearer {token}"})
    try:
        with urlopen(f"{base_url}/", timeout=TIMEOUT_SECONDS) as answer:
            page_status, page = answer.status, answer.read().decode()
        with urlopen(keys, timeout=TIMEOUT_SECONDS) as answer:
            keys_status, keys_body = answer.status, answer.read().decode()
    finally:
        stop()
    if "<title>" not in page:
        sys.exit("GET / did not answer with the page")
    print(f"GET / {page_status}")
    print(f"GET /api/keys {keys_status} {keys_body}")
    if answers(f"{base_url}/"):
        sys.exit("the server still answers after its stop")
    print("stopped: nothing answers")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]))
