"""A small regression check of r3con on Loong and CorpusQA: ``uv run python -m evals``.

See ``evals/README.md``. Lives outside ``src/`` so it never ships in the wheel.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from r3con.logging_setup import get_logger
from r3con.runs import StageRun
from r3con.runtime.llm import litellm_chat_completion

JUDGE_MODEL = "openai/gpt-6-luna"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DOWNLOAD_TIMEOUT_SECONDS = 60

log = get_logger("evals")

Verdict = TypeVar("Verdict", bound=BaseModel)


def download(url: str, path: Path, sha256: str) -> Path:
    """Fetch ``url`` to ``path`` once, refusing a file whose SHA-256 is not ``sha256``,
    so a changed or truncated upstream file fails loudly instead of moving the numbers."""
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    log.info("downloading %s", url)
    with (
        urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response,
        partial.open("wb") as out,
    ):
        shutil.copyfileobj(response, out)
    with partial.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != sha256:
        partial.unlink()
        raise ValueError(f"{url} has SHA-256 {digest}, expected {sha256}")
    os.replace(partial, path)
    return path


def rows_by_id(path: Path) -> dict[str, dict[str, Any]]:
    """Each JSON line of ``path`` by its ``id``, read a line at a time."""
    with path.open(encoding="utf-8") as file:
        return {row["id"]: row for row in map(json.loads, file)}


def ask_judge(
    system: str, user: str, schema: type[Verdict], run: StageRun | None
) -> Verdict:
    """The judge's verdict as ``schema``, its call recorded in ``run``."""
    verdict = litellm_chat_completion(
        system_prompt=system,
        user_prompt=user,
        model=JUDGE_MODEL,
        schema=schema,
        run=run,
        kind="judge",
    )
    return cast(Verdict, verdict)
