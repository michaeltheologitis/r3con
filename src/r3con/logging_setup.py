"""Standard `logging` for the pipeline.

Library modules log through ``get_logger(...)``; an application calls
``configure_logging()`` once at startup. Enabling per-stage progress output is
one knob:

- ``R3CON_LOG_LEVEL`` env (``DEBUG`` / ``INFO`` / ``WARNING``; default
  ``WARNING`` = quiet), or
- ``--verbose`` on the CLI, which selects ``INFO`` for that run.

No handler is attached until ``configure_logging`` runs, so importing the library
in a notebook/test stays silent.

Each logger masks keys in its records' messages and tracebacks before any handler sees
them (:func:`r3con.masking.masked`), so they reach the application's own handlers
masked too: tenacity logs each error it resends, and a provider's error may echo a key.
"""

from __future__ import annotations

import logging
import os
import sys

from r3con.masking import masked

_NAME = "r3con"


def get_logger(name: str | None = None) -> logging.Logger:
    """The pipeline logger (``r3con`` or ``r3con.<name>``), masking what it logs."""
    logger = logging.getLogger(_NAME if name is None else f"{_NAME}.{name}")
    logger.addFilter(_masked_record)
    return logger


def _masked_record(record: logging.LogRecord) -> bool:
    """Mask ``record``'s message and its traceback, as a formatter writes it."""
    record.msg, record.args = masked(record.getMessage()), ()
    if record.exc_info and not record.exc_text:
        record.exc_text = masked(logging.Formatter().formatException(record.exc_info))
    return True


def configure_logging(level: str | int | None = None) -> None:
    """Attach a single stderr handler to the ``r3con`` logger (idempotent).

    Level comes from ``level``, else ``R3CON_LOG_LEVEL`` (in any case), else
    ``WARNING``.

    Raises:
        ValueError: naming the variable, if ``R3CON_LOG_LEVEL`` is not a level name.
    """
    logger = logging.getLogger(_NAME)
    if level is not None:
        logger.setLevel(level)
    else:
        raw = os.environ.get("R3CON_LOG_LEVEL") or "WARNING"
        try:
            logger.setLevel(raw.strip().upper())
        except ValueError:
            raise ValueError(
                f"R3CON_LOG_LEVEL must be a level name such as INFO, got {raw!r}."
            ) from None
    if not logger.handlers:
        # stderr, not stdout: the CLI writes the answer to stdout, so progress must not
        # pollute a pipe.
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(name)s: %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
