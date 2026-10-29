"""Standard `logging` for the pipeline.

Library modules log through ``get_logger(...)``; an application calls
``configure_logging()`` once at startup. Enabling per-stage progress output is
one knob:

- ``R3CON_LOG_LEVEL`` env (``DEBUG`` / ``INFO`` / ``WARNING``; default
  ``WARNING`` = quiet), or
- ``--verbose`` on the CLI, which selects ``INFO`` for that run.

No handler is attached until ``configure_logging`` runs, so importing the library
in a notebook/test stays silent.

Each logger masks keys in its records' messages, tracebacks and stacks before any
handler sees them (:func:`r3con.masking.masked`), so they reach the application's own
handlers masked too: tenacity logs each error it resends, and a provider's error may
echo a key.
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
    """Mask ``record``'s message (:func:`_message`), its traceback and its stack. The
    traceback is left only as the text a formatter appends, so a handler that renders
    ``exc_info`` itself never sees the error unmasked."""
    record.msg, record.args = masked(_message(record)), ()
    if record.exc_info:
        record.exc_text = masked(logging.Formatter().formatException(record.exc_info))
        record.exc_info = None
    if record.stack_info:
        record.stack_info = masked(record.stack_info)
    return True


def _message(record: logging.LogRecord) -> str:
    """``record``'s message; when it does not format, as ``("doc %d", "seven")`` or an
    argument whose ``str`` raises, its format and its arguments' ``repr``, or
    ``<unprintable args>`` when that raises too."""
    # Logging does not guard a filter, so what this raised would fail the log call,
    # and the run with it.
    try:
        return record.getMessage()
    except Exception:  # noqa: BLE001 — see above
        try:
            return f"{record.msg} {record.args!r}"
        except Exception:  # noqa: BLE001 — see above
            return f"{record.msg} <unprintable args>"


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
