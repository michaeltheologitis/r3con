"""r3con's app: a page that runs r3con, and the local server behind it (``r3con app``).

It needs the ``app`` extra: ``pip install "r3context[app]"``.
"""

from r3con.app.server import create_app, serve

__all__ = ["create_app", "serve"]
