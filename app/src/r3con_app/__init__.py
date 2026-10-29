"""The local server that connects r3con's app to r3con."""

from r3con_app.server import create_app, serve

__all__ = ["create_app", "serve"]
