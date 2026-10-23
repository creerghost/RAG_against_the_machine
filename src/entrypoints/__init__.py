"""Ways into the program: the Fire CLI and the HTTP server (``server.py``,
imported only by ``serve`` so the CLI never loads fastapi); both only talk
to ``RagService``."""

from .cli import RagCli

__all__ = ["RagCli"]
