"""Ways into the program: the Fire CLI (and later the HTTP server); they
only talk to ``RagService``."""

from .cli import RagCli

__all__ = ["RagCli"]
