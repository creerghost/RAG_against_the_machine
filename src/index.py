"""Build the on-disk index: load files, chunk them, build BM25."""

from .config import RagConfig


def build_index(config: RagConfig) -> int:
    """Chunk every indexed file under ``config.raw_dir``, persist the index.

    Returns the number of chunks; stored paths keep the ``raw_dir`` prefix.
    """
    