"""Build the on-disk index: load files, chunk them, build BM25."""

from pathlib import Path


def build_index(
    raw_dir: Path, processed_dir: Path, max_chunk_size: int
) -> int:
    """Chunk every useful file under ``raw_dir`` and persist the index.

    Args:
        raw_dir: Corpus root, e.g. ``data/raw``. Stored file paths must start
            with this exact (relative) prefix.
        processed_dir: Directory the index files are written to.
        max_chunk_size: Maximum chunk width in characters (<= 2000).

    Returns:
        int: Number of chunks indexed.
    """
    raise NotImplementedError("build_index")
