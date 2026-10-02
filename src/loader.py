"""Corpus loader: sorted ``(file_path, text)`` pairs from ``raw_dir``."""

import sys
from pathlib import Path
from typing import Iterable, Optional

from tqdm import tqdm


class CorpusLoader:
    """Turns ``raw_dir`` into sorted ``(file_path, text)`` pairs."""
    def __init__(self, raw_dir: str, extensions: Iterable[str]) -> None:
        """Store the corpus root and the (case-insensitive) extensions."""
        self.raw_dir = raw_dir
        self.extensions = {e.lower() for e in extensions}

    def all_files(self) -> list[Path]:
        """Return every regular file under ``raw_dir``; [] if it is missing."""
        base_path = Path(self.raw_dir)
        if not base_path.is_dir():
            return []
        return [p for p in base_path.rglob("*") if p.is_file()]

    def paths(self) -> list[str]:
        """Return sorted relative paths of files with a wanted extension."""
        return sorted(str(f) for f in self.all_files()
                      if f.suffix.lower() in self.extensions)

    def read(self, path: str) -> Optional[str]:
        """Return the UTF-8 text of ``path``, or None (with a warning)."""
        try:
            with open(path, encoding="utf-8") as f:
                return f.read()
        except (OSError, UnicodeDecodeError) as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)
            return None

    def load(self) -> list[tuple[str, str]]:
        """Return ``(path, text)`` for every readable, non-blank file."""
        result: list[tuple[str, str]] = []
        for path in tqdm(self.paths(), desc="Loading", unit="file"):
            text = self.read(path)
            if text is None or not text.strip():
                continue
            result.append((path, text))
        return result
