"""Loader draft: fill the TODOs step by step, run, read the checks.

Run from the project root:  uv run python rest/loader_draft.py
"""
from pathlib import Path


class LoaderDraft:
    """Turns ``raw_dir`` into sorted ``(file_path, text)`` pairs."""
    def __init__(self, raw_dir: str = "data/raw",
                 extensions: tuple[str, ...] = (".py", ".md", ".txt")) -> None:
        """Store the corpus root and the extensions to keep."""
        self.raw_dir = raw_dir
        self.extensions = extensions

    def all_files(self) -> list[Path]:
        """Return every regular file under ``raw_dir``."""
        base_path = Path(self.raw_dir)
        if not base_path.is_dir():
            return []
        return [p for p in base_path.rglob("*") if p.is_file()]

    def paths(self) -> list[str]:
        """Return sorted corpus paths with a wanted extension."""
        return sorted(str(f) for f in self.all_files()
                      if f.suffix.lower() in self.extensions)

    def read(self, path: str) -> str | None:
        """Return the text of ``path``, or None if it cannot be read."""
        try:
            with open(path, encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeDecodeError):
            return None
        return text

    def load(self) -> list[tuple[str, str]]:
        """Return ``(path, text)`` for every useful file."""
        result = []
        for path in self.paths():
            text = self.read(path)
            if not text or not text.strip():
                continue
            result.append((path, text))
        return result


def check(name: str, ok: bool, detail: object = "") -> None:
    """Print one check line."""
    print(f"[{'OK' if ok else '--'}] {name} {detail}")


if __name__ == "__main__":
    loader = LoaderDraft()
    lora = "data/raw/vllm-0.10.1/docs/features/lora.md"

    print("== step 1 ==")
    files = loader.all_files()
    check("finds all 2874 files", len(files) == 2874, f"(found {len(files)})")
    check("paths are relative", all(not p.is_absolute() for p in files))

    print("== step 2 ==")
    paths = loader.paths()
    check("1969 .py/.md/.txt files", len(paths) == 1969, f"(got {len(paths)})")
    check("prefix data/raw/vllm-0.10.1/",
          all(p.startswith("data/raw/vllm-0.10.1/") for p in paths))
    check("sorted", paths == sorted(paths))
    check("lora.md included", lora in paths)
    check("root files included",
          "data/raw/vllm-0.10.1/CMakeLists.txt" in paths)

    print("== step 3 ==")
    text = loader.read(lora)
    check("lora.md is 15148 chars", text is not None and len(text) == 15148,
          f"(got {None if text is None else len(text)})")
    check("missing file -> None", loader.read("nope/missing.md") is None)

    print("== step 4 ==")
    pairs = loader.load()
    check("loads files", len(pairs) > 1500, f"(got {len(pairs)})")
    check("no blank texts", all(t.strip() for _, t in pairs))
    check("same order as paths()",
          [p for p, _ in pairs] == [p for p in paths if p in dict(pairs)])
