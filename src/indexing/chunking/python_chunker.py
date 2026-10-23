"""Python chunking at top-level definitions, found with ``ast``."""

import ast

from .base import Chunker, Span

Definition = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


class PythonChunker(Chunker):
    """Sections at top-level defs/classes (methods of big classes), packed."""
    kind = "python"

    def _sections(self, text: str) -> list[Span]:
        """Return contiguous sections of ``text`` cut at definitions."""
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return [(0, len(text), "")]
        line_start = [0]
        for line in text.splitlines(keepends=True):
            line_start.append(line_start[-1] + len(line))

        def start_of(node: Definition) -> int:
            """Return the offset of ``node``'s first decorator or keyword."""
            lines = [node.lineno] + [d.lineno for d in node.decorator_list]
            return line_start[min(lines) - 1]

        functions = (ast.FunctionDef, ast.AsyncFunctionDef)
        cuts: list[tuple[int, str]] = []
        for node in tree.body:
            if isinstance(node, functions):
                cuts.append((start_of(node), node.name))
            elif isinstance(node, ast.ClassDef):
                start = start_of(node)
                end = line_start[node.end_lineno or node.lineno]
                cuts.append((start, node.name))
                if end - start > self.max_chunk_size:  # big class
                    for sub in node.body:
                        if isinstance(sub, functions):
                            title = f"{node.name}.{sub.name}"
                            cuts.append((start_of(sub), title))
            else:
                cuts.append((line_start[node.lineno - 1], ""))
        merged: list[tuple[int, str]] = []
        for offset, title in cuts:
            if merged and title == "" and merged[-1][1] == "":
                continue
            merged.append((offset, title))
        if merged and merged[0][1] == "":
            merged[0] = (0, "")  # leading comments join the first glue
        elif not merged or merged[0][0] != 0:
            merged.insert(0, (0, ""))
        result: list[Span] = []
        ends = [offset for offset, _ in merged[1:]] + [len(text)]
        for (curr, title), nxt in zip(merged, ends):
            if nxt > curr:  # skip empty sections (and empty files)
                result.append((curr, nxt, title))
        # grow the last section while the merged span still fits
        packed: list[Span] = []
        for start, end, title in result:
            if packed and end - packed[-1][0] <= self.max_chunk_size:
                p_start, _, p_title = packed[-1]
                joined = " ".join(t for t in (p_title, title) if t)
                packed[-1] = (p_start, end, joined)
            else:
                packed.append((start, end, title))
        return packed
