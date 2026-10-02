import re

class ChunkerDraft:
    def __init__(self, filename: str = "data/raw/vllm-0.10.1/docs/features/lora.md"
    ) -> None:
        self.filename = filename

    def retrieve_lora_md(self) -> str:
        with open(self.filename, "r", encoding="utf-8") as f:
            text = f.read()
        return text

    def find_headings(self, text: str) -> list[tuple[int, str]]:
        result = []
        in_code = False
        pos = 0
        # for m in re.finditer(r'^(#+) (.*)', text, re.M):
        #     start = m.start()
        #     heading = m.group(2)
        #     result.append((start, heading))
        for line in text.splitlines(keepends=True):
            if line.lstrip().startswith("```"):
                in_code = not in_code
            elif not in_code:
                match = re.match(r'(#+) (.*)', line)
                if match:
                    title = match.group(2).strip()
                    result.append((pos, title))
            pos += len(line)
        return result

    def headings_to_chunks(self, text: str, headings: list[tuple[int, str]]
        ) -> list[tuple[int, int, str]]:
        if not headings:
            return [(0, len(text), "")]
        result = []
        if headings[0][0] != 0:
            result.append((0, headings[0][0], ""))
        for (c_start, c_str), (n_start, _) in zip(headings, headings[1:]):
            result.append((c_start, n_start, c_str))
        result.append((headings[-1][0], len(text), headings[-1][1]))
            
        return result

    def _split(self, text: str, start: int, end: int, title: str,
               seps: tuple[str, ...], max_s: int
               ) -> list[tuple[int, int, str]]:
        if end - start <= max_s:
            return [(start, end, title)]
        if not seps:
            return [(x, min(x + max_s, end), title)
                    for x in range(start, end, max_s)]
        sep, rest = seps[0], seps[1:]
        pieces: list[tuple[int, int, str]] = []
        parts = text[start:end].split(sep)
        last_pos = start
        pos = start
        for j, part in enumerate(parts):
            is_last = j == len(parts) - 1
            # split removes separators! that's why there is addition in the end
            part_end = pos + len(part) + (0 if is_last else len(sep))
            if part_end - last_pos >  max_s and pos > last_pos:
                pieces.append((last_pos, pos, title))
                last_pos = pos
            pos = part_end
        if last_pos < end:  # text ending with sep leaves an empty last part
            pieces.append((last_pos, end, title))
        result: list[tuple[int, int, str]] = []
        for p in pieces:
            if p[1] - p[0] <= max_s:
                result.append(p)
            else:
                result.extend(self._split(text, p[0], p[1], p[2], rest, max_s))
        return result
        
    def split_oversized_chunks(self, text: str, chunks: list[tuple[int, int, str]],
                               max_chunk_size: int = 2000) -> list[tuple[int, int, str]]:
        res: list[tuple[int, int, str]] = []
        for start, end, title in chunks:
            res.extend(
                self._split(text, start, end, title, ("\n\n", "\n"), max_chunk_size))
        return res
            

if __name__ == "__main__":
    chunker = ChunkerDraft()
    text = chunker.retrieve_lora_md()
    headings = chunker.find_headings(text)
    # [print(x) for x in headings]
    chunks = chunker.headings_to_chunks(text, headings)
    print("\n=== Pre-processed chunks #, ##, ... ===")
    for i, x in enumerate(chunks):
        print(f"Chunk {i + 1}: ")
        print(f"start: {x[0]}, end: {x[1]}, len: {x[1] - x[0]}, title: {x[2]}")
    final_chunks = chunker.split_oversized_chunks(text, chunks)
    print ("\n=== Post-processed chunk \\n\\n ===")
    for i, x in enumerate(final_chunks):
        print(f"Chunk {i + 1}: ")
        print(f"start: {x[0]}, end: {x[1]}, len: {x[1] - x[0]}, title: {x[2]}")