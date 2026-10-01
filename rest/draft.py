import re

class ChunkerDraft:
    def __init__(
        self, filename: str = "data/raw/vllm-0.10.1/docs/features/lora.md"
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
                    title = match.group(2).strip("\r\n")
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
            
            

if __name__ == "__main__":
    chunker = ChunkerDraft()
    text = chunker.retrieve_lora_md()
    headings = chunker.find_headings(text)
    # [print(x) for x in headings]
    chunks = chunker.headings_to_chunks(text, headings)
    for i, x in enumerate(chunks):
        print(f"Chunk {i + 1}: ")
        print(f"start: {x[0]}, end: {x[1]}, len: {x[1] - x[0]}, title: {x[2]}")