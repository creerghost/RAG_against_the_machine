from ..interfaces import BaseChunker
from ..models import Chunk
import re


class MarkdownChunker(BaseChunker):
    def chunk(
        self, text: str, file_path: str,
        max_chunk_size: int
    ) -> list[Chunk]:
        # \n matches newline, #+ matches one or more hashes, space ensures
        # that this is a complete header
        # re.finditer return iterable (idx, content)
        matches = list(re.finditer(r"\n#+ ", text))
        self.chunks = []
        for i, match in enumerate(matches):
            fst_char_idx = match.start()
            # if next header exists
            if i + 1 < len(matches):
                # end of chunk is the start of next header
                last_char_idx = matches[i+1].start()
            else:
                # end of chunk is the length of text (obviously)
                last_char_idx = len(text)
            if (last_char_idx - fst_char_idx) <= max_chunk_size:
                self.chunks.append(
                    Chunk(
                        file_path=file_path,
                        content=text[fst_char_idx:last_char_idx],
                        first_char_idx=fst_char_idx,
                        last_char_idx=last_char_idx
                    )
                )
            else:
                pass
        return self.chunks

    def __repr__(self):
        if not hasattr(self, "chunks"):
            return "MarkdownChunker(empty)"
        output = f"MarkdownChunker with {len(self.chunks)} chunks:\n"
        for i, chunk in enumerate(self.chunks):
            output += (f"  [{i}] {chunk.file_path} "
                       f"(Chars: {chunk.first_char_idx}-"
                       f"{chunk.last_char_idx})\n")
        return output
