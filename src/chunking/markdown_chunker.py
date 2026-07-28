from ..interfaces import BaseChunker
from ..models import Chunk
import re


class MarkdownChunker(BaseChunker):
    def chunk(
        self, text: str, file_path: str, max_chunk_size: int
    ) -> list[Chunk]:
        # Matches the start of the file OR a newline,
        # followed by hashes and a space
        matches = list(re.finditer(r"(?:^|\n)#+ ", text))
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
            new_chunks = self._process_chunk(
                text=text,
                start=fst_char_idx,
                end=last_char_idx,
                file_path=file_path,
                max_size=max_chunk_size
            )
            self.chunks.extend(new_chunks)
        return self.chunks

    def _process_chunk(
        self, text: str, start: int, end: int,
        file_path: str, max_size: int
    ) -> list[Chunk]:
        """
        Decides if a section fits, or if it needs fallback splitting.
        """
        if (end - start) <= max_size:
            return [Chunk(
                        file_path=file_path,
                        content=text[start:end],
                        first_char_idx=start,
                        last_char_idx=end
                    )]
        else:
            return self._split_by_paragraph(
                text[start:end], start, file_path, max_size
            )

    def _split_by_paragraph(
        self, text: str, original_start_idx: int,
        file_path: str, max_size: int
    ) -> list[Chunk]:
        """Split text by \n\n and groups them up to max_size."""
        fallback_chunks = []
        matches = list(re.finditer(r"\n\n", text))

        current_start = 0
        last_valid_end = 0

        for match in matches:
            if (match.end() - current_start) > max_size:
                # if a single paragraph is too big, just force split it
                if last_valid_end == current_start:
                    last_valid_end = match.end()

                fallback_chunks.append(Chunk(
                    file_path=file_path,
                    content=text[current_start:last_valid_end],
                    first_char_idx=original_start_idx + current_start,
                    last_char_idx=original_start_idx + last_valid_end,
                ))
                current_start = last_valid_end

            last_valid_end = match.end()

        # grab any leftover text after the last newline
        if current_start < len(text):
            fallback_chunks.append(Chunk(
                file_path=file_path,
                content=text[current_start:len(text)],
                first_char_idx=original_start_idx + current_start,
                last_char_idx=original_start_idx + len(text),
            ))

        return fallback_chunks

    def __repr__(self):
        if not hasattr(self, "chunks"):
            return "MarkdownChunker(empty)"
        output = f"MarkdownChunker with {len(self.chunks)} chunks:\n"
        for i, chunk in enumerate(self.chunks):
            output += (f"  [{i}] {chunk.file_path} "
                       f"(Chars: {chunk.first_char_idx}-"
                       f"{chunk.last_char_idx})\n")
        return output
