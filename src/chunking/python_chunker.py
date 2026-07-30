from ..interfaces import BaseChunker
from ..models import Chunk
import ast


class PythonChunker(BaseChunker):
    def chunk(
        self, text: str, file_path: str, max_chunk_size: int
    ) -> list[Chunk]:
        parsed_text = ast.parse(text)
        self.chunks: list[Chunk] = []
        search_start = 0

        for node in parsed_text.body:
            search_start = self._process_node(
                node, text, file_path, max_chunk_size, search_start
            )

        return self.chunks

    def _process_node(
        self, node: ast.AST, text: str, file_path: str,
        max_size: int, search_start: int
    ) -> int:
        segment = ast.get_source_segment(text, node)
        if not segment:
            return search_start

        start_idx = text.find(segment, search_start)
        if start_idx == -1:
            return search_start

        last_char_idx = start_idx + len(segment)

        if len(segment) <= max_size:
            self.chunks.append(Chunk(
                file_path=file_path,
                content=segment,
                first_char_idx=start_idx,
                last_char_idx=last_char_idx,
            ))
            # return the end of this node so the next sibling searches forward
            return last_char_idx
        else:
            if hasattr(node, "body"):
                # if too big, dive into its body
                # children search starts exactly where the parent starts.
                current_start = start_idx
                for child_node in node.body:
                    current_start = self._process_node(
                        child_node, text, file_path, max_size, current_start
                    )
                # return the end of the parent so the next sibling works
                # correctly
                return last_char_idx
            else:
                # if it's huge but has no body (e.g. massive string/dict),
                # just force it
                self.chunks.append(Chunk(
                    file_path=file_path,
                    content=segment,
                    first_char_idx=start_idx,
                    last_char_idx=last_char_idx,
                ))
                return last_char_idx
