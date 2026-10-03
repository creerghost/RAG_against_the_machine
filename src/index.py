"""Build the on-disk index: load files, chunk them, build BM25."""

from .config import RagConfig
from .loader import CorpusLoader
from .bm25 import BM25Index
from .tokenizer import Tokenizer
from .chunking import chunker_for
from .models import Chunk, ChunkTable
from pathlib import Path
from .io_utils import save_model


def build_index(config: RagConfig) -> int:
    """Chunk every indexed file under ``config.raw_dir``, persist the index.

    Returns the number of chunks; stored paths keep the ``raw_dir`` prefix.
    """
    loader = CorpusLoader(config.raw_dir, config.chunker_by_extension.keys())
    chunks: list[Chunk] = []
    docs: list[list[str]] = []
    tokenizer = Tokenizer(config.stopwords, config.min_token_length)
    for path, text in loader.load():
        # this will create around 1800 tiny objects,
        # which are negligible to cache
        chunker = chunker_for(path, config)
        if chunker is None:
            continue  # never happens, just for mypy

        for chunk in chunker.chunk(path, text):
            tokens = tokenizer.tokenize(
                text[chunk.first_character_index:chunk.last_character_index])
            tokens += tokenizer.tokenize(chunk.title or "")
            rel_path = chunk.file_path.removeprefix(config.raw_dir)
            tokens += tokenizer.tokenize(chunk.file_path)
            chunks.append(chunk)
            docs.append(tokens)
    bm25 = BM25Index.build(docs, config.bm25_k1, config.bm25_b)
    out_path = Path(config.processed_dir)
    bm25.save(out_path / "bm25.npz")
    save_model(ChunkTable(chunks=chunks), out_path / "chunks.json")
    return len(chunks)
