"""Local HTTP API over ``RagService``: search and answers as JSON (bonus).

Started by the ``serve`` command; a search page at ``/``, interactive docs
at ``/docs``.
"""

import threading
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StringConstraints

from src.models import MinimalSource
from src.retrieval import Retriever
from src.service import RagService
from .utils import INDEX_ERRORS, index_error


class QueryRequest(BaseModel):
    """Body of /search and /answer; ``k`` defaults to retrieval.default_k."""
    query: Annotated[str, StringConstraints(strip_whitespace=True,
                                            min_length=1)]
    k: Optional[int] = Field(default=None, ge=1)


class SourceText(MinimalSource):
    """A source with its text, read from the corpus ("" if unreadable)."""
    text: str

    @classmethod
    def read(cls, source: MinimalSource) -> "SourceText":
        """Return ``source`` with the text of its span from its file."""
        try:
            with open(source.file_path, encoding="utf-8") as f:
                text = f.read()[source.first_character_index:
                                source.last_character_index]
        except (OSError, UnicodeDecodeError):  # corpus changed since index
            text = ""
        return cls(**source.model_dump(), text=text)


class SearchResponse(BaseModel):
    """The query, the ``k`` used and the sources found, best first."""
    query: str
    k: int
    sources: list[SourceText]


class AnswerResponse(SearchResponse):
    """A search response plus the answer generated from its sources."""
    answer: str


class RagServer:
    """HTTP API over ``RagService``; one lock serialises requests, so lazy
    loading never races and one CPU generation runs at a time."""
    def __init__(self, service: RagService) -> None:
        """Build the FastAPI ``app`` and register the routes on ``service``."""
        self.service = service
        self._lock = threading.Lock()
        self.app = FastAPI(title="RAG against the machine")
        # plain def routes: FastAPI runs them in a thread pool; async def
        # would block the event loop (and /health) during a generation
        self.app.add_api_route("/", self.page, methods=["GET"],
                               include_in_schema=False)
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/search", self.search, methods=["POST"])
        self.app.add_api_route("/answer", self.answer, methods=["POST"])

    def _load_index(self) -> Retriever:
        """Return the retriever, or fail with 503 and the fix-it message."""
        cfg = self.service.config
        try:
            return self.service.load_index()
        except INDEX_ERRORS as e:
            raise HTTPException(503, index_error(
                e, cfg.paths.processed_dir, cfg.retrieval.mode)) from e

    def _search(self, request: QueryRequest
                ) -> tuple[int, list[MinimalSource]]:
        """Return the ``k`` used and the sources for ``request`` (caller
        holds the lock)."""
        default_k = self.service.config.retrieval.default_k
        k = default_k if request.k is None else request.k
        return k, self._load_index().search(request.query, k)

    def page(self) -> FileResponse:
        """Return the search page (static HTML calling /search, /answer)."""
        return FileResponse(Path(__file__).parent / "static" / "index.html")

    def health(self) -> dict[str, Any]:
        """Report that the server is up, how many chunks it searches, the
        ranking mode and the answer model."""
        cfg = self.service.config
        with self._lock:
            chunks = len(self._load_index().chunks)
        return {"status": "ok", "chunks": chunks, "mode": cfg.retrieval.mode,
                "model": cfg.generation.model}

    def search(self, request: QueryRequest) -> SearchResponse:
        """Return the top ``k`` sources for the query, with their text."""
        with self._lock:
            k, sources = self._search(request)
        return SearchResponse(query=request.query, k=k,
                              sources=[SourceText.read(s) for s in sources])

    def answer(self, request: QueryRequest) -> AnswerResponse:
        """Search, then answer the query from the sources found."""
        with self._lock:
            k, sources = self._search(request)
            answer = self.service.answer(request.query, sources)
        return AnswerResponse(query=request.query, k=k,
                              sources=[SourceText.read(s) for s in sources],
                              answer=answer)
