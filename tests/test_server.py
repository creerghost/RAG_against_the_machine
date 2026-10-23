"""Tests for the HTTP API, over a fake service (no index, no model)."""

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.config import PathsConfig, RagConfig
from src.entrypoints.server import RagServer, SourceText
from src.models import Chunk, MinimalSource
from src.retrieval import Retriever
from src.service import RagService

SOURCE = MinimalSource(file_path="data/raw/vllm-0.10.1/a.md",
                       first_character_index=0, last_character_index=10)


class FakeRetriever(Retriever):
    """Returns ``k`` copies of one source; records the ``k`` asked for."""
    def __init__(self) -> None:
        """Hold two chunks and no ranker."""
        chunk = Chunk(file_path=SOURCE.file_path, first_character_index=0,
                      last_character_index=10, kind="markdown")
        self.chunks = [chunk, chunk]
        self.last_k = 0

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return ``k`` copies of the fixed source."""
        self.last_k = k
        return [SOURCE] * k


class FakeService(RagService):
    """A service whose index and answer model are fakes."""
    def __init__(self, missing: bool = False) -> None:
        """Use a config over a non-existent index directory."""
        paths = PathsConfig(raw_dir="raw", processed_dir="nowhere")
        super().__init__(RagConfig(paths=paths))
        self.missing = missing
        self.retriever = FakeRetriever()

    def load_index(self) -> Retriever:
        """Return the fake retriever, or fail like a missing index."""
        if self.missing:
            raise FileNotFoundError(2, "No such file", "nowhere/bm25.npz")
        return self.retriever

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Echo the question and the number of sources."""
        return f"{question}: {len(sources)}"


@pytest.fixture
def service() -> FakeService:
    """Return a fake service with an index."""
    return FakeService()


def post(service: RagService, path: str, body: dict[str, Any]) -> Any:
    """POST ``body`` to ``path`` of an app over ``service``."""
    return TestClient(RagServer(service).app).post(path, json=body)


def test_health(service: FakeService) -> None:
    """/health reports the chunk count, ranking mode and answer model."""
    response = TestClient(RagServer(service).app).get("/health")
    assert response.json() == {"status": "ok", "chunks": 2,
                               "mode": "lexical", "model": "Qwen/Qwen3-0.6B"}


def test_search(service: FakeService) -> None:
    """/search returns the query, k and the sources (missing file: no
    text)."""
    response = post(service, "/search", {"query": " lora ", "k": 2})
    assert response.status_code == 200
    source = {**SOURCE.model_dump(), "text": ""}
    assert response.json() == {"query": "lora", "k": 2,
                               "sources": [source] * 2}


def test_search_default_k(service: FakeService) -> None:
    """Without k, retrieval.default_k is used."""
    response = post(service, "/search", {"query": "lora"})
    default_k = service.config.retrieval.default_k
    assert response.json()["k"] == default_k
    assert service.retriever.last_k == default_k


def test_answer(service: FakeService) -> None:
    """/answer adds the answer to the search response."""
    body = post(service, "/answer", {"query": "lora", "k": 3}).json()
    assert body["answer"] == "lora: 3" and len(body["sources"]) == 3


@pytest.mark.parametrize("body", [{"query": ""}, {"query": "   "},
                                  {"query": "x", "k": 0}, {"k": 3},
                                  {"query": "x", "k": "many"}])
def test_invalid_request(service: FakeService, body: dict[str, Any]) -> None:
    """Empty queries, bad k and missing fields are 422, as JSON."""
    response = post(service, "/search", body)
    assert response.status_code == 422 and "detail" in response.json()


def test_missing_index() -> None:
    """A missing index is 503 with the CLI's fix-it message."""
    response = post(FakeService(missing=True), "/search", {"query": "x"})
    assert response.status_code == 503
    assert response.json()["detail"].startswith("No index in nowhere")


def test_source_text(tmp_path: Path) -> None:
    """A source's text is its span of the file."""
    path = tmp_path / "a.md"
    path.write_text("# LoRA\nload adapters\n", encoding="utf-8")
    source = MinimalSource(file_path=str(path), first_character_index=7,
                           last_character_index=11)
    assert SourceText.read(source).text == "load"


def test_page(service: FakeService) -> None:
    """/ serves the search page, which is not in the API docs."""
    client = TestClient(RagServer(service).app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert 'id="ask"' in response.text
    assert "/" not in client.get("/openapi.json").json()["paths"]
