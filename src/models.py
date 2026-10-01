"""Pydantic models exchanged between the pipeline stages.

The first block mirrors the subject verbatim (see docs/subject.md, VI.4);
``Chunk`` is an internal addition used by the indexer and retriever.
"""

import uuid
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class MinimalSource(BaseModel):
    """A character span inside one corpus file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without ground truth."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions, answered or not."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Retrieved sources plus a generated answer for one question."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output of ``search_dataset``."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of ``answer_dataset``."""

    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """One indexed span of a corpus file.

    Only offsets are stored; the text is re-read from ``file_path`` when
    needed. ``kind`` is the chunker that produced it (e.g. ``"python"`` or
    ``"text"``) and ``title`` an optional symbol name or heading path.
    """

    file_path: str
    first_character_index: int = Field(ge=0)
    last_character_index: int = Field(ge=0)
    kind: str
    title: Optional[str] = None

    @model_validator(mode="after")
    def _check_span(self) -> "Chunk":
        """Reject spans that end before they start."""
        if self.last_character_index < self.first_character_index:
            raise ValueError("last_character_index < first_character_index")
        return self

    def to_source(self) -> MinimalSource:
        """Convert to the ``MinimalSource`` returned to the grader.

        Returns:
            MinimalSource: Same file path and character span.
        """
        return MinimalSource(
            file_path=self.file_path,
            first_character_index=self.first_character_index,
            last_character_index=self.last_character_index,
        )
