from typing import List
from pydantic import BaseModel
from .minimal_source import MinimalSource


class MinimalSearchResults(BaseModel):
    """Store ranked source locations for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]
