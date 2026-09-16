from typing import List
from pydantic import BaseModel
from .minimal_answer import MinimalAnswer


class StudentSearchResultsAndAnswer(BaseModel):
    """Store batch search results together with generated answers."""

    search_results: List[MinimalAnswer]
    k: int
