from typing import List
from pydantic import BaseModel
from .minimal_search_results import MinimalSearchResults


class StudentSearchResults(BaseModel):
    """Store batch search results and the requested result count."""

    search_results: List[MinimalSearchResults]
    k: int
