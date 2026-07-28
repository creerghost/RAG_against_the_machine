from typing import List
from pydantic import BaseModel
from .minimal_answer import MinimalAnswer


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
