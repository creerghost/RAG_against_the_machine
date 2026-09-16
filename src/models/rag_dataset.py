from typing import List
from pydantic import BaseModel
from .answered_question import AnsweredQuestion
from .unanswered_question import UnansweredQuestion


class RagDataset(BaseModel):
    """Validate a collection containing answered or unanswered RAG questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
