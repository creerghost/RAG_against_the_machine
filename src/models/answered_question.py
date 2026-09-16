from typing import List
from .unanswered_question import UnansweredQuestion
from .minimal_source import MinimalSource


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with reference sources and a ground-truth answer."""

    sources: List[MinimalSource]
    answer: str
