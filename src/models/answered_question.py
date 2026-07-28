from typing import List
from .unanswered_question import UnansweredQuestion
from .minimal_source import MinimalSource


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str
