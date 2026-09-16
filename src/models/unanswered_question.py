import uuid
from pydantic import BaseModel, Field


class UnansweredQuestion(BaseModel):
    """Represent a question before its supporting sources are known."""

    question_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )
    question: str
