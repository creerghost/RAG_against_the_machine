from ..models import MinimalSource
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Interface for components that generate answers from retrieved context."""

    @abstractmethod
    def generate(self, question: str, sources: list[MinimalSource]) -> str:
        """Generate a natural-language answer grounded in sources."""
        pass
