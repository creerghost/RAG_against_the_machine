from ..models import MinimalSource
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, sources: list[MinimalSource]) -> str:
        """Generate a natural-language answer grounded in sources."""
        pass
