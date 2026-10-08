from abc import ABC, abstractmethod


class Ranker(ABC):
    @abstractmethod
    def rank(query: str, k: int) -> list[tuple[int, float]]:
        pass
