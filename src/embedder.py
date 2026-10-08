from .config import RagConfig


class Embedder:
    def __init__(self, config: RagConfig) -> None:
        self.config = config
