"""Answer generation with a local causal LM."""

from .models import MinimalSource


class Generator:
    """Wraps the LLM: loads it once, builds prompts, generates answers."""
    def __init__(self, model_name: str) -> None:
        """Load the tokenizer and weights of ``model_name`` (a HF model id)."""
        self.model_name = model_name
        raise NotImplementedError("Generator.__init__")

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Return an answer to ``question`` grounded in ``sources``' text."""
        raise NotImplementedError("Generator.answer")
