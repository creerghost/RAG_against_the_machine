"""Answer generation with a local causal LM."""

from typing import List

from .models import MinimalSource


class Generator:
    """Wraps the LLM: loads it once, builds prompts, generates answers."""

    def __init__(self, model_name: str) -> None:
        """Load tokenizer and model weights.

        Args:
            model_name: Hugging Face model id, e.g. ``Qwen/Qwen3-0.6B``.
        """
        self.model_name = model_name
        raise NotImplementedError("Generator.__init__")

    def answer(self, question: str, sources: List[MinimalSource]) -> str:
        """Generate an answer grounded in ``sources``.

        Args:
            question: The user question.
            sources: Retrieved spans; their text is read from disk.

        Returns:
            str: The generated answer.
        """
        raise NotImplementedError("Generator.answer")
