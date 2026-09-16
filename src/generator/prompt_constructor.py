class PromptConstructor:
    """Build a grounded question-and-context prompt for the language model."""

    def __init__(self, question: str) -> None:
        """Initialize a prompt builder for one question."""
        self.question = question
        self.contexts: list[str] = []

    def add_context(self, file_path: str, content: str) -> None:
        """Append a labeled source excerpt to the prompt context."""
        self.contexts.append(f"[Source: {file_path}]\n{content}\n")

    def build(self) -> str:
        """Return the final instruction, context, question, and answer prompt."""
        parts = [
            "You are an expert programming assistant.",
            "Answer the user's question based ONLY on the following context.",
            "\n",
            "CONTEXT:\n"
        ]
        # Append all collected contexts
        parts.extend(self.contexts)

        # Append the user's question
        parts.append(f"QUESTION:\n{self.question}\n")
        parts.append("ANSWER:\n")

        return "\n".join(parts)
