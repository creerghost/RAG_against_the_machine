from ..interfaces import BaseGenerator
from transformers import pipeline
from ..models import MinimalSource
from .prompt_constructor import PromptConstructor
import torch

class QwenGenerator(BaseGenerator):
    def __init__(self) -> None:
        print("Loading Qwen with Transformers (safe mode)...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen3-0.6B",
            device=device,
            max_new_tokens=256,
            temperature=0.2,
            do_sample=True,
        )

    def generate(self, question: str, sources: list[MinimalSource]) -> str:
        prompt_builder = PromptConstructor(question)

        for source in sources:
            with open(source.file_path, "r", encoding="utf-8") as f:
                content = f.read()[source.first_char_idx:source.last_char_idx]
                prompt_builder.add_context(source.file_path, content)

        final_prompt = prompt_builder.build()

        # Generate the answer and return only the newly generated text (not the prompt)
        outputs = self.pipe(final_prompt, return_full_text=False)
        return outputs[0]["generated_text"].strip()
