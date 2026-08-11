from ..interfaces import BaseGenerator
from vllm import LLM, SamplingParams
from ..models import MinimalSource
from .prompt_constructor import PromptConstructor


class QwenGenerator(BaseGenerator):
    def __init__(self) -> None:
        self.llm = LLM(
            model="Qwen/Qwen3-0.6B",
            max_model_len=4096,
            gpu_memory_utilization=0.5,
            enforce_eager=True
        )
        self.params = SamplingParams(temperature=0.2, max_tokens=256)

    def generate(self, question: str, sources: list[MinimalSource]) -> str:
        prompt_builder = PromptConstructor(question)

        for source in sources:
            with open(source.file_path, "r", encoding="utf-8") as f:
                content = f.read()[source.first_char_idx:source.last_char_idx]
                prompt_builder.add_context(source.file_path, content)

        final_prompt = prompt_builder.build()

        outputs = self.llm.generate([final_prompt], self.params)
        # vllm returns a list of RequestOutputs.
        # We extract the generated text from the first one.
        return outputs[0].outputs[0].text
