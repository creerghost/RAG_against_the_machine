"""Answer generation with a local causal LM (Qwen3 by default)."""

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding

from .config import RagConfig
from .models import MinimalSource


class Generator:
    """Wraps the LLM: loads it once, builds prompts, generates answers."""
    def __init__(self, config: RagConfig) -> None:
        """Load the tokenizer and weights of ``config.model_name``.

        Weights use ``config.model_dtype`` (float32 or bfloat16).
        """
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        # Any: transformers' stubs reject generate() on the Auto* type
        self.model: Any = AutoModelForCausalLM.from_pretrained(
            config.model_name, dtype=getattr(torch, config.model_dtype))

    def _n_tokens(self, text: str) -> int:
        """Return how many model tokens ``text`` takes."""
        return len(self.tokenizer(text, add_special_tokens=False)
                   ["input_ids"])

    def _build_prompt(self, question: str,
                      sources: list[MinimalSource]) -> tuple[str, str]:
        """Return ``(system, user)`` messages; sources fill a token budget.

        Sources are numbered in rank order; unreadable ones are skipped.
        """
        budget = self.config.max_context_tokens
        text = ""
        for i, s in enumerate(sources, start=1):
            try:
                with open(s.file_path, encoding="utf-8") as f:
                    file_text = f.read()
            except (OSError, UnicodeDecodeError):
                continue
            chunk = file_text[s.first_character_index:s.last_character_index]
            block = f"[{i}] {s.file_path}\n{chunk}\n\n"
            cost = self._n_tokens(block)
            if cost > budget:
                break  # sources are ranked: drop the rest
            text += block
            budget -= cost
        user = f"Sources:\n\n{text}Question: {question}"
        return self.config.system_prompt, user

    def answer(self, question: str, sources: list[MinimalSource]) -> str:
        """Return an answer to ``question`` grounded in ``sources``' text."""
        if not sources:
            return self.config.no_answer
        system, user = self._build_prompt(question, sources)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, enable_thinking=False,
            return_tensors="pt", return_dict=True)
        if not isinstance(inputs, BatchEncoding):  # narrows the type
            raise TypeError("chat template did not return a BatchEncoding")
        input_len = inputs["input_ids"].shape[1]
        lookup = self.config.prompt_lookup_tokens or None  # 0: plain decoding
        with torch.inference_mode():
            out = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens,
                do_sample=False, prompt_lookup_num_tokens=lookup)
        reply = self.tokenizer.decode(out[0][input_len:],
                                      skip_special_tokens=True)
        return str(reply).strip()
