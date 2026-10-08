from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

Float32Array = npt.NDArray[np.float32]


class Embedder:
    def __init__(self, model_name: str, max_length: int,
                 batch_size: int, dtype: str) -> None:
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name, dtype=getattr(torch, dtype))
        # Vector width: 384 for MiniLM.
        self.dim: int = self.model.config.hidden_size

    def embed(self, texts: list[str],
              desc: Optional[str] = None) -> Float32Array:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        # optimization: sort texts by length before batching
        # similar lengths share a batch, so less padding
        order = np.argsort([len(text) for text in texts])

        with torch.inference_mode():
            parts: list[Float32Array] = []
            starts = range(0, len(texts), self.batch_size)
            # progress bar only when a ``desc`` is given (indexing)
            for start in tqdm(starts, desc=desc, unit="batch",
                              disable=desc is None):
                batch_texts = [
                    texts[i] for i in order[start:start+self.batch_size]]
                batch = self.tokenizer(
                    batch_texts, padding=True,
                    truncation=True, max_length=self.max_length,
                    return_tensors="pt")
                out = self.model(**batch)
                tokens = out.last_hidden_state  # (b, seq, dim)
                mask = batch["attention_mask"].unsqueeze(-1)  # (b, seq, 1)
                summed = (tokens * mask).sum(dim=1)  # (b, dim)
                counts = mask.sum(dim=1).clamp(min=1e-9)  # (b, 1)
                vectors = torch.nn.functional.normalize(summed / counts, dim=1)
                # np can't hold bf16 (added .float())
                parts.append(vectors.float().numpy())
        vectors_sorted = np.concatenate(parts)
        result = np.empty((len(texts), self.dim), dtype=np.float32)
        result[order] = vectors_sorted
        return result
