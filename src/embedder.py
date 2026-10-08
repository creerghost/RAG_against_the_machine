import numpy as np
import numpy.typing as npt

Float32Array = npt.NDArray[np.float32]


class Embedder:
    def __init__(self, model_name: str, max_length: int,
                 batch_size: int) -> None:
        self.max_length = max_length
        self.batch_size = batch_size

        from transformers import AutoModel, AutoTokenizer  # heavy
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Vector width: 384 for MiniLM.
        self.dim: int = self.model.config.hidden_size

    def embed(self, texts: list[str]) -> Float32Array:
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)
        import torch  # heavy (python also caches imports)

        with torch.inference_mode():
            parts: list[Float32Array] = []
            for start in range(0, len(texts), self.batch_size):
                batch = self.tokenizer(
                    texts[start:start+self.batch_size], padding=True,
                    truncation=True, max_length=self.max_length,
                    return_tensors="pt")
                out = self.model(**batch)

                tokens = out.last_hidden_state  # (b, seq, dim)
                mask = batch["attention_mask"].unsqueeze(-1)  # (b, seq, 1)
                summed = (tokens * mask).sum(dim=1)  # (b, 384)
                counts = mask.sum(dim=1).clamp(min=1e-9)  # (b, 1)
                vectors = torch.nn.functional.normalize(summed / counts, dim=1)
                parts.append(vectors.numpy())
        return np.concatenate(parts)
