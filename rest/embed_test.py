from src.embedder import Embedder


e = Embedder("sentence-transformers/all-MiniLM-L6-v2", 256, 32)
v = e.embed(["load a LoRA adapter", "LoRA adapter loading", "pip install vllm"])
print(v.shape, v @ v.T)    # (3, 384); [0,1] should be clearly above [0,2]
