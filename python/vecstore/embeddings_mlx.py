"""MLX embedding backend for VecStore — Apple-Silicon-native embeddings (MLX / M5 GPU + Neural
Accelerators) as a drop-in for the sentence-transformers `EmbeddingModel`.

Same model family (all-MiniLM-L6-v2), same 384-dim vectors, same interface (encode / encode_query /
dimension) — but the compute runs on MLX instead of PyTorch, so it's faster + lower-power on Mac and
needs no torch. Optional: `pip install mlx-embeddings`.

  from vecstore.embeddings_mlx import MLXEmbeddingModel
  store = VecStoreWithEmbeddings("./db")
  store._embedding_model = MLXEmbeddingModel()        # swap in the MLX backend
"""
from typing import List

DEFAULT_MLX_MODEL = "mlx-community/all-MiniLM-L6-v2-4bit"


def is_available() -> bool:
    try:
        import mlx_embeddings  # noqa: F401
        return True
    except Exception:
        return False


class MLXEmbeddingModel:
    """Drop-in replacement for vecstore.embeddings.EmbeddingModel, computed on MLX (Apple Silicon)."""

    DEFAULT_MODEL = DEFAULT_MLX_MODEL

    def __init__(self, model_name: str = DEFAULT_MLX_MODEL, **_ignored):
        import mlx.core as mx  # noqa: F401
        from mlx_embeddings.utils import load
        self.model_name = model_name
        self._model, self._tokenizer = load(model_name)
        self._dimension = len(self.encode(["probe"])[0])

    @property
    def dimension(self) -> int:
        return self._dimension

    def _embed(self, texts: List[str]):
        import mlx.core as mx
        import numpy as np
        toks = self._tokenizer.batch_encode_plus(
            list(texts), return_tensors="mlx", padding=True, truncation=True, max_length=512)
        out = self._model(toks["input_ids"], attention_mask=toks.get("attention_mask"))
        emb = getattr(out, "text_embeds", None)
        if emb is None:                                  # fallback: mean-pool last hidden state
            h = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state
            m = toks["attention_mask"][..., None].astype(h.dtype)
            emb = (h * m).sum(axis=1) / mx.maximum(m.sum(axis=1), 1e-9)
        emb = emb / mx.maximum(mx.linalg.norm(emb, axis=1, keepdims=True), 1e-9)  # L2-normalize
        return np.asarray(emb, dtype="float32")

    def encode(self, texts, **_ignored) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return self._embed(texts).tolist()

    def encode_query(self, query: str) -> List[float]:
        return self.encode([query])[0]

    def __repr__(self) -> str:
        return f"MLXEmbeddingModel(model='{self.model_name}', dim={self.dimension}, backend='mlx')"
