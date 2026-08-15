"""Embedding backend for the atommem upgrades.

Default: local sentence-transformers all-MiniLM-L6-v2 (384d) — the exact model
AtomMem uses, already cached on this host, fully local/hermetic. Lazy-loaded
singleton so importing the package is cheap and offline hosts that lack the
model degrade gracefully (embed() returns []).
"""

from __future__ import annotations

from typing import List, Optional

_MODEL_NAME = "all-MiniLM-L6-v2"
_model = None
_load_failed = False


def _get_model():
    global _model, _load_failed
    if _model is not None or _load_failed:
        return _model
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(_MODEL_NAME)
    except Exception:
        _load_failed = True
        _model = None
    return _model


def embed(text: str) -> List[float]:
    """Return a dense embedding for text, or [] if the model is unavailable."""
    if not text:
        return []
    model = _get_model()
    if model is None:
        return []
    try:
        vec = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist()
    except Exception:
        return []


def embed_batch(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    if model is None:
        return [[] for _ in texts]
    try:
        vecs = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=32
        )
        return [v.tolist() for v in vecs]
    except Exception:
        return [[] for _ in texts]


def available() -> bool:
    return _get_model() is not None


def dimension() -> Optional[int]:
    model = _get_model()
    if model is None:
        return None
    try:
        dim = model.get_sentence_embedding_dimension()
        return int(dim) if dim is not None else None
    except Exception:
        return None


if __name__ == "__main__":
    print("embedder available:", available(), "dim:", dimension())
    if available():
        v = embed("Emma aced her chemistry midterm.")
        print("vector len:", len(v), "first 4:", [round(x, 4) for x in v[:4]])
