from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


def load_model(
    model_name: str,
    device,
    cache_folder: str | None,
) -> SentenceTransformer:
    kwargs: dict = {"device": str(device)}
    if cache_folder:
        kwargs["cache_folder"] = cache_folder
    return SentenceTransformer(model_name, **kwargs)


def token_length(model: SentenceTransformer, text: str) -> int:
    return len(model.tokenizer.encode(text, add_special_tokens=True))


def max_context_tokens(model: SentenceTransformer) -> int:
    tok = model.tokenizer
    msl = getattr(model, "max_seq_length", None)
    tmax = getattr(tok, "model_max_length", None) if tok is not None else None
    candidates = [x for x in (msl, tmax) if x is not None and int(x) > 0]
    if not candidates:
        return 512
    return min(int(x) for x in candidates)


def encode_text(model: SentenceTransformer, text: str) -> np.ndarray:
    vec = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if vec.ndim > 1:
        vec = vec[0]
    return vec.astype(np.float64)
