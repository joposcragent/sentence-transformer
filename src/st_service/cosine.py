import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_01(left: np.ndarray, right: np.ndarray) -> float:
    """Cosine similarity in [0, 1] from sklearn output in [-1, 1]."""
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("left and right must have the same shape")
    x = a.reshape(1, -1)
    y = b.reshape(1, -1)
    raw = float(cosine_similarity(x, y)[0, 0])
    return (raw + 1.0) / 2.0
