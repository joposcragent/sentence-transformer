from __future__ import annotations

from typing import Sequence

import numpy as np

PREVIEW_MAX_CHARS = 500


def preview_text(s: str) -> str:
    if len(s) <= PREVIEW_MAX_CHARS:
        return s
    return s[:PREVIEW_MAX_CHARS] + "..."


def preview_vector(arr: Sequence[float] | np.ndarray) -> str:
    if isinstance(arr, np.ndarray):
        flat = arr.astype(float).ravel()
        s = str(flat.tolist())
    else:
        s = str(list(arr))
    if len(s) <= PREVIEW_MAX_CHARS:
        return s
    return s[:PREVIEW_MAX_CHARS] + "..."
