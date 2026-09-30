"""School files on disk, and the bands a training draw balances over.

These helpers touch no library: both the record-patch trainer (cadence 0.12) and the
deep trainer (cadence 0.50) read the same school format through them.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

BANDS = ((0, 8), (8, 16), (16, 24), (24, 43))


def load(files: list[Path]) -> dict[str, np.ndarray]:
    parts = [np.load(f) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("own", "other", "z", "n", "plies")}


def keys_of(data: dict[str, np.ndarray]) -> np.ndarray:
    return np.stack((data["own"], data["other"]), axis=1)


def packed(keys: np.ndarray) -> np.ndarray:
    """Each key as one structured value, for set operations."""
    return np.ascontiguousarray(keys).view([("a", keys.dtype), ("b", keys.dtype)]).reshape(-1)
