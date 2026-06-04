"""Lightweight retrieval utilities (no heavy ML imports)."""

import numpy as np


def cosine_sim(a, b) -> float:
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    denom = np.linalg.norm(a_arr) * np.linalg.norm(b_arr) + 1e-12
    return float(np.dot(a_arr, b_arr) / denom)
