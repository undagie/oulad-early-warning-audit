"""Metrik early-warning utility: recall/precision@top-k."""

from __future__ import annotations

import numpy as np


def recall_at_top_k(y_true: np.ndarray, p: np.ndarray, k_frac: float = 0.10) -> float:
    """Fraksi seluruh mahasiswa at-risk yang tertangkap di top-k% skor risiko tertinggi."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y_true)
    pos = int((y_true == 1).sum())
    if pos == 0 or n == 0:
        return float("nan")
    k = max(1, int(round(n * k_frac)))
    top_idx = np.argsort(-p)[:k]
    return float((y_true[top_idx] == 1).sum() / pos)


def precision_at_top_k(y_true: np.ndarray, p: np.ndarray, k_frac: float = 0.10) -> float:
    """Fraksi mahasiswa at-risk di antara top-k% skor risiko tertinggi."""
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    k = max(1, int(round(n * k_frac)))
    top_idx = np.argsort(-p)[:k]
    return float((y_true[top_idx] == 1).mean())
