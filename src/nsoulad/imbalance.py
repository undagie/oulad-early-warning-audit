"""Penanganan class imbalance + tuning decision threshold.

Fokus desain: recall at-risk. Threshold dipilih di validation, bukan default 0.5.
"""

from __future__ import annotations

import numpy as np


def class_pos_weight(y: np.ndarray) -> float:
    """Bobot kelas positif untuk BCEWithLogitsLoss = n_neg / n_pos."""
    y = np.asarray(y)
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    return n_neg / max(n_pos, 1.0)


def tune_threshold(
    y_true: np.ndarray,
    p: np.ndarray,
    objective: str = "balanced_accuracy",
    min_precision: float | None = None,
) -> float:
    """Cari threshold optimal pada validation.

    objective:
      - "balanced_accuracy" (default): maksimalkan (TPR+TNR)/2 (robust utk imbalance);
      - "recall_at_precision": maksimalkan recall dengan precision >= min_precision;
      - "f1": maksimalkan F1.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    candidates = np.unique(np.concatenate([[0.0], p, [1.0]]))

    best_t, best_score = 0.5, -1.0
    P = max(int((y_true == 1).sum()), 1)
    N = max(int((y_true == 0).sum()), 1)
    for t in candidates:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tpr = tp / P
        tnr = (N - fp) / N
        precision = tp / max(tp + fp, 1)

        if objective == "balanced_accuracy":
            score = (tpr + tnr) / 2
        elif objective == "f1":
            recall = tpr
            score = 2 * precision * recall / max(precision + recall, 1e-9)
        elif objective == "recall_at_precision":
            if min_precision is None:
                raise ValueError("min_precision wajib untuk objective recall_at_precision")
            score = tpr if precision >= min_precision else -1.0
        else:
            raise ValueError(f"objective tak dikenal: {objective}")

        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t
