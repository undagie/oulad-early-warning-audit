"""Metrik predictive performance (fokus recall at-risk + AUC + PR-AUC)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calibration_metrics(y_true: np.ndarray, p: np.ndarray, n_bins: int = 15) -> dict:
    """Kalibrasi: ECE & MCE (bin equal-frequency) + Brier score. Lower = better.

    Equal-frequency binning menghindari bin kosong pada distribusi p yang timpang.
    """
    y_true = np.asarray(y_true).astype(float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    ys, ps = y_true[order], p[order]
    ece = mce = 0.0
    for b in np.array_split(np.arange(len(ps)), n_bins):
        if len(b) == 0:
            continue
        gap = abs(ps[b].mean() - ys[b].mean())
        ece += (len(b) / len(ps)) * gap
        mce = max(mce, gap)
    return {"ece": float(ece), "mce": float(mce), "brier": float(np.mean((p - y_true) ** 2))}


def predictive_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """Hitung metrik prediksi pada satu split.

    Returns dict: auc, pr_auc, f1, recall, precision, balanced_acc, threshold,
    base_rate, n, tn/fp/fn/tp, ece, mce, brier.
    """
    y_true = np.asarray(y_true).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)

    # AUC butuh kedua kelas hadir.
    both = len(np.unique(y_true)) == 2
    auc = float(roc_auc_score(y_true, p)) if both else float("nan")
    pr_auc = float(average_precision_score(y_true, p)) if both else float("nan")

    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "auc": auc,
        "pr_auc": pr_auc,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
        "threshold": float(threshold),
        "base_rate": float(y_true.mean()),
        "n": int(len(y_true)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        **calibration_metrics(y_true, p),
    }
