"""Baseline post-hoc XAI: SHAP (TreeExplainer atas XGBoost) untuk perbandingan actionability.

SHAP memberi atribusi fitur, bukan recourse actionable. Metrik di sini sejajar dgn
perbandingan counterfactual: seberapa sering atribusi teratas jatuh pada fitur
**immutable** (demografi enrollment) -> tidak bisa ditindaklanjuti. Rule-based = 0%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _features_for_mass(abs_v: np.ndarray, frac: float) -> float:
    """Rata-rata #fitur untuk mencapai `frac` dari total |atribusi| per kasus (sparsity)."""
    out = []
    for row in abs_v:
        s = np.sort(row)[::-1]
        tot = s.sum()
        if tot <= 0:
            out.append(0)
            continue
        c = np.cumsum(s) / tot
        out.append(int(np.searchsorted(c, frac) + 1))
    return float(np.mean(out)) if out else float("nan")


def shap_actionability(model, X_df: pd.DataFrame, immutable_cols: set[str],
                       pred: np.ndarray, top_k: int = 3) -> dict:
    """Hitung metrik actionability SHAP untuk kasus yang diprediksi at-risk.

    model: XGBClassifier terlatih; X_df: fitur (kolom bernama) untuk subset uji;
    immutable_cols: nama kolom fitur immutable; pred: prediksi biner (1=at-risk).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    vals = np.asarray(explainer.shap_values(X_df))
    if vals.ndim == 3:  # beberapa versi: [n,features,classes] atau [classes,n,features]
        vals = vals[..., -1] if vals.shape[-1] == 2 else vals[-1]
    abs_v = np.abs(vals)

    cols = list(X_df.columns)
    imm = np.array([c in immutable_cols for c in cols])
    mask = np.asarray(pred).astype(bool)
    if mask.sum() == 0:
        return {"n": 0, "shap_top1_immutable_rate": float("nan"),
                f"shap_top{top_k}_immutable_rate": float("nan"), "shap_sparsity80": float("nan")}

    sub = abs_v[mask]
    top1 = sub.argmax(axis=1)
    topk = np.argsort(-sub, axis=1)[:, :top_k]
    return {
        "n": int(mask.sum()),
        "shap_top1_immutable_rate": float(imm[top1].mean()),
        f"shap_top{top_k}_immutable_rate": float(np.mean([imm[r].any() for r in topk])),
        "shap_sparsity80": _features_for_mass(sub, 0.80),
    }
