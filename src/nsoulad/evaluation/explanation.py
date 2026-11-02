"""Metrik kualitas explanation (proxy actionability tanpa human eval).

Karena tidak ada penilai manusia, actionability diukur lewat proxy yang tidak
sirkular: rule diagnostic validity, coverage, sparsity, contradiction, stability,
dan (bila tersedia) perbandingan terhadap counterfactual recourse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nsoulad.symbolic.rules import RULE_IDS, RULE_META


def coverage(triggered: pd.DataFrame, pred: np.ndarray) -> float:
    """% prediksi at-risk yang memiliki >=1 rule aktif (actionable)."""
    pred = np.asarray(pred).astype(bool)
    if pred.sum() == 0:
        return float("nan")
    any_rule = triggered[RULE_IDS].sum(axis=1).to_numpy() > 0
    return float(any_rule[pred].mean())


def explanation_sparsity(triggered: pd.DataFrame, pred: np.ndarray) -> float:
    """Rata-rata jumlah rule aktif per prediksi at-risk (makin ringkas makin baik)."""
    pred = np.asarray(pred).astype(bool)
    if pred.sum() == 0:
        return float("nan")
    counts = triggered[RULE_IDS].sum(axis=1).to_numpy()
    return float(counts[pred].mean())


def rule_diagnostic_validity(triggered: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Untuk tiap rule: precision = P(at-risk | rule), recall = P(rule | at-risk).

    Mengukur apakah rule benar-benar menandai mahasiswa at-risk (bukan sekadar
    'punya aksi'). Ini proxy actionability yang TIDAK sirkular.
    """
    y = np.asarray(y).astype(int)
    pos = max(int((y == 1).sum()), 1)
    rows = []
    for r in RULE_IDS:
        fired = triggered[r].to_numpy().astype(bool)
        n_fired = int(fired.sum())
        tp = int(((fired) & (y == 1)).sum())
        rows.append({
            "rule": r,
            "name": RULE_META[r].name,
            "severity": RULE_META[r].severity,
            "activation": float(fired.mean()),
            "precision": tp / n_fired if n_fired else float("nan"),
            "recall": tp / pos,
        })
    return pd.DataFrame(rows)


def contradiction_rate(p_neural: np.ndarray, triggered: pd.DataFrame, neural_threshold: float = 0.5,
                       min_high_severity: int = 2) -> float:
    """% kasus: neural memprediksi risiko rendah tetapi >= min_high_severity rule high aktif."""
    p_neural = np.asarray(p_neural, dtype=float)
    high_rules = [r for r in RULE_IDS if RULE_META[r].severity == "high"]
    high_count = triggered[high_rules].sum(axis=1).to_numpy()
    contradict = (p_neural < neural_threshold) & (high_count >= min_high_severity)
    return float(contradict.mean())


def jaccard_stability(trig_a: pd.DataFrame, keys_a: pd.DataFrame,
                      trig_b: pd.DataFrame, keys_b: pd.DataFrame) -> float:
    """Rata-rata Jaccard similarity himpunan rule aktif antar dua window berdekatan.

    Hanya untuk mahasiswa yang muncul di kedua window (irisan cohort).
    """
    from nsoulad.data.schema import UNIT_KEYS
    keys = list(UNIT_KEYS)

    a = keys_a.reset_index(drop=True).copy()
    b = keys_b.reset_index(drop=True).copy()
    a["_sa"] = [frozenset(r for r in RULE_IDS if trig_a.iloc[i][r]) for i in range(len(a))]
    b["_sb"] = [frozenset(r for r in RULE_IDS if trig_b.iloc[i][r]) for i in range(len(b))]
    merged = a.merge(b, on=keys, how="inner")
    if len(merged) == 0:
        return float("nan")

    sims = []
    for sa, sb in zip(merged["_sa"], merged["_sb"]):
        union = sa | sb
        sims.append(1.0 if not union else len(sa & sb) / len(union))
    return float(np.mean(sims))


def immutable_violation_rate(changed_features: list[set[str]], immutable: set[str]) -> float:
    """Untuk baseline counterfactual recourse: % rekomendasi yang menyentuh fitur immutable.

    `changed_features[i]` = himpunan nama fitur yang diubah counterfactual untuk kasus i.
    Rule-based selalu 0 (tidak pernah menyarankan ubah fitur immutable) -> pembanding adil.
    """
    if not changed_features:
        return float("nan")
    viol = [1.0 if (s & immutable) else 0.0 for s in changed_features]
    return float(np.mean(viol))
