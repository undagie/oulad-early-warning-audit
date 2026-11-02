"""Test fungsi evaluasi (metrics, early-warning, explanation, statistik)."""

import numpy as np
import pandas as pd
import pytest

from nsoulad.evaluation.early_warning import precision_at_top_k, recall_at_top_k
from nsoulad.evaluation.explanation import (
    contradiction_rate,
    coverage,
    rule_diagnostic_validity,
)
from nsoulad.evaluation.metrics import predictive_metrics
from nsoulad.evaluation.shap_baseline import _features_for_mass
from nsoulad.evaluation.stats import (
    aggregate_seeds,
    corrected_resampled_t_ci,
    holm_bonferroni,
    mcnemar_test,
    paired_bootstrap_auc_diff,
)
from nsoulad.symbolic.rules import RULE_IDS


def test_predictive_metrics_perfect():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = predictive_metrics(y, p, threshold=0.5)
    assert m["auc"] == 1.0
    assert m["recall"] == 1.0
    assert m["precision"] == 1.0
    assert m["base_rate"] == 0.5


def test_recall_precision_at_top_k():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert recall_at_top_k(y, p, 0.5) == 1.0    # top-2 = dua positif
    assert precision_at_top_k(y, p, 0.5) == 1.0


def _triggered(rows):
    return pd.DataFrame(rows, columns=RULE_IDS)


def test_coverage_and_contradiction():
    # 3 mahasiswa; R1 col index 0.
    trig = _triggered([
        [1, 0, 0, 0, 0, 0, 0],   # ada rule
        [0, 0, 0, 0, 0, 0, 0],   # tak ada rule
        [1, 0, 0, 1, 0, 0, 1],   # 2 high (R1,R4,R7 -> R1,R4,R7 high)
    ])
    pred = np.array([1, 1, 1])
    assert coverage(trig, pred) == 2 / 3
    # contradiction: neural rendah & >=2 high severity -> hanya baris ke-3.
    p_neural = np.array([0.9, 0.1, 0.2])
    assert contradiction_rate(p_neural, trig) == 1 / 3


def test_rule_diagnostic_validity():
    trig = _triggered([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
    ])
    y = np.array([1, 1, 0])
    dv = rule_diagnostic_validity(trig, y).set_index("rule")
    assert dv.loc["R1", "precision"] == 1.0   # R1 menyala -> selalu at-risk
    assert dv.loc["R1", "recall"] == 1.0      # menangkap semua positif


def test_mcnemar_and_bootstrap():
    y = np.array([0, 0, 1, 1, 0, 1])
    pred1 = np.array([0, 0, 1, 1, 0, 1])   # sempurna
    pred2 = np.array([1, 1, 0, 0, 1, 0])   # selalu salah
    mc = mcnemar_test(y, pred1, pred2)
    assert mc["p_value"] < 0.1

    p1 = np.array([0.1, 0.2, 0.9, 0.8, 0.05, 0.95])
    p2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    res = paired_bootstrap_auc_diff(y, p1, p2, n_boot=200, seed=0)
    assert res["auc_diff"] > 0


def test_holm_bonferroni():
    r = holm_bonferroni([0.01, 0.04, 0.03], alpha=0.05)
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monoton -> [0.03, 0.06, 0.06]
    assert r["p_adjusted"][0] == pytest.approx(0.03)
    assert r["reject"][0] is True
    assert r["reject"][1] is False
    # adjusted >= raw
    assert all(adj >= raw for adj, raw in zip(r["p_adjusted"], [0.01, 0.04, 0.03]))


def test_features_for_mass():
    v = np.array([[10.0, 0, 0, 0], [1.0, 1, 1, 1]])
    # baris0: 1 fitur cukup utk 80%; baris1: butuh 4 -> mean 2.5
    assert _features_for_mass(v, 0.8) == pytest.approx(2.5)


def test_aggregate_seeds():
    a = pd.DataFrame({"window": [4], "model": ["X"], "seed": [1], "auc": [0.8]})
    b = pd.DataFrame({"window": [4], "model": ["X"], "seed": [2], "auc": [0.6]})
    agg = aggregate_seeds([a, b])
    assert agg.loc[0, "auc_mean"] == 0.7


def test_corrected_resampled_t_ci():
    d = [0.01, 0.02, 0.03, 0.04, 0.05]
    # ratio 0 -> t-interval biasa: se = s / sqrt(J)
    plain = corrected_resampled_t_ci(d, test_train_ratio=0.0)
    assert plain["se"] == pytest.approx(np.std(d, ddof=1) / np.sqrt(5))
    # koreksi Nadeau-Bengio: se = sqrt((1/J + ratio) * s^2), t(0.975, 4) = 2.7764
    r = corrected_resampled_t_ci(d, test_train_ratio=0.25)
    assert r["mean"] == pytest.approx(0.03)
    assert r["se"] == pytest.approx(np.sqrt(0.45 * 0.00025))
    assert r["ci_high"] - r["mean"] == pytest.approx(2.776445 * r["se"], rel=1e-5)
    assert r["ci_low"] < plain["ci_low"] and r["p_value"] > plain["p_value"]
