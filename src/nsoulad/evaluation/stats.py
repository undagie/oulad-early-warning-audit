"""Statistik: agregasi multi-seed, bootstrap CI, dan uji signifikansi.

Memakai paired bootstrap untuk selisih AUC (robust, setara DeLong untuk tujuan
prosiding), McNemar untuk perbandingan klasifikasi, dan corrected resampled t
(Nadeau & Bengio, 2003) untuk CI selisih berpasangan antar split acak berulang.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def aggregate_seeds(metrics_list: list[pd.DataFrame], by: str = "model",
                    keys: tuple[str, ...] = ("window", "model")) -> pd.DataFrame:
    """Gabungkan beberapa DataFrame metrik (per seed) -> mean & std per (window, model)."""
    allm = pd.concat(metrics_list, ignore_index=True)
    num = allm.select_dtypes(include=[np.number]).columns.difference(["seed"])
    agg = allm.groupby(list(keys))[list(num)].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def bootstrap_ci(y_true: np.ndarray, p: np.ndarray, metric=roc_auc_score,
                 n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """CI bootstrap (persentil) untuk sebuah metrik skalar."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); p = np.asarray(p)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(metric(y_true[idx], p[idx]))
    if not vals:
        return (float("nan"), float("nan"))
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi


def paired_bootstrap_auc_diff(y_true: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                              n_boot: int = 2000, seed: int = 0) -> dict:
    """Uji selisih AUC (model1 - model2) via paired bootstrap. Returns diff & p-value 2-sisi."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); p1 = np.asarray(p1); p2 = np.asarray(p2)
    n = len(y_true)
    obs = roc_auc_score(y_true, p1) - roc_auc_score(y_true, p2)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y_true[idx], p1[idx]) - roc_auc_score(y_true[idx], p2[idx]))
    diffs = np.asarray(diffs)
    # p-value 2-sisi: proporsi bootstrap yang menyeberang nol relatif arah observasi.
    if obs >= 0:
        p_val = 2 * float((diffs <= 0).mean())
    else:
        p_val = 2 * float((diffs >= 0).mean())
    return {"auc_diff": float(obs), "p_value": min(p_val, 1.0)}


def corrected_resampled_t_ci(diffs, test_train_ratio: float, alpha: float = 0.05) -> dict:
    """CI selisih berpasangan atas J split acak berulang (Nadeau & Bengio, 2003).

    Split acak berulang berbagi sebagian data train, sehingga selisih antar split
    berkorelasi dan t biasa terlalu sempit. Varians sampel dikalikan
    (1/J + n_test/n_train) lalu dipakai distribusi t dengan J-1 derajat bebas.
    """
    from scipy.stats import t as t_dist

    d = np.asarray(diffs, dtype=float)
    J = len(d)
    if J < 2:
        raise ValueError("Butuh minimal 2 split untuk CI.")
    mean = float(d.mean())
    se = float(np.sqrt((1.0 / J + test_train_ratio) * d.var(ddof=1)))
    half = float(t_dist.ppf(1 - alpha / 2, J - 1)) * se
    stat = mean / se if se > 0 else float("inf") * np.sign(mean)
    p = float(2 * t_dist.sf(abs(stat), J - 1)) if se > 0 else float(mean == 0)
    return {"mean": mean, "se": se, "ci_low": mean - half, "ci_high": mean + half,
            "t": float(stat), "p_value": p, "n_splits": J}


def tost_corrected_t(diffs, test_train_ratio: float, margin: float) -> dict:
    """Uji ekuivalensi TOST (two one-sided tests) dengan SE corrected resampled t.

    H0: |selisih| >= margin. Ekuivalen pada alpha 0.05 bila p_tost < 0.05, setara dengan
    CI 90% seluruhnya di dalam (-margin, +margin).
    """
    from scipy.stats import t as t_dist

    d = np.asarray(diffs, dtype=float)
    J = len(d)
    mean = float(d.mean())
    se = float(np.sqrt((1.0 / J + test_train_ratio) * d.var(ddof=1)))
    if se == 0:
        p = 0.0 if abs(mean) < margin else 1.0
        return {"mean": mean, "se": se, "ci90_low": mean, "ci90_high": mean, "p_tost": p,
                "margin": margin}
    p_lower = float(t_dist.sf((mean + margin) / se, J - 1))   # H0: diff <= -margin
    p_upper = float(t_dist.cdf((mean - margin) / se, J - 1))  # H0: diff >= +margin
    half = float(t_dist.ppf(0.95, J - 1)) * se
    return {"mean": mean, "se": se, "ci90_low": mean - half, "ci90_high": mean + half,
            "p_tost": max(p_lower, p_upper), "margin": margin}


def cluster_bootstrap(clusters, stat_fn, n_boot: int = 1000, alpha: float = 0.05,
                      seed: int = 0) -> dict:
    """CI bootstrap persentil yang me-resample KLASTER (mis. id_student), bukan baris.

    clusters: array label klaster per baris. stat_fn(idx) menerima indeks baris hasil
    resample dan mengembalikan skalar (NaN diabaikan). Enrolment dari mahasiswa yang sama
    ikut atau keluar bersama, sehingga ketergantungan dalam mahasiswa dihormati.
    """
    rng = np.random.default_rng(seed)
    clusters = np.asarray(clusters)
    uniq, inv = np.unique(clusters, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    starts = np.searchsorted(inv[order], np.arange(len(uniq)))
    ends = np.append(starts[1:], len(order))
    point = float(stat_fn(np.arange(len(clusters))))
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([order[starts[c]:ends[c]] for c in pick])
        v = stat_fn(idx)
        if v == v:  # bukan NaN
            vals.append(v)
    if not vals:
        return {"estimate": point, "ci_low": float("nan"), "ci_high": float("nan")}
    return {"estimate": point,
            "ci_low": float(np.percentile(vals, 100 * alpha / 2)),
            "ci_high": float(np.percentile(vals, 100 * (1 - alpha / 2)))}


def holm_bonferroni(pvalues: list[float], alpha: float = 0.05) -> dict:
    """Koreksi Holm-Bonferroni untuk multiple comparisons.

    Returns dict berisi p-value terkoreksi (urutan asli) dan keputusan reject pada alpha.
    """
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = min((m - rank) * p[idx], 1.0)
        running_max = max(running_max, val)  # jaga monotonisitas
        adjusted[idx] = running_max
    return {
        "p_adjusted": adjusted.tolist(),
        "reject": (adjusted < alpha).tolist(),
    }


def mcnemar_test(y_true: np.ndarray, pred1: np.ndarray, pred2: np.ndarray) -> dict:
    """McNemar untuk dua klasifier (benar/salah). Returns b, c, dan p-value chi2 (1 dof)."""
    from scipy.stats import chi2

    y_true = np.asarray(y_true).astype(int)
    c1 = (np.asarray(pred1).astype(int) == y_true)
    c2 = (np.asarray(pred2).astype(int) == y_true)
    b = int((c1 & ~c2).sum())  # model1 benar, model2 salah
    c = int((~c1 & c2).sum())  # model1 salah, model2 benar
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)  # koreksi kontinuitas
    return {"b": b, "c": c, "statistic": float(stat), "p_value": float(chi2.sf(stat, 1))}
