"""Analisis revisi SOIC 4706 dari cache prediksi per baris (outputs/revision/preds).

Semua fungsi menerima DataFrame hasil `load_predictions` (kolom: condition, window, seed,
split, kunci unit, y, final_result, demografi, p_<model>, p_symbolic, rule R1..R7, input
rule mentah) sehingga tidak ada model yang dilatih ulang di sini, kecuali fusion logistik
kecil pada analisis rule (fit di split validasi seed yang sama).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from nsoulad.evaluation.metrics import calibration_metrics
from nsoulad.evaluation.stats import cluster_bootstrap, corrected_resampled_t_ci, tost_corrected_t
from nsoulad.imbalance import tune_threshold
from nsoulad.symbolic.rules import RULE_IDS, RULE_META, evaluate_rules

RATIO = 0.15 / 0.70          # n_test / n_train untuk corrected resampled t
MODELS = ["NS-GRU-2in", "NS-GRU", "GRU", "LSTM", "Transformer", "XGBoost", "NS-GRU-RA",
          "Rules-only", "Static-only"]
HIGH = [r for r in RULE_IDS if RULE_META[r].severity == "high"]
WEIGHTS = {r: RULE_META[r].weight for r in RULE_IDS}


def load_predictions(pred_dir: Path, condition: str) -> pd.DataFrame:
    files = sorted((pred_dir / condition).glob("w*_s*.parquet"))
    if not files:
        raise FileNotFoundError(f"Tidak ada prediksi untuk kondisi {condition} di {pred_dir}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def _groups(df: pd.DataFrame):
    for (w, s), g in df.groupby(["window", "seed"]):
        yield w, s, g[g.split == "val"], g[g.split == "test"]


def models_in(df: pd.DataFrame) -> list[str]:
    return [m for m in MODELS if f"p_{m}" in df.columns]


# ------------------------------------------------------------------ prediksi & ekuivalensi
def per_seed_metrics(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    rows = []
    for w, s, _, te in _groups(df):
        for m in models or models_in(df):
            p = te[f"p_{m}"]
            rows.append({"window": w, "seed": s, "model": m, "auc": _auc(te.y, p),
                         "pr_auc": float(average_precision_score(te.y, p)),
                         "n_test": len(te), "base_rate": float(te.y.mean())})
    return pd.DataFrame(rows)


def mean_sd(rows: pd.DataFrame, values=("auc", "pr_auc"), by=("window", "model")) -> pd.DataFrame:
    agg = rows.groupby(list(by))[list(values)].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    return agg.reset_index()


def paired_differences(rows: pd.DataFrame, pairs, metric: str = "auc", margin: float = 0.01) -> pd.DataFrame:
    """Selisih berpasangan antar split: corrected resampled t 95% + TOST (margin ekuivalensi)."""
    out = []
    piv = rows.pivot_table(index=["window", "seed"], columns="model", values=metric)
    for a, b in pairs:
        for w, g in piv.groupby(level="window"):
            d = (g[a] - g[b]).to_numpy()
            ci = corrected_resampled_t_ci(d, RATIO)
            eq = tost_corrected_t(d, RATIO, margin)
            out.append({"window": w, "comparison": f"{a} - {b}", "metric": metric, "mean": ci["mean"],
                        "ci95_low": ci["ci_low"], "ci95_high": ci["ci_high"], "p_diff": ci["p_value"],
                        "ci90_low": eq["ci90_low"], "ci90_high": eq["ci90_high"],
                        "p_tost": eq["p_tost"], "margin": margin, "equivalent": eq["p_tost"] < 0.05})
    return pd.DataFrame(out)


def clustered_delta_auc(df: pd.DataFrame, pairs, n_boot: int = 1000) -> pd.DataFrame:
    """Per seed: CI bootstrap selisih AUC yang me-resample mahasiswa (paired, klaster)."""
    out = []
    for w, s, _, te in _groups(df):
        y = te.y.to_numpy()
        for a, b in pairs:
            pa, pb = te[f"p_{a}"].to_numpy(), te[f"p_{b}"].to_numpy()
            res = cluster_bootstrap(te.id_student.to_numpy(),
                                    lambda i: _auc(y[i], pa[i]) - _auc(y[i], pb[i]),
                                    n_boot=n_boot, seed=int(s))
            out.append({"window": w, "seed": s, "comparison": f"{a} - {b}", **res})
    return pd.DataFrame(out)


def rank_agreement(df: pd.DataFrame, a: str = "NS-GRU-2in", b: str = "GRU") -> pd.DataFrame:
    """Seberapa monoton skor fusion terhadap skor neural (menjelaskan AUC identik)."""
    out = []
    for w, s, _, te in _groups(df):
        rho = spearmanr(te[f"p_{a}"], te[f"p_{b}"]).statistic
        tau = kendalltau(te[f"p_{a}"], te[f"p_{b}"]).statistic
        out.append({"window": w, "seed": s, "pair": f"{a} vs {b}", "spearman": float(rho),
                    "kendall_tau": float(tau), "discordant_pair_share": float((1 - tau) / 2)})
    return pd.DataFrame(out)


def flag_overlap(df: pd.DataFrame, pairs: list[tuple[str, str]], capacities=(0.05, 0.10, 0.20)) -> pd.DataFrame:
    """Share of the top-c flagged set of model a that model b also flags (same test split)."""
    out = []
    for w, s, _, te in _groups(df):
        for c in capacities:
            k = int(round(c * len(te)))
            for a, b in pairs:
                fa = set(te.nlargest(k, f"p_{a}").index)
                fb = set(te.nlargest(k, f"p_{b}").index)
                out.append({"window": w, "seed": s, "capacity": c, "pair": f"{a} vs {b}", "k": k,
                            "overlap": len(fa & fb) / k})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ kalibrasi & operating point
def calibrated(val: pd.DataFrame, te: pd.DataFrame, col: str) -> np.ndarray:
    iso = IsotonicRegression(out_of_bounds="clip").fit(val[col], val.y)
    return iso.predict(te[col])


def calibration_table(df: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    """ECE (15 bin equal-frequency) & Brier sebelum dan sesudah isotonic (fit di val) untuk SEMUA model."""
    rows = []
    for w, s, va, te in _groups(df):
        for m in models or models_in(df):
            raw = calibration_metrics(te.y, te[f"p_{m}"])
            cal = calibration_metrics(te.y, calibrated(va, te, f"p_{m}"))
            rows.append({"window": w, "seed": s, "model": m, "ece_raw": raw["ece"], "brier_raw": raw["brier"],
                         "ece_cal": cal["ece"], "brier_cal": cal["brier"],
                         "auc_cal": _auc(te.y, calibrated(va, te, f"p_{m}"))})
    return pd.DataFrame(rows)


def operating_points(df: pd.DataFrame, models: list[str], capacities=(0.05, 0.10, 0.20, 0.30)) -> pd.DataFrame:
    rows = []
    for w, s, _, te in _groups(df):
        y = te.y.to_numpy()
        for m in models:
            order = np.argsort(-te[f"p_{m}"].to_numpy(), kind="stable")
            for c in capacities:
                k = int(round(c * len(y)))
                tp = int(y[order[:k]].sum())
                rows.append({"window": w, "seed": s, "model": m, "capacity": c, "flagged": k,
                             "precision": tp / k, "recall": tp / y.sum(),
                             "missed_at_risk_share": 1 - tp / y.sum()})
    return pd.DataFrame(rows)


def net_benefit(df: pd.DataFrame, models: list[str], thresholds=(0.2, 0.3, 0.4, 0.5, 0.6)) -> pd.DataFrame:
    """Decision-curve net benefit pada probabilitas terkalibrasi (isotonic, fit di val)."""
    rows = []
    for w, s, va, te in _groups(df):
        y = te.y.to_numpy()
        n, prev = len(y), y.mean()
        for pt in thresholds:
            odds = pt / (1 - pt)
            rows.append({"window": w, "seed": s, "model": "Treat all", "threshold": pt,
                         "net_benefit": prev - (1 - prev) * odds})
            for m in models:
                flag = calibrated(va, te, f"p_{m}") >= pt
                tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
                rows.append({"window": w, "seed": s, "model": m, "threshold": pt,
                             "net_benefit": tp / n - fp / n * odds})
    return pd.DataFrame(rows)


def outcome_type_auc(df: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """AUC terpisah: Fail vs Pass/Distinction dan Withdrawn vs Pass/Distinction."""
    rows = []
    for w, s, _, te in _groups(df):
        neg = te.final_result.isin(["Pass", "Distinction"])
        for outcome in ("Fail", "Withdrawn"):
            sub = te[neg | (te.final_result == outcome)]
            for m in models:
                rows.append({"window": w, "seed": s, "model": m, "outcome": outcome,
                             "auc": _auc(sub.y, sub[f"p_{m}"]), "n_pos": int((sub.y == 1).sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ fairness
def _subgroups(te: pd.DataFrame) -> dict[str, pd.Series]:
    imd = te.imd_band.astype(str).str.extract(r"(\d+)")[0].astype(float)
    edu = te.highest_education.map({
        "No Formal quals": "Below A level", "Lower Than A Level": "Below A level",
        "A Level or Equivalent": "A level", "HE Qualification": "HE or above",
        "Post Graduate Qualification": "HE or above"})
    return {
        "gender": te.gender,
        "disability": te.disability.map({"Y": "Disability", "N": "No disability"}),
        "age_band": te.age_band,
        "imd_quintile": pd.cut(imd, [-1, 19, 39, 59, 79, 100],
                               labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]).astype(str)
                                .replace("nan", "Missing"),
        "highest_education": edu,
        "prior_attempt": np.where(te.num_of_prev_attempts > 0, "Repeat", "First attempt"),
        "region": te.region,
    }


def fairness_audit(df: pd.DataFrame, model: str = "NS-GRU-2in", capacity: float = 0.10,
                   n_boot: int = 500) -> pd.DataFrame:
    """Kinerja, kalibrasi, dan error per subkelompok (test digabung lintas seed, CI per mahasiswa)."""
    parts = []
    for w, s, va, te in _groups(df):
        t = te.copy()
        t["p_cal"] = calibrated(va, te, f"p_{model}")
        k = int(round(capacity * len(t)))
        t["flag"] = 0
        t.iloc[np.argsort(-t[f"p_{model}"].to_numpy(), kind="stable")[:k], t.columns.get_loc("flag")] = 1
        t["any_alert"] = (t[RULE_IDS].sum(axis=1) > 0).astype(int)
        parts.append(t)
    pooled = pd.concat(parts, ignore_index=True)
    rows = []
    for w, pw in pooled.groupby("window"):
        for attr, labels in _subgroups(pw).items():
            for level in pd.Series(labels).dropna().unique():
                g = pw[np.asarray(labels) == level]
                if len(g) < 100 or g.y.nunique() < 2:
                    continue
                y, p = g.y.to_numpy(), g.p_cal.to_numpy()
                ci = cluster_bootstrap(g.id_student.to_numpy(), lambda i: _auc(y[i], p[i]),
                                       n_boot=n_boot, seed=int(w))
                pos, neg = g.y == 1, g.y == 0
                flagged = g.flag == 1
                rows.append({
                    "window": w, "attribute": attr, "group": str(level), "n": len(g),
                    "base_rate": float(g.y.mean()), "auc": ci["estimate"], "auc_ci_low": ci["ci_low"],
                    "auc_ci_high": ci["ci_high"], "ece": calibration_metrics(y, p)["ece"],
                    "mean_pred": float(p.mean()),
                    "flag_rate": float(flagged.mean()),
                    "fpr_top10": float((flagged & neg).sum() / max(neg.sum(), 1)),
                    "fnr_top10": float((~flagged & pos).sum() / max(pos.sum(), 1)),
                    "alert_coverage_flagged": float(g.loc[flagged, "any_alert"].mean()) if flagged.any() else np.nan,
                })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rationale / rule layer
def disagreement(df: pd.DataFrame) -> pd.DataFrame:
    """Ketidaksesuaian rule vs risiko neural TERKALIBRASI, tanpa cut-point 0,5.

    - kendall_tau: korelasi peringkat P_symbolic vs P_neural terkalibrasi (tanpa ambang);
    - alert_low_risk: proporsi enrolment dengan >= 1 rule high-severity yang risiko
      terkalibrasinya di bawah base rate cohort;
    - silent_high_risk: proporsi decile risiko teratas tanpa satu pun rule aktif.
    """
    rows = []
    for w, s, va, te in _groups(df):
        p = calibrated(va, te, "p_GRU")
        base = float(va.y.mean())
        high = te.high_severity_count.to_numpy() >= 1
        top = p >= np.quantile(p, 0.9)
        none = te[RULE_IDS].sum(axis=1).to_numpy() == 0
        rows.append({"window": w, "seed": s,
                     "kendall_tau": float(kendalltau(te.p_symbolic, p).statistic),
                     "alert_low_risk": float((p[high] < base).mean()) if high.any() else np.nan,
                     "silent_high_risk": float(none[top].mean()),
                     "legacy_contradiction": float(((te.p_GRU < 0.5) & (te.high_severity_count >= 2)).mean())})
    return pd.DataFrame(rows)


def band_diagnostic(df: pd.DataFrame, lo_q: float = 0.30, hi_q: float = 0.70, n_boot: int = 500) -> pd.DataFrame:
    """Sinyal simbolik residual di band tengah skor neural (per seed), digabung lintas seed.

    Fit logistik [P_neural, P_symbolic, rule_count, high_severity_count] dilakukan in-sample pada
    band gabungan (deskriptif); CI selisih AUC me-resample mahasiswa.
    """
    te = df[df.split == "test"]
    rows = []
    for w, g in te.groupby("window"):
        parts = []
        for _, gs in g.groupby("seed"):
            lo, hi = gs.p_GRU.quantile([lo_q, hi_q])
            parts.append(gs[(gs.p_GRU >= lo) & (gs.p_GRU <= hi)])
        b = pd.concat(parts)
        y, pn = b.y.to_numpy(), b.p_GRU.to_numpy()
        X = np.column_stack([pn, b.p_symbolic, b.rule_count, b.high_severity_count])
        pc = LogisticRegression(max_iter=2000).fit(X, y).predict_proba(X)[:, 1]
        d = cluster_bootstrap(b.id_student.to_numpy(), lambda i: _auc(y[i], pc[i]) - _auc(y[i], pn[i]),
                              n_boot=n_boot, seed=int(w))
        rows.append({"window": w, "n_band": len(b), "auc_neural": _auc(y, pn), "auc_symbolic": _auc(y, b.p_symbolic),
                     "delta_auc": d["estimate"], "delta_ci_low": d["ci_low"], "delta_ci_high": d["ci_high"]})
    return pd.DataFrame(rows)


def rule_metrics(df: pd.DataFrame, n_boot: int = 500) -> pd.DataFrame:
    """Aktivasi, precision, recall, lift per rule (test digabung lintas seed, CI klaster mahasiswa)."""
    te = df[df.split == "test"]
    rows = []
    for w, g in te.groupby("window"):
        y = g.y.to_numpy()
        base = float(y.mean())
        cl = g.id_student.to_numpy()
        for r in RULE_IDS:
            f = g[r].to_numpy() == 1
            prec = cluster_bootstrap(cl, lambda i: y[i][f[i]].mean() if f[i].any() else np.nan,
                                     n_boot=n_boot, seed=int(w))
            rows.append({"window": w, "rule": r, "name": RULE_META[r].name, "activation": float(f.mean()),
                         "precision": prec["estimate"], "precision_ci_low": prec["ci_low"],
                         "precision_ci_high": prec["ci_high"], "recall": float(f[y == 1].mean()),
                         "base_rate": base, "lift": prec["estimate"] / base,
                         "above_base_rate": prec["ci_low"] > base})
    return pd.DataFrame(rows)


def rule_dependence(df: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Phi, Jaccard, dan aktivasi bersyarat P(Rj | Ri) antar pasangan rule (test, lintas seed)."""
    te = df[(df.split == "test") & (df.window == window)]
    rows = []
    for a, b in combinations(RULE_IDS, 2):
        x, z = te[a].to_numpy(bool), te[b].to_numpy(bool)
        n11, n10, n01, n00 = (x & z).sum(), (x & ~z).sum(), (~x & z).sum(), (~x & ~z).sum()
        denom = np.sqrt(float((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
        rows.append({"window": window, "rule_a": a, "rule_b": b,
                     "phi": float((n11 * n00 - n10 * n01) / denom) if denom else np.nan,
                     "jaccard": float(n11 / max(n11 + n10 + n01, 1)),
                     "p_b_given_a": float(n11 / max(n11 + n10, 1)),
                     "p_a_given_b": float(n11 / max(n11 + n01, 1))})
    return pd.DataFrame(rows)


def _sym_from(trig: pd.DataFrame, weights: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.array([weights.get(r, 0.0) for r in RULE_IDS])
    t = trig[RULE_IDS].to_numpy(float)
    p_sym = t @ w / w.sum() if w.sum() > 0 else np.zeros(len(t))
    high = t[:, [RULE_IDS.index(r) for r in HIGH]].sum(axis=1)
    return p_sym, t.sum(axis=1), high


def _fusion_auc(va, te, trig_va, trig_te, weights) -> dict:
    """Refit fusion 2-input (standar) & 4-input pada val dengan rule set/bobot tertentu."""
    sv, rc_v, hi_v = _sym_from(trig_va, weights)
    st, rc_t, hi_t = _sym_from(trig_te, weights)
    out = {"rules_only_auc": _auc(te.y, st)}
    X2v, X2t = np.column_stack([va.p_GRU, sv]), np.column_stack([te.p_GRU, st])
    sc = StandardScaler().fit(X2v)
    lr2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X2v), va.y)
    p2v, p2t = lr2.predict_proba(sc.transform(X2v))[:, 1], lr2.predict_proba(sc.transform(X2t))[:, 1]
    out["fusion2_auc"] = _auc(te.y, p2t)
    out["gamma_sym"] = float(lr2.coef_[0][1])
    X4v = np.column_stack([va.p_GRU, sv, rc_v, hi_v])
    X4t = np.column_stack([te.p_GRU, st, rc_t, hi_t])
    lr4 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(X4v, va.y)
    out["fusion4_auc"] = _auc(te.y, lr4.predict_proba(X4t)[:, 1])
    thr = tune_threshold(va.y.to_numpy(), p2v)
    flagged = p2t >= thr
    active = trig_te[RULE_IDS].to_numpy().sum(axis=1)
    out["coverage"] = float((active[flagged] > 0).mean()) if flagged.any() else np.nan
    out["sparsity"] = float(active[flagged & (active > 0)].mean()) if (flagged & (active > 0)).any() else np.nan
    return out


def leave_one_rule_out(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w, s, va, te in _groups(df):
        rows.append({"window": w, "seed": s, "rule_set": "all", **_fusion_auc(va, te, va, te, WEIGHTS)})
        for r in RULE_IDS:
            wts = {k: v for k, v in WEIGHTS.items() if k != r}
            tv, tt = va[RULE_IDS].copy(), te[RULE_IDS].copy()
            tv[r] = 0
            tt[r] = 0
            rows.append({"window": w, "seed": s, "rule_set": f"without {r}", **_fusion_auc(va, te, tv, tt, wts)})
    return pd.DataFrame(rows)


def weight_sensitivity(df: pd.DataFrame, n_random: int = 100, seed: int = 0) -> pd.DataFrame:
    """Bobot seragam, 100 perturbasi acak (x U[0.5, 1.5]), dan bobot terpelajar (logistik di val)."""
    rng = np.random.default_rng(seed)
    draws = [{r: WEIGHTS[r] * rng.uniform(0.5, 1.5) for r in RULE_IDS} for _ in range(n_random)]
    rows = []
    for w, s, va, te in _groups(df):
        rows.append({"window": w, "seed": s, "weights": "design", **_fusion_auc(va, te, va, te, WEIGHTS)})
        rows.append({"window": w, "seed": s, "weights": "uniform",
                     **_fusion_auc(va, te, va, te, {r: 1.0 for r in RULE_IDS})})
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(va[RULE_IDS], va.y)
        learned = {r: max(float(c), 0.0) for r, c in zip(RULE_IDS, lr.coef_[0])}
        rows.append({"window": w, "seed": s, "weights": "learned (val)", **_fusion_auc(va, te, va, te, learned)})
        for i, d in enumerate(draws):
            rows.append({"window": w, "seed": s, "weights": f"random_{i}", **_fusion_auc(va, te, va, te, d)})
    return pd.DataFrame(rows)


THRESHOLD_GRID = {
    "consecutive_weeks": [1, 2, 3],
    "scheduled_ratio": [0.2, 0.3, 0.4],
    "low_score": [30, 40, 50],
    "min_late": [1, 2],
}
_RULE_COLS = ["run_zero_active", "run_below_q1", "scheduled_resource_ratio", "first_score", "n_scored",
              "has_assessment", "n_late", "early_engagement", "early_median_course", "has_due_assessment",
              "active_days_7d_pre_deadline"]


def _retrigger(part: pd.DataFrame, params: dict) -> pd.DataFrame:
    rf = part[_RULE_COLS].copy()
    rf["num_of_prev_attempts"] = part["num_of_prev_attempts_rule"] if "num_of_prev_attempts_rule" in part \
        else part["num_of_prev_attempts"]
    return evaluate_rules(rf.reset_index(drop=True), params=params).set_index(part.index)


def threshold_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Ubah satu ambang rule pada satu waktu; ukur aktivasi/precision rule terkait & AUC."""
    affected = {"consecutive_weeks": ["R1", "R2"], "scheduled_ratio": ["R3"], "low_score": ["R4"],
                "min_late": ["R5"]}
    rows = []
    for w, s, va, te in _groups(df):
        for param, values in THRESHOLD_GRID.items():
            for v in values:
                tv, tt = _retrigger(va, {param: v}), _retrigger(te, {param: v})
                res = _fusion_auc(va, te, tv, tt, WEIGHTS)
                for r in affected[param]:
                    f = tt[r].to_numpy() == 1
                    rows.append({"window": w, "seed": s, "parameter": param, "value": v, "rule": r,
                                 "activation": float(f.mean()),
                                 "precision": float(te.y.to_numpy()[f].mean()) if f.any() else np.nan,
                                 "val_precision": float(va.y.to_numpy()[tv[r].to_numpy() == 1].mean())
                                 if (tv[r] == 1).any() else np.nan,
                                 "base_rate": float(te.y.mean()), **res})
    return pd.DataFrame(rows)
