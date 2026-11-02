"""Phase 0 diagnosis (pre-registered) — runs on CACHED predictions, no retraining.

Answers, per window (mean over seeds), from outputs/test_predictions.csv + cohort_triggered.csv:
  (D1) Calibration: ECE(15 equal-freq bins), Brier, MCE per model.
  (H-A1) Conditional complementarity: within the neural-uncertain band P_neural in [Q30,Q70],
         does {P_symbolic, rule_count} add ROC-AUC over P_neural? Paired-bootstrap p.
  Complementarity map: per P_neural decile, base rate + AUC of rule_count for the label.

Outputs: outputs/diagnosis_calibration.csv, diagnosis_residual.csv, diagnosis_decile_map.csv
Writes a human verdict to stdout and outputs/diagnosis_log.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
KEYS = ["code_module", "code_presentation", "id_student"]
RULES = ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
RNG = np.random.default_rng(42)


def ece_brier_mce(y: np.ndarray, p: np.ndarray, n_bins: int = 15):
    order = np.argsort(p)
    y, p = y[order], p[order]
    e = mce = 0.0
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b) == 0:
            continue
        gap = abs(p[b].mean() - y[b].mean())
        e += (len(b) / len(p)) * gap
        mce = max(mce, gap)
    brier = float(np.mean((p - y) ** 2))
    return float(e), brier, float(mce)


def load_merged():
    pred = pd.read_csv(OUT / "test_predictions.csv")
    trig = pd.read_csv(OUT / "cohort_triggered.csv")
    trig["rule_count"] = trig[RULES].sum(axis=1)
    return pred, trig[["seed", "window", *KEYS, "rule_count"]]


def per_sw_frame(pred, trig, seed, window):
    sub = pred[(pred.seed == seed) & (pred.window == window)]
    wide = sub.pivot_table(index=KEYS, columns="model", values="p").reset_index()
    y = sub.drop_duplicates(KEYS).set_index(KEYS)["y_true"]
    wide = wide.merge(y, on=KEYS)
    tr = trig[(trig.seed == seed) & (trig.window == window)]
    wide = wide.merge(tr[[*KEYS, "rule_count"]], on=KEYS, how="left")
    wide["rule_count"] = wide["rule_count"].fillna(0.0)
    return wide


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan


def paired_bootstrap_delta(y, p_base, p_aug, B=2000):
    """P(AUC_aug - AUC_base <= 0) via paired resampling."""
    n = len(y)
    base0, aug0 = auc(y, p_base), auc(y, p_aug)
    if np.isnan(base0) or np.isnan(aug0):
        return np.nan, np.nan
    cnt = 0
    for _ in range(B):
        idx = RNG.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        if auc(yy, p_aug[idx]) - auc(yy, p_base[idx]) <= 0:
            cnt += 1
    return float(aug0 - base0), cnt / B


def main():
    pred, trig = load_merged()
    seeds = sorted(pred.seed.unique())
    windows = sorted(pred.window.unique())
    models = ["GRU", "NS-GRU", "XGBoost", "Rules-only", "Static-only"]

    cal_rows, res_rows, dec_rows = [], [], []
    log = []

    for w in windows:
        # ---- Calibration (per model, mean over seeds) ----
        for m in models:
            vals = []
            for s in seeds:
                sub = pred[(pred.seed == s) & (pred.window == w) & (pred.model == m)]
                if sub.empty:
                    continue
                vals.append(ece_brier_mce(sub.y_true.to_numpy(float), sub.p.to_numpy(float)))
            vals = np.array(vals)
            cal_rows.append(dict(window=w, model=m,
                                 ece=vals[:, 0].mean(), ece_std=vals[:, 0].std(),
                                 brier=vals[:, 1].mean(), mce=vals[:, 2].mean()))

        # ---- H-A1 residual signal in neural-uncertain band (pool band rows over seeds) ----
        band_y, band_pn, band_ps, band_rc = [], [], [], []
        full_dauc = []
        for s in seeds:
            f = per_sw_frame(pred, trig, s, w)
            pn, ps, rc, y = (f["GRU"].to_numpy(float), f["Rules-only"].to_numpy(float),
                             f["rule_count"].to_numpy(float), f["y_true"].to_numpy(float))
            # full-cohort marginal value of symbolic (in-sample LR, optimism comparable)
            base = LogisticRegression(max_iter=1000).fit(pn.reshape(-1, 1), y).predict_proba(pn.reshape(-1, 1))[:, 1]
            Xa = np.column_stack([pn, ps, rc])
            augf = LogisticRegression(max_iter=1000).fit(Xa, y).predict_proba(Xa)[:, 1]
            full_dauc.append(auc(y, augf) - auc(y, base))
            lo, hi = np.quantile(pn, [0.30, 0.70])
            mask = (pn >= lo) & (pn <= hi)
            band_y.append(y[mask]); band_pn.append(pn[mask]); band_ps.append(ps[mask]); band_rc.append(rc[mask])
        by, bpn, bps, brc = map(lambda a: np.concatenate(a), (band_y, band_pn, band_ps, band_rc))
        # base = P_neural within band; aug = LR(P_neural + P_sym + rule_count) within band
        base_b = bpn
        Xb = np.column_stack([bpn, bps, brc])
        aug_b = LogisticRegression(max_iter=1000).fit(Xb, by).predict_proba(Xb)[:, 1]
        dauc_band, p_boot = paired_bootstrap_delta(by, base_b, aug_b)
        res_rows.append(dict(
            window=w, n_band=len(by), base_rate_band=float(by.mean()),
            auc_neural_band=auc(by, bpn), auc_sym_alone_band=auc(by, bps),
            auc_rulecount_alone_band=auc(by, brc), auc_aug_band=auc(by, aug_b),
            dAUC_band=dauc_band, bootstrap_p=p_boot,
            dAUC_fullcohort=float(np.mean(full_dauc))))

        # ---- Complementarity map: per P_neural decile (seed 42) ----
        f0 = per_sw_frame(pred, trig, seeds[0], w)
        pn0, y0, rc0, ps0 = (f0["GRU"].to_numpy(float), f0["y_true"].to_numpy(float),
                             f0["rule_count"].to_numpy(float), f0["Rules-only"].to_numpy(float))
        dec = pd.qcut(pn0, 10, labels=False, duplicates="drop")
        for d in np.unique(dec):
            mm = dec == d
            dec_rows.append(dict(window=w, decile=int(d), n=int(mm.sum()),
                                 base_rate=float(y0[mm].mean()), mean_pneural=float(pn0[mm].mean()),
                                 mean_rulecount=float(rc0[mm].mean()),
                                 auc_rulecount=auc(y0[mm], rc0[mm]) if mm.sum() > 20 else np.nan))

    cal = pd.DataFrame(cal_rows); res = pd.DataFrame(res_rows); dmap = pd.DataFrame(dec_rows)
    cal.to_csv(OUT / "diagnosis_calibration.csv", index=False)
    res.to_csv(OUT / "diagnosis_residual.csv", index=False)
    dmap.to_csv(OUT / "diagnosis_decile_map.csv", index=False)

    # ---- Verdict on H-A1 ----
    sig = ((res.dAUC_band > 0) & (res.bootstrap_p < 0.05)).sum()
    verdict = ("CONFIRMED" if sig >= 2 else "REJECTED")
    log.append("=== Phase 0 diagnosis ===")
    log.append(f"Seeds={seeds} windows={windows}")
    log.append("\n-- Calibration (ECE lower=better), mean over seeds --")
    log.append(cal.pivot(index="model", columns="window", values="ece").round(4).to_string())
    log.append("\n-- Residual signal in neural-uncertain band [Q30,Q70] --")
    log.append(res[["window", "n_band", "base_rate_band", "auc_neural_band",
                    "auc_sym_alone_band", "auc_rulecount_alone_band", "auc_aug_band",
                    "dAUC_band", "bootstrap_p", "dAUC_fullcohort"]].round(4).to_string(index=False))
    log.append(f"\nH-A1 (residual signal in >=2 windows, p<0.05): {verdict}  "
               f"[{sig}/{len(windows)} windows significant]")
    text = "\n".join(log)
    print(text)
    (OUT / "diagnosis_log.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
