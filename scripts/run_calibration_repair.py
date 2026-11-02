"""Track D1 — calibration repair (pre-registered secondary route).

Phase 0 showed NS-GRU is WORSE calibrated than GRU (the balanced-weight logistic fusion
distorts probabilities). Here we test, honestly, whether post-hoc calibration recovers it.

Method: cross-fitted (out-of-fold) isotonic calibration on the cached NS-GRU test scores
(5-fold; fit on 4 folds, predict the held-out fold) — no leakage, no in-sample optimism.
Compare ECE/Brier: raw NS-GRU vs calibrated NS-GRU vs GRU vs XGBoost (mean over seeds).

This is a proof-of-concept on cache; the production version calibrates on the validation
split inside the pipeline (a one-line change). Output: outputs/diagnosis_calibration_repair.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def ece_brier(y, p, n_bins=15):
    order = np.argsort(p)
    y, p = y[order], p[order]
    e = 0.0
    for b in np.array_split(np.arange(len(p)), n_bins):
        if len(b):
            e += (len(b) / len(p)) * abs(p[b].mean() - y[b].mean())
    return float(e), float(np.mean((p - y) ** 2))


def oof_isotonic(y, p, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    cal = np.zeros_like(p)
    for tr, te in skf.split(p, y):
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[tr], y[tr])
        cal[te] = iso.predict(p[te])
    return cal


def main():
    pred = pd.read_csv(OUT / "test_predictions.csv")
    seeds = sorted(pred.seed.unique())
    windows = sorted(pred.window.unique())
    rows = []
    for w in windows:
        acc = {k: [] for k in ["NSGRU_raw", "NSGRU_cal", "GRU", "XGB"]}
        bri = {k: [] for k in acc}
        for s in seeds:
            def get(m):
                d = pred[(pred.seed == s) & (pred.window == w) & (pred.model == m)]
                return d.y_true.to_numpy(float), d.p.to_numpy(float)
            y, p_ns = get("NS-GRU")
            _, p_gru = get("GRU")
            _, p_xgb = get("XGBoost")
            p_cal = oof_isotonic(y, p_ns, s)
            for k, p in [("NSGRU_raw", p_ns), ("NSGRU_cal", p_cal), ("GRU", p_gru), ("XGB", p_xgb)]:
                e, b = ece_brier(y, p)
                acc[k].append(e); bri[k].append(b)
        rows.append(dict(window=w,
                         ece_NSGRU_raw=np.mean(acc["NSGRU_raw"]),
                         ece_NSGRU_cal=np.mean(acc["NSGRU_cal"]),
                         ece_GRU=np.mean(acc["GRU"]),
                         ece_XGB=np.mean(acc["XGB"]),
                         brier_NSGRU_raw=np.mean(bri["NSGRU_raw"]),
                         brier_NSGRU_cal=np.mean(bri["NSGRU_cal"]),
                         brier_GRU=np.mean(bri["GRU"])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "diagnosis_calibration_repair.csv", index=False)
    show = df[["window", "ece_NSGRU_raw", "ece_NSGRU_cal", "ece_GRU", "ece_XGB"]].round(4)
    print("-- ECE: raw NS-GRU vs calibrated NS-GRU vs GRU vs XGBoost (mean over seeds) --")
    print(show.to_string(index=False))
    beats_gru = (df.ece_NSGRU_cal < df.ece_GRU).sum()
    beats_xgb = (df.ece_NSGRU_cal < df.ece_XGB).sum()
    print(f"\nCalibrated NS-GRU beats GRU ECE in {beats_gru}/{len(df)} windows; "
          f"beats XGBoost in {beats_xgb}/{len(df)} windows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
