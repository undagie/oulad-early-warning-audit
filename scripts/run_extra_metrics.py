"""Dump prediksi per-mahasiswa lalu hitung metrik yang butuh prediksi mentah:
ROC/PR curve, detection lead time, dan Jaccard explanation stability.

Melatih ulang NS-GRU/GRU/XGBoost/Static-only/Rules-only untuk tiap (seed, window),
menyimpan prediksi test + triggered-rules cohort, lalu menurunkan:
  - outputs/test_predictions{sfx}.csv     (untuk ROC/PR + lead time)
  - outputs/cohort_triggered{sfx}.csv     (untuk Jaccard stability)
  - outputs/detection_leadtime{sfx}.csv   (+ detection_curve{sfx}.csv)
  - outputs/jaccard_stability{sfx}.csv

    python scripts/run_extra_metrics.py --seeds 42 43 44
    python scripts/run_extra_metrics.py --split presentation --seeds 42 43 44
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nsoulad.config import load_config  # noqa: E402
from nsoulad.data.loader import load_oulad  # noqa: E402
from nsoulad.data.schema import UNIT_KEYS  # noqa: E402
from nsoulad.data.windowing import build_window_view  # noqa: E402
from nsoulad.evaluation.explanation import jaccard_stability  # noqa: E402
from nsoulad.features.build import presentation_split_datasets, split_window_datasets  # noqa: E402
from nsoulad.imbalance import tune_threshold  # noqa: E402
from nsoulad.models import (  # noqa: E402
    GRUModel, LogisticFusion, StaticOnly, XGBoostFull, align_symbolic,
)
from nsoulad.symbolic import RuleThresholds, SymbolicEngine  # noqa: E402
from nsoulad.symbolic.rules import RULE_IDS  # noqa: E402

KEYS = list(UNIT_KEYS)
TAU = 0.10  # operating point top-10% (konsisten dgn recall@top-k)
PRED_MODELS = ["NS-GRU", "GRU", "XGBoost", "Static-only", "Rules-only"]
LEADTIME_MODELS = ["NS-GRU", "GRU", "XGBoost"]


def _train_window(tables, k, seed, cfg, split_mode):
    view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
    if split_mode == "presentation":
        parts = presentation_split_datasets(
            view, tables.student_info,
            train_presentations=cfg.split.train_presentations,
            test_presentations=cfg.split.test_presentations,
            seed=seed, days_per_week=cfg.days_per_week)
    else:
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios,
                                      seed=seed, days_per_week=cfg.days_per_week)
    train, val, test = parts["train"], parts["val"], parts["test"]

    th = RuleThresholds.fit(view, set(train.keys["id_student"]), days_per_week=cfg.days_per_week)
    rule_res = SymbolicEngine(th, days_per_week=cfg.days_per_week).apply(view)
    sf = rule_res.to_frame()
    sym_val, sym_test = align_symbolic(sf, val.keys), align_symbolic(sf, test.keys)

    gru = GRUModel(seed=seed).fit(train, val)
    pv_gru, pt_gru = gru.predict_proba(val), gru.predict_proba(test)
    fusion = LogisticFusion(seed=seed).fit(pv_gru, sym_val, val.y)
    xgb = XGBoostFull(seed=seed).fit(train)
    sto = StaticOnly(seed=seed).fit(train)

    probs = {
        "NS-GRU": (fusion.predict_proba(pv_gru, sym_val), fusion.predict_proba(pt_gru, sym_test)),
        "GRU": (pv_gru, pt_gru),
        "XGBoost": (xgb.predict_proba(val), xgb.predict_proba(test)),
        "Static-only": (sto.predict_proba(val), sto.predict_proba(test)),
        "Rules-only": (sym_val.p_symbolic, sym_test.p_symbolic),
    }

    pred_rows = []
    for name, (pv, pt) in probs.items():
        t = tune_threshold(val.y, pv)
        df = test.keys[KEYS].reset_index(drop=True).copy()
        df.insert(0, "model", name)
        df.insert(0, "window", k)
        df.insert(0, "seed", seed)
        df["y_true"] = np.asarray(test.y).astype(int)
        df["p"] = np.asarray(pt, dtype=float)
        df["threshold"] = float(t)
        pred_rows.append(df)

    # Triggered-rules seluruh cohort (untuk Jaccard antar-window).
    trig = rule_res.keys[KEYS].copy()
    trig.insert(0, "window", k)
    trig.insert(0, "seed", seed)
    for r in RULE_IDS:
        trig[r] = rule_res.triggered[r].to_numpy().astype(int)

    return pd.concat(pred_rows, ignore_index=True), trig


def detection_leadtime(preds: pd.DataFrame, windows: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lead time & detection curve per model, dirata-rata atas seed.

    Untuk tiap (seed, window) tandai top-TAU sebagai 'terdeteksi'. Untuk mahasiswa
    at-risk, cari window deteksi pertama; lead = k_max - first_window.
    """
    k_max = max(windows)
    lt_rows, curve_rows = [], []
    for model in LEADTIME_MODELS:
        pm = preds[preds["model"] == model]
        per_seed_lead, per_seed_rate = [], []
        cum_by_seed = {k: [] for k in windows}
        for seed, g in pm.groupby("seed"):
            g = g.copy()
            # tandai top-TAU per window
            g["detected"] = 0
            for k, gk in g.groupby("window"):
                kk = max(1, int(round(len(gk) * TAU)))
                thr = gk["p"].nlargest(kk).min()
                g.loc[gk.index[gk["p"] >= thr], "detected"] = 1
            atrisk = g[g["y_true"] == 1]
            # first detection window per mahasiswa at-risk
            det = atrisk[atrisk["detected"] == 1]
            first = det.groupby(KEYS)["window"].min()
            total_ar = atrisk.drop_duplicates(KEYS).shape[0]
            if total_ar == 0:
                continue
            per_seed_rate.append(len(first) / total_ar)
            if len(first):
                per_seed_lead.append(float((k_max - first).mean()))
            for k in windows:
                cum_by_seed[k].append(float((first <= k).sum()) / total_ar)
        if not per_seed_rate:
            continue
        lt_rows.append({
            "model": model,
            "detection_rate_mean": round(float(np.mean(per_seed_rate)), 4),
            "detection_rate_std": round(float(np.std(per_seed_rate)), 4),
            "mean_lead_weeks_mean": round(float(np.mean(per_seed_lead)), 3) if per_seed_lead else float("nan"),
            "mean_lead_weeks_std": round(float(np.std(per_seed_lead)), 3) if per_seed_lead else float("nan"),
        })
        for k in windows:
            curve_rows.append({"model": model, "window": k,
                               "cum_detect_frac": round(float(np.mean(cum_by_seed[k])), 4)})
    return pd.DataFrame(lt_rows), pd.DataFrame(curve_rows)


def jaccard_table(trig: pd.DataFrame, windows: list[int]) -> pd.DataFrame:
    """Jaccard stability rata-rata atas seed untuk pasangan window berdekatan."""
    pairs = list(zip(windows[:-1], windows[1:]))
    rows = []
    for a, b in pairs:
        vals = []
        for seed, g in trig.groupby("seed"):
            ga, gb = g[g["window"] == a], g[g["window"] == b]
            if len(ga) == 0 or len(gb) == 0:
                continue
            j = jaccard_stability(ga[RULE_IDS].reset_index(drop=True), ga[KEYS].reset_index(drop=True),
                                  gb[RULE_IDS].reset_index(drop=True), gb[KEYS].reset_index(drop=True))
            if not np.isnan(j):
                vals.append(j)
        if vals:
            rows.append({"window_pair": f"{a}->{b}", "jaccard_mean": round(float(np.mean(vals)), 4),
                         "jaccard_std": round(float(np.std(vals)), 4), "n_seed": len(vals)})
    return pd.DataFrame(rows)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--windows", type=int, nargs="+", default=list(cfg.windows))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--split", choices=["student", "presentation"], default="student")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sfx = "" if args.split == "student" else "_gen"
    tables = load_oulad(args.data_dir)

    all_preds, all_trig = [], []
    for seed in args.seeds:
        for k in args.windows:
            print(f"... predictions: window {k}, seed {seed} (split={args.split})", flush=True)
            preds, trig = _train_window(tables, k, seed, cfg, args.split)
            all_preds.append(preds); all_trig.append(trig)

    preds = pd.concat(all_preds, ignore_index=True)
    trig = pd.concat(all_trig, ignore_index=True)
    preds.to_csv(out / f"test_predictions{sfx}.csv", index=False)
    trig.to_csv(out / f"cohort_triggered{sfx}.csv", index=False)

    lt, curve = detection_leadtime(preds, args.windows)
    lt.to_csv(out / f"detection_leadtime{sfx}.csv", index=False)
    curve.to_csv(out / f"detection_curve{sfx}.csv", index=False)
    jac = jaccard_table(trig, args.windows)
    jac.to_csv(out / f"jaccard_stability{sfx}.csv", index=False)

    print("\n=== Detection lead time (top-10%, ref=week 8) ===")
    print(lt.to_string(index=False))
    print("\n=== Jaccard explanation stability (antar-window) ===")
    print(jac.to_string(index=False))
    print(f"\nPrediksi + metrik disimpan ke {out.resolve()} (sfx='{sfx or 'student'}')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
