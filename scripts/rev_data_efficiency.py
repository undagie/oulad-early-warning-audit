"""Revisi SOIC 4706: kurva data-efficiency dengan BUDGET VALIDASI YANG SAMA.

Versi asli mengecilkan train tetapi mempertahankan validasi penuh untuk early stopping dan
fusion, sehingga label validasi menjadi supervisi tambahan. Di sini train DAN validasi
di-subsample pada fraksi yang sama (stratified), sehingga total label yang dipakai cocok
dengan fraksi. Encoder statis, threshold rule, GRU, fusion, dan threshold keputusan semuanya
di-fit hanya pada subsample. Test tetap penuh. Hyperparameter = pilihan per-seed kondisi main.

    python scripts/rev_data_efficiency.py --window 2 --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.data.windowing import build_window_view
from nsoulad.features.build import split_window_datasets
from nsoulad.models import GRUModel, LogisticFusion, XGBoostFull, align_symbolic
from nsoulad.symbolic import RuleThresholds, SymbolicEngine

ROOT = Path(__file__).resolve().parents[1]


def _stratified_mask(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        mask[idx[: max(2, int(round(len(idx) * frac)))]] = True
    return mask


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = ap.parse_args()

    tables = load_oulad(str(ROOT / cfg.data_dir))
    view = build_window_view(tables, args.window, days_per_week=cfg.days_per_week)
    out = ROOT / "outputs" / "revision" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        meta = json.loads((ROOT / "outputs" / "revision" / "preds" / "main" / f"w{args.window}_s{seed}.json")
                          .read_text(encoding="utf-8"))
        hp_gru, hp_xgb = meta["selected"]["gru"], meta["selected"]["xgboost"]
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios, seed=seed,
                                      days_per_week=cfg.days_per_week)
        train, val_full, test = parts["train"], parts["val"], parts["test"]
        for frac in args.fractions:
            tr = train.subset(_stratified_mask(train.y, frac, seed))
            va = val_full.subset(_stratified_mask(val_full.y, frac, seed + 1000))
            th = RuleThresholds.fit(view, set(tr.keys["id_student"]), days_per_week=cfg.days_per_week)
            sf = SymbolicEngine(th, days_per_week=cfg.days_per_week).apply(view).to_frame()
            s_tr, s_va, s_te = (align_symbolic(sf, d.keys) for d in (tr, va, test))

            gru = GRUModel(seed=seed, arch="gru", **hp_gru).fit(tr, va)
            pv, pt = gru.predict_proba(va), gru.predict_proba(test)
            preds = {"GRU": pt, "Rules-only": s_te.p_symbolic}
            for fs, name in (("full", "NS-GRU"), ("compact", "NS-GRU-2in")):
                fus = LogisticFusion(seed=seed, feature_set=fs).fit(pv, s_va, va.y)
                preds[name] = fus.predict_proba(pt, s_te)
            ra = GRUModel(seed=seed, arch="gru", **hp_gru).fit(
                tr, va, train_extra=s_tr.as_features(), val_extra=s_va.as_features())
            preds["NS-GRU-RA"] = ra.predict_proba(test, extra=s_te.as_features())
            preds["XGBoost"] = XGBoostFull(seed=seed, **hp_xgb).fit(tr).predict_proba(test)
            for name, p in preds.items():
                rows.append({"window": args.window, "seed": seed, "frac": frac, "n_train": tr.n, "n_val": va.n,
                             "model": name, "auc": float(roc_auc_score(test.y, p))})
            print(f"seed {seed} frac {frac:.2f}: n_train={tr.n} n_val={va.n} "
                  + " ".join(f"{k}={roc_auc_score(test.y, v):.4f}" for k, v in preds.items()), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "data_efficiency_matched.csv", index=False)
    print(df.groupby(["frac", "model"]).auc.agg(["mean", "std"]).unstack("model").round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
