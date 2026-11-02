"""Kurva data-efficiency: apakah neuro-symbolic menang di rezim low-data?

Untuk window early (default Week 2), latih model pada fraksi train yang mengecil
{1%,5%,10%,25%,100%}, evaluasi di test penuh. Hipotesis (tesis disertasi):
prior simbolik mengisi kekosongan data -> NS-GRU/Rules-only mengungguli GRU murni
saat data sedikit; gap menutup pada data penuh.

Semua yang di-fit (encoder, threshold rule, GRU, fusion) memakai HANYA subsample
train -> simulasi low-data yang jujur. Val/test tetap penuh.

    python scripts/run_data_efficiency.py --window 2 --seeds 42 43 44
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from nsoulad.config import load_config  # noqa: E402
from nsoulad.data.loader import load_oulad  # noqa: E402
from nsoulad.data.windowing import build_window_view  # noqa: E402
from nsoulad.evaluation.metrics import predictive_metrics  # noqa: E402
from nsoulad.features.build import split_window_datasets  # noqa: E402
from nsoulad.imbalance import tune_threshold  # noqa: E402
from nsoulad.models import GRUModel, LogisticFusion, XGBoostFull, align_symbolic  # noqa: E402
from nsoulad.symbolic import RuleThresholds, SymbolicEngine  # noqa: E402


def _stratified_mask(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = np.zeros(len(y), dtype=bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n = max(2, int(round(len(idx) * frac)))
        mask[idx[:n]] = True
    return mask


def _auc_recall(val_p, test_p, val_y, test_y):
    t = tune_threshold(val_y, val_p)
    m = predictive_metrics(test_y, test_p, threshold=t)
    return m["auc"], m["recall"]


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    tables = load_oulad(args.data_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []

    for seed in args.seeds:
        view = build_window_view(tables, args.window, days_per_week=cfg.days_per_week)
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios, seed=seed,
                                      days_per_week=cfg.days_per_week)
        train, val, test = parts["train"], parts["val"], parts["test"]

        for frac in args.fractions:
            mask = _stratified_mask(train.y, frac, seed)
            tr = train.subset(mask)
            sub_ids = set(tr.keys["id_student"])

            # Symbolic threshold di-fit pada subsample (low-data jujur).
            th = RuleThresholds.fit(view, sub_ids, days_per_week=cfg.days_per_week)
            rr = SymbolicEngine(th, days_per_week=cfg.days_per_week).apply(view)
            sf = rr.to_frame()
            sym_train = align_symbolic(sf, tr.keys)
            sym_val = align_symbolic(sf, val.keys)
            sym_test = align_symbolic(sf, test.keys)

            # GRU + NS-GRU (late fusion)
            gru = GRUModel(seed=seed).fit(tr, val)
            pv_gru, pt_gru = gru.predict_proba(val), gru.predict_proba(test)
            fusion = LogisticFusion(seed=seed).fit(pv_gru, sym_val, val.y)
            pv_ns, pt_ns = fusion.predict_proba(pv_gru, sym_val), fusion.predict_proba(pt_gru, sym_test)

            # NS-GRU-RA (integrasi lebih erat: fitur symbolic disuntik ke GRU)
            ra = GRUModel(seed=seed, arch="gru").fit(
                tr, val, train_extra=sym_train.as_features(), val_extra=sym_val.as_features())
            pv_ra = ra.predict_proba(val, extra=sym_val.as_features())
            pt_ra = ra.predict_proba(test, extra=sym_test.as_features())

            # XGBoost (neural-free) + Rules-only (training-free)
            xgb = XGBoostFull(seed=seed).fit(tr)
            pv_xgb, pt_xgb = xgb.predict_proba(val), xgb.predict_proba(test)

            for name, (pv, pt) in {
                "GRU": (pv_gru, pt_gru),
                "NS-GRU": (pv_ns, pt_ns),
                "NS-GRU-RA": (pv_ra, pt_ra),
                "XGBoost": (pv_xgb, pt_xgb),
                "Rules-only": (sym_val.p_symbolic, sym_test.p_symbolic),
            }.items():
                auc, rec = _auc_recall(pv, pt, val.y, test.y)
                rows.append({"seed": seed, "frac": frac, "n_train": tr.n,
                             "model": name, "auc": auc, "recall": rec})
            print(f"seed {seed} frac {frac:.2f} (n={tr.n}) done")

    df = pd.DataFrame(rows)
    df.to_csv(out / "data_efficiency.csv", index=False)

    agg = df.groupby(["frac", "model"])["auc"].agg(["mean", "std"]).reset_index()
    print("\n=== AUC vs train fraction (mean atas seed) ===")
    piv = agg.pivot(index="frac", columns="model", values="mean").round(4)
    print(piv.to_string())

    # Figur
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for model, g in agg.groupby("model"):
        g = g.sort_values("frac")
        ax.plot(g["frac"], g["mean"], marker="o", label=model)
        ax.fill_between(g["frac"], g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.12)
    ax.set_xscale("log")
    ax.set_xlabel("Training-set fraction (log scale)")
    ax.set_ylabel(f"ROC-AUC (test, Week {args.window})")
    ax.set_title(f"Data efficiency across training-set fractions (Week {args.window})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "fig6_data_efficiency.png", dpi=150)
    print(f"\nDisimpan: {out/'data_efficiency.csv'} + figures/fig6_data_efficiency.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
