"""E3 — Mekanisme fusion: apakah NS-GRU bersandar lebih ke symbolic saat data sedikit?

Sepanjang kurva data (Week 2, fraksi 1-100%), latih GRU + fusion, lalu catat
KONTRIBUSI TERSTANDARISASI tiap fitur fusion = |koef| x std(fitur). Standarisasi
perlu karena P_neural in [0,1] tak sebanding dengan rule_count (0-7).

Hipotesis NeSy (RQ3): rasio reliance symbolic / neural NAIK saat data mengecil ->
fusion belajar menyandarkan diri pada pengetahuan pedagogis saat bukti neural lemah.

    python scripts/run_fusion_analysis.py --window 2 --seeds 42 43 44
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
from nsoulad.features.build import split_window_datasets  # noqa: E402
from nsoulad.models import GRUModel, LogisticFusion, align_symbolic  # noqa: E402
from nsoulad.models.fusion import FUSION_FEATURES  # noqa: E402
from nsoulad.symbolic import RuleThresholds, SymbolicEngine  # noqa: E402

SYMBOLIC_FEATS = ["P_symbolic", "rule_count", "high_severity_count"]


def _stratified_mask(y, frac, seed):
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
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.01, 0.05, 0.10, 0.25, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    tables = load_oulad(args.data_dir)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, ratio_rows = [], []

    for seed in args.seeds:
        view = build_window_view(tables, args.window, days_per_week=cfg.days_per_week)
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios, seed=seed,
                                      days_per_week=cfg.days_per_week)
        train, val = parts["train"], parts["val"]

        for frac in args.fractions:
            tr = train.subset(_stratified_mask(train.y, frac, seed))
            th = RuleThresholds.fit(view, set(tr.keys["id_student"]), days_per_week=cfg.days_per_week)
            sf = SymbolicEngine(th, days_per_week=cfg.days_per_week).apply(view).to_frame()
            sym_val = align_symbolic(sf, val.keys)

            gru = GRUModel(seed=seed).fit(tr, val)
            pv = gru.predict_proba(val)
            fusion = LogisticFusion(seed=seed).fit(pv, sym_val, val.y)

            X = sym_val.fusion_matrix(pv)              # [n,4] selaras FUSION_FEATURES
            coef = fusion.clf.coef_[0]
            std = X.std(axis=0) + 1e-9
            contrib = np.abs(coef) * std               # kontribusi terstandarisasi

            d = dict(zip(FUSION_FEATURES, contrib))
            for f in FUSION_FEATURES:
                rows.append({"seed": seed, "frac": frac, "feature": f, "std_contrib": d[f]})
            sym_total = sum(d[f] for f in SYMBOLIC_FEATS)
            ratio = sym_total / (d["P_neural"] + 1e-9)
            ratio_rows.append({"seed": seed, "frac": frac, "symbolic_reliance_ratio": ratio,
                               "neural_contrib": d["P_neural"], "symbolic_contrib": sym_total})
            print(f"seed {seed} frac {frac:.2f}: reliance(sym/neural)={ratio:.3f}")

    df = pd.DataFrame(rows)
    rr = pd.DataFrame(ratio_rows)
    df.to_csv(out / "fusion_contrib.csv", index=False)
    rr.to_csv(out / "fusion_reliance.csv", index=False)

    print("\n=== Kontribusi terstandarisasi tiap fitur fusion (mean atas seed) ===")
    piv = df.groupby(["frac", "feature"])["std_contrib"].mean().reset_index().pivot(
        index="frac", columns="feature", values="std_contrib")
    print(piv[FUSION_FEATURES].round(4).to_string())

    print("\n=== Rasio reliance symbolic/neural vs fraksi (mean±std) ===")
    agg = rr.groupby("frac")["symbolic_reliance_ratio"].agg(["mean", "std"])
    print(agg.round(3).to_string())

    # Figur: rasio reliance vs fraksi.
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    a = agg.reset_index().sort_values("frac")
    ax.plot(a["frac"], a["mean"], marker="o", color="#7c3aed")
    ax.fill_between(a["frac"], a["mean"] - a["std"], a["mean"] + a["std"], alpha=0.15, color="#7c3aed")
    ax.set_xscale("log")
    ax.set_xlabel("Training-set fraction (log scale)")
    ax.set_ylabel("Symbolic / neural reliance\n(standardized contribution)")
    ax.set_title("Fusion reliance on the symbolic branch by training-set size")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    (out / "figures").mkdir(exist_ok=True)
    fig.savefig(out / "figures" / "fig7_fusion_reliance.png", dpi=150)
    print(f"\nDisimpan: fusion_contrib.csv, fusion_reliance.csv, figures/fig7_fusion_reliance.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
