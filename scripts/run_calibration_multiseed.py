"""Full multi-seed calibration table (5 seeds x 4 windows) — publication-ready mean+/-std.

Runs the pipeline (GRU-only archs, for speed) and captures AUC/ECE/Brier for GRU, XGBoost,
NS-GRU, and NS-GRU-cal. Confirms the Phase-0 finding at scale: NS-GRU-cal is the best-
calibrated model while AUC is preserved. Non-destructive: writes its own CSVs.

Output: outputs/calibration_multiseed_raw.csv, outputs/table_calibration.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.pipeline import run_window

SEEDS = [42, 43, 44, 45, 46]
WINDOWS = [2, 4, 6, 8]
MODELS = ["GRU", "XGBoost", "NS-GRU", "NS-GRU-cal"]
OUT = Path(__file__).resolve().parents[1] / "outputs"
RAW = OUT / "calibration_multiseed_raw.csv"


def main():
    cfg = load_config()
    tables = load_oulad(str(cfg.data_dir))
    done = set()
    if RAW.exists():
        prev = pd.read_csv(RAW)
        done = set(zip(prev.window, prev.seed))
        print(f"Resume: {len(done)} (window,seed) done.", flush=True)

    for k in WINDOWS:
        for s in SEEDS:
            if (k, s) in done:
                continue
            print(f"... window {k}, seed {s}", flush=True)
            out = run_window(tables, k, s, cfg, include_gru=True, neural_archs=("gru",))
            m = out["metrics"]
            m = m[m.model.isin(MODELS)][["window", "seed", "model", "auc", "ece", "brier"]]
            m.to_csv(RAW, mode="a", header=not RAW.exists(), index=False)

    raw = pd.read_csv(RAW)
    agg = raw.groupby(["window", "model"])[["auc", "ece", "brier"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv(OUT / "table_calibration.csv", index=False)

    print("\n== Calibration (mean over 5 seeds) ==")
    piv = raw.groupby(["model", "window"])[["auc", "ece"]].mean().reset_index()
    print("ECE (lower=better):")
    print(piv.pivot(index="model", columns="window", values="ece").round(4).to_string())
    print("\nAUC (preserved?):")
    print(piv.pivot(index="model", columns="window", values="auc").round(4).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
