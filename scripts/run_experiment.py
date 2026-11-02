"""Jalankan eksperimen (MVP: satu window, satu seed).

Contoh:
    python scripts/run_experiment.py --window 4 --seed 42
    python scripts/run_experiment.py --window 4 --no-gru     # tanpa GRU/NS-GRU (cepat)
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.pipeline import run_window

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

REPORT_COLS = ["model", "auc", "pr_auc", "recall", "precision", "f1", "balanced_acc", "threshold"]


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--no-gru", action="store_true")
    args = ap.parse_args()

    tables = load_oulad(args.data_dir)
    print(f"Window {args.window}, seed {args.seed} — melatih model...")
    out = run_window(tables, args.window, args.seed, cfg, include_gru=not args.no_gru)

    print(f"\nn_train={out['n_train']} n_val={out['n_val']} n_test={out['n_test']}")
    print("\n=== Predictive performance (test) ===")
    print(out["metrics"][REPORT_COLS].round(4).to_string(index=False))

    if out["fusion_coef"]:
        print("\n=== Koefisien fusion NS-GRU (learned weights) ===")
        print({k: round(v, 4) for k, v in out["fusion_coef"].items()})

    if out.get("explanation"):
        ex = out["explanation"]
        print("\n=== Explanation quality (NS-GRU, test) ===")
        print(f"coverage={ex['coverage']:.3f}  sparsity={ex['sparsity']:.2f}  "
              f"contradiction_rate={ex['contradiction_rate']:.3f}")
        print("\nRule diagnostic validity (precision = P(at-risk | rule fired)):")
        print(ex["diagnostic_validity"].round(3).to_string(index=False))

    print("\n=== Rule-activation rate ===")
    print({r: round(float(v), 3) for r, v in out["activation"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
