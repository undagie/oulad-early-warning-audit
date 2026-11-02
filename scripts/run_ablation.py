"""Ablation leakage (strict vs leaky) — kuantifikasi inflasi metrik ala LEAP.

Membandingkan model yang sama di bawah protokol strict (cutoff-first) vs leaky
(fitur melihat data masa depan), cohort & split identik. Selisih AUC = besarnya
inflasi akibat leakage.

Contoh:
    python scripts/run_ablation.py --windows 2 4 6 8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.pipeline import run_window

pd.set_option("display.width", 200)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--windows", type=int, nargs="+", default=list(cfg.windows))
    ap.add_argument("--seed", type=int, default=cfg.seed)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tables = load_oulad(args.data_dir)

    rows = []
    for k in args.windows:
        strict = run_window(tables, k, args.seed, cfg, include_gru=True, leaky=False)["metrics"]
        leaky = run_window(tables, k, args.seed, cfg, include_gru=True, leaky=True)["metrics"]
        s = strict.set_index("model")["auc"]
        l = leaky.set_index("model")["auc"]
        for model in s.index:
            rows.append({
                "window": k, "model": model,
                "auc_strict": round(float(s[model]), 4),
                "auc_leaky": round(float(l[model]), 4),
                "inflation": round(float(l[model] - s[model]), 4),
            })

    df = pd.DataFrame(rows)
    df.to_csv(out / "ablation_strict_vs_leaky.csv", index=False)
    print("=== Ablation leakage: strict vs leaky (AUC) ===")
    print(df.to_string(index=False))
    print("\nRata-rata inflasi per window:")
    print(df.groupby("window")["inflation"].mean().round(4).to_string())
    print(f"\nDisimpan ke {out / 'ablation_strict_vs_leaky.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
