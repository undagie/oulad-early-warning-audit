"""CI 95% selisih AUC berpasangan antar 5 split (corrected resampled t, Nadeau & Bengio 2003).

Memakai AUC test per (window, seed, model) dari outputs/metrics_raw.csv (hasil run_full.py,
5 seed). Tidak ada pelatihan ulang. Rasio n_test/n_train = 0.15/0.70 mengikuti split
student-level 70/15/15.

    python scripts/run_auc_diff_ci.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd  # noqa: E402

from nsoulad.config import load_config  # noqa: E402
from nsoulad.evaluation.stats import corrected_resampled_t_ci  # noqa: E402

COMPARISONS = [("NS-GRU", "GRU"), ("NS-GRU", "XGBoost"), ("NS-GRU", "Static-only")]


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    out = Path(args.out)

    r_train, _, r_test = cfg.split.ratios
    ratio = r_test / r_train
    raw = pd.read_csv(out / "metrics_raw.csv")
    auc = raw.pivot_table(index=["window", "seed"], columns="model", values="auc")

    rows = []
    for k in sorted(auc.index.get_level_values("window").unique()):
        a = auc.loc[k]
        for m1, m2 in COMPARISONS:
            d = (a[m1] - a[m2]).dropna()
            ci = corrected_resampled_t_ci(d.to_numpy(), test_train_ratio=ratio)
            rows.append({"window": k, "comparison": f"{m1} - {m2}", **ci})

    df = pd.DataFrame(rows)
    df.to_csv(out / "auc_diff_ci.csv", index=False)
    pd.set_option("display.width", 160)
    print(f"n_test/n_train = {ratio:.4f}\n")
    print(df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
