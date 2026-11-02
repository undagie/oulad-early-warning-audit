"""Tabel ringkasan turunan dari metrics_raw*.csv (TANPA retrain).

Menghasilkan, untuk split student ('') dan generalisasi ('_gen') bila ada:
  - table1_full_metrics{sfx}.csv  : auc/pr_auc/f1/precision/recall/balanced_acc mean±std
  - table4_confusion{sfx}.csv     : confusion matrix (tp/fp/fn/tn) rata-rata seed + turunannya
  - table2b_precision_top10{sfx}.csv : precision@top-10% per model×window

    python scripts/make_summary_tables.py
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

OUT = Path("outputs")

FULL_METRICS = ["auc", "pr_auc", "f1", "precision", "recall", "balanced_acc"]


def _mean_std(raw: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    agg = raw.groupby(["window", "model"])[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def build(sfx: str) -> bool:
    raw_p = OUT / f"metrics_raw{sfx}.csv"
    if not raw_p.exists():
        return False
    raw = pd.read_csv(raw_p)
    n_seed = raw["seed"].nunique()

    # --- Table 1 full: semua metrik prediktif mean±std ---
    _mean_std(raw, FULL_METRICS).round(4).to_csv(OUT / f"table1_full_metrics{sfx}.csv", index=False)

    # --- Table 4 confusion matrix (rata-rata atas seed) ---
    conf = raw.groupby(["window", "model"]).agg(
        n=("n", "mean"), base_rate=("base_rate", "mean"),
        tp=("tp", "mean"), fp=("fp", "mean"), fn=("fn", "mean"), tn=("tn", "mean"),
    ).reset_index()
    # Turunan dari count rata-rata (konsisten dgn confusion matrix yang dilaporkan).
    conf["precision"] = conf["tp"] / (conf["tp"] + conf["fp"]).clip(lower=1e-9)
    conf["recall"] = conf["tp"] / (conf["tp"] + conf["fn"]).clip(lower=1e-9)
    conf["f1"] = 2 * conf["precision"] * conf["recall"] / (conf["precision"] + conf["recall"]).clip(lower=1e-9)
    tpr = conf["recall"]
    tnr = conf["tn"] / (conf["tn"] + conf["fp"]).clip(lower=1e-9)
    conf["balanced_acc"] = (tpr + tnr) / 2
    for c in ["n", "tp", "fp", "fn", "tn"]:
        conf[c] = conf[c].round(0).astype(int)
    conf.round({"base_rate": 3, "precision": 3, "recall": 3, "f1": 3, "balanced_acc": 3}).to_csv(
        OUT / f"table4_confusion{sfx}.csv", index=False)

    # --- Table 2b precision@top-10% ---
    if "precision_top10" in raw.columns:
        raw.pivot_table(index="model", columns="window", values="precision_top10",
                        aggfunc="mean").round(4).to_csv(OUT / f"table2b_precision_top10{sfx}.csv")

    tag = "student-split" if sfx == "" else "generalization (2013->2014)"
    print(f"[{tag}] {n_seed} seed -> table1_full_metrics{sfx}.csv, table4_confusion{sfx}.csv, "
          f"table2b_precision_top10{sfx}.csv")
    return True


def main() -> int:
    any_done = False
    for sfx in ("", "_gen"):
        any_done |= build(sfx)
    if not any_done:
        print("Tidak ada metrics_raw*.csv. Jalankan scripts/run_full.py dulu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
