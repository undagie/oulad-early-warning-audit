"""Evaluasi ulang baseline Transformer setelah penambahan positional embedding.

Protokol identik dengan Tabel 1 (run_full.py): student-level split per seed, HP dari
outputs/hparams.json (Transformer mewarisi HP GRU), threshold dituning di val. Hasil
dibandingkan berpasangan per (window, seed) dengan baris Transformer lama (tanpa
positional embedding) di outputs/metrics_raw.csv.

    python scripts/run_transformer_posenc.py --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd  # noqa: E402

from nsoulad.config import load_config  # noqa: E402
from nsoulad.data.loader import load_oulad  # noqa: E402
from nsoulad.data.windowing import build_window_view  # noqa: E402
from nsoulad.features.build import split_window_datasets  # noqa: E402
from nsoulad.models import GRUModel  # noqa: E402
from nsoulad.pipeline import _evaluate  # noqa: E402


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--windows", type=int, nargs="+", default=list(cfg.windows))
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    hp_all = json.loads((out / "hparams.json").read_text(encoding="utf-8"))
    tables = load_oulad(args.data_dir)
    rows = []

    for k in args.windows:
        view = build_window_view(tables, k, days_per_week=cfg.days_per_week)
        hp = hp_all.get(str(k), {}).get("transformer", {})
        for s in args.seeds:
            parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios,
                                          seed=s, days_per_week=cfg.days_per_week)
            train, val, test = parts["train"], parts["val"], parts["test"]
            mdl = GRUModel(seed=s, arch="transformer", **hp).fit(train, val)
            m, _ = _evaluate("Transformer+pos", mdl.predict_proba(val), mdl.predict_proba(test),
                             val.y, test.y)
            m.update({"window": k, "seed": s})
            rows.append(m)
            print(f"window {k} seed {s}: AUC {m['auc']:.4f}", flush=True)

    new = pd.DataFrame(rows)
    new.to_csv(out / "transformer_posenc_raw.csv", index=False)

    old = pd.read_csv(out / "metrics_raw.csv")
    old = old[old["model"] == "Transformer"][["window", "seed", "auc", "pr_auc"]]
    gru = pd.read_csv(out / "metrics_raw.csv")
    gru = gru[gru["model"] == "GRU"][["window", "seed", "auc"]].rename(columns={"auc": "auc_gru"})
    cmp = (new[["window", "seed", "auc", "pr_auc"]]
           .merge(old, on=["window", "seed"], suffixes=("_pos", "_nopos"))
           .merge(gru, on=["window", "seed"]))
    cmp["delta_vs_nopos"] = cmp["auc_pos"] - cmp["auc_nopos"]
    cmp["delta_vs_gru"] = cmp["auc_pos"] - cmp["auc_gru"]
    summary = cmp.groupby("window").agg(
        auc_pos_mean=("auc_pos", "mean"), auc_pos_std=("auc_pos", "std"),
        auc_nopos_mean=("auc_nopos", "mean"), auc_nopos_std=("auc_nopos", "std"),
        auc_gru_mean=("auc_gru", "mean"),
        delta_vs_nopos_mean=("delta_vs_nopos", "mean"), delta_vs_nopos_std=("delta_vs_nopos", "std"),
        delta_vs_gru_mean=("delta_vs_gru", "mean"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    summary.to_csv(out / "transformer_posenc_summary.csv", index=False)
    print("\n=== Transformer: dengan vs tanpa positional embedding (AUC test) ===")
    print(summary.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
