"""Jalankan eksperimen lengkap: semua window x seed, tulis Tabel 1-3 ke outputs/.

Checkpoint INKREMENTAL + resume: tiap (window, seed) langsung di-append ke CSV, jadi
interupsi tidak menghilangkan progres; menjalankan ulang melanjutkan dari yang tersisa.

    python scripts/run_full.py --seeds 42 43 44 45 46     # multi-seed
    python scripts/run_full.py --restart                  # mulai bersih (hapus checkpoint)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import pandas as pd

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.pipeline import run_window

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)


def _append(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--windows", type=int, nargs="+", default=list(cfg.windows))
    ap.add_argument("--seeds", type=int, nargs="+", default=[cfg.seed])
    ap.add_argument("--no-gru", action="store_true")
    ap.add_argument("--split", choices=["student", "presentation"], default="student")
    ap.add_argument("--restart", action="store_true", help="hapus checkpoint & mulai bersih")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tables = load_oulad(args.data_dir)

    hp_path = out / "hparams.json"
    hparams_all = json.loads(hp_path.read_text(encoding="utf-8")) if hp_path.exists() else {}
    if hparams_all:
        print(f"Memakai hyperparameter tuned dari {hp_path}")

    sfx = "" if args.split == "student" else "_gen"
    raw_p = out / f"metrics_raw{sfx}.csv"
    act_p = out / f"rule_activation{sfx}.csv"
    expl_p = out / f"explanation_raw{sfx}.csv"
    diag_p = out / f"rule_diagnostic_raw{sfx}.csv"
    fus_p = out / f"fusion_coefficients{sfx}.csv"
    checkpoints = [raw_p, act_p, expl_p, diag_p, fus_p]

    if args.restart:
        for p in checkpoints:
            p.unlink(missing_ok=True)

    done = set()
    if raw_p.exists():
        prev = pd.read_csv(raw_p)
        done = set(zip(prev["window"], prev["seed"]))
        print(f"Resume: {len(done)} (window,seed) sudah selesai, dilewati.")

    for k in args.windows:
        for s in args.seeds:
            if (k, s) in done:
                continue
            print(f"... window {k}, seed {s} (split={args.split})", flush=True)
            res = run_window(tables, k, s, cfg, include_gru=not args.no_gru, split_mode=args.split,
                             hparams=hparams_all.get(str(k)))
            _append(res["metrics"], raw_p)
            _append(pd.DataFrame([{"window": k, "seed": s, "rule": r, "activation": float(v)}
                                  for r, v in res["activation"].items()]), act_p)
            if res.get("explanation"):
                ex = res["explanation"]
                _append(pd.DataFrame([{"window": k, "seed": s, "coverage": ex["coverage"],
                                       "sparsity": ex["sparsity"],
                                       "contradiction_rate": ex["contradiction_rate"]}]), expl_p)
                dv = ex["diagnostic_validity"].copy()
                dv.insert(0, "seed", s); dv.insert(0, "window", k)
                _append(dv, diag_p)
            if res.get("fusion_coef"):
                fr = dict(res["fusion_coef"]); fr.update({"window": k, "seed": s})
                _append(pd.DataFrame([fr]), fus_p)

    # --- Agregasi tabel dari checkpoint ---
    raw = pd.read_csv(raw_p)
    metric_cols = ["auc", "pr_auc", "recall", "precision", "f1", "balanced_acc", "base_rate"]
    t1 = raw.groupby(["model", "window"])[metric_cols].mean().reset_index().sort_values(["window", "model"])
    t1.to_csv(out / f"table1_predictive{sfx}.csv", index=False)
    # Tabel 1 mean±std (untuk paper)
    agg = raw.groupby(["window", "model"])[["auc", "pr_auc", "recall", "balanced_acc"]].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.reset_index().to_csv(out / f"table1_predictive_multiseed{sfx}.csv", index=False)

    raw.pivot_table(index="model", columns="window", values="recall_top10", aggfunc="mean").to_csv(
        out / f"table2_early_warning{sfx}.csv")

    if expl_p.exists():
        pd.read_csv(expl_p).groupby("window")[["coverage", "sparsity", "contradiction_rate"]].mean(
        ).reset_index().to_csv(out / f"table3_explanation{sfx}.csv", index=False)
        pd.read_csv(diag_p).groupby(["window", "rule", "name", "severity"])[
            ["activation", "precision", "recall"]].mean().reset_index().to_csv(
            out / f"table3b_rule_diagnostic{sfx}.csv", index=False)

    print("\n================ TABEL 1: Predictive (mean atas seed) ================")
    print(t1.round(4).to_string(index=False))
    print(f"\nSeed selesai: {sorted(raw['seed'].unique())} | Hasil di: {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
