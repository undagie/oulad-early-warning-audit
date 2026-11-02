"""Revisi SOIC 4706: hitung semua tabel revisi dari cache prediksi (tanpa melatih ulang).

Menulis CSV ke outputs/revision/tables/. Kondisi yang belum selesai dilewati.

    python scripts/rev_analyze.py
    python scripts/rev_analyze.py --only-conditions   # sensitivitas, transfer, LOMO saja (cepat)
"""

from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from nsoulad.evaluation import revision_analysis as ra
from nsoulad.evaluation.stats import corrected_resampled_t_ci

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "outputs" / "revision" / "preds"
TAB = ROOT / "outputs" / "revision" / "tables"
PAIRS = [("NS-GRU-2in", "GRU"), ("NS-GRU-2in", "XGBoost"), ("NS-GRU-2in", "Static-only"),
         ("NS-GRU", "GRU"), ("NS-GRU-RA", "GRU"), ("LSTM", "GRU"), ("Transformer", "GRU")]
CORE = ["NS-GRU-2in", "GRU", "XGBoost", "Static-only"]


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB / f"{name}.csv", index=False)
    print(f"  {name}: {len(df)} rows")


def _cond(name: str) -> pd.DataFrame | None:
    try:
        return ra.load_predictions(PRED, name)
    except FileNotFoundError:
        print(f"[skip] {name}: belum ada prediksi")
        return None


def strict_vs(cond: str, main_rows: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """AUC kondisi sensitivitas vs kondisi utama (seed sama), selisih dengan corrected t."""
    df = _cond(cond)
    if df is None:
        return pd.DataFrame()
    rows = ra.per_seed_metrics(df, models)
    m = rows.merge(main_rows, on=["window", "seed", "model"], suffixes=("", "_main"))
    out = []
    for (w, mod), g in m.groupby(["window", "model"]):
        d = (g.auc - g.auc_main).to_numpy()
        ci = corrected_resampled_t_ci(d, ra.RATIO) if len(d) > 1 else {"ci_low": float("nan"), "ci_high": float("nan")}
        out.append({"condition": cond, "window": w, "model": mod, "n_seeds": len(g),
                    "auc_mean": g.auc.mean(), "auc_sd": g.auc.std(), "auc_main_mean": g.auc_main.mean(),
                    "delta_mean": d.mean(), "delta_ci_low": ci["ci_low"], "delta_ci_high": ci["ci_high"],
                    "n_test_mean": g.n_test.mean(), "base_rate_mean": g.base_rate.mean()})
    return pd.DataFrame(out)


def main() -> int:
    TAB.mkdir(parents=True, exist_ok=True)
    main_df = _cond("main")
    if main_df is None:
        return 1
    print(f"main: windows {sorted(main_df.window.unique())}, seeds {sorted(main_df.seed.unique())}")

    rows = ra.per_seed_metrics(main_df)
    if "--only-conditions" in sys.argv:
        return _conditions(rows)
    save(rows, "main_per_seed")
    save(ra.mean_sd(rows), "main_predictive")
    save(ra.paired_differences(rows, PAIRS, "auc"), "main_paired_auc")
    save(ra.paired_differences(rows, PAIRS, "pr_auc"), "main_paired_prauc")
    save(ra.clustered_delta_auc(main_df, PAIRS[:3]), "main_clustered_delta_auc")
    save(pd.concat([ra.rank_agreement(main_df, "NS-GRU-2in", "GRU"),
                    ra.rank_agreement(main_df, "NS-GRU", "GRU")]), "rank_agreement")
    save(ra.flag_overlap(main_df, [("NS-GRU-2in", "GRU"), ("NS-GRU-2in", "XGBoost"), ("GRU", "XGBoost")]),
         "flag_overlap")
    save(ra.calibration_table(main_df), "calibration_all_models")
    save(ra.operating_points(main_df, CORE), "operating_points")
    save(ra.net_benefit(main_df, CORE), "net_benefit")
    save(ra.outcome_type_auc(main_df, CORE), "outcome_type_auc")
    save(ra.fairness_audit(main_df), "fairness")
    save(ra.disagreement(main_df), "disagreement")
    save(ra.band_diagnostic(main_df), "band_diagnostic")
    save(ra.rule_metrics(main_df), "rule_metrics")
    save(pd.concat([ra.rule_dependence(main_df, w) for w in sorted(main_df.window.unique())]), "rule_dependence")
    save(ra.leave_one_rule_out(main_df), "rule_leave_one_out")
    save(ra.weight_sensitivity(main_df), "rule_weight_sensitivity")
    save(ra.threshold_sensitivity(main_df), "rule_threshold_sensitivity")

    tuning = []
    for f in sorted((PRED / "main").glob("*.json")):
        meta = pd.read_json(f, typ="series")
        for entry in meta["tuning_log"]:
            tuning.append({"window": meta["window"], "seed": meta["seed"], **entry})
    save(pd.DataFrame(tuning), "tuning_log")
    return _conditions(rows)


def _conditions(rows: pd.DataFrame) -> int:
    sens = [strict_vs(c, rows, ["GRU", "NS-GRU-2in", "XGBoost", "Static-only", "Rules-only"])
            for c in ("leaky", "lag7", "lag14", "lag21", "fixed_cohort", "enrolment_cohort")]
    if any(len(s) for s in sens):
        save(pd.concat([s for s in sens if len(s)]), "sensitivity_vs_main")

    for cond in ("transfer_disjoint",):
        df = _cond(cond)
        if df is not None:
            r = ra.per_seed_metrics(df, ["NS-GRU-2in", "NS-GRU", "GRU", "XGBoost", "Static-only", "Rules-only"])
            save(ra.mean_sd(r), f"{cond}_predictive")
            save(ra.paired_differences(r, [("NS-GRU-2in", "GRU"), ("NS-GRU-2in", "XGBoost")]), f"{cond}_paired")

    lomo = [_cond(f"lomo_{m}") for m in ("AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG")]
    lomo = [d for d in lomo if d is not None]
    if lomo:
        parts = []
        for d in lomo:
            r = ra.per_seed_metrics(d, ["NS-GRU-2in", "GRU", "XGBoost", "Static-only", "Rules-only"])
            r.insert(0, "held_out_module", d.condition.iloc[0].removeprefix("lomo_"))
            parts.append(r)
        save(pd.concat(parts), "lomo_per_seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
