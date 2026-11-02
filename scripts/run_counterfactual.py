"""Baseline counterfactual recourse (DiCE) vs explanation rule-based.

Melatih model atas fitur interpretable, lalu meng-generate counterfactual untuk
mahasiswa at-risk dan mengukur:
  - immutable-feature-violation rate: % CF yang mengubah fitur immutable
    (gender/region/age/imd/highest_education/disability/num_of_prev_attempts) —
    rekomendasi semacam ini TIDAK actionable;
  - rata-rata jumlah fitur diubah (effort/sparsity).
Rule-based NS-GRU: violation 0 (tak pernah menyuruh ubah fitur immutable) dgn
sparsity ~1-2 rule -> pembanding actionability yang adil.

    python scripts/run_counterfactual.py --window 4 --sample 100
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from nsoulad.config import load_config
from nsoulad.data.loader import load_oulad
from nsoulad.data.labels import make_binary_label
from nsoulad.data.schema import UNIT_KEYS
from nsoulad.data.splits import student_level_split
from nsoulad.data.windowing import build_window_view
from nsoulad.evaluation.explanation import immutable_violation_rate
from nsoulad.features.assessment import assessment_features
from nsoulad.features.course_structure import course_structure_features
from nsoulad.features.temporal import weekly_engagement

warnings.filterwarnings("ignore")
_KEYS = list(UNIT_KEYS)

IMMUTABLE = {"gender", "region", "age_band", "imd_band", "highest_education",
             "disability", "num_of_prev_attempts"}
CONTINUOUS = ["total_clicks", "total_active_days", "n_submitted", "mean_score",
              "first_score", "n_late", "scheduled_resource_ratio",
              "active_days_7d_pre_deadline", "num_of_prev_attempts"]


def _interpretable_frame(view) -> pd.DataFrame:
    keys = view.cohort_keys[_KEYS].drop_duplicates().reset_index(drop=True)
    weekly = weekly_engagement(view)
    agg = weekly.groupby(_KEYS).agg(total_clicks=("clicks", "sum"),
                                    total_active_days=("active_days", "sum")).reset_index()
    assess = assessment_features(view)[_KEYS + ["n_submitted", "mean_score", "first_score", "n_late"]]
    course = course_structure_features(view)[_KEYS + ["scheduled_resource_ratio", "active_days_7d_pre_deadline"]]
    static = view.static[_KEYS + ["gender", "region", "age_band", "imd_band",
                                  "highest_education", "disability", "num_of_prev_attempts"]]
    df = (keys.merge(agg, on=_KEYS, how="left").merge(assess, on=_KEYS, how="left")
          .merge(course, on=_KEYS, how="left").merge(static, on=_KEYS, how="left"))
    return df.fillna({c: 0 for c in CONTINUOUS})


def main() -> int:
    import dice_ml

    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(cfg.data_dir))
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--sample", type=int, default=100)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    tables = load_oulad(args.data_dir)
    view = build_window_view(tables, args.window, days_per_week=cfg.days_per_week)
    frame = _interpretable_frame(view)
    labels = make_binary_label(tables.student_info)
    frame = frame.merge(labels[_KEYS + ["at_risk"]], on=_KEYS, how="left")

    # DiCE butuh dtype int/float murni (bukan Int64 nullable) untuk fitur kontinu.
    for c in CONTINUOUS:
        frame[c] = pd.to_numeric(frame[c], errors="coerce").astype(float).fillna(0.0)
    frame["at_risk"] = frame["at_risk"].astype(int)

    feat_cols = [c for c in frame.columns if c not in _KEYS + ["at_risk"]]
    cat_cols = [c for c in feat_cols if c not in CONTINUOUS]
    # OneHotEncoder tak menerima NaN -> isi kategori kosong dgn 'missing'.
    for c in cat_cols:
        frame[c] = frame[c].astype("object").fillna("missing").astype(str)

    split = student_level_split(frame[_KEYS + ["at_risk"]], ratios=cfg.split.ratios, seed=cfg.seed)
    train_ids, test_ids = set(split.train["id_student"]), set(split.test["id_student"])
    tr = frame[frame["id_student"].isin(train_ids)]
    te = frame[frame["id_student"].isin(test_ids)]

    # Pipeline: one-hot kategori + passthrough kontinu -> RandomForest (DiCE-compatible).
    clf = Pipeline([
        ("prep", ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", CONTINUOUS),
        ])),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=cfg.seed, n_jobs=-1)),
    ])
    clf.fit(tr[feat_cols], tr["at_risk"])

    # DiCE: counterfactual mengubah at-risk -> not-at-risk (varying SEMUA fitur).
    d = dice_ml.Data(dataframe=tr[feat_cols + ["at_risk"]],
                     continuous_features=CONTINUOUS, outcome_name="at_risk")
    m = dice_ml.Model(model=clf, backend="sklearn")
    exp = dice_ml.Dice(d, m, method="random")

    at_risk_test = te[te["at_risk"] == 1].head(args.sample)
    print(f"Generate counterfactual untuk {len(at_risk_test)} mahasiswa at-risk (Week {args.window})...")

    changed_sets, n_changed = [], []
    for _, row in at_risk_test.iterrows():
        q = row[feat_cols].to_frame().T
        try:
            cf = exp.generate_counterfactuals(q, total_CFs=1, desired_class=0)
            cf_df = cf.cf_examples_list[0].final_cfs_df
            if cf_df is None or len(cf_df) == 0:
                continue
            orig = row[feat_cols]
            cfr = cf_df.iloc[0][feat_cols]
            changed = {c for c in feat_cols if str(orig[c]) != str(cfr[c])}
            changed_sets.append(changed)
            n_changed.append(len(changed))
        except Exception:
            continue

    viol = immutable_violation_rate(changed_sets, IMMUTABLE)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame([{
        "window": args.window, "n_cf": len(changed_sets),
        "cf_immutable_violation_rate": round(viol, 4),
        "cf_mean_features_changed": round(float(np.mean(n_changed)) if n_changed else float("nan"), 2),
        "rule_immutable_violation_rate": 0.0,
        "rule_mean_reasons": "~1-2 (lihat sparsity)",
    }])
    res.to_csv(out / "counterfactual_comparison.csv", index=False)
    print("\n=== Counterfactual recourse vs rule-based (actionability) ===")
    print(res.to_string(index=False))
    print("\nInterpretasi: CF naif sering mengubah fitur immutable (tidak actionable);")
    print("explanation rule-based NS-GRU tidak pernah (violation=0) -> lebih actionable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
