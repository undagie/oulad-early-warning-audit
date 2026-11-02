"""Orkestrasi satu window x satu seed: latih semua model, evaluasi di test.

Mengembalikan baris metrik per model (+ koefisien fusion & activation symbolic).
Dipakai oleh scripts/run_experiment.py (MVP) dan akan diperluas untuk multi-seed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from nsoulad.config import Config
from nsoulad.data.loader import OuladTables
from nsoulad.data.windowing import build_window_view
from nsoulad.evaluation.early_warning import precision_at_top_k, recall_at_top_k
from nsoulad.evaluation.explanation import (
    contradiction_rate,
    coverage,
    explanation_sparsity,
    rule_diagnostic_validity,
)
from nsoulad.evaluation.metrics import predictive_metrics
from nsoulad.features.build import presentation_split_datasets, split_window_datasets
from nsoulad.imbalance import tune_threshold
from nsoulad.models import (
    GRUModel,
    LogisticFull,
    LogisticFusion,
    RulesOnly,
    StaticOnly,
    XGBoostFull,
    align_symbolic,
)
from nsoulad.symbolic import RuleThresholds, SymbolicEngine


def _evaluate(name, val_p, test_p, val_y, test_y, threshold_obj="balanced_accuracy") -> tuple[dict, float]:
    t = tune_threshold(val_y, val_p, objective=threshold_obj)
    m = predictive_metrics(test_y, test_p, threshold=t)
    m["recall_top10"] = recall_at_top_k(test_y, test_p, 0.10)
    m["precision_top10"] = precision_at_top_k(test_y, test_p, 0.10)
    m["model"] = name
    return m, t


def run_window(
    tables: OuladTables,
    k: int,
    seed: int,
    cfg: Config,
    include_gru: bool = True,
    leaky: bool = False,
    split_mode: str = "student",
    active_rules: list[str] | None = None,
    hparams: dict | None = None,
    neural_archs: tuple[str, ...] = ("gru", "lstm", "transformer"),
) -> dict:
    """Latih & evaluasi semua model untuk window k, seed tertentu.

    leaky=True membangun fitur dari seluruh course (data masa depan) sementara cohort
    & split tetap di window k -> kondisi 'leaky' untuk ablation LEAP (strict vs leaky).
    split_mode "student" (stratified) atau "presentation" (generalisasi 2013->2014).

    Returns dict: metrics (DataFrame), fusion_coef (dict), activation (Series).
    """
    fcut = 10**6 if leaky else None
    view = build_window_view(tables, k, days_per_week=cfg.days_per_week, feature_cutoff_day=fcut)
    if split_mode == "presentation":
        parts = presentation_split_datasets(
            view, tables.student_info,
            train_presentations=cfg.split.train_presentations,
            test_presentations=cfg.split.test_presentations,
            seed=seed, days_per_week=cfg.days_per_week,
        )
    else:
        parts = split_window_datasets(view, tables.student_info, ratios=cfg.split.ratios, seed=seed,
                                      days_per_week=cfg.days_per_week)
    train, val, test = parts["train"], parts["val"], parts["test"]

    # --- Symbolic ---
    train_ids = set(train.keys["id_student"])
    thresholds = RuleThresholds.fit(view, train_ids, days_per_week=cfg.days_per_week)
    rule_res = SymbolicEngine(thresholds, days_per_week=cfg.days_per_week,
                              active_rules=active_rules).apply(view)
    sym_frame = rule_res.to_frame()
    sym = {name: align_symbolic(sym_frame, p.keys) for name, p in parts.items()}

    rows = []
    thresholds_used: dict[str, float] = {}
    hp = hparams or {}

    # --- Baselines ML ---
    for name, model in [
        ("XGBoost", XGBoostFull(seed=seed, **hp.get("xgboost", {}))),
        ("LogReg", LogisticFull(seed=seed)),
        ("Static-only", StaticOnly(seed=seed)),
    ]:
        model.fit(train, val)
        m, t = _evaluate(name, model.predict_proba(val), model.predict_proba(test), val.y, test.y)
        rows.append(m)
        thresholds_used[name] = t

    # --- Rules-only ---
    m, t = _evaluate("Rules-only", sym["val"].p_symbolic, sym["test"].p_symbolic, val.y, test.y)
    rows.append(m)
    thresholds_used["Rules-only"] = t

    fusion_coef: dict[str, float] = {}
    explanation: dict = {}
    if include_gru:
        arch_label = {"gru": "GRU", "lstm": "LSTM", "transformer": "Transformer"}

        # --- Baseline neural: GRU / LSTM / Transformer ---
        p_val_gru = p_test_gru = None
        for arch in neural_archs:
            mdl = GRUModel(seed=seed, arch=arch, **hp.get(arch, {}))
            mdl.fit(train, val)
            pv, pt = mdl.predict_proba(val), mdl.predict_proba(test)
            m, _ = _evaluate(arch_label[arch], pv, pt, val.y, test.y)
            rows.append(m)
            if arch == "gru":
                p_val_gru, p_test_gru = pv, pt

        # --- NS-GRU (late/logistic fusion) ---
        fusion = LogisticFusion(seed=seed).fit(p_val_gru, sym["val"], val.y)
        p_val_ns = fusion.predict_proba(p_val_gru, sym["val"])
        p_test_ns = fusion.predict_proba(p_test_gru, sym["test"])
        m, t_ns = _evaluate("NS-GRU", p_val_ns, p_test_ns, val.y, test.y)
        rows.append(m)
        fusion_coef = fusion.coefficients()

        # --- NS-GRU-cal: kalibrasi isotonic output fusion (fit val, terapkan test) ---
        # Memperbaiki distorsi probabilitas akibat class-weighting pada fusion (lihat
        # diagnosis Fase 0: NS-GRU mentah under-calibrated). AUC invariant thd isotonic
        # monoton; yang membaik = ECE/Brier. Estimasi test = out-of-sample (jujur).
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_val_ns, val.y)
        p_val_nsc, p_test_nsc = iso.predict(p_val_ns), iso.predict(p_test_ns)
        m, _ = _evaluate("NS-GRU-cal", p_val_nsc, p_test_nsc, val.y, test.y)
        rows.append(m)

        # --- NS-GRU-RA (rule-augmented: fitur symbolic disuntik ke cabang tabular GRU) ---
        ra = GRUModel(seed=seed, arch="gru", **hp.get("gru", {}))
        ra.fit(train, val, train_extra=sym["train"].as_features(), val_extra=sym["val"].as_features())
        p_val_ra = ra.predict_proba(val, extra=sym["val"].as_features())
        p_test_ra = ra.predict_proba(test, extra=sym["test"].as_features())
        m, _ = _evaluate("NS-GRU-RA", p_val_ra, p_test_ra, val.y, test.y)
        rows.append(m)

        # --- Explanation quality (proxy actionability) di test ---
        trig_test = sym["test"].triggered
        pred_ns = (p_test_ns >= t_ns).astype(int)
        explanation = {
            "coverage": coverage(trig_test, pred_ns),
            "sparsity": explanation_sparsity(trig_test, pred_ns),
            "contradiction_rate": contradiction_rate(p_test_gru, trig_test),
            "diagnostic_validity": rule_diagnostic_validity(trig_test, test.y),
        }

    metrics = pd.DataFrame(rows)
    metrics.insert(0, "window", k)
    metrics.insert(1, "seed", seed)
    return {
        "metrics": metrics,
        "fusion_coef": fusion_coef,
        "explanation": explanation,
        "activation": rule_res.activation_rate(),
        "n_train": train.n, "n_val": val.n, "n_test": test.n,
    }
