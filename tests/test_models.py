"""Smoke-test komponen model ringan (tanpa training berat)."""

import numpy as np
import pandas as pd

from nsoulad.data.windowing import build_window_view
from nsoulad.features import build_window_dataset
from nsoulad.imbalance import class_pos_weight, tune_threshold
from nsoulad.models import LogisticFusion, RulesOnly, align_symbolic, full_features, static_features
from nsoulad.models.util import SymbolicBundle
from nsoulad.symbolic import RuleThresholds, SymbolicEngine


def test_feature_assembly(oulad):
    view = build_window_view(oulad, 4)
    ds = build_window_dataset(view, oulad.student_info, fit_static=True)
    ff = full_features(ds)
    sf = static_features(ds)
    assert len(ff) == ds.n and len(sf) == ds.n
    assert sf.shape[1] < ff.shape[1]  # static subset dari fitur penuh


def test_align_symbolic_and_fusion_matrix(oulad):
    view = build_window_view(oulad, 4)
    eng = SymbolicEngine(RuleThresholds.fit(view, oulad.student_info["id_student"].tolist()))
    res = eng.apply(view)
    bundle = align_symbolic(res.to_frame(), res.keys)
    assert len(bundle.p_symbolic) == len(res.keys)
    fm = bundle.fusion_matrix(np.zeros(len(res.keys)))
    assert fm.shape == (len(res.keys), 4)


def test_logistic_fusion_toy():
    rng = np.random.default_rng(0)
    n = 80
    p_neural = rng.random(n)
    sym = SymbolicBundle(
        p_symbolic=rng.random(n),
        rule_count=rng.integers(0, 5, n).astype(float),
        high_severity_count=rng.integers(0, 3, n).astype(float),
        triggered=pd.DataFrame(),
    )
    y = (p_neural > 0.5).astype(int)
    f = LogisticFusion(seed=0).fit(p_neural, sym, y)
    p = f.predict_proba(p_neural, sym)
    assert p.shape == (n,) and ((p >= 0) & (p <= 1)).all()
    assert set(f.coefficients()) == {"P_neural", "P_symbolic", "rule_count", "high_severity_count"}


def test_logistic_fusion_compact_is_standardized():
    rng = np.random.default_rng(0)
    n = 200
    p_neural = rng.random(n)
    sym = SymbolicBundle(
        p_symbolic=rng.random(n),
        rule_count=rng.integers(0, 5, n).astype(float),
        high_severity_count=rng.integers(0, 3, n).astype(float),
        triggered=pd.DataFrame(),
    )
    y = (p_neural > 0.5).astype(int)
    f = LogisticFusion(seed=0, feature_set="compact").fit(p_neural, sym, y)
    assert set(f.coefficients()) == {"P_neural", "P_symbolic"}
    # Koefisien per 1 std: invarian terhadap penskalaan ulang input.
    sym10 = SymbolicBundle(sym.p_symbolic * 10, sym.rule_count, sym.high_severity_count, sym.triggered)
    f10 = LogisticFusion(seed=0, feature_set="compact").fit(p_neural, sym10, y)
    assert np.isclose(f.coefficients()["P_symbolic"], f10.coefficients()["P_symbolic"])
    assert np.allclose(f.predict_proba(p_neural, sym), f10.predict_proba(p_neural, sym10))


def test_symbolic_bundle_as_features():
    from nsoulad.symbolic.rules import RULE_IDS
    trig = pd.DataFrame(np.zeros((3, 7), dtype=int), columns=RULE_IDS)
    trig.iloc[0, 0] = 1
    b = SymbolicBundle(np.array([0.1, 0.2, 0.3]), np.array([1.0, 2, 3]), np.array([0.0, 1, 2]), trig)
    feats = b.as_features()
    assert feats.shape == (3, 10)  # 7 triggered + p_symbolic + rule_count + high_sev


def test_rules_only_and_imbalance():
    sym = SymbolicBundle(np.array([0.1, 0.9]), np.array([1.0, 3.0]), np.array([0.0, 2.0]), pd.DataFrame())
    assert list(RulesOnly().predict_proba(sym)) == [0.1, 0.9]
    assert class_pos_weight(np.array([0, 0, 0, 1])) == 3.0
    t = tune_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert 0.0 <= t <= 1.0
