"""Test symbolic engine (Fase 2) pada fixture sintetis terkontrol."""

import numpy as np

from nsoulad.data.windowing import build_window_view
from nsoulad.symbolic import RULE_IDS, RuleThresholds, SymbolicEngine
from nsoulad.symbolic.rules import max_possible_weight


def _apply(oulad, k=4, train_ids=None):
    view = build_window_view(oulad, k)
    ids = train_ids if train_ids is not None else oulad.student_info["id_student"].tolist()
    eng = SymbolicEngine(RuleThresholds.fit(view, ids))
    return view, eng.apply(view)


def _by_student(result):
    idx = {sid: i for i, sid in enumerate(result.keys["id_student"])}
    return lambda sid, rule: result.triggered.iloc[idx[sid]][rule]


def test_r1_persistent_inactivity(oulad):
    _, res = _apply(oulad)
    trig = _by_student(res)
    assert trig(3, "R1") == 1   # s3 tak pernah aktif -> run zero panjang
    assert trig(1, "R1") == 0   # s1 aktif berturut di minggu awal


def test_r7_no_activity_before_assessment(oulad):
    _, res = _apply(oulad)
    trig = _by_student(res)
    # s3: course punya assessment due (1001) tapi nol aktivitas pra-deadline.
    assert trig(3, "R7") == 1
    # s1: aktif (date 10) dalam 7 hari sebelum deadline 12 -> tidak terpicu.
    assert trig(1, "R7") == 0


def test_symbolic_score_range(oulad):
    _, res = _apply(oulad)
    assert res.p_symbolic.min() >= 0.0
    assert res.p_symbolic.max() <= 1.0
    assert len(res.p_symbolic) == len(res.keys)


def test_activation_rate_shape(oulad):
    _, res = _apply(oulad)
    ar = res.activation_rate()
    assert list(ar.index) == RULE_IDS
    assert ((ar >= 0) & (ar <= 1)).all()


def test_explanations_have_actions(oulad):
    _, res = _apply(oulad)
    expls = res.explanations()
    assert len(expls) == len(res.keys)
    idx = {sid: i for i, sid in enumerate(res.keys["id_student"])}
    s3 = expls[idx[3]]
    assert len(s3) >= 1
    assert all("action" in e and "reason" in e for e in s3)


def test_severity_and_weight(oulad):
    _, res = _apply(oulad)
    assert max_possible_weight() == 1.30
    assert (res.severity["rule_count"] >= res.severity["high_severity_count"]).all()


def test_thresholds_leak_free_subset(oulad):
    # Fit hanya pada train subset -> tetap berjalan & menghasilkan p_symbolic valid.
    view, res = _apply(oulad, train_ids=[1, 2, 3])
    assert np.isfinite(res.p_symbolic).all()
