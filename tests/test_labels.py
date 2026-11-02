"""Test konstruksi label biner at-risk."""

import pytest

from nsoulad.data.labels import base_rate, make_binary_label


def test_binary_mapping(oulad):
    labels = make_binary_label(oulad.student_info)
    by_id = labels.set_index("id_student")["at_risk"].to_dict()
    # s1 Pass=0, s2 Fail=1, s3 Withdrawn=1, s4 Distinction=0, s5 Fail=1, s6 Pass=0
    assert by_id == {1: 0, 2: 1, 3: 1, 4: 0, 5: 1, 6: 0}
    assert "final_result" in labels.columns  # disimpan sbg label, bukan dibuang


def test_base_rate(oulad):
    labels = make_binary_label(oulad.student_info)
    assert base_rate(labels) == pytest.approx(3 / 6)


def test_unknown_final_result_raises(oulad):
    bad = oulad.student_info.copy()
    bad.loc[0, "final_result"] = "Unknown"
    with pytest.raises(ValueError, match="tak dikenal"):
        make_binary_label(bad)


def test_incomplete_mapping_raises(oulad):
    with pytest.raises(ValueError, match="tidak menutup"):
        make_binary_label(oulad.student_info, at_risk=("Fail",), not_at_risk=("Pass",))
