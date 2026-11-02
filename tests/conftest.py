"""Fixture OULAD sintetis kecil yang terkontrol untuk menguji pipeline data.

Skenario dirancang agar nilai harapan dapat dihitung manual (lihat komentar).
Module AAA dengan dua presentation: 2013J dan 2014J. Enam mahasiswa (id 1..6).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nsoulad.data.loader import OuladTables


@pytest.fixture
def raw_frames() -> dict[str, pd.DataFrame]:
    courses = pd.DataFrame(
        {
            "code_module": ["AAA", "AAA"],
            "code_presentation": ["2013J", "2014J"],
            "module_presentation_length": [268, 268],
        }
    )

    # final_result: s1 Pass, s2 Fail, s3 Withdrawn, s4 Distinction, s5 Fail, s6 Pass
    student_info = pd.DataFrame(
        {
            "code_module": ["AAA"] * 6,
            "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2014J", "2014J"],
            "id_student": [1, 2, 3, 4, 5, 6],
            "gender": ["M", "F", "M", "F", "M", "F"],
            "region": ["East", "West", "East", "West", "East", "West"],
            "highest_education": [
                "HE Qualification",
                "A Level or Equivalent",
                "HE Qualification",
                "Lower Than A Level",
                "HE Qualification",
                "A Level or Equivalent",
            ],
            "imd_band": ["20-30%", "40-50%", np.nan, "60-70%", "20-30%", "40-50%"],
            "age_band": ["0-35", "35-55", "0-35", "35-55", "0-35", "35-55"],
            "num_of_prev_attempts": [0, 1, 0, 0, 2, 0],
            "studied_credits": [60, 60, 90, 60, 120, 60],
            "disability": ["N", "N", "Y", "N", "N", "N"],
            "final_result": ["Pass", "Fail", "Withdrawn", "Distinction", "Fail", "Pass"],
        }
    )

    # Registrasi: kontrol cohort per window.
    #   s3 unregister hari 30 -> in cohort week2 (cutoff13), out week6 (cutoff41)
    #   s4 registrasi hari 20 -> out week2 (cutoff13), in week4 (cutoff27)
    student_registration = pd.DataFrame(
        {
            "code_module": ["AAA"] * 6,
            "code_presentation": ["2013J", "2013J", "2013J", "2013J", "2014J", "2014J"],
            "id_student": [1, 2, 3, 4, 5, 6],
            "date_registration": [-30.0, -20.0, -10.0, 20.0, -15.0, -15.0],
            "date_unregistration": [np.nan, np.nan, 30.0, np.nan, np.nan, np.nan],
        }
    )

    assessments = pd.DataFrame(
        {
            "code_module": ["AAA", "AAA", "AAA", "AAA"],
            "code_presentation": ["2013J", "2013J", "2013J", "2014J"],
            "id_assessment": [1001, 1002, 1003, 2001],
            "assessment_type": ["TMA", "TMA", "Exam", "TMA"],
            "date": [12.0, 40.0, np.nan, 12.0],  # 1003 exam tanpa tanggal
            "weight": [10.0, 20.0, 100.0, 10.0],
        }
    )

    student_assessment = pd.DataFrame(
        {
            "id_assessment": [1001, 1001, 1002, 2001],
            "id_student": [1, 2, 1, 5],
            "date_submitted": [11, 30, 38, 10],  # hanya 11 & 10 yang <= cutoff13
            "is_banked": [0, 0, 0, 0],
            "score": [80.0, 30.0, 55.0, 70.0],
        }
    )

    student_vle = pd.DataFrame(
        {
            "code_module": ["AAA"] * 9,
            "code_presentation": [
                "2013J", "2013J", "2013J", "2013J", "2013J", "2013J", "2013J", "2014J", "2014J"
            ],
            "id_student": [1, 1, 1, 1, 2, 2, 4, 5, 6],
            "id_site": [100, 100, 101, 100, 100, 101, 100, 200, 200],
            "date": [-5, 3, 10, 20, 5, 40, 25, 2, 4],  # 20,40,25 > cutoff13
            "sum_click": [3, 5, 2, 9, 4, 7, 6, 8, 1],
        }
    )

    vle = pd.DataFrame(
        {
            "id_site": [100, 101, 200],
            "code_module": ["AAA", "AAA", "AAA"],
            "code_presentation": ["2013J", "2013J", "2014J"],
            "activity_type": ["resource", "forumng", "resource"],
            "week_from": [0.0, 0.0, 0.0],
            "week_to": [np.nan, np.nan, np.nan],
        }
    )

    return {
        "courses": courses,
        "student_info": student_info,
        "student_registration": student_registration,
        "assessments": assessments,
        "student_assessment": student_assessment,
        "student_vle": student_vle,
        "vle": vle,
    }


@pytest.fixture
def oulad(raw_frames) -> OuladTables:
    return OuladTables(**raw_frames)
