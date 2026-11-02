"""Skema 7 tabel OULAD: nama file, kolom wajib, dan kolom numerik/integer.

Dipakai oleh loader untuk validasi struktur dan koersi tipe. Referensi struktur:
Kuzilek et al. (2017), Open University Learning Analytics Dataset (sdata2017171).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableSchema:
    """Skema satu tabel OULAD."""

    filename: str
    required: tuple[str, ...]
    # Kolom yang dikoersi ke numerik float (memuat NaN dengan aman, mis. date).
    numeric: tuple[str, ...] = field(default_factory=tuple)
    # Kolom id/hitung yang dikoersi ke Int64 nullable (rapi untuk join).
    integer: tuple[str, ...] = field(default_factory=tuple)


# Nilai yang diperlakukan sebagai missing pada CSV OULAD.
NA_VALUES = ["?", "", "NA", "nan"]

# Nilai final_result yang valid.
FINAL_RESULT_VALUES = frozenset({"Pass", "Fail", "Withdrawn", "Distinction"})

OULAD_SCHEMA: dict[str, TableSchema] = {
    "courses": TableSchema(
        filename="courses.csv",
        required=("code_module", "code_presentation", "module_presentation_length"),
        integer=("module_presentation_length",),
    ),
    "student_info": TableSchema(
        filename="studentInfo.csv",
        required=(
            "code_module",
            "code_presentation",
            "id_student",
            "gender",
            "region",
            "highest_education",
            "imd_band",
            "age_band",
            "num_of_prev_attempts",
            "studied_credits",
            "disability",
            "final_result",
        ),
        integer=("id_student", "num_of_prev_attempts", "studied_credits"),
    ),
    "student_registration": TableSchema(
        filename="studentRegistration.csv",
        required=(
            "code_module",
            "code_presentation",
            "id_student",
            "date_registration",
            "date_unregistration",
        ),
        numeric=("date_registration", "date_unregistration"),
        integer=("id_student",),
    ),
    "assessments": TableSchema(
        filename="assessments.csv",
        required=(
            "code_module",
            "code_presentation",
            "id_assessment",
            "assessment_type",
            "date",
            "weight",
        ),
        numeric=("date", "weight"),
        integer=("id_assessment",),
    ),
    "student_assessment": TableSchema(
        filename="studentAssessment.csv",
        required=("id_assessment", "id_student", "date_submitted", "is_banked", "score"),
        numeric=("score",),
        integer=("id_assessment", "id_student", "date_submitted", "is_banked"),
    ),
    "student_vle": TableSchema(
        filename="studentVle.csv",
        required=(
            "code_module",
            "code_presentation",
            "id_student",
            "id_site",
            "date",
            "sum_click",
        ),
        integer=("id_student", "id_site", "date", "sum_click"),
    ),
    "vle": TableSchema(
        filename="vle.csv",
        required=("id_site", "code_module", "code_presentation", "activity_type"),
        numeric=("week_from", "week_to"),
        integer=("id_site",),
    ),
}

# Kunci unit analisis: satu mahasiswa pada satu module-presentation.
UNIT_KEYS = ("code_module", "code_presentation", "id_student")
