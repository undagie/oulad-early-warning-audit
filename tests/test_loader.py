"""Test loader: round-trip CSV dan validasi skema."""

import pytest

from nsoulad.data.loader import SchemaError, load_oulad
from nsoulad.data.schema import OULAD_SCHEMA


def _write_csvs(raw_frames, data_dir):
    for name, df in raw_frames.items():
        df.to_csv(data_dir / OULAD_SCHEMA[name].filename, index=False)


def test_load_oulad_roundtrip(tmp_path, raw_frames):
    _write_csvs(raw_frames, tmp_path)
    tables = load_oulad(tmp_path)
    assert len(tables.student_info) == 6
    assert len(tables.student_vle) == 9
    # id_student dikoersi ke Int64 nullable.
    assert str(tables.student_info["id_student"].dtype) == "Int64"


def test_missing_file_raises(tmp_path, raw_frames):
    # Tulis semua kecuali courses.csv.
    partial = {k: v for k, v in raw_frames.items() if k != "courses"}
    _write_csvs(partial, tmp_path)
    with pytest.raises(FileNotFoundError):
        load_oulad(tmp_path)


def test_missing_column_raises(tmp_path, raw_frames):
    _write_csvs(raw_frames, tmp_path)
    # Rusak salah satu file: hapus kolom wajib.
    bad = raw_frames["courses"].drop(columns=["module_presentation_length"])
    bad.to_csv(tmp_path / OULAD_SCHEMA["courses"].filename, index=False)
    with pytest.raises(SchemaError, match="kehilangan kolom wajib"):
        load_oulad(tmp_path)
