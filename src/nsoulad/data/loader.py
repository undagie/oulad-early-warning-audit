"""Loader + validasi struktur 7 tabel OULAD."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from nsoulad.data.schema import NA_VALUES, OULAD_SCHEMA, TableSchema


class SchemaError(ValueError):
    """Struktur file OULAD tidak sesuai skema yang diharapkan."""


@dataclass
class OuladTables:
    """Tujuh tabel OULAD yang sudah dimuat dan dikoersi tipenya."""

    courses: pd.DataFrame
    student_info: pd.DataFrame
    student_registration: pd.DataFrame
    assessments: pd.DataFrame
    student_assessment: pd.DataFrame
    student_vle: pd.DataFrame
    vle: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Ringkasan jumlah baris per tabel."""
        return pd.DataFrame(
            {"table": list(self.__dict__), "rows": [len(df) for df in self.__dict__.values()]}
        )


def _coerce_types(df: pd.DataFrame, schema: TableSchema) -> pd.DataFrame:
    """Koersi kolom numerik ke float dan kolom integer ke Int64 nullable."""
    for col in schema.numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in schema.integer:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


def _validate(df: pd.DataFrame, schema: TableSchema, name: str) -> None:
    missing = [c for c in schema.required if c not in df.columns]
    if missing:
        raise SchemaError(
            f"Tabel '{name}' ({schema.filename}) kehilangan kolom wajib: {missing}. "
            f"Kolom tersedia: {list(df.columns)}"
        )


def load_table(data_dir: str | Path, name: str) -> pd.DataFrame:
    """Muat satu tabel OULAD berdasarkan nama logis (lihat OULAD_SCHEMA)."""
    if name not in OULAD_SCHEMA:
        raise KeyError(f"Tabel tidak dikenal: {name}. Pilihan: {list(OULAD_SCHEMA)}")
    schema = OULAD_SCHEMA[name]
    path = Path(data_dir) / schema.filename
    if not path.exists():
        raise FileNotFoundError(
            f"File '{schema.filename}' tidak ditemukan di {data_dir}. "
            f"Unduh OULAD dan letakkan CSV di folder tersebut (lihat README)."
        )
    df = pd.read_csv(path, na_values=NA_VALUES, keep_default_na=True)
    _validate(df, schema, name)
    return _coerce_types(df, schema)


def load_oulad(data_dir: str | Path) -> OuladTables:
    """Muat seluruh 7 tabel OULAD menjadi satu objek OuladTables."""
    tables = {name: load_table(data_dir, name) for name in OULAD_SCHEMA}
    return OuladTables(**tables)
