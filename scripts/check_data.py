"""Verifikasi keberadaan & struktur dataset OULAD, lalu cetak ringkasan.

Jalankan:
    python scripts/check_data.py [--data-dir data/raw]
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (menyuntik src/ ke sys.path)

from nsoulad.config import load_config
from nsoulad.data.labels import base_rate, make_binary_label
from nsoulad.data.loader import load_oulad


def main() -> int:
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Cek dataset OULAD")
    parser.add_argument("--data-dir", default=str(cfg.data_dir))
    args = parser.parse_args()

    print(f"Memeriksa OULAD di: {args.data_dir}")
    try:
        tables = load_oulad(args.data_dir)
    except FileNotFoundError as e:
        print(f"[GAGAL] {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[GAGAL] {type(e).__name__}: {e}")
        return 1

    print("\nJumlah baris per tabel:")
    print(tables.summary().to_string(index=False))

    labels = make_binary_label(
        tables.student_info, at_risk=cfg.label.at_risk, not_at_risk=cfg.label.not_at_risk
    )
    print("\nDistribusi final_result:")
    print(tables.student_info["final_result"].value_counts().to_string())
    print(f"\nBase rate at-risk (keseluruhan): {base_rate(labels):.4f}  (n={len(labels)})")
    print("\n[OK] Struktur dataset valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
