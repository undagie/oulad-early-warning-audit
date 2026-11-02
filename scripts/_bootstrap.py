"""Tambahkan src/ ke sys.path agar skrip bisa dijalankan tanpa `pip install -e .`."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
