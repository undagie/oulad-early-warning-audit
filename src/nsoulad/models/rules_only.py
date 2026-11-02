"""Baseline symbolic-only: prediksi langsung dari P_symbolic."""

from __future__ import annotations

import numpy as np

from nsoulad.models.util import SymbolicBundle


class RulesOnly:
    """Skor = P_symbolic. Tidak ada pelatihan; threshold dituning terpisah di val."""

    def fit(self, sym_train: SymbolicBundle, y_train=None) -> "RulesOnly":
        return self

    def predict_proba(self, sym: SymbolicBundle) -> np.ndarray:
        return np.asarray(sym.p_symbolic, dtype=float)
