"""NS-GRU: logistic fusion atas output neural + ringkasan symbolic.

Fusion sengaja decoupled & auditable (lihat positioning desain): koefisien logistik
menjadi 'learned weights' yang dapat diperiksa.

feature_set:
  - "full" (default, dipakai di paper): [P_neural, P_symbolic, rule_count,
    high_severity_count], skala mentah. Ketiga fitur symbolic adalah kombinasi linear
    dari rule yang sama (kolinear), sehingga koefisien individualnya tidak stabil
    antar-seed dan TIDAK boleh dibaca sebagai tingkat ketergantungan.
  - "compact": [P_neural, P_symbolic] yang distandarisasi (fit pada data fusion).
    Satu fitur per cabang -> koefisien terstandarisasi dapat dibandingkan langsung
    sebagai reliance neural vs symbolic.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from nsoulad.models.util import SymbolicBundle

FUSION_FEATURES = ["P_neural", "P_symbolic", "rule_count", "high_severity_count"]
COMPACT_FUSION_FEATURES = ["P_neural", "P_symbolic"]
FEATURE_SETS = {"full": FUSION_FEATURES, "compact": COMPACT_FUSION_FEATURES}


class LogisticFusion:
    def __init__(self, seed: int = 42, feature_set: str = "full"):
        if feature_set not in FEATURE_SETS:
            raise ValueError(f"feature_set tak dikenal: {feature_set}. Pilihan: {list(FEATURE_SETS)}")
        self.feature_set = feature_set
        self.feature_names = FEATURE_SETS[feature_set]
        self.clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed)
        self.scaler: StandardScaler | None = StandardScaler() if feature_set == "compact" else None
        self._fitted = False

    def _matrix(self, p_neural: np.ndarray, sym: SymbolicBundle, fit: bool = False) -> np.ndarray:
        X = sym.fusion_matrix(p_neural)
        if self.feature_set == "compact":
            X = X[:, [FUSION_FEATURES.index(f) for f in COMPACT_FUSION_FEATURES]]
            X = self.scaler.fit_transform(X) if fit else self.scaler.transform(X)
        return X

    def fit(self, p_neural: np.ndarray, sym: SymbolicBundle, y: np.ndarray) -> "LogisticFusion":
        self.clf.fit(self._matrix(p_neural, sym, fit=True), y)
        self._fitted = True
        return self

    def predict_proba(self, p_neural: np.ndarray, sym: SymbolicBundle) -> np.ndarray:
        return self.clf.predict_proba(self._matrix(p_neural, sym))[:, 1]

    def coefficients(self) -> dict[str, float]:
        """Koefisien fusion (learned weights) untuk audit/interpretasi.

        Untuk "compact" koefisien sudah terstandarisasi (per 1 std fitur).
        """
        if not self._fitted:
            raise RuntimeError("LogisticFusion belum di-fit.")
        return {name: float(c) for name, c in zip(self.feature_names, self.clf.coef_[0])}
