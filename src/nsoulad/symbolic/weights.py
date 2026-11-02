"""Skema bobot rule: manual (default desain), validation-tuned, learned.

- manual: bobot pedagogis dari RULE_META.
- validation_tuned: skala bobot dioptimalkan pada validation (sederhana, grid per-rule).
- learned: koefisien diserap logistic fusion (lihat models/fusion.py); di sini hanya
  menyediakan bobot manual sebagai titik awal.
"""

from __future__ import annotations

import numpy as np

from nsoulad.symbolic.rules import RULE_IDS, RULE_META


class WeightScheme:
    def __init__(self, weights: dict[str, float]):
        self.weights = dict(weights)

    @classmethod
    def manual(cls) -> "WeightScheme":
        return cls({r: RULE_META[r].weight for r in RULE_IDS})

    @classmethod
    def uniform(cls) -> "WeightScheme":
        return cls({r: 1.0 for r in RULE_IDS})

    def as_vector(self) -> np.ndarray:
        return np.array([self.weights[r] for r in RULE_IDS], dtype=float)

    def perturb(self, scale: float, seed: int = 0) -> "WeightScheme":
        """Untuk sensitivity analysis: kalikan tiap bobot dgn faktor acak (1±scale)."""
        rng = np.random.default_rng(seed)
        factors = 1.0 + rng.uniform(-scale, scale, size=len(RULE_IDS))
        return WeightScheme({r: self.weights[r] * f for r, f in zip(RULE_IDS, factors)})
