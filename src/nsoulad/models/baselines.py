"""Baseline ML: Logistic Regression, XGBoost (fitur penuh), dan Static-only.

Semua mengikuti antarmuka seragam fit(train, val=None) / predict_proba(ds).
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from nsoulad.features.build import WindowDataset
from nsoulad.imbalance import class_pos_weight
from nsoulad.models.util import full_features, static_features


class LogisticFull:
    """Logistic Regression atas fitur penuh (temporal-flat + tabular), class-balanced."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )

    def fit(self, train: WindowDataset, val: WindowDataset | None = None) -> "LogisticFull":
        self.pipe.fit(full_features(train), train.y)
        return self

    def predict_proba(self, ds: WindowDataset) -> np.ndarray:
        return self.pipe.predict_proba(full_features(ds))[:, 1]


class XGBoostFull:
    """XGBoost atas fitur penuh; scale_pos_weight dari distribusi train. Hyperparam dapat dituning."""

    def __init__(self, seed: int = 42, max_depth: int = 5, learning_rate: float = 0.05,
                 n_estimators: int = 300):
        self.seed = seed
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.model: XGBClassifier | None = None

    def fit(self, train: WindowDataset, val: WindowDataset | None = None) -> "XGBoostFull":
        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=0.9,
            colsample_bytree=0.8,
            scale_pos_weight=class_pos_weight(train.y),
            eval_metric="aucpr",
            random_state=self.seed,
            n_jobs=0,
        )
        self.model.fit(full_features(train), train.y)
        return self

    def predict_proba(self, ds: WindowDataset) -> np.ndarray:
        return self.model.predict_proba(full_features(ds))[:, 1]


class StaticOnly:
    """Kontrol static-feature dominance: XGBoost hanya pada fitur demografi statis."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model: XGBClassifier | None = None

    def fit(self, train: WindowDataset, val: WindowDataset | None = None) -> "StaticOnly":
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            scale_pos_weight=class_pos_weight(train.y),
            eval_metric="aucpr",
            random_state=self.seed,
            n_jobs=0,
        )
        self.model.fit(static_features(train), train.y)
        return self

    def predict_proba(self, ds: WindowDataset) -> np.ndarray:
        return self.model.predict_proba(static_features(ds))[:, 1]
