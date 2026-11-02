"""Grid search hyperparameter kecil di validation (PR-AUC). Dijalankan sekali per window."""

from __future__ import annotations

from sklearn.metrics import average_precision_score

from nsoulad.features.build import WindowDataset
from nsoulad.models.baselines import XGBoostFull
from nsoulad.models.gru import GRUModel

GRU_GRID = [dict(hidden=h, dropout=0.3, lr=l) for h in (16, 32, 64) for l in (1e-3, 5e-4)]
XGB_GRID = [dict(max_depth=d, learning_rate=l) for d in (4, 5, 6) for l in (0.03, 0.05, 0.1)]


def _prauc(y, p) -> float:
    return float(average_precision_score(y, p)) if len(set(y.tolist())) > 1 else 0.0


def tune_gru(train: WindowDataset, val: WindowDataset, seed: int = 42, grid=None) -> dict:
    grid = grid or GRU_GRID
    best, best_s = grid[0], -1.0
    for kw in grid:
        m = GRUModel(seed=seed, arch="gru", **kw).fit(train, val)
        s = _prauc(val.y, m.predict_proba(val))
        if s > best_s:
            best_s, best = s, kw
    return {"best": best, "score": round(best_s, 4)}


def tune_xgboost(train: WindowDataset, val: WindowDataset, seed: int = 42, grid=None) -> dict:
    grid = grid or XGB_GRID
    best, best_s = grid[0], -1.0
    for kw in grid:
        m = XGBoostFull(seed=seed, **kw).fit(train)
        s = _prauc(val.y, m.predict_proba(val))
        if s > best_s:
            best_s, best = s, kw
    return {"best": best, "score": round(best_s, 4)}
