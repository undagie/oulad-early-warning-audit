"""Test arsitektur sekuensial (GRU/LSTM/Transformer) + rule-augmented (fitur ekstra)."""

import numpy as np
import pandas as pd

from nsoulad.features.build import WindowDataset
from nsoulad.features.static import StaticEncoder
from nsoulad.models.gru import GRUModel


def _ds(n, K=4, F=12, seed=0):
    rng = np.random.default_rng(seed)
    keys = pd.DataFrame({
        "code_module": ["AAA"] * n, "code_presentation": ["2013J"] * n,
        "id_student": list(range(seed * 1000, seed * 1000 + n)),
    })
    X_tab = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"f{i}" for i in range(5)])
    enc = StaticEncoder()
    enc.feature_names_ = list(X_tab.columns)
    y = np.array([0, 1] * (n // 2))
    return WindowDataset(k=K, keys=keys, X_seq=rng.normal(size=(n, K, F)),
                         seq_features=[f"s{i}" for i in range(F)], X_tab=X_tab, y=y, encoder=enc)


def test_arch_variants_predict():
    tr, va = _ds(40, seed=1), _ds(16, seed=2)
    for arch in ["gru", "lstm", "transformer"]:
        m = GRUModel(arch=arch, hidden=8, n_heads=2, epochs=3, seed=0).fit(tr, va)
        p = m.predict_proba(va)
        assert p.shape == (16,)
        assert ((p >= 0) & (p <= 1)).all()


def test_rule_augmented_extra_features():
    tr, va = _ds(40, seed=1), _ds(16, seed=2)
    ex_tr = np.random.default_rng(0).normal(size=(40, 3))
    ex_va = np.random.default_rng(1).normal(size=(16, 3))
    m = GRUModel(arch="gru", hidden=8, epochs=3, seed=0).fit(tr, va, train_extra=ex_tr, val_extra=ex_va)
    p = m.predict_proba(va, extra=ex_va)
    assert p.shape == (16,)
    assert m.n_tab_ == 5 + 3  # X_tab + extra


def test_transformer_is_order_sensitive():
    # Tanpa positional embedding, self-attention + mean-pool invarian thd urutan minggu.
    import torch

    tr, va = _ds(40, seed=1), _ds(16, seed=2)
    m = GRUModel(arch="transformer", hidden=8, n_heads=2, epochs=2, seed=0).fit(tr, va)
    rev = WindowDataset(k=va.k, keys=va.keys, X_seq=va.X_seq[:, ::-1, :].copy(),
                        seq_features=va.seq_features, X_tab=va.X_tab, y=va.y, encoder=va.encoder)
    with torch.no_grad():
        m.net.pos_emb.mul_(50.0)  # perbesar efek posisi agar selisih jelas
    assert not np.allclose(m.predict_proba(va), m.predict_proba(rev), atol=1e-6)
