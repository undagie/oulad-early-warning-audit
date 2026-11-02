"""Sequential early-risk predictors (PyTorch): GRU / LSTM / Transformer.

Arsitektur: weekly sequence -> encoder (GRU/LSTM/Transformer) -> concat fitur
tabular -> Dense -> Dropout -> logit. Transformer memakai learned positional embedding
per minggu; tanpa itu self-attention + mean-pool invarian terhadap urutan minggu. Loss BCE berbobot (pos_weight), early stopping
pada PR-AUC validation, normalisasi pakai statistik train.

Mendukung **fitur tabular ekstra** (parameter `extra`) untuk varian neuro-symbolic
yang lebih erat (rule-augmented): aktivasi rule disuntikkan langsung ke cabang tabular.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from nsoulad.features.build import WindowDataset
from nsoulad.imbalance import class_pos_weight


# Batas atas panjang sekuens (minggu) untuk positional embedding Transformer.
MAX_WEEKS = 64


class _SeqNet(nn.Module):
    def __init__(self, n_seq: int, n_tab: int, hidden: int = 32, dropout: float = 0.3,
                 arch: str = "gru", n_heads: int = 4, n_layers: int = 1):
        super().__init__()
        self.arch = arch
        if arch == "gru":
            self.rnn = nn.GRU(n_seq, hidden, batch_first=True)
        elif arch == "lstm":
            self.rnn = nn.LSTM(n_seq, hidden, batch_first=True)
        elif arch == "transformer":
            self.input_proj = nn.Linear(n_seq, hidden)
            self.pos_emb = nn.Parameter(torch.zeros(1, MAX_WEEKS, hidden))
            nn.init.normal_(self.pos_emb, std=0.02)
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=n_heads, dim_feedforward=hidden * 2,
                dropout=dropout, batch_first=True)
            self.rnn = nn.TransformerEncoder(layer, num_layers=n_layers)
        else:
            raise ValueError(f"arch tak dikenal: {arch}")
        self.head = nn.Sequential(
            nn.Linear(hidden + n_tab, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        if self.arch == "gru":
            _, h = self.rnn(x_seq)
            h = h.squeeze(0)
        elif self.arch == "lstm":
            _, (h, _c) = self.rnn(x_seq)
            h = h.squeeze(0)
        else:  # transformer: embedding posisi minggu, lalu mean-pool atas token
            z = self.input_proj(x_seq) + self.pos_emb[:, : x_seq.shape[1]]
            z = self.rnn(z)
            h = z.mean(dim=1)
        return self.head(torch.cat([h, x_tab], dim=1)).squeeze(1)


class GRUModel:
    """Wrapper fit/predict_proba untuk encoder sekuensial (GRU/LSTM/Transformer).

    arch in {"gru","lstm","transformer"}. `extra` (opsional) = fitur tabular tambahan
    [n, m] yang di-concat ke X_tab (dipakai untuk rule-augmented neuro-symbolic).
    """

    def __init__(self, hidden: int = 32, dropout: float = 0.3, lr: float = 1e-3,
                 epochs: int = 40, patience: int = 6, batch_size: int = 256, seed: int = 42,
                 arch: str = "gru", n_heads: int = 4, n_layers: int = 1):
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size
        self.seed = seed
        self.arch = arch
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net: _SeqNet | None = None
        self.seq_mean_ = self.seq_std_ = None
        self.tab_mean_ = self.tab_std_ = None
        self.n_tab_ = None

    def _tab_matrix(self, ds: WindowDataset, extra) -> np.ndarray:
        tab = ds.X_tab.to_numpy(dtype=float)
        if extra is not None:
            tab = np.column_stack([tab, np.asarray(extra, dtype=float)])
        return tab

    def _fit_scalers(self, train: WindowDataset, train_extra) -> None:
        seq = train.X_seq.reshape(-1, train.X_seq.shape[-1])
        self.seq_mean_ = seq.mean(axis=0)
        self.seq_std_ = seq.std(axis=0) + 1e-6
        tab = self._tab_matrix(train, train_extra)
        self.tab_mean_ = tab.mean(axis=0)
        self.tab_std_ = tab.std(axis=0) + 1e-6
        self.n_tab_ = tab.shape[1]

    def _tensors(self, ds: WindowDataset, extra):
        xs = (ds.X_seq - self.seq_mean_) / self.seq_std_
        xt = (self._tab_matrix(ds, extra) - self.tab_mean_) / self.tab_std_
        return (torch.tensor(xs, dtype=torch.float32, device=self.device),
                torch.tensor(xt, dtype=torch.float32, device=self.device))

    def fit(self, train: WindowDataset, val: WindowDataset | None = None,
            train_extra=None, val_extra=None) -> "GRUModel":
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self._fit_scalers(train, train_extra)

        self.net = _SeqNet(train.X_seq.shape[-1], self.n_tab_, self.hidden, self.dropout,
                           self.arch, self.n_heads, self.n_layers).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        pos_weight = torch.tensor([class_pos_weight(train.y)], dtype=torch.float32, device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        xs, xt = self._tensors(train, train_extra)
        y = torch.tensor(train.y, dtype=torch.float32, device=self.device)
        n = len(y)

        best_score, best_state, bad = -np.inf, None, 0
        for _ in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(xs[idx], xt[idx]), y[idx])
                loss.backward()
                opt.step()

            ref, ref_extra = (val, val_extra) if val is not None else (train, train_extra)
            p = self._predict(ref, ref_extra)
            score = average_precision_score(ref.y, p) if len(np.unique(ref.y)) == 2 else 0.0
            if score > best_score:
                best_score, bad = score, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _predict(self, ds: WindowDataset, extra) -> np.ndarray:
        self.net.eval()
        xs, xt = self._tensors(ds, extra)
        return torch.sigmoid(self.net(xs, xt)).cpu().numpy()

    def predict_proba(self, ds: WindowDataset, extra=None) -> np.ndarray:
        return self._predict(ds, extra)
