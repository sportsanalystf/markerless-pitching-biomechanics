"""Study C: does a model that predicts velocity from the motion generalise to new pitchers?

Each delivery is the 45-channel OBP joint-angle time series (15 joints x 3
axes) from peak knee height to ball release, resampled to 64 steps. A small
bidirectional LSTM regresses pitch speed from it.

The point is the evaluation, not the architecture. The same model is scored
two ways:

* random split   pitches shuffled into folds, so other pitches from the same
                 pitcher sit in training. The model can recognise the pitcher.
* pitcher-out    GroupKFold by pitcher, so every test pitcher is unseen. This
                 is the only split that matches how a scout uses the model.

A "pitcher memory" baseline makes the leak explicit: it predicts the average
speed of the same pitcher's training pitches and knows nothing about mechanics.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

N_STEPS = 64


def build_sequences(angles: pd.DataFrame, trials: List[tuple]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return X (n, 64, 45), y (n,), groups (n,), channel names."""
    chans = [c for c in angles.columns if c.endswith(("_x", "_y", "_z")) and "angle" in c]
    by = dict(tuple(angles.groupby("session_pitch")))
    X, y, g = [], [], []
    for row, _t, _kp, ev in trials:
        a = by.get(row.name)
        if a is None:
            continue
        a = a.sort_values("time")
        tt = a["time"].to_numpy()
        m = (tt >= ev["pkh"]) & (tt <= ev["br"])
        seg = a.loc[m, chans].to_numpy(float)
        if len(seg) < 8 or not np.isfinite(seg).all():
            continue
        src = np.linspace(0, 1, len(seg))
        dst = np.linspace(0, 1, N_STEPS)
        X.append(np.column_stack([np.interp(dst, src, seg[:, j]) for j in range(seg.shape[1])]))
        y.append(row["pitch_speed_mph"])
        g.append(row["pitcher_id"])
    return np.stack(X), np.asarray(y, float), np.asarray(g), chans


def _metrics(y, p) -> Dict[str, float]:
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"mae": float(np.mean(np.abs(y - p))), "r2": 1.0 - ss_res / ss_tot}


def fit_predict_bilstm(Xtr, ytr, Xte, seed: int = 0, epochs: int = 150, hidden: int = 32) -> np.ndarray:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    ym, ys = ytr.mean(), ytr.std() + 1e-6
    xt = torch.tensor((Xtr - mu) / sd, dtype=torch.float32)
    yt = torch.tensor((ytr - ym) / ys, dtype=torch.float32)
    xe = torch.tensor((Xte - mu) / sd, dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.lstm = nn.LSTM(d, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(0.2)
            self.head = nn.Linear(4 * hidden, 1)

        def forward(self, x):
            h, _ = self.lstm(x)
            z = torch.cat([h.mean(1), h[:, -1]], dim=1)
            return self.head(self.drop(z)).squeeze(-1)

    net = Net(Xtr.shape[-1])
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-3)
    lossf = nn.MSELoss()
    n = len(xt)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(n)
        for i in range(0, n, 32):
            b = perm[i:i + 32]
            opt.zero_grad()
            loss = lossf(net(xt[b]), yt[b])
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return net(xe).numpy() * ys + ym


def fit_predict_ridge(Xtr, ytr, Xte, alpha: float = 10.0) -> np.ndarray:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    ym = ytr.mean()
    w = np.linalg.solve(A.T @ A + alpha * np.eye(A.shape[1]), A.T @ (ytr - ym))
    return B @ w + ym


def pitcher_memory(ytr, gtr, gte) -> np.ndarray:
    means = pd.Series(ytr).groupby(gtr).mean()
    return np.array([means.get(g, ytr.mean()) for g in gte])
