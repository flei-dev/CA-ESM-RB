"""Neural power model (ESM-RB's local playback energy predictor), refitted per fold.

Features per (chunk, bitrate level), all available from the DASH manifest of any video:
bitrate, chunk size, phone VMAF and pixel rate (the original model also used a motion
feature, which only exists for Tears of Steel).  Target: measured local playback energy of
the training device.  Architecture as in the paper: 7 and 20 ReLU units, linear output,
MSE + Adam.  The model is fitted only on the training videos of the fold, so it never
sees the held-out video.  Its predictions drive the training reward/state; the final
evaluation always uses the measured energy of both phones.
"""

from __future__ import annotations

import numpy as np

from .data import BITRATES, N_CHUNKS, Video
from .nn import Adam

PIXELS = np.array([320 * 180, 400 * 360, 768 * 432, 1024 * 576, 1280 * 720, 1920 * 1080], float)
E_SCALE = 1000.0


def features(v: Video) -> np.ndarray:
    """(N_CHUNKS, 6, 4) feature tensor."""
    b = np.broadcast_to(np.array(BITRATES, float) / 4300.0, (N_CHUNKS, 6))
    p = np.broadcast_to(PIXELS / PIXELS.max(), (N_CHUNKS, 6))
    return np.stack([b, v.sizes / 2.5e6, v.vmaf / 100.0, p], axis=-1)


class PowerModel:
    def __init__(self, seed: int, hidden=(7, 20)):
        rng = np.random.default_rng(seed + 777)
        dims = (4,) + tuple(hidden) + (1,)
        self.p = {}
        for i in range(len(dims) - 1):
            self.p[f"w{i}"] = rng.normal(0, np.sqrt(2.0 / dims[i]), (dims[i], dims[i + 1]))
            self.p[f"b{i}"] = np.zeros(dims[i + 1])
        self.n = len(dims) - 1

    def _forward(self, x):
        hs = [x]
        for i in range(self.n):
            z = hs[-1] @ self.p[f"w{i}"] + self.p[f"b{i}"]
            hs.append(np.maximum(z, 0) if i < self.n - 1 else z)
        return hs

    def predict(self, x):
        return self._forward(x)[-1][..., 0] * E_SCALE

    def fit(self, x, y, iters=3000, lr=3e-3):
        y = y / E_SCALE
        opt = Adam(self.p, lr=lr, eps=1e-8)
        for _ in range(iters):
            hs = self._forward(x)
            d = 2.0 * (hs[-1][:, 0] - y)[:, None] / len(y)
            g = {}
            for i in reversed(range(self.n)):
                g[f"w{i}"] = hs[i].T @ d
                g[f"b{i}"] = d.sum(0)
                if i > 0:
                    d = (d @ self.p[f"w{i}"].T) * (hs[i] > 0)
            opt.step(self.p, g)
        return self

    def table(self, v: Video) -> np.ndarray:
        return self.predict(features(v).reshape(-1, 4)).reshape(N_CHUNKS, 6)


def r2(y, yhat) -> float:
    return float(1 - np.mean((y - yhat) ** 2) / np.var(y))


def fit_for_fold(train_videos: list[Video], device: str, seed: int, chunks=slice(1, 44)):
    x = np.concatenate([features(v)[chunks].reshape(-1, 4) for v in train_videos])
    y = np.concatenate([v.local[device][chunks].reshape(-1) for v in train_videos])
    m = PowerModel(seed).fit(x, y)
    return m, {"device": device, "train_videos": [v.name for v in train_videos], "n_samples": int(len(y)),
               "r2_train": r2(y, m.predict(x))}


def generalization_report(m: PowerModel, videos: list[Video], chunks=slice(1, 44)) -> dict:
    """R^2 of the fitted model against measured energy of every video/device."""
    out = {}
    for v in videos:
        for dev, table in v.local.items():
            pred = m.table(v)[chunks]
            meas = table[chunks]
            out[f"{v.name}_{dev}_r2"] = r2(meas.reshape(-1), pred.reshape(-1))
            a = meas - meas.mean(1, keepdims=True)
            b = pred - pred.mean(1, keepdims=True)
            out[f"{v.name}_{dev}_within_chunk_corr"] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    return out
