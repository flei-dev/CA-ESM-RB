"""Dueling Q-network and Adam optimizer in NumPy.

Same architecture and hyper-parameters as the Keras network used by ESM-RB
(``esm_rb/agent.py``): input -> 256 ReLU -> 256 ReLU, then a value stream
(128 ReLU -> 1) and an advantage stream (128 ReLU -> n_actions), combined as
Q = V + A - mean(A).  Hidden layers use RandomNormal(0, 0.05) for weights and
biases, output layers use Glorot-uniform weights and zero biases (the Keras
defaults of the original code).  The loss is Keras' sample-weighted MSE.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

LAYERS = ("h1", "h2", "v1", "v2", "a1", "a2")


class DuelingQNet:
    def __init__(self, state_dim: int, action_dim: int, rng: np.random.Generator,
                 fc1: int = 256, fc2: int = 256, stream: int = 128):
        self.state_dim, self.action_dim = state_dim, action_dim

        def normal(shape):
            return rng.normal(0.0, 0.05, size=shape).astype(np.float32)

        def glorot(fan_in, fan_out):
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            return rng.uniform(-limit, limit, size=(fan_in, fan_out)).astype(np.float32)

        self.p = {
            "h1_w": normal((state_dim, fc1)), "h1_b": normal((fc1,)),
            "h2_w": normal((fc1, fc2)), "h2_b": normal((fc2,)),
            "v1_w": normal((fc2, stream)), "v1_b": normal((stream,)),
            "v2_w": glorot(stream, 1), "v2_b": np.zeros(1, np.float32),
            "a1_w": normal((fc2, stream)), "a1_b": normal((stream,)),
            "a2_w": glorot(stream, action_dim), "a2_b": np.zeros(action_dim, np.float32),
        }

    # ------------------------------------------------------------------ forward
    def forward(self, x: np.ndarray, cache: bool = False):
        p = self.p
        x = np.asarray(x, dtype=np.float32)
        z1 = x @ p["h1_w"] + p["h1_b"]; h1 = np.maximum(z1, 0)
        z2 = h1 @ p["h2_w"] + p["h2_b"]; h2 = np.maximum(z2, 0)
        zv = h2 @ p["v1_w"] + p["v1_b"]; hv = np.maximum(zv, 0)
        v = hv @ p["v2_w"] + p["v2_b"]
        za = h2 @ p["a1_w"] + p["a1_b"]; ha = np.maximum(za, 0)
        a = ha @ p["a2_w"] + p["a2_b"]
        q = v + a - a.mean(axis=1, keepdims=True)
        if cache:
            return q, (x, z1, h1, z2, h2, zv, hv, za, ha)
        return q

    __call__ = forward

    # ----------------------------------------------------------------- backward
    def gradients(self, cache, dq: np.ndarray) -> dict:
        p = self.p
        x, z1, h1, z2, h2, zv, hv, za, ha = cache
        dv = dq.sum(axis=1, keepdims=True)
        da = dq - dq.mean(axis=1, keepdims=True)
        g = {}
        g["v2_w"] = hv.T @ dv; g["v2_b"] = dv.sum(0)
        dhv = (dv @ p["v2_w"].T) * (zv > 0)
        g["v1_w"] = h2.T @ dhv; g["v1_b"] = dhv.sum(0)
        g["a2_w"] = ha.T @ da; g["a2_b"] = da.sum(0)
        dha = (da @ p["a2_w"].T) * (za > 0)
        g["a1_w"] = h2.T @ dha; g["a1_b"] = dha.sum(0)
        dh2 = (dhv @ p["v1_w"].T + dha @ p["a1_w"].T) * (z2 > 0)
        g["h2_w"] = h1.T @ dh2; g["h2_b"] = dh2.sum(0)
        dh1 = (dh2 @ p["h2_w"].T) * (z1 > 0)
        g["h1_w"] = x.T @ dh1; g["h1_b"] = dh1.sum(0)
        return g

    def loss_and_grads(self, x, targets, sample_weight):
        """Keras MSE with sample weights: sum_i w_i * mean_j (q-t)^2 / batch."""
        q, cache = self.forward(x, cache=True)
        diff = q - targets
        n, k = diff.shape
        w = np.asarray(sample_weight, dtype=np.float32)[:, None]
        loss = float(np.sum(w * diff ** 2) / (n * k))
        dq = (2.0 * w * diff / (n * k)).astype(np.float32)
        return loss, self.gradients(cache, dq)

    # ------------------------------------------------------------------- params
    def get_weights(self) -> dict:
        return {k: v.copy() for k, v in self.p.items()}

    def set_weights(self, weights: dict) -> None:
        for k in self.p:
            self.p[k] = np.asarray(weights[k], dtype=np.float32).copy()

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **self.p)

    @classmethod
    def load(cls, path: Path | str) -> "DuelingQNet":
        data = np.load(path)
        net = cls.__new__(cls)
        net.p = {k: data[k].astype(np.float32) for k in data.files}
        net.state_dim, net.action_dim = net.p["h1_w"].shape[0], net.p["a2_w"].shape[1]
        return net


class Adam:
    """Keras-style Adam (epsilon added to sqrt(v_hat))."""

    def __init__(self, params: dict, lr: float = 1e-4, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-7):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.b2 ** self.t) / (1 - self.b1 ** self.t)
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            params[k] -= (lr_t * self.m[k] / (np.sqrt(self.v[k]) + self.eps)).astype(np.float32)
