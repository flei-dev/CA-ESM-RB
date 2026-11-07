"""Self-test before launching the full experiment: python -m esm_tmm.selftest

1. finite-difference gradient check of the NumPy Q-network;
2. the metric code reproduces evaluation/rep_6/summaryResults for existing logs;
3. a short train + evaluate run completes.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig
from .data import DEVICES, Paths
from .metrics import DeviceEnergy, read_log, session_metrics
from .nn import DuelingQNet

REPO = Path(__file__).resolve().parents[2]


def check_gradients() -> float:
    rng = np.random.default_rng(0)
    net = DuelingQNet(9, 6, rng, fc1=16, fc2=12, stream=8)
    for k in net.p:
        net.p[k] = net.p[k].astype(np.float64) * 10
    x, t, w = rng.normal(size=(5, 9)), rng.normal(size=(5, 6)), rng.uniform(0.2, 1, 5)
    _, g = net.loss_and_grads(x, t, w)

    def loss():
        return np.sum(w[:, None] * (net.forward(x) - t) ** 2) / 30

    worst = 0.0
    for k, v in net.p.items():
        num = np.zeros_like(v)
        for i in np.ndindex(v.shape):
            o = v[i]; v[i] = o + 1e-6; lp = loss(); v[i] = o - 1e-6; lm = loss(); v[i] = o
            num[i] = (lp - lm) / 2e-6
        worst = max(worst, np.abs(num - g[k]).max() / (np.abs(num).max() + 1e-12))
    return worst


def check_metrics(paths: Paths, n: int = 10) -> float:
    worst = 0.0
    for video in ("tos", "bbb", "doc"):
        ref_path = REPO / "evaluation" / "rep_6" / "summaryResults" / f"{video}_BB_summaryResults.csv"
        if not ref_path.exists():
            continue
        ref = pd.read_csv(ref_path, index_col=0)
        dev = {d: DeviceEnergy(paths.power_files, video, d) for d in DEVICES}
        logs = sorted((paths.baseline_logs / video).glob("log_BB_reward_*"))[:n]
        for f in logs:
            m = session_metrics(read_log(f), dev)
            r = ref.loc[f.name.split("_reward_")[1]]
            for k in ("QoE", "Energy_G", "Energy_X"):
                worst = max(worst, abs(m[k] - r[k]) / max(abs(r[k]), 1e-9))
    return worst


def main():
    paths = Paths(REPO)
    g = check_gradients()
    print(f"[1] gradient check: max relative error {g:.2e} -> {'OK' if g < 1e-4 else 'FAIL'}")
    m = check_metrics(paths)
    print(f"[2] metrics vs original summaryResults: max relative error {m:.2e} -> {'OK' if m < 1e-9 else 'FAIL'}")
    from .run import run
    tmp = Path(tempfile.mkdtemp())
    try:
        cfg = RunConfig(variant="full", test_video="tos", episodes=300, calib_episodes=60)
        d = run(cfg, REPO, tmp, log_every=100)
        ok = (d / "done.json").exists() and len(pd.read_csv(d / "results.csv")) > 0
        print(f"[3] short train+evaluate run -> {'OK' if ok else 'FAIL'}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
