"""Data loading: videos (rep_6 ladder), measured device energy, network traces."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BITRATES = (300, 750, 1200, 1850, 2850, 4300)   # kbps, rep_6 ladder
VIDEOS = ("tos", "bbb", "doc")
DEVICES = ("galaxy", "xcover")
N_CHUNKS = 45          # chunks available per file
EVAL_CHUNKS = 44       # chunks streamed per session (as in rep_6 evaluate.py)


@dataclass
class Paths:
    repo: Path

    @property
    def train_traces(self) -> Path:
        return self.repo / "training" / "cooked_traces"

    @property
    def test_traces(self) -> Path:
        return self.repo / "evaluation" / "test_sim_traces"

    @property
    def eval_files(self) -> Path:
        return self.repo / "evaluation" / "evaluationFiles"

    @property
    def power_files(self) -> Path:
        return self.repo / "evaluation" / "powerMeasurementFiles"

    @property
    def baseline_logs(self) -> Path:
        return self.repo / "evaluation" / "rep_6" / "test_results"


class Video:
    """Per-chunk sizes, phone VMAF and measured local energy for one video."""

    def __init__(self, paths: Paths, name: str):
        self.name = name
        root = paths.eval_files / name / "rep_6"
        self.sizes = np.zeros((N_CHUNKS, len(BITRATES)), dtype=np.float64)
        for q in range(len(BITRATES)):
            with (root / f"video_size_{q}").open() as f:
                vals = [int(line.split()[0]) for line in f if line.split()]
            self.sizes[:, q] = vals[:N_CHUNKS]
        vmaf = pd.read_csv(root / "vmaf_phone.csv", index_col=0)
        self.vmaf = np.stack([vmaf[f"VMAF_{q + 1}"].to_numpy()[:N_CHUNKS] for q in range(len(BITRATES))], axis=1)
        self.local = {}
        for dev in DEVICES:
            t = pd.read_csv(paths.power_files / f"{dev}_{name.upper()}_streaming.csv", index_col=0)
            t.columns = [str(c) for c in t.columns]
            self.local[dev] = np.stack([t[str(b)].to_numpy()[:N_CHUNKS] for b in BITRATES], axis=1)


def load_traces(trace_dir: Path):
    times, bws, names = [], [], []
    for path in sorted(p for p in Path(trace_dir).iterdir() if p.is_file() and not p.name.startswith(".")):
        t, b = [], []
        with path.open() as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    t.append(float(parts[0])); b.append(float(parts[1]))
        if len(t) >= 2:
            times.append(t); bws.append(b); names.append(path.name)
    if not names:
        raise FileNotFoundError(f"No traces in {trace_dir}")
    return times, bws, names


def split_train_val(names: list[str], val_fraction: float = 0.15):
    """Deterministic split of training traces (by name hash) into train / validation."""
    train, val = [], []
    for i, n in enumerate(names):
        h = int(hashlib.md5(n.encode()).hexdigest(), 16) % 1000
        (val if h < val_fraction * 1000 else train).append(i)
    return train, val
