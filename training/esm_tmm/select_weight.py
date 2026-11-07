"""Choose the energy weight of each method on the TRAINING videos (no retraining).

python -m esm_tmm.select_weight [--runs results/esm_tmm/runs_v2 --v1 results/esm_tmm/runs]

For every finished galaxy run of ESM-RB and CA-ESM-RB with energy weight 0.25 / 0.5 / 1.0
(seeds 0-2), the saved policy is replayed on validation sessions: the two training videos of
its fold x 40 traces drawn from the training trace set, scored with the measured energy of
the training device.  The weight with the highest mean QoE/energy is selected per method.
The held-out video and the test traces are never touched, so the selection is legitimate
model selection.  Writes results/esm_tmm/weight_selection.json, which `experiments` reads to
add seeds 3-4 for the selected weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import RunConfig
from .data import Paths, Video
from .env import Calibration
from .metrics import DeviceEnergy, session_metrics
from .nn import DuelingQNet
from .powermodel import fit_for_fold
from .run import _val_envs, rollout

REPO = Path(__file__).resolve().parents[2]
TAG_W = {"": 0.5, "ew0.25": 0.25, "ew1": 1.0}
OUT = REPO / "results" / "esm_tmm" / "weight_selection.json"


class Greedy:
    def __init__(self, q: DuelingQNet):
        self.q = q

    def act(self, state, explore: bool = False) -> int:
        return int(np.argmax(self.q(state[None, :])[0]))


def validate(run_dir: Path, paths: Paths) -> dict:
    cfg = RunConfig.from_dict(json.loads((run_dir / "config.json").read_text()))
    calib = Calibration(**json.loads((run_dir / "calibration.json").read_text()))
    tables = None
    if cfg.energy_source == "model":
        videos = [Video(paths, v) for v in cfg.train_videos]
        power, _ = fit_for_fold(videos, cfg.train_device, cfg.seed)
        tables = {v.name: power.table(v) for v in videos}
    agent = Greedy(DuelingQNet.load(run_dir / "model_final.npz"))
    envs, ids = _val_envs(cfg, paths, calib, tables)
    key = {"galaxy": "G", "xcover": "X"}[cfg.train_device]
    eff, q, e = [], [], []
    for vname, env in envs:
        dev = {cfg.train_device: DeviceEnergy(paths.power_files, vname, cfg.train_device)}
        for t in ids:
            recs, _, _ = rollout(env, agent, t, 0.0, 0.0)
            m = session_metrics(recs, dev)
            eff.append(m[f"Efficiency_{key}"]); q.append(m["QoE"]); e.append(m[f"Energy_{key}"])
    return {"variant": cfg.variant, "weight": TAG_W[cfg.tag], "seed": cfg.seed, "fold": cfg.test_video,
            "efficiency": float(np.mean(eff)), "QoE": float(np.mean(q)), "energy": float(np.mean(e))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "results" / "esm_tmm" / "runs_v2"))
    ap.add_argument("--v1", default=str(REPO / "results" / "esm_tmm" / "runs"))
    a = ap.parse_args()
    paths = Paths(REPO)
    rows = []
    for root in (Path(a.v1), Path(a.runs)):
        for d in sorted(root.iterdir()):
            if not (d / "done.json").exists() or not (d / "model_final.npz").exists():
                continue
            c = json.loads((d / "config.json").read_text())
            if (c["variant"] in ("esmrb", "ctx") and c["train_device"] == "galaxy"
                    and c.get("tag", "") in TAG_W and c["seed"] <= 2
                    and (root == Path(a.runs) or c.get("tag", "") == "")):
                rows.append(validate(d, paths))
                print(rows[-1], flush=True)
    out = {"rows": rows, "selected": {}, "summary": {}}
    for v in ("esmrb", "ctx"):
        per_w = {}
        for w in sorted(set(TAG_W.values())):
            vals = [r["efficiency"] for r in rows if r["variant"] == v and r["weight"] == w]
            if vals:
                per_w[w] = float(np.mean(vals))
        if per_w:
            out["summary"][v] = per_w
            out["selected"][v] = max(per_w, key=per_w.get)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print("validation efficiency by weight:", json.dumps(out["summary"], indent=1))
    print("selected:", out["selected"])


if __name__ == "__main__":
    main()
