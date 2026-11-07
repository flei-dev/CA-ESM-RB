"""Evaluate the released NetLLM (Llama2-7B, ABR) checkpoint in the ESM-RB/CA-ESM-RB simulator.

The policy, its state encoding, reward and return-conditioning are exactly those of NetLLM's
`plm_special/test.py`; only the environment is ours (same trace replay as all other methods:
92 test traces, held-out videos tos/bbb/doc, 44 chunks, first chunk at the lowest level, no
delay noise).  NetLLM's bitrate ladder (300...4300 kbps) is identical to ours.  Each session
starts with empty context (the official test loop does not reset the context between traces;
we reset it so sessions are independent, as for every other method).

Output: rep_6-style logs  <out>/<video>/log_NetLLM<seed>_reward_<trace>  which
esm_tmm.analyze scores with the common metric code (method name "NetLLM*").

python run_netllm.py --plm-path <llama2-7b-hf dir> --model-dir <netllm ckpt dir> --device cuda:0 --seed 0
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ABR = HERE / "NetLLM" / "adaptive_bitrate_streaming"
ESM = HERE / "ESM"
sys.path.insert(0, str(ABR))
sys.path.insert(0, str(ESM / "training"))
os.chdir(ABR)                                           # NetLLM's config uses relative paths

from config import cfg                                   # noqa: E402  (NetLLM)
from plm_special.data.dataset import ExperienceDataset   # noqa: E402
from plm_special.models.rl_policy import OfflineRLPolicy  # noqa: E402
from plm_special.models.state_encoder import EncoderNetwork  # noqa: E402
from plm_special.models.low_rank import peft_model       # noqa: E402
from plm_special.utils.plm_utils import load_plm         # noqa: E402

from esm_tmm.config import RunConfig                     # noqa: E402  (ours)
from esm_tmm.data import Paths, Video, load_traces       # noqa: E402
from esm_tmm.env import ABREnv                           # noqa: E402
from esm_tmm.metrics import write_log                    # noqa: E402

BR = [300, 750, 1200, 1850, 2850, 4300]
REBUF_PENALTY, SMOOTH_PENALTY, CAP = 4.3, 1.0, 48.0


def build_policy(a):
    exp_pool = pickle.load(open(ABR / "artifacts" / "exp_pools" / "exp_pool.pkl", "rb"))
    info = ExperienceDataset(exp_pool, gamma=1.0, scale=1000, max_length=20).exp_dataset_info
    plm, *_ = load_plm("llama", a.plm_path, device_input_side=a.device, device_output_side=a.device,
                       device_middle_side=None)
    plm = peft_model(plm, "llama", rank=128)
    enc = EncoderNetwork(embed_dim=256).to(a.device)
    pol = OfflineRLPolicy(state_feature_dim=256, bitrate_levels=6, state_encoder=enc, plm=plm,
                          plm_embed_size=cfg.plm_embed_sizes["llama"]["base"], max_length=20,
                          max_ep_len=info["max_timestep"] + 1, device=a.device, device_out=a.device,
                          which_layer=-1)
    pol.plm.load_adapter(a.model_dir, adapter_name="default")
    pol.modules_except_plm.load_state_dict(torch.load(os.path.join(a.model_dir, "modules_except_plm.bin"),
                                                      map_location=a.device))
    pol.eval()
    return pol, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plm-path", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "netllm_logs"))
    ap.add_argument("--videos", default="tos,bbb,doc")
    ap.add_argument("--max-traces", type=int, default=0, help="debug: only the first N traces")
    a = ap.parse_args()

    pol, info = build_policy(a)
    target0 = info["max_return"] * 1.0
    lo, hi = info["min_reward"], info["max_reward"]

    def proc(r):
        r = min(hi, max(lo, r))
        return (r - lo) / (hi - lo) / 1000

    paths = Paths(ESM)
    times, bws, names = load_traces(paths.test_traces)
    n = a.max_traces or len(names)
    cfg_esm = RunConfig(variant="esmrb")
    t0 = time.time()
    with torch.no_grad():
        for vname in a.videos.split(","):
            video = Video(paths, vname)
            env = ABREnv(cfg_esm, [video], times, bws, list(range(len(names))), "galaxy",
                         np.random.default_rng(0), train=False, calib=None)
            for t in range(n):
                random.seed(a.seed * 100003 + t); np.random.seed(a.seed); torch.manual_seed(a.seed)
                env.reset(trace_idx=t)
                pol.clear_dq()
                state = torch.zeros((1, 1, 6, 6), dtype=torch.float32, device=a.device)
                act, last, target, step, records = 0, 0, target0, 0, []
                while True:
                    _s, _r, done, inf = env.step(act)
                    rec = inf["record"]
                    records.append(rec)
                    reward = (BR[act] / 1000 - REBUF_PENALTY * rec.rebuf
                              - SMOOTH_PENALTY * abs(BR[act] - BR[last]) / 1000)
                    last = act
                    state = torch.roll(state, -1, dims=-1)
                    state[..., 0, -1] = BR[act] / BR[-1]
                    state[..., 1, -1] = rec.buffer / 10.0
                    state[..., 2, -1] = rec.size / rec.delay / 1000.0
                    state[..., 3, -1] = rec.delay / 1000.0 / 10.0
                    nxt = video.sizes[env._c(env.chunk)] if not done else np.zeros(6)
                    state[..., 4, :6] = torch.as_tensor(nxt / 1e6, dtype=torch.float32, device=a.device)
                    state[..., 5, -1] = min(env.total_chunks - env.chunk, CAP) / CAP
                    if step > 0:
                        target -= proc(reward)
                    if done:
                        break
                    act = int(pol.sample(state, target, step))
                    step += 1
                write_log(Path(a.out) / vname / f"log_NetLLM{a.seed}_reward_{names[t]}", records)
            print(f"[seed {a.seed}] {vname}: {n} traces done ({time.time() - t0:.0f}s)", flush=True)
    print("all done ->", a.out)


if __name__ == "__main__":
    main()
