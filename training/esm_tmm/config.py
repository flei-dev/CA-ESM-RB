"""Experiment configuration and method variants."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace

# Variant flags:
#   context      throughput statistics in the state
#   lookahead    next-chunk VMAF / size of every ladder level in the state
#   constraint   lambda-conditioned policy: reward = ESM-RB reward - lambda * cost, lambda is a
#                state input and is sampled per training episode; the budget is enforced at run
#                time by choosing lambda (offline calibration + session feedback)
#   online       session-level feedback: lambda_t = lambda_k + kappa * (running cost - budget)
#   stall_cost   the budget cost includes the baseline energy spent while stalled
_C = dict(context=True, lookahead=True, constraint=True, online=True, stall_cost=True)
_U = dict(context=False, lookahead=False, constraint=False, online=False, stall_cost=False)
VARIANTS = {
    "esmrb":        dict(_U),                                                   # ESM-RB (retrained baseline)
    "ctx":          dict(_U, context=True, lookahead=True),                     # CA-ESM-RB (proposed method in the paper)
    "ctx_nolook":   dict(_U, context=True),                                     # ablation: throughput stats only
    "look_only":    dict(_U, lookahead=True),                                   # ablation: lookahead only
    # exploratory energy-budget variants; NOT used in the paper
    "cons":         dict(_C, context=False, lookahead=False),                   # + constraint only
    "full":         dict(_C),                                                   # context + constraint
    "full_nolook":  dict(_C, lookahead=False),                                  # ablation
    "full_noonline": dict(_C, online=False),                                    # ablation: no session feedback
    "full_nostall": dict(_C, stall_cost=False),                                 # ablation: content energy only
    # learned baselines re-implemented and retrained under our protocol (esm_tmm.baselines)
    "greenabr":     dict(_U),
    "pensieve":     dict(_U),
    "genet":        dict(_U),
}


@dataclass
class RunConfig:
    variant: str = "full"
    test_video: str = "tos"          # held-out video; the other two are used for training
    train_device: str = "galaxy"     # device whose measured energy drives reward/state
    seed: int = 0
    episodes: int = 30_000
    # DQN (same as ESM-RB)
    gamma: float = 0.99
    learning_rate: float = 1e-4
    batch_size: int = 32
    memory_size: int = 50_000
    warmup_episodes: int = 200
    learning_frequency: int = 4
    target_update_steps: int = 1_000
    n_step: int = 3
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 100_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.9999
    train_chunks: int = 45
    random_offset: bool = True      # random content start in training episodes
    energy_source: str = "model"    # "model": refitted power model (ESM-RB); "measured": raw per-chunk
    # reward (ESM-RB weights)
    quality_weight: float = 1.5
    rebuffer_penalty: float = 1.0
    smoothness_penalty: float = 0.5
    energy_weight: float = 0.5
    # constraint (Lagrangian)
    lambda0: float = 0.5             # v1 only (unused)
    eta: float = 0.05                # v1 only (unused)
    lambda_max: float = 5.0
    lam_zero_prob: float = 0.3       # training: share of episodes with lambda = 0
    lam_min: float = 0.05            # otherwise lambda ~ log-uniform[lam_min, lambda_max]
    kappa: float = 2.0               # gain of the session-level lambda feedback
    kappa_warm: int = 8              # feedback ramps in over the first chunks of a session
    budget_fracs: tuple = (0.2, 0.3, 0.4, 0.5, 0.6)
    default_budget_frac: float = 0.4  # used only if validation selection is disabled
    select_budget: bool = True       # pick the operating budget on training videos (validation)
    val_traces: int = 40
    calib_episodes: int = 300
    a2c_epochs: int = 15_000         # Pensieve / Genet: epochs of 16 agents x 100 steps
    tag: str = ""                    # extra label for sensitivity runs

    @property
    def flags(self) -> dict:
        return VARIANTS[self.variant]

    @property
    def train_videos(self) -> list[str]:
        return [v for v in ("tos", "bbb", "doc") if v != self.test_video]

    @property
    def run_id(self) -> str:
        parts = [self.variant, f"test-{self.test_video}", f"dev-{self.train_device}", f"seed{self.seed}"]
        if self.tag:
            parts.append(self.tag)
        return "__".join(parts)

    def to_json(self) -> str:
        d = asdict(self)
        d["budget_fracs"] = list(self.budget_fracs)
        return json.dumps(d, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "RunConfig":
        known = set(cls.__dataclass_fields__)
        d = {k: v for k, v in d.items() if k in known}
        if "budget_fracs" in d:
            d["budget_fracs"] = tuple(d["budget_fracs"])
        return cls(**d)

    def with_(self, **kw) -> "RunConfig":
        return replace(self, **kw)
