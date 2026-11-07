"""ABR simulator with measured device energy, content context and an energy budget.

The chunk download dynamics (`_download`) are copied line for line from
``esm_rb/environment.py`` (Mahimahi-style trace replay, 80 ms RTT, 4 s chunks,
60 s buffer cap), so trained policies are evaluated on exactly the simulator
that reproduced the paper's rep_6 results.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import RunConfig
from .data import BITRATES, EVAL_CHUNKS, N_CHUNKS, Video
from .metrics import BASE_POWER, ChunkRecord, network_energy

MAX_BITRATE = float(max(BITRATES))
SIZE_NORM = 2.5e6


@dataclass
class Calibration:
    """Energy statistics of the training videos/device, used to normalise energy.

    Percentiles are taken over playback + network ("content") energy, i.e. without the
    baseline energy spent while stalled, which would otherwise dominate the scale.
    Reward, state and the budget constraint use content energy for every variant (as the
    original ESM-RB reward did); stall energy is only added in the final evaluation metric.
    """
    p5: float
    p75: float
    p95: float
    e_lo: float      # mean normalised energy of the always-lowest policy
    e_hi: float      # mean normalised energy of the always-highest policy

    @property
    def scale(self) -> float:
        return max(self.p95 - self.p5, 1e-6)

    def norm(self, energy: float) -> float:
        return (energy - self.p5) / self.scale

    def stall(self, stall_energy: float) -> float:
        """Stall (baseline) energy on the same scale as `norm` (no offset: zero stall costs zero)."""
        return stall_energy / self.scale

    def budget(self, frac: float) -> float:
        return self.e_lo + frac * (self.e_hi - self.e_lo)


def state_dim(cfg: RunConfig) -> int:
    f = cfg.flags
    d = 7
    if f["context"]:
        d += 2
    if f["lookahead"]:
        d += 12
    if f["constraint"]:
        d += 1
    return d


class ABREnv:
    def __init__(self, cfg: RunConfig, videos: list[Video], times, bws, trace_ids: list[int],
                 device: str, rng: np.random.Generator, *, train: bool, calib: Calibration | None = None,
                 energy_tables: dict | None = None):
        self.cfg, self.videos, self.times, self.bws = cfg, videos, times, bws
        self.trace_ids, self.device, self.rng, self.train = list(trace_ids), device, rng, train
        self.calib = calib
        # Local playback energy used by the agent: power-model predictions (default) or measured.
        self.energy_tables = energy_tables or {v.name: v.local[device] for v in videos}
        self.total_chunks = cfg.train_chunks if train else EVAL_CHUNKS
        self.budget_value = 0.0
        self.lam = 0.0          # lambda in effect for the next decision
        self.lam_base = 0.0     # per-budget multiplier learned across episodes
        self._eval_pos = 0

    # ------------------------------------------------------------ trace replay
    def _select(self, trace_idx: int, randomize_ptr: bool) -> None:
        self.cooked_time = self.times[trace_idx]
        self.cooked_bw = self.bws[trace_idx]
        self.ptr = int(self.rng.integers(1, len(self.cooked_bw))) if randomize_ptr else 1
        self.last_time = self.cooked_time[self.ptr - 1]

    def _download(self, size: float):
        delay = 0.0
        sent = 0.0
        while True:
            throughput = self.cooked_bw[self.ptr] * 1_000_000.0 / 8.0
            duration = self.cooked_time[self.ptr] - self.last_time
            payload = throughput * duration * 0.95
            if sent + payload > size:
                fractional = (size - sent) / throughput / 0.95
                delay += fractional
                self.last_time += fractional
                break
            sent += payload
            delay += duration
            self.last_time = self.cooked_time[self.ptr]
            self.ptr += 1
            if self.ptr >= len(self.cooked_bw):
                self.ptr = 1
                self.last_time = 0.0
        delay = delay * 1000.0 + 80.0
        if self.train:
            delay *= self.rng.uniform(0.9, 1.1)
        rebuffer = max(delay - self.buffer, 0.0)
        self.buffer = max(self.buffer - delay, 0.0) + 4000.0
        if self.buffer > 60000.0:
            sleep = np.ceil((self.buffer - 60000.0) / 500.0) * 500.0
            self.buffer -= sleep
            while True:
                duration = self.cooked_time[self.ptr] - self.last_time
                if duration > sleep / 1000.0:
                    self.last_time += sleep / 1000.0
                    break
                sleep -= duration * 1000.0
                self.last_time = self.cooked_time[self.ptr]
                self.ptr += 1
                if self.ptr >= len(self.cooked_bw):
                    self.ptr = 1
                    self.last_time = 0.0
        return delay, rebuffer

    # --------------------------------------------------------------- episodes
    def reset(self, *, trace_idx: int | None = None, budget: float | None = None, lam: float = 0.0) -> np.ndarray:
        if self.train:
            self.video = self.videos[int(self.rng.integers(len(self.videos)))]
            self._select(self.trace_ids[int(self.rng.integers(len(self.trace_ids)))], True)
        else:
            self.video = self.videos[0]
            self._select(trace_idx if trace_idx is not None else self.trace_ids[0], False)
        self.chunk = 0
        # Training episodes start at a random point of the (looped) video, so the chunk index
        # carries no information about which content is played; evaluation starts at chunk 0.
        self.offset = int(self.rng.integers(N_CHUNKS)) if (self.train and self.cfg.random_offset) else 0
        self.buffer = 0.0
        self.last_action = 1
        self.thr_hist: list[float] = []
        self.energy_hist: list[float] = []      # per-chunk budget cost
        self.budget_value, self.lam, self.lam_base = budget, lam, lam
        base = np.zeros(7, dtype=np.float32)
        return self._state(base)

    def _c(self, i: int) -> int:
        return (self.offset + i) % N_CHUNKS

    def _state(self, base: np.ndarray) -> np.ndarray:
        f = self.cfg.flags
        parts = [base]
        if f["context"]:
            h = np.asarray(self.thr_hist[-5:]) if self.thr_hist else np.zeros(1)
            parts.append(np.array([h.mean() / 10.0, h.std() / 10.0], dtype=np.float32))
        if f["lookahead"]:
            nxt = self._c(self.chunk if self.chunk < self.total_chunks else 0)
            parts.append((self.video.vmaf[nxt] / 100.0).astype(np.float32))
            parts.append((self.video.sizes[nxt] / SIZE_NORM).astype(np.float32))
        if f["constraint"]:
            parts.append(np.array([self.lam / self.cfg.lambda_max], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def step(self, action: int):
        cfg, v = self.cfg, self.video
        idx = self.chunk
        c = self._c(idx)                 # content index of this chunk
        size = float(v.sizes[c, action])
        delay, rebuffer_ms = self._download(size)
        rebuf = rebuffer_ms / 1000.0
        self.chunk += 1
        done = self.chunk >= self.total_chunks
        thr_mbytes = size / delay / 1000.0            # MB/s, as in ESM-RB
        thr_mbps = size / delay * 8.0 * 1000.0 / 1e6  # Mbit/s, as in the evaluation script
        self.thr_hist.append(thr_mbytes)
        content_energy = float(self.energy_tables[v.name][c, action]) + network_energy(thr_mbps, size / 1e6)
        stall_energy = BASE_POWER[self.device] * rebuf
        energy = content_energy + stall_energy
        e_norm = self.calib.norm(content_energy) if self.calib else 0.0
        cost = e_norm + (self.calib.stall(stall_energy) if (self.calib and cfg.flags["stall_cost"]) else 0.0)
        self.energy_hist.append(cost)
        quality = float(v.vmaf[c, action])
        smooth = 0.0 if idx == 0 else abs(quality - float(v.vmaf[self._c(idx - 1), self.last_action]))

        base = np.zeros(7, dtype=np.float32)
        base[0] = BITRATES[action] / MAX_BITRATE
        base[1] = self.buffer / 1000.0 / 10.0
        base[2] = np.clip(thr_mbytes / 10.0, 0.0, 2.0)
        base[3] = delay / 1000.0 / 10.0
        base[4] = min(self.total_chunks - self.chunk, 45) / 45.0
        base[5] = np.clip(e_norm, 0.0, 2.0)
        base[6] = quality / 100.0

        reward = (cfg.quality_weight * quality / 100.0
                  - (cfg.rebuffer_penalty * np.sqrt(rebuf) if rebuf > 0 else 0.0)
                  - cfg.smoothness_penalty * smooth / 100.0
                  + min(self.buffer / 1000.0 / 30.0, 0.3))
        if self.calib:
            # ESM-RB efficiency shaping (all variants); the constrained variants add a
            # Lagrangian budget term, so with lambda = 0 they reduce to the unconstrained reward.
            pen = e_norm
            if content_energy > self.calib.p75:
                pen += 2.0 * np.log1p((content_energy - self.calib.p75) / self.calib.scale)
            reward -= cfg.energy_weight * pen
        lam_used = self.lam
        if cfg.flags["constraint"]:
            reward -= lam_used * cost
            if cfg.flags["online"] and self.budget_value is not None:
                # session-level feedback on the running budget deviation (uses past chunks only)
                dev = float(np.mean(self.energy_hist)) - self.budget_value
                ramp = min(1.0, len(self.energy_hist) / max(cfg.kappa_warm, 1))
                self.lam = float(np.clip(self.lam_base + cfg.kappa * ramp * dev, 0.0, cfg.lambda_max))

        record = ChunkRecord(idx, BITRATES[action], self.buffer / 1000.0, rebuf, size, delay, quality)
        self.last_action = action
        return self._state(base), float(reward), done, {"energy": energy, "e_norm": e_norm,
                                                         "content_energy": content_energy,
                                                         "cost": cost, "lam": lam_used,
                                                         "quality": quality, "rebuf": rebuf,
                                                         "smooth": smooth, "record": record}


def calibrate(cfg: RunConfig, videos, times, bws, trace_ids, device, seed: int,
              energy_tables: dict | None = None) -> Calibration:
    """Energy percentiles under a random policy, plus always-lowest / always-highest levels."""
    rng = np.random.default_rng(seed + 12345)
    env = ABREnv(cfg, videos, times, bws, trace_ids, device, rng, train=True, calib=None,
                 energy_tables=energy_tables)
    energies = []
    per_policy = {"lo": [], "hi": []}
    n = cfg.calib_episodes
    for ep in range(n):
        mode = "rand" if ep < n // 2 else ("lo" if ep < 3 * n // 4 else "hi")
        env.reset()
        done = False
        ep_e = []
        while not done:
            a = int(rng.integers(len(BITRATES))) if mode == "rand" else (0 if mode == "lo" else len(BITRATES) - 1)
            _, _, done, info = env.step(a)
            ep_e.append(info["content_energy"])
        energies.extend(ep_e)
        if mode in per_policy:
            per_policy[mode].append(ep_e)
    p5, p75, p95 = (float(np.percentile(energies, q)) for q in (5, 75, 95))
    scale = max(p95 - p5, 1e-6)
    e_lo = float(np.mean([(np.mean(e) - p5) / scale for e in per_policy["lo"]]))
    e_hi = float(np.mean([(np.mean(e) - p5) / scale for e in per_policy["hi"]]))
    return Calibration(p5, p75, p95, e_lo, e_hi)
