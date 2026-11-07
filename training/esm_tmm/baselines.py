"""Learned baselines re-implemented from their papers and retrained under OUR protocol
(leave-one-video-out, same training traces, same simulator, same evaluation).

- greenabr : GreenABR (Turkkan et al., 2022) - double DQN, MLP 256-256-128-64, batch 64,
             lr 1e-4, gamma 0.99, update every step, target copy every 100 transitions,
             epsilon x0.9995 per episode; 7-dim state; reward = nonlinear VMAF/20 term
             - 4.3*rebuffer - |dVMAF|/20 - energy penalty ((E-1000)/1000 + 2^((E-3000)/1000)
             above 3000).  Energy = per-fold power model + network model (as ESM-RB).
- pensieve : Pensieve (Mao et al., 2017) - actor-critic (synchronous A3C, 16 agents,
             100-step rollouts), state 6x8, conv1d(128,k=4) / dense(128) branches -> 128 ->
             softmax; actor lr 1e-4, critic lr 1e-3, gamma 0.99, entropy 0.5 -> 0.05;
             linear QoE reward (Mbps - 4.3*rebuffer - |dMbps|).
- genet    : Genet (Xia et al., 2022) - Pensieve trained with a curriculum: every round the
             environment configuration (synthetic trace parameters) with the largest gap
             between a rule-based baseline (BBA) and the current policy is added to the
             training distribution.  Genet searches configurations with Bayesian
             optimisation; we use random search with the same budget (15 candidates/round).

All three use our random content offset in training (as every learned method here) and are
evaluated exactly like the other methods: 92 test traces, held-out video, first chunk at the
lowest level, greedy actions.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from .config import RunConfig
from .data import BITRATES, DEVICES, Paths, Video, load_traces
from .env import ABREnv
from .metrics import DeviceEnergy, session_metrics, write_log
from .nn import Adam
from .powermodel import fit_for_fold

BR = np.array(BITRATES, float)
A = len(BITRATES)
BASELINE_VARIANTS = ("greenabr", "pensieve", "genet")


# ============================================================================ small nets
def glorot(rng, fan_in, fan_out, shape=None):
    lim = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-lim, lim, size=shape or (fan_in, fan_out)).astype(np.float32)


class MLP:
    """ReLU MLP with a linear output (Keras Dense defaults: glorot-uniform, zero bias)."""

    def __init__(self, dims, rng):
        self.n = len(dims) - 1
        self.p = {}
        for i in range(self.n):
            self.p[f"w{i}"] = glorot(rng, dims[i], dims[i + 1])
            self.p[f"b{i}"] = np.zeros(dims[i + 1], np.float32)

    def forward(self, x, cache=False):
        hs, zs = [np.asarray(x, np.float32)], []
        for i in range(self.n):
            z = hs[-1] @ self.p[f"w{i}"] + self.p[f"b{i}"]
            zs.append(z)
            hs.append(np.maximum(z, 0) if i < self.n - 1 else z)
        return (hs[-1], (hs, zs)) if cache else hs[-1]

    __call__ = forward

    def backward(self, cache, dout):
        hs, zs = cache
        g, d = {}, dout
        for i in reversed(range(self.n)):
            g[f"w{i}"] = hs[i].T @ d
            g[f"b{i}"] = d.sum(0)
            if i > 0:
                d = (d @ self.p[f"w{i}"].T) * (zs[i - 1] > 0)
        return g

    def get(self):
        return {k: v.copy() for k, v in self.p.items()}

    def set(self, w):
        for k in self.p:
            self.p[k] = w[k].copy()


class PensieveNet:
    """Pensieve's network on a (B, 6, 8) state; out = 6 logits (actor) or 1 value (critic)."""
    K, F, H = 4, 128, 128

    def __init__(self, out_dim, rng):
        K, F, H = self.K, self.F, self.H
        self.out_dim = out_dim
        self.p = {}
        for n in ("d0", "d1", "d5"):
            self.p[n + "_w"] = glorot(rng, 1, F); self.p[n + "_b"] = np.zeros(F, np.float32)
        for n in ("c2", "c3", "c4"):
            self.p[n + "_w"] = glorot(rng, K, F, (K, F)); self.p[n + "_b"] = np.zeros(F, np.float32)
        merged = 3 * F + 2 * (8 - K + 1) * F + (6 - K + 1) * F
        self.p["m_w"] = glorot(rng, merged, H); self.p["m_b"] = np.zeros(H, np.float32)
        self.p["o_w"] = glorot(rng, H, out_dim); self.p["o_b"] = np.zeros(out_dim, np.float32)

    @staticmethod
    def _unfold(x, k):                       # (B, L) -> (B, L-k+1, k)
        L = x.shape[1]
        return np.stack([x[:, i:i + k] for i in range(L - k + 1)], axis=1)

    def forward(self, s, cache=False):
        p, K = self.p, self.K
        s = np.asarray(s, np.float32)
        B = s.shape[0]
        parts, c = [], {}
        for n, row in (("d0", 0), ("d1", 1), ("d5", 5)):
            x = s[:, row, -1:]
            z = x @ p[n + "_w"] + p[n + "_b"]; c[n] = (x, z); parts.append(np.maximum(z, 0))
        for n, row, L in (("c2", 2, 8), ("c3", 3, 8), ("c4", 4, 6)):
            u = self._unfold(s[:, row, :L], K)                        # (B, P, K)
            z = u @ p[n + "_w"] + p[n + "_b"]                        # (B, P, F)
            c[n] = (u, z); parts.append(np.maximum(z, 0).reshape(B, -1))
        h = np.concatenate(parts, axis=1)
        zm = h @ p["m_w"] + p["m_b"]; hm = np.maximum(zm, 0)
        out = hm @ p["o_w"] + p["o_b"]
        if cache:
            c["h"], c["zm"], c["hm"], c["sizes"] = h, zm, hm, [q.shape[1] for q in parts]
            return out, c
        return out

    __call__ = forward

    def backward(self, c, dout):
        p, g = self.p, {}
        g["o_w"] = c["hm"].T @ dout; g["o_b"] = dout.sum(0)
        dzm = (dout @ p["o_w"].T) * (c["zm"] > 0)
        g["m_w"] = c["h"].T @ dzm; g["m_b"] = dzm.sum(0)
        dh = dzm @ p["m_w"].T
        offs = np.cumsum([0] + c["sizes"])
        names = ["d0", "d1", "d5", "c2", "c3", "c4"]
        for i, n in enumerate(names):
            d = dh[:, offs[i]:offs[i + 1]]
            x, z = c[n]
            if n.startswith("d"):
                dz = d * (z > 0)
                g[n + "_w"] = x.T @ dz; g[n + "_b"] = dz.sum(0)
            else:
                dz = d.reshape(z.shape) * (z > 0)                     # (B, P, F)
                g[n + "_w"] = np.einsum("bpk,bpf->kf", x, dz); g[n + "_b"] = dz.sum((0, 1))
        return g


# ============================================================================ states
class PensieveState:
    """Pensieve's 6x8 history state built from our simulator's chunk records."""

    def __init__(self):
        self.s = np.zeros((6, 8), np.float32)

    def reset(self):
        self.s[:] = 0.0

    def update(self, env: ABREnv, action: int, rec):
        s = np.roll(self.s, -1, axis=1)
        s[0, -1] = BR[action] / BR[-1]
        s[1, -1] = rec.buffer / 10.0
        s[2, -1] = rec.size / rec.delay / 1000.0
        s[3, -1] = rec.delay / 1000.0 / 10.0
        nxt = env.video.sizes[env._c(env.chunk)] if env.chunk < env.total_chunks else np.zeros(A)
        s[4, :] = 0.0
        s[4, :A] = nxt / 1e6
        s[5, -1] = min(env.total_chunks - env.chunk, 48) / 48.0
        self.s = s
        return s.copy()


def pensieve_reward(a, last, rebuf):
    return BR[a] / 1000.0 - 4.3 * rebuf - abs(BR[a] - BR[last]) / 1000.0


E_CAP = 10_000.0   # GreenABR's 2^((E-3000)/1000) term overflows when the network-energy model sees
                   # near-zero throughput on our traces; energy is capped at 10 J (penalty <= 135).
                   # The raw energy and VMAF state features are also scaled (/1000, /100): unscaled,
                   # the Q-network diverged within a few hundred episodes on our traces.


def greenabr_state(env, a, rec, info):
    return np.array([BR[a] / BR[-1], rec.buffer / 10.0, rec.size / rec.delay / 1000.0,
                     rec.delay / 1000.0 / 10.0, min(env.total_chunks - env.chunk, 45) / 45.0,
                     min(info["content_energy"], E_CAP) / 1000.0, info["quality"] / 100.0], np.float32)


def greenabr_reward(info, last_vmaf, first):
    q = info["quality"] / 20.0
    qr = q + 2 ** (q - 3) if q > 3 else q
    smooth = 0.0 if first else abs(info["quality"] - last_vmaf) / 20.0
    e = min(info["content_energy"], E_CAP)
    pen = (e - 1000.0) / 1000.0 + (2 ** ((e - 3000.0) / 1000.0) if e > 3000 else 0.0)
    return qr - 4.3 * info["rebuf"] - smooth - pen


# ============================================================================ GreenABR
class GreenABRAgent:
    def __init__(self, rng, mem=500_000, batch=64, lr=1e-4, gamma=0.99, replace=100):
        self.rng, self.batch, self.gamma, self.replace = rng, batch, gamma, replace
        self.q = MLP([7, 256, 256, 128, 64, A], rng)
        self.qt = MLP([7, 256, 256, 128, 64, A], rng); self.qt.set(self.q.get())
        self.opt = Adam(self.q.p, lr=lr)
        self.S = np.zeros((mem, 7), np.float32); self.S2 = np.zeros((mem, 7), np.float32)
        self.Aa = np.zeros(mem, np.int64); self.R = np.zeros(mem, np.float32); self.NT = np.zeros(mem, np.float32)
        self.mem, self.cnt, self.eps = mem, 0, 1.0

    def act(self, s, explore=True):
        if explore and self.rng.random() < self.eps:
            return int(self.rng.integers(A))
        return int(np.argmax(self.q(s[None])[0]))

    def remember(self, s, a, r, s2, done):
        i = self.cnt % self.mem
        self.S[i], self.Aa[i], self.R[i], self.S2[i], self.NT[i] = s, a, r, s2, 1.0 - float(done)
        self.cnt += 1

    def learn(self):
        if self.cnt <= self.batch:
            return
        idx = self.rng.integers(min(self.cnt, self.mem), size=self.batch)
        s, a, r, s2, nt = self.S[idx], self.Aa[idx], self.R[idx], self.S2[idx], self.NT[idx]
        qn, qe = self.qt(s2), self.q(s2)
        qp, cache = self.q.forward(s, cache=True)
        tgt = qp.copy()
        b = np.arange(self.batch)
        tgt[b, a] = r + self.gamma * qn[b, np.argmax(qe, 1)] * nt
        dout = (2.0 * (qp - tgt) / qp.size).astype(np.float32)        # Keras mse
        self.opt.step(self.q.p, self.q.backward(cache, dout))
        if self.cnt % self.replace == 0:
            self.qt.set(self.q.get())


def train_greenabr(cfg, env, rng, log):
    ag = GreenABRAgent(rng)
    for ep in range(cfg.episodes):
        env.reset()
        last, last_vmaf, first, done = 1, 0.0, True, False
        a = 1                                            # GreenABR starts at quality 1
        _s, _r, done, info = env.step(a)
        s = greenabr_state(env, a, info["record"], info)
        last_vmaf, first, tot = info["quality"], False, 0.0
        while not done:
            a = ag.act(s)
            _s, _r, done, info = env.step(a)
            r = greenabr_reward(info, last_vmaf, False)
            s2 = greenabr_state(env, a, info["record"], info)
            ag.remember(s, a, r, s2, done)
            ag.learn()
            s, last_vmaf, tot = s2, info["quality"], tot + r
        ag.eps = max(0.01, ag.eps * 0.9995)
        log(ep, tot, ag.eps)
    return ag


# ============================================================================ Pensieve / Genet
class A2C:
    def __init__(self, rng, actor_lr=1e-4, critic_lr=1e-3, gamma=0.99):
        self.rng, self.gamma = rng, gamma
        self.actor, self.critic = PensieveNet(A, rng), PensieveNet(1, rng)
        self.oa, self.oc = Adam(self.actor.p, lr=actor_lr), Adam(self.critic.p, lr=critic_lr)

    def probs(self, s):
        z = self.actor(s)
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def act(self, s, explore=False):
        p = self.probs(s[None])[0]
        return int(self.rng.choice(A, p=p)) if explore else int(np.argmax(p))

    def update(self, S, Aa, R, beta):
        n = len(Aa)
        v, cc = self.critic.forward(S, cache=True)
        adv = (R - v[:, 0]).astype(np.float32)
        self.oc.step(self.critic.p, self.critic.backward(cc, (2.0 * (v[:, 0] - R) / n)[:, None].astype(np.float32)))
        z, ca = self.actor.forward(S, cache=True)
        z = z - z.max(1, keepdims=True)
        p = np.exp(z); p /= p.sum(1, keepdims=True)
        logp = np.log(np.clip(p, 1e-8, 1.0))
        onehot = np.eye(A, dtype=np.float32)[Aa]
        d_pg = -(onehot - p) * adv[:, None]
        d_ent = p * (logp - (p * logp).sum(1, keepdims=True))          # d(sum p log p)/dz
        dz = ((d_pg + beta * d_ent) / n).astype(np.float32)
        self.oa.step(self.actor.p, self.actor.backward(ca, dz))


def bba_action(buffer_s):
    if buffer_s < 5.0:
        return 0
    if buffer_s >= 15.0:
        return A - 1
    return int((buffer_s - 5.0) * (A - 1) / 10.0)


def synth_traces(rng, cfgp, n=10, duration=400.0):
    max_bw, min_ratio, interval = cfgp
    out_t, out_b = [], []
    for _ in range(n):
        t, b, now = [], [], 0.0
        level = rng.uniform(max_bw * min_ratio, max_bw)
        nxt = rng.exponential(interval)
        while now < duration:
            if now >= nxt:
                level = rng.uniform(max_bw * min_ratio, max_bw); nxt = now + rng.exponential(interval)
            t.append(now); b.append(max(level * rng.uniform(0.9, 1.1), 0.05)); now += 1.0
        out_t.append(t); out_b.append(b)
    return out_t, out_b


def sample_config(rng):
    return (float(rng.uniform(0.5, 10.0)), float(rng.uniform(0.1, 1.0)), float(rng.uniform(1.0, 20.0)))


def rollout_reward(env, policy, trace_idx):
    """Mean Pensieve reward of one greedy session (policy: 'bba' or an A2C agent)."""
    env.reset(trace_idx=trace_idx)
    st = PensieveState()
    a, last, total, done, n = 1, 1, 0.0, False, 0
    while not done:
        _s, _r, done, info = env.step(a)
        rec = info["record"]
        total += pensieve_reward(a, last, info["rebuf"]); n += 1
        last = a
        s = st.update(env, a, rec)
        a = bba_action(rec.buffer) if policy == "bba" else policy.act(s, explore=False)
    return total / n


def train_pensieve(cfg, mk_env, videos, rng, log, curriculum=False, n_agents=16, seq=100):
    ag = A2C(rng)
    envs = [mk_env() for _ in range(n_agents)]
    synth_envs = []                                          # Genet: added configurations
    states = [PensieveState() for _ in range(n_agents)]
    cur = [None] * n_agents                                  # (state, action) pending per agent
    last = [1] * n_agents

    def new_episode(i):
        e = envs[i]
        if curriculum and synth_envs and rng.random() < 0.3:
            e = synth_envs[int(rng.integers(len(synth_envs)))]
            envs[i] = e
        elif curriculum:
            envs[i] = e = base_envs[i]
        e.reset(); states[i].reset(); last[i] = 1
        _s, _r, _d, info = e.step(1)                         # Pensieve: first chunk at default quality
        s = states[i].update(e, 1, info["record"])
        cur[i] = s

    base_envs = list(envs)
    for i in range(n_agents):
        new_episode(i)
    E = cfg.a2c_epochs
    rounds = 9
    round_every = max(1, E // (rounds + 1))
    for ep in range(E):
        beta = 0.5 - 0.45 * min(1.0, ep / (0.75 * E))
        S, Aa, Rw, D = [], [], [], []
        for t in range(seq):
            batch = np.stack(cur)
            p = ag.probs(batch)
            acts = [int(rng.choice(A, p=pi)) for pi in p]
            S.append(batch); Aa.append(acts); r_t, d_t = [], []
            for i, a in enumerate(acts):
                _s, _r, done, info = envs[i].step(a)
                r_t.append(pensieve_reward(a, last[i], info["rebuf"])); last[i] = a
                d_t.append(done)
                if done:
                    new_episode(i)
                else:
                    cur[i] = states[i].update(envs[i], a, info["record"])
            Rw.append(r_t); D.append(d_t)
        Rw, D = np.array(Rw, np.float32), np.array(D)
        boot = ag.critic(np.stack(cur))[:, 0]
        R = np.zeros_like(Rw); run = boot.copy()
        for t in reversed(range(seq)):
            run = Rw[t] + ag.gamma * run * (~D[t])
            R[t] = run
        ag.update(np.concatenate(S), np.array(Aa).reshape(-1), R.reshape(-1), beta)
        if curriculum and ep > 0 and ep % round_every == 0 and len(synth_envs) < rounds:
            best, best_gap = None, -1e9
            for _ in range(15):
                c = sample_config(rng)
                tt, bb = synth_traces(rng, c, n=5)
                env_c = ABREnv(cfg, [videos[int(rng.integers(len(videos)))]], tt, bb, list(range(5)),
                               cfg.train_device, rng, train=False, calib=None, energy_tables=base_envs[0].energy_tables)
                gap = np.mean([rollout_reward(env_c, "bba", k) - rollout_reward(env_c, ag, k) for k in range(5)])
                if gap > best_gap:
                    best, best_gap = c, gap
            tt, bb = synth_traces(rng, best, n=20)
            synth_envs.append(ABREnv(cfg, videos, tt, bb, list(range(20)), cfg.train_device, rng,
                                     train=True, calib=None, energy_tables=base_envs[0].energy_tables))
            log(ep, float(Rw.mean()), beta, extra=f"curriculum: added {best} gap={best_gap:.3f}")
        if ep % 100 == 0:
            log(ep, float(Rw.mean()), beta)
    return ag


# ============================================================================ run
def run_baseline(cfg: RunConfig, repo: Path, out: Path, log_every: int = 500) -> Path:
    paths = Paths(repo)
    run_dir = out / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.to_json())
    rng = np.random.default_rng(cfg.seed)
    videos = [Video(paths, v) for v in cfg.train_videos]
    times, bws, _ = load_traces(paths.train_traces)
    power, info = fit_for_fold(videos, cfg.train_device, cfg.seed)
    test_video = Video(paths, cfg.test_video)
    tables = {v.name: power.table(v) for v in videos + [test_video]}
    (run_dir / "power_model.json").write_text(json.dumps(info, indent=2))

    def mk_env():
        return ABREnv(cfg, videos, times, bws, list(range(len(times))), cfg.train_device, rng,
                      train=True, calib=None, energy_tables=tables)

    t0 = time.time()
    fh = (run_dir / "train_log.csv").open("w", newline="")
    w = csv.writer(fh); w.writerow(["episode", "score", "param", "note"])

    def log(ep, score, param, extra=""):
        w.writerow([ep, round(score, 4), round(param, 5), extra])
        if extra or ep % log_every == 0:
            fh.flush()
            print(f"[{cfg.run_id}] ep={ep} score={score:.3f} p={param:.3f} {extra} {time.time() - t0:.0f}s", flush=True)

    if cfg.variant == "greenabr":
        ag = train_greenabr(cfg, mk_env(), rng, lambda e, s, p: log(e, s, p))
        state_fn = "greenabr"
    else:
        ag = train_pensieve(cfg, mk_env, videos, rng, log, curriculum=(cfg.variant == "genet"))
        state_fn = "pensieve"
    fh.close()
    secs = time.time() - t0
    np.savez(run_dir / "model_final.npz", **(ag.q.p if state_fn == "greenabr" else
                                              {f"actor_{k}": v for k, v in ag.actor.p.items()}))

    # ---------------------------------------------------------------- evaluation (same protocol)
    ttimes, tbws, names = load_traces(paths.test_traces)
    devices = {d: DeviceEnergy(paths.power_files, cfg.test_video, d) for d in DEVICES}
    env = ABREnv(cfg, [test_video], ttimes, tbws, list(range(len(ttimes))), cfg.train_device,
                 np.random.default_rng(0), train=False, calib=None, energy_tables=tables)
    rows = []
    for t, name in enumerate(names):
        env.reset(trace_idx=t)
        st = PensieveState()
        a, done, records = 0, False, []
        while not done:
            _s, _r, done, info = env.step(a)
            rec = info["record"]; records.append(rec)
            if done:
                break
            if state_fn == "greenabr":
                a = ag.act(greenabr_state(env, a, rec, info), explore=False)
            else:
                a = ag.act(st.update(env, a, rec), explore=False)
        m = session_metrics(records, devices)
        rows.append({"run_id": cfg.run_id, "variant": cfg.variant, "test_video": cfg.test_video,
                     "train_device": cfg.train_device, "seed": cfg.seed, "tag": cfg.tag,
                     "budget_frac": "none", "budget": 0.0, "lambda": 0.0, "trace": name,
                     "e_norm_mean": 0.0, "cost_mean": 0.0, "violation": "", "selected": True, **m})
        write_log(run_dir / "eval_logs" / "none" / f"log_{name}", records)
    with (run_dir / "results.csv").open("w", newline="") as f2:
        wr = csv.DictWriter(f2, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    summary = {k: float(np.mean([r[k] for r in rows])) for k in ("QoE", "Energy_G", "Energy_X",
                                                                 "Efficiency_Avg", "Rebuffer_Time")}
    (run_dir / "done.json").write_text(json.dumps({"train_seconds": secs, **summary}, indent=2))
    print(f"[{cfg.run_id}] DONE in {secs / 60:.1f} min  " +
          "  ".join(f"{k}={v:.3f}" for k, v in summary.items()), flush=True)
    return run_dir
