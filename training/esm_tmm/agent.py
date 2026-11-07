"""DQN agent of ESM-RB (dueling + double DQN, prioritized replay, 3-step returns)."""

from __future__ import annotations

import numpy as np

from .config import RunConfig
from .nn import Adam, DuelingQNet


class PrioritizedReplay:
    def __init__(self, size: int, dim: int, alpha: float, rng: np.random.Generator):
        self.size, self.alpha, self.rng = size, alpha, rng
        self.counter = 0
        self.s = np.zeros((size, dim), np.float32)
        self.s2 = np.zeros((size, dim), np.float32)
        self.a = np.zeros(size, np.int64)
        self.r = np.zeros(size, np.float32)
        self.nt = np.zeros(size, np.float32)
        self.prio = np.zeros(size, np.float64)
        self.max_prio = 1.0

    def store(self, s, a, r, s2, done):
        i = self.counter % self.size
        self.s[i], self.a[i], self.r[i], self.s2[i] = s, a, r, s2
        self.nt[i] = 1.0 - float(done)
        self.prio[i] = self.max_prio
        self.counter += 1

    def sample(self, batch: int, beta: float):
        n = min(self.counter, self.size)
        p = self.prio[:n] ** self.alpha
        p /= p.sum()
        idx = self.rng.choice(n, batch, p=p)
        w = (n * p[idx]) ** (-beta)
        w /= w.max()
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.nt[idx], idx, w.astype(np.float32)

    def update(self, idx, prios):
        prios = np.maximum(np.asarray(prios, np.float64), 1e-6)
        self.prio[idx] = prios
        self.max_prio = max(self.max_prio, float(prios.max()))


class NStep:
    def __init__(self, n: int, gamma: float):
        self.n, self.gamma, self.items = n, gamma, []

    def _make(self, length):
        total = sum((self.gamma ** i) * self.items[i][2] for i in range(length))
        return self.items[0][0], self.items[0][1], total, self.items[length - 1][3], self.items[length - 1][4]

    def add(self, s, a, r, s2, done):
        self.items.append((s, a, r, s2, done))
        if len(self.items) < self.n:
            return None
        out = self._make(len(self.items))
        self.items.pop(0)
        return out

    def flush(self):
        out = []
        while self.items:
            out.append(self._make(len(self.items)))
            self.items.pop(0)
        return out


class Agent:
    def __init__(self, cfg: RunConfig, state_dim: int, action_dim: int, rng: np.random.Generator):
        self.cfg, self.rng, self.action_dim = cfg, rng, action_dim
        self.q = DuelingQNet(state_dim, action_dim, rng)
        self.q_target = DuelingQNet(state_dim, action_dim, rng)
        self.q_target.set_weights(self.q.get_weights())
        self.opt = Adam(self.q.p, lr=cfg.learning_rate)
        self.replay = PrioritizedReplay(cfg.memory_size, state_dim, cfg.priority_alpha, rng)
        self.nstep = NStep(cfg.n_step, cfg.gamma)
        self.epsilon = cfg.epsilon_start
        self.beta = cfg.priority_beta_start
        self.beta_inc = (1.0 - self.beta) / cfg.priority_beta_frames
        self.learn_steps = 0

    def act(self, state, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_dim))
        return int(np.argmax(self.q(state[None, :])[0]))

    def remember(self, s, a, r, s2, done):
        t = self.nstep.add(s, a, r, s2, done)
        if t is not None:
            self.replay.store(*t)

    def end_episode(self):
        for t in self.nstep.flush():
            self.replay.store(*t)

    def learn(self):
        cfg = self.cfg
        if self.replay.counter < cfg.batch_size:
            return None
        self.beta = min(1.0, self.beta + self.beta_inc)
        s, a, r, s2, nt, idx, w = self.replay.sample(cfg.batch_size, self.beta)
        q_next = self.q_target(s2)
        q_eval_next = self.q(s2)
        q_pred = self.q(s)
        b = np.arange(cfg.batch_size)
        best = np.argmax(q_eval_next, axis=1)
        targets = q_pred.copy()
        targets[b, a] = r + (cfg.gamma ** cfg.n_step) * q_next[b, best] * nt
        td = np.abs(targets[b, a] - q_pred[b, a])
        loss, grads = self.q.loss_and_grads(s, targets, w)
        self.opt.step(self.q.p, grads)
        self.replay.update(idx, td + 1e-6)
        self.learn_steps += 1
        if self.learn_steps % cfg.target_update_steps == 0:
            self.q_target.set_weights(self.q.get_weights())
        return loss

    def decay_epsilon(self):
        self.epsilon = max(self.cfg.epsilon_end, self.epsilon * self.cfg.epsilon_decay)
