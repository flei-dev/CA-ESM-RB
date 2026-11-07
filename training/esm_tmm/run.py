"""Train one configuration and evaluate it on the held-out video.

python -m esm_tmm.run --config '<json>' --out <runs_dir>
Writes <runs_dir>/<run_id>/{config.json, calibration.json, train_log.csv,
model_final.npz, lambdas.json, results.csv, eval_logs/..., done.json}.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .agent import Agent
from .config import RunConfig
from .data import BITRATES, DEVICES, Paths, Video, load_traces
from .env import ABREnv, Calibration, calibrate, state_dim
from .metrics import DeviceEnergy, session_metrics, write_log
from .powermodel import fit_for_fold, generalization_report


def train(cfg: RunConfig, paths: Paths, run_dir: Path, log_every: int = 500):
    rng = np.random.default_rng(cfg.seed)
    videos = [Video(paths, v) for v in cfg.train_videos]
    times, bws, _names = load_traces(paths.train_traces)
    ids = list(range(len(times)))
    power = None
    tables = None
    if cfg.energy_source == "model":
        power, info = fit_for_fold(videos, cfg.train_device, cfg.seed)
        all_videos = videos + [Video(paths, cfg.test_video)]
        info["r2_and_within_chunk_corr"] = generalization_report(power, all_videos)
        (run_dir / "power_model.json").write_text(json.dumps(info, indent=2))
        tables = {v.name: power.table(v) for v in all_videos}
    calib = calibrate(cfg, videos, times, bws, ids, cfg.train_device, cfg.seed, tables)
    (run_dir / "calibration.json").write_text(json.dumps(asdict(calib), indent=2))

    env = ABREnv(cfg, videos, times, bws, ids, cfg.train_device, rng, train=True, calib=calib,
                 energy_tables=tables)
    agent = Agent(cfg, state_dim(cfg), len(BITRATES), rng)
    cons = cfg.flags["constraint"]
    budgets = [calib.budget(f) for f in cfg.budget_fracs]
    lambdas = [0.0 for _ in budgets]          # set after training by calibrate_lambdas
    log_lo, log_hi = np.log(cfg.lam_min), np.log(cfg.lambda_max)

    t0 = time.time()
    with (run_dir / "train_log.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "score", "avg_score", "quality", "rebuffer", "energy", "e_norm",
                    "budget_idx", "budget", "lambda", "epsilon", "loss", "cost"])
        scores = []
        for ep in range(cfg.episodes):
            k = -1
            lam = 0.0
            if cons and rng.random() >= cfg.lam_zero_prob:
                lam = float(np.exp(rng.uniform(log_lo, log_hi)))
            state = env.reset(budget=None, lam=lam)
            agent.nstep.items.clear()
            done, steps = False, 0
            score = qual = reb = energy = 0.0
            e_norms, costs, losses = [], [], []
            while not done:
                a = agent.act(state, explore=True)
                nxt, r, done, info = env.step(a)
                agent.remember(state, a, r, nxt, done)
                state = nxt
                score += r; qual += info["quality"]; reb += info["rebuf"]; energy += info["energy"]
                e_norms.append(info["e_norm"])
                costs.append(info["cost"])
                steps += 1
                if ep >= cfg.warmup_episodes and steps % cfg.learning_frequency == 0:
                    loss = agent.learn()
                    if loss is not None:
                        losses.append(loss)
            agent.end_episode()
            ep_e = float(np.mean(e_norms))
            ep_c = float(np.mean(costs))
            if ep >= cfg.warmup_episodes:
                agent.decay_epsilon()
            scores.append(score)
            avg = float(np.mean(scores[-100:]))
            w.writerow([ep, round(score, 4), round(avg, 4), round(qual / steps, 3), round(reb, 4),
                        round(energy, 2), round(ep_e, 4), k, "",
                        round(lam, 4) if cons else "", round(agent.epsilon, 5),
                        round(float(np.mean(losses)), 6) if losses else "", round(ep_c, 4)])
            if ep % log_every == 0:
                fh.flush()
                lam_txt = "sampled" if cons else "-"
                print(f"[{cfg.run_id}] ep={ep} avg={avg:.3f} eps={agent.epsilon:.3f} "
                      f"lambda=[{lam_txt}] {time.time() - t0:.0f}s", flush=True)

    agent.q.save(run_dir / "model_final.npz")
    (run_dir / "lambdas.json").write_text(json.dumps({"budget_fracs": list(cfg.budget_fracs),
                                                      "budgets": budgets, "lambdas": lambdas}, indent=2))
    return agent, calib, budgets, lambdas, time.time() - t0, tables


def rollout(env: ABREnv, agent: Agent, trace_idx: int, budget: float, lam: float):
    """One deterministic session (rep_6 protocol: first chunk at the lowest level)."""
    state = env.reset(trace_idx=trace_idx, budget=budget, lam=lam)
    done, first = False, True
    records, e_norms, costs = [], [], []
    while not done:
        a = 0 if first else agent.act(state, explore=False)
        first = False
        state, _r, done, info = env.step(a)
        records.append(info["record"])
        e_norms.append(info["e_norm"])
        costs.append(info["cost"])
    return records, float(np.mean(e_norms)), float(np.mean(costs))


def _val_envs(cfg: RunConfig, paths: Paths, calib: Calibration, tables: dict | None):
    """Validation sessions: training videos x a fixed subset of the training traces."""
    times, bws, _ = load_traces(paths.train_traces)
    rng = np.random.default_rng(cfg.seed + 999)
    ids = sorted(rng.choice(len(times), size=min(cfg.val_traces, len(times)), replace=False).tolist())
    envs = []
    for vname in cfg.train_videos:
        env = ABREnv(cfg, [Video(paths, vname)], times, bws, ids, cfg.train_device,
                     np.random.default_rng(0), train=False, calib=calib, energy_tables=tables)
        envs.append((vname, env))
    return envs, ids


def calibrate_lambdas(cfg: RunConfig, paths: Paths, agent: Agent, calib: Calibration, budgets, lambdas,
                      tables: dict | None, iters: int = 8) -> list[float]:
    """Offline part of the budget controller: for each budget, the smallest lambda in
    [0, lambda_max] whose mean budget cost on validation sessions of the TRAINING videos does
    not exceed the budget (bisection; the cost decreases with lambda).  The session feedback is
    active during calibration, exactly as at deployment."""
    envs, ids = _val_envs(cfg, paths, calib, tables)

    def mean_cost(budget, lam):
        return float(np.mean([rollout(env, agent, t, budget, lam)[2] for _, env in envs for t in ids]))

    out = []
    for budget in budgets:
        if mean_cost(budget, 0.0) <= budget:
            out.append(0.0); continue
        lo, hi = 0.0, cfg.lambda_max
        if mean_cost(budget, hi) > budget:
            out.append(hi); continue
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if mean_cost(budget, mid) > budget:
                lo = mid
            else:
                hi = mid
        out.append(hi)
    return out


def select_budget(cfg: RunConfig, paths: Paths, agent: Agent, calib: Calibration, budgets, lambdas,
                  tables: dict | None) -> dict:
    """Pick the operating budget on the TRAINING videos (validation traces drawn from the
    training trace set, measured energy of the training device): the budget with the highest
    QoE/energy.  The held-out video is never used."""
    envs, ids = _val_envs(cfg, paths, calib, tables)
    scores = []
    for frac, budget, lam in zip(cfg.budget_fracs, budgets, lambdas):
        eff, q, e = [], [], []
        for vname, env in envs:
            dev = {cfg.train_device: DeviceEnergy(paths.power_files, vname, cfg.train_device)}
            for t in ids:
                recs, _, _ = rollout(env, agent, t, budget, lam)
                m = session_metrics(recs, dev)
                key = {"galaxy": "G", "xcover": "X"}[cfg.train_device]
                eff.append(m[f"Efficiency_{key}"]); q.append(m["QoE"]); e.append(m[f"Energy_{key}"])
        scores.append({"budget_frac": frac, "efficiency": float(np.mean(eff)), "QoE": float(np.mean(q)),
                       "energy": float(np.mean(e))})
    best = max(scores, key=lambda r: r["efficiency"])
    return {"selected_budget_frac": best["budget_frac"], "validation": scores}


def evaluate(cfg: RunConfig, paths: Paths, run_dir: Path, agent: Agent, calib: Calibration,
             budgets: list[float], lambdas: list[float], tables: dict | None = None,
             write_logs: bool = True) -> list[dict]:
    video = Video(paths, cfg.test_video)
    times, bws, names = load_traces(paths.test_traces)
    devices = {d: DeviceEnergy(paths.power_files, cfg.test_video, d) for d in DEVICES}
    rng = np.random.default_rng(0)
    env = ABREnv(cfg, [video], times, bws, list(range(len(times))), cfg.train_device, rng,
                 train=False, calib=calib, energy_tables=tables)
    settings = (list(zip(cfg.budget_fracs, budgets, lambdas)) if cfg.flags["constraint"]
                else [("none", 0.0, 0.0)])
    selected = "none"
    if cfg.flags["constraint"]:
        lambdas = calibrate_lambdas(cfg, paths, agent, calib, budgets, lambdas, tables)
        (run_dir / "lambdas.json").write_text(json.dumps({"budget_fracs": list(cfg.budget_fracs),
                                                          "budgets": budgets, "lambdas": lambdas}, indent=2))
        settings = list(zip(cfg.budget_fracs, budgets, lambdas))
    if cfg.flags["constraint"]:
        if cfg.select_budget:
            sel = select_budget(cfg, paths, agent, calib, budgets, lambdas, tables)
            (run_dir / "budget_selection.json").write_text(json.dumps(sel, indent=2))
            selected = sel["selected_budget_frac"]
        else:
            selected = cfg.default_budget_frac
    rows = []
    for frac, budget, lam in settings:
        for t, name in enumerate(names):
            records, mean_e, mean_c = rollout(env, agent, t, budget, lam)
            m = session_metrics(records, devices)
            rows.append({"run_id": cfg.run_id, "variant": cfg.variant, "test_video": cfg.test_video,
                         "train_device": cfg.train_device, "seed": cfg.seed, "tag": cfg.tag,
                         "budget_frac": frac, "budget": budget, "lambda": lam, "trace": name,
                         "e_norm_mean": mean_e, "cost_mean": mean_c,
                         "violation": (mean_c > budget) if frac != "none" else "",
                         "selected": frac == selected, **m})
            if write_logs:
                write_log(run_dir / "eval_logs" / str(frac) / f"log_{name}", records)
    with (run_dir / "results.csv").open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return rows


def run(cfg: RunConfig, repo: Path, out: Path, log_every: int = 500) -> Path:
    from .baselines import BASELINE_VARIANTS, run_baseline
    if cfg.variant in BASELINE_VARIANTS:
        return run_baseline(cfg, repo, out, log_every)
    paths = Paths(repo)
    run_dir = out / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(cfg.to_json())
    agent, calib, budgets, lambdas, secs, tables = train(cfg, paths, run_dir, log_every)
    rows = evaluate(cfg, paths, run_dir, agent, calib, budgets, lambdas, tables)
    main = [r for r in rows if r["selected"]]
    summary = {k: float(np.mean([r[k] for r in main])) for k in ("QoE", "Energy_G", "Energy_X", "Efficiency_Avg",
                                                                 "Rebuffer_Time")}
    (run_dir / "done.json").write_text(json.dumps({"train_seconds": secs, **summary}, indent=2))
    print(f"[{cfg.run_id}] DONE in {secs / 60:.1f} min  " +
          "  ".join(f"{k}={v:.3f}" for k, v in summary.items()), flush=True)
    return run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="RunConfig as JSON")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--out", required=True)
    ap.add_argument("--log-every", type=int, default=500)
    a = ap.parse_args()
    run(RunConfig.from_dict(json.loads(a.config)), Path(a.repo), Path(a.out), a.log_every)


if __name__ == "__main__":
    main()
