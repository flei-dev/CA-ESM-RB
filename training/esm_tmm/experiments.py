"""Experiment matrix and a resumable two-machine scheduler.

Usage (from ~/ESM/training):
  python -m esm_tmm.experiments list
  python -m esm_tmm.experiments run --worker 0 --workers 2 --jobs 16     # machine A
  python -m esm_tmm.experiments run --worker 1 --workers 2 --jobs 16     # machine B
  python -m esm_tmm.experiments status
Runs already finished (done.json present) are skipped, so the command can simply be
restarted after an interruption.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .config import RunConfig

REPO = Path(__file__).resolve().parents[2]
# Round 2: the context-aware variant ("ctx") is the proposed method.  The v1 runs of ESM-RB and
# ctx (main comparison, ESM-RB cross-device / protocol) are reused from results/esm_tmm/runs;
# this round adds the experiments ctx still needs.
DEFAULT_OUT = REPO / "results" / "esm_tmm" / "runs_v2"
FOLDS = ("tos", "bbb", "doc")
SEEDS5 = (0, 1, 2, 3, 4)
SEEDS3 = (0, 1, 2)


def matrix() -> list[tuple[str, RunConfig]]:
    """(group, config) in priority order.  96 runs."""
    runs = []
    for seed in SEEDS5:                                   # B. ablation (30 runs)
        for fold in FOLDS:
            for v in ("ctx_nolook", "look_only"):
                runs.append(("B_ablation", RunConfig(variant=v, test_video=fold, seed=seed)))
    for seed in SEEDS5:                                   # D. cross-device (15 runs)
        for fold in FOLDS:
            runs.append(("D_xdevice", RunConfig(variant="ctx", test_video=fold, seed=seed,
                                                train_device="xcover")))
    for seed in SEEDS5:                                   # E. training protocol (15 runs)
        for fold in FOLDS:
            runs.append(("E_protocol", RunConfig(variant="ctx", test_video=fold, seed=seed,
                                                 random_offset=False, tag="nooffset")))
    for seed in SEEDS3:                                   # C. energy-weight sweep (36 runs)
        for fold in FOLDS:
            for v in ("ctx", "esmrb"):
                for w in (0.25, 1.0):
                    runs.append(("C_energy_weight", RunConfig(variant=v, test_video=fold, seed=seed,
                                                              energy_weight=w, tag=f"ew{w:g}")))
    for seed in SEEDS5:                                   # H. retrained learned baselines (45 runs)
        for fold in FOLDS:
            for v in ("greenabr", "pensieve", "genet"):
                runs.append(("H_baselines", RunConfig(variant=v, test_video=fold, seed=seed)))
    # G. seeds 3-4 for the energy weight chosen on the training videos (select_weight.py)
    sel_file = REPO / "results" / "esm_tmm" / "weight_selection.json"
    if sel_file.exists():
        import json
        for v, w in json.loads(sel_file.read_text())["selected"].items():
            if abs(w - 0.5) < 1e-9:
                continue                                  # default weight already has 5 seeds
            for seed in (3, 4):
                for fold in FOLDS:
                    runs.append(("G_selected", RunConfig(variant=v, test_video=fold, seed=seed,
                                                         energy_weight=w, tag=f"ew{w:g}")))
    return runs


def is_done(out: Path, cfg: RunConfig) -> bool:
    return (out / cfg.run_id / "done.json").exists()


def _launch(cfg: RunConfig, out: Path) -> tuple[str, int, float]:
    run_dir = out / cfg.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    t0 = time.time()
    with (run_dir / "stdout.log").open("w") as log:
        rc = subprocess.call([sys.executable, "-m", "esm_tmm.run", "--config", cfg.to_json(),
                              "--repo", str(REPO), "--out", str(out)],
                             cwd=str(REPO / "training"), stdout=log, stderr=subprocess.STDOUT, env=env)
    return cfg.run_id, rc, time.time() - t0


def cmd_run(a):
    out = Path(a.out)
    todo = [(g, c) for i, (g, c) in enumerate(matrix()) if i % a.workers == a.worker]
    if a.group:
        todo = [(g, c) for g, c in todo if g.startswith(a.group)]
    pending = [c for g, c in todo if not is_done(out, c)]
    print(f"worker {a.worker}/{a.workers}: {len(todo)} runs assigned, {len(pending)} pending, "
          f"{a.jobs} in parallel", flush=True)
    finished = failed = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(_launch, c, out) for c in pending]
        for f in as_completed(futures):
            rid, rc, secs = f.result()
            if rc == 0:
                finished += 1
            else:
                failed += 1
            el = (time.time() - t0) / 3600
            print(f"[{time.strftime('%H:%M:%S')}] {'OK  ' if rc == 0 else 'FAIL'} {rid} "
                  f"({secs / 60:.1f} min)  progress {finished + failed}/{len(pending)}  "
                  f"failed={failed}  elapsed={el:.2f} h", flush=True)
    print("all assigned runs processed" + (f"; {failed} FAILED - see stdout.log in their folders"
                                            if failed else ""), flush=True)


def cmd_status(a):
    out = Path(a.out)
    groups: dict[str, list[int]] = {}
    for g, c in matrix():
        d = groups.setdefault(g, [0, 0])
        d[1] += 1
        d[0] += is_done(out, c)
    for g, (done, total) in groups.items():
        print(f"{g:15s} {done:4d}/{total}")
    print(f"{'TOTAL':15s} {sum(v[0] for v in groups.values()):4d}/{sum(v[1] for v in groups.values())}")


def cmd_list(a):
    for i, (g, c) in enumerate(matrix()):
        print(f"{i:4d} worker{i % a.workers} {g:14s} {c.run_id}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "status", "list"):
        p = sub.add_parser(name)
        p.add_argument("--out", default=str(DEFAULT_OUT))
        p.add_argument("--workers", type=int, default=2)
        if name == "run":
            p.add_argument("--worker", type=int, required=True)
            p.add_argument("--jobs", type=int, default=16)
            p.add_argument("--group", default="", help="only runs whose group starts with this, e.g. A")
    a = ap.parse_args()
    {"run": cmd_run, "status": cmd_status, "list": cmd_list}[a.cmd](a)


if __name__ == "__main__":
    main()
