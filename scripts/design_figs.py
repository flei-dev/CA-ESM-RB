"""Reproduce Fig. 1 (motivation), Fig. 4 (design insights) and Fig. 6 (energy penalty) and dump
all plotted data.

Run from the ESM repository's training directory (needs esm_tmm and the repo data):
    cd ESM/training && python /path/to/design_figs.py --repo .. --out design_fig_data

Outputs (in --out):
  fig_motivation.png, fig_design_insight.png, fig_energy_penalty.png
  fig1_chunk_level_data.csv      per video, chunk 0-44, level: VMAF, size, measured Galaxy playback
                                 energy, network energy at 3 Mbps, total
  fig4a_example_chunks.csv       the three example chunks (all six levels)
  fig4b_throughput_windows.csv   the two training-trace windows (trace name, start index/time, samples)
  fold_power_table.csv           fold (train bbb+doc, Galaxy, seed 0): predicted playback energy per
                                 video/chunk/level, plus network energy and estimate e_hat
  fold_calibration.json          p5/p75/p95 (full precision), e_lo, e_hi and the settings used
  fig4c_upgrade_test.csv         per chunk: actual gain, required gain (w=1), pass flag
  fig6_histogram_bins.csv        histogram bin edges and counts per level (25 bins per level)
  settings.json                  all constants used
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import os
import sys

import matplotlib
import numpy as np

sys.path.insert(0, os.getcwd())   # run from ESM/training
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from esm_tmm.config import RunConfig  # noqa: E402
from esm_tmm.data import BITRATES, Paths, Video, load_traces  # noqa: E402
from esm_tmm.env import calibrate  # noqa: E402
from esm_tmm.metrics import DeviceEnergy  # noqa: E402
from esm_tmm.powermodel import fit_for_fold  # noqa: E402

THR = 3.0                       # Mbps used for illustrative network energy
NET = lambda mb: mb * (210.0 / THR + 28.0)  # noqa: E731
FOLD_TRAIN, FOLD_TEST, SEED, W = ["bbb", "doc"], "tos", 0, 1.0
EXAMPLES = {"tos": 12, "bbb": 10, "doc": 17}
COL = {"tos": "#1f5fbf", "bbb": "#d9822b", "doc": "#3a8f5c"}
NAME = {"tos": "Tears of Steel", "bbb": "Big Buck Bunny", "doc": "Documentary"}
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="..")
    ap.add_argument("--out", default="design_fig_data")
    a = ap.parse_args()
    P, out = Paths(Path(a.repo).resolve()), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    V = {v: Video(P, v) for v in ("tos", "bbb", "doc")}

    # ---------------- Fig. 1 data + figure
    rows = []
    for v in V:
        dev = DeviceEnergy(P.power_files, v, "galaxy")
        for c in range(45):
            for i, b in enumerate(BITRATES):
                loc = dev.local(b, c)
                net = NET(V[v].sizes[c, i] / 1e6)
                rows.append({"video": v, "chunk": c, "bitrate_kbps": b, "vmaf": V[v].vmaf[c, i],
                             "size_bytes": V[v].sizes[c, i], "galaxy_playback_energy": loc,
                             "network_energy_3mbps": net, "total_energy": loc + net})
    f1 = pd.DataFrame(rows)
    f1.to_csv(out / "fig1_chunk_level_data.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for v in V:
        d = f1[f1.video == v]
        m = d.groupby("bitrate_kbps")[["total_energy", "vmaf"]].mean()
        ax[0].plot(m.total_energy, m.vmaf, "-o", ms=3, color=COL[v], label=NAME[v])
        p = d.pivot(index="chunk", columns="bitrate_kbps")
        ax[1].scatter(p["total_energy"][4300] - p["total_energy"][2850], p["vmaf"][4300] - p["vmaf"][2850],
                      s=9, alpha=.75, color=COL[v], edgecolors="none", label=NAME[v])
    ax[0].set_xlabel("Mean energy per chunk"); ax[0].set_ylabel("Mean VMAF")
    ax[0].set_title("(a) Perceptual quality vs. energy, six levels", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6.5, loc="lower right")
    ax[1].set_xlabel("Extra energy, 2850→4300 kbps"); ax[1].set_ylabel("VMAF gain")
    ax[1].set_title("(b) Per-chunk gain of the top level", fontsize=8)
    fig.savefig(out / "fig_motivation.png", dpi=300); plt.close(fig)

    # ---------------- fold power model + calibration (shared by Fig. 4c and Fig. 6)
    cfg = RunConfig(variant="ctx", test_video=FOLD_TEST, seed=SEED)
    vids = [V[v] for v in FOLD_TRAIN]
    power, _ = fit_for_fold(vids, "galaxy", SEED)
    tables = {v.name: power.table(v) for v in vids}
    t, b, names = load_traces(P.train_traces)
    cal = calibrate(cfg, vids, t, b, list(range(len(names))), "galaxy", SEED, energy_tables=tables)
    p5, p75, p95 = cal.p5, cal.p75, cal.p95
    D = p95 - p5
    json.dump({**cal.__dict__, "delta": D, "fold_train_videos": FOLD_TRAIN, "fold_test_video": FOLD_TEST,
               "device": "galaxy", "seed": SEED, "calib_episodes": cfg.calib_episodes,
               "note": "recomputed with the experiment code for seed 0; the stored calibration.json of "
                       "the corresponding run on the experiment server is the authoritative record"},
              open(out / "fold_calibration.json", "w"), indent=2)
    rows = []
    for v in vids:
        for c in range(45):
            for i, br in enumerate(BITRATES):
                pred = tables[v.name][c, i]
                net = NET(v.sizes[c, i] / 1e6)
                rows.append({"video": v.name, "chunk": c, "bitrate_kbps": br, "predicted_playback": pred,
                             "network_energy_3mbps": net, "e_hat": pred + net})
    ft = pd.DataFrame(rows)
    ft.to_csv(out / "fold_power_table.csv", index=False)

    def pen(e):
        eb = (e - p5) / D
        return eb + (2 * np.log1p((e - p75) / D) if e > p75 else 0.0)

    # ---------------- Fig. 4 data + figure
    pd.concat([f1[(f1.video == v) & (f1.chunk == c)] for v, c in EXAMPLES.items()]).to_csv(
        out / "fig4a_example_chunks.csv", index=False)

    def find(cond):
        for ti, (tt, bb) in enumerate(zip(t, b)):
            tt, bb = np.asarray(tt), np.asarray(bb)
            for s in range(0, len(tt) - 25):
                idx = [int(np.searchsorted(tt, tt[s] + 4 * k)) for k in range(5)]
                if idx[-1] >= len(bb):
                    break
                w = bb[idx]
                if cond(w):
                    return w, names[ti], s, tt[s], idx
    win = {"stable": find(lambda w: 2.8 <= w.mean() <= 3.2 and w.std() < 0.15),
           "variable": find(lambda w: 2.8 <= w.mean() <= 3.2 and w.std() > 2.3)}
    pd.DataFrame([{"window": k, "trace": n, "start_index": s, "start_time_s": st, "sample_indices": idx,
                   **{f"t-{5 - j}": w[j] for j in range(5)}, "mean": w.mean(), "std_population": w.std()}
                  for k, (w, n, s, st, idx) in win.items()]).to_csv(out / "fig4b_throughput_windows.csv", index=False)
    rows = []
    for v in vids:
        d = ft[ft.video == v.name].pivot(index="chunk", columns="bitrate_kbps")["e_hat"]
        for c in range(45):
            req = 100 * W / 1.5 * (pen(d[4300][c]) - pen(d[2850][c]))
            act = v.vmaf[c, 5] - v.vmaf[c, 4]
            rows.append({"video": v.name, "chunk": c, "actual_gain": act, "required_gain_w1": req,
                         "passes": act > req})
    up = pd.DataFrame(rows)
    up.to_csv(out / "fig4c_upgrade_test.csv", index=False)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5))
    fig.subplots_adjust(wspace=0.45)
    for v, c in EXAMPLES.items():
        ax[0].plot(V[v].sizes[c] / 1e6, V[v].vmaf[c], "-o", ms=3, color=COL[v], label=f"{NAME[v]} #{c}")
    ax[0].set_xlabel("Chunk size (MB)"); ax[0].set_ylabel("VMAF")
    ax[0].set_title("(a) Lookahead: quality vs. size", fontsize=8)
    ax[0].legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.02, 0.12))
    for k, colr in (("stable", "#1f5fbf"), ("variable", "#c0392b")):
        w = win[k][0]
        ax[1].plot(range(1, 6), w, "-o", ms=3, color=colr, label=f"$\\mu$={w.mean():.2f}, $\\sigma$={w.std():.2f}")
    ax[1].set_xticks(range(1, 6)); ax[1].set_xticklabels([f"t-{5 - i}" for i in range(5)])
    ax[1].set_xlabel("Previous chunk"); ax[1].set_ylabel("Throughput (Mbps)"); ax[1].set_ylim(0, 8.5)
    ax[1].set_title("(b) Same mean, different variability", fontsize=8)
    ax[1].legend(frameon=False, fontsize=6, loc="upper left")
    mx = up.required_gain_w1.max()
    for v in FOLD_TRAIN:
        d = up[up.video == v]
        ax[2].scatter(d.required_gain_w1, d.actual_gain, s=9, color=COL[v], alpha=.8, edgecolors="none", label=NAME[v])
    ax[2].plot([0, mx * 1.05], [0, mx * 1.05], "k--", lw=.8)
    ax[2].text(mx * 0.25, mx * 0.85, "upgrade\nworthwhile", fontsize=6.5, ha="center")
    ax[2].set_xlabel("Required VMAF gain ($w=1$)"); ax[2].set_ylabel("Actual VMAF gain")
    ax[2].set_title("(c) Upgrade test, 2850→4300 kbps", fontsize=8)
    ax[2].legend(frameon=False, fontsize=6, loc="lower right", bbox_to_anchor=(1.02, 0.3))
    fig.savefig(out / "fig_design_insight.png", dpi=300); plt.close(fig)

    # ---------------- Fig. 6 data + figure
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    cols = plt.cm.viridis(np.linspace(0.1, 0.9, 6))
    hist_rows = []
    for i, br in enumerate(BITRATES):
        e = ft[ft.bitrate_kbps == br].e_hat.to_numpy()
        cnt, edges, _ = ax[0].hist(e, bins=25, alpha=.6, color=cols[i], label=f"{br}")
        hist_rows += [{"bitrate_kbps": br, "bin_left": edges[j], "bin_right": edges[j + 1], "count": int(cnt[j])}
                      for j in range(len(cnt))]
    pd.DataFrame(hist_rows).to_csv(out / "fig6_histogram_bins.csv", index=False)
    ax[0].set_xlim(350, 1350)
    for q, l in ((p5, "$p_5$"), (p75, "$p_{75}$"), (p95, "$p_{95}$")):
        ax[0].axvline(q, color="k", ls="--", lw=.8)
        ax[0].annotate(l, (q, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=7)
    ax[0].set_xlabel("Estimated energy per chunk (playback + network)"); ax[0].set_ylabel("Chunks")
    ax[0].set_title("(a) Estimated chunk energy by representation", fontsize=8, pad=12)
    ax[0].legend(title="kbps", title_fontsize=6.5, frameon=False, fontsize=6.5, loc="center right",
                 bbox_to_anchor=(1.0, 0.55))
    e = np.linspace(p5 - 100, p95 + 500, 400)
    eb = (e - p5) / D
    ax[1].plot(e, eb, "--", color="grey", lw=1, label=r"linear part $\bar e$")
    ax[1].plot(e, [pen(x) for x in e], color="#1f5fbf", lw=1.6, label=r"penalty $P(e)$")
    ax[1].axvspan(p75, e.max(), color="#f4c27a", alpha=.25, lw=0)
    ax[1].text(1380, 0.2, "log-amplified region", fontsize=7, ha="center")
    for q, l in ((p5, "$p_5$"), (p75, "$p_{75}$"), (p95, "$p_{95}$")):
        ax[1].axvline(q, color="k", ls=":", lw=.7)
        ax[1].annotate(l, (q, 1.0), xycoords=("data", "axes fraction"), ha="center", va="bottom", fontsize=7)
    ax[1].set_xlabel(r"Estimated chunk energy $\hat e$"); ax[1].set_ylabel("Penalty")
    ax[1].set_title("(b) Calibrated energy penalty", fontsize=8, pad=12)
    ax[1].legend(frameon=False, fontsize=6.5, loc="upper left")
    fig.savefig(out / "fig_energy_penalty.png", dpi=300); plt.close(fig)

    json.dump({"network_energy": "size_MB * (210 / 3.0 + 28)", "throughput_mbps": THR, "example_chunks": EXAMPLES,
               "fold": {"train": FOLD_TRAIN, "test": FOLD_TEST, "device": "galaxy", "seed": SEED},
               "energy_weight_fig4c": W, "fig6_hist_bins_per_level": 25, "fig6_xlim": [350, 1350],
               "trace_window_rule": "5 samples spaced 4 s apart, mean in [2.8, 3.2] Mbps; stable: std < 0.15, "
                                    "variable: std > 2.3 (first match scanning training traces in order)"},
              open(out / "settings.json", "w"), indent=2)
    print("written to", out)


if __name__ == "__main__":
    main()
