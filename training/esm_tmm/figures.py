"""Paper figures (PNG + PDF) from the aggregated results (round 2: CA-ESM-RB is ours)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .analyze import NAMES, seed_level  # noqa: E402

ACCENT, GREY, MID = "#1f5fbf", "#9a9a9a", "#5a5a5a"
VCOL = {"esmrb": "#555555", "ctx": ACCENT, "ctx_nolook": "#74add1", "look_only": "#8c6bb1"}
EW = {"": 0.5, "ew0.25": 0.25, "ew1": 1.0}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


def _save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def _seed_stats(df, keys, metric):
    s = seed_level(df, keys)
    return s.groupby(keys)[metric].agg(["mean", "std"]).reset_index()


def make_figures(df: pd.DataFrame, base: pd.DataFrame, runs, out: Path):
    galaxy = df.train_device == "galaxy"

    # ---- Pareto: QoE vs energy, with the energy-weight sweep of ESM-RB and ours ------------
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    if not base.empty:
        b = base.groupby(["method", "test_video"])[["QoE", "Energy_Avg"]].mean().groupby("method").mean()
        ax.scatter(b.Energy_Avg, b.QoE, color=GREY, s=28, zorder=2)
        for m, r in b.iterrows():
            ax.annotate(m, (r.Energy_Avg, r.QoE), textcoords="offset points", xytext=(4, 3),
                        fontsize=7.5, color=MID)
    sweep = df[galaxy & df.variant.isin(["esmrb", "ctx"]) & df.tag.isin(list(EW)) & (df.seed <= 2)]
    for v in ("esmrb", "ctx"):
        d = sweep[sweep.variant == v]
        if d.empty:
            continue
        d = d.assign(w=d.tag.map(EW))
        e = _seed_stats(d, ["w"], "Energy_Avg").sort_values("w")
        q = _seed_stats(d, ["w"], "QoE").sort_values("w")
        ax.errorbar(e["mean"], q["mean"], xerr=e["std"], yerr=q["std"], fmt="-o", color=VCOL[v],
                    label=f"{NAMES[v]} (energy weight 0.25/0.5/1)", capsize=2, ms=4, zorder=4)
    ax.set_xlabel("Energy consumption (mean of Galaxy S4 and XCover Pro)")
    ax.set_ylabel("QoE (SQoE-III model)")
    ax.set_title("QoE–energy trade-off on held-out videos")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out, "fig_pareto")

    # ---- Ablation bars --------------------------------------------------------------------
    abl = df[galaxy & (df.tag == "") & df.variant.isin(list(NAMES))]
    if abl.variant.nunique() > 1:
        order = [v for v in ("ctx", "ctx_nolook", "look_only", "esmrb") if v in set(abl.variant)]
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        for ax, m in zip(axes, ("Efficiency_Avg", "QoE", "Energy_Avg")):
            st = _seed_stats(abl, ["variant"], m).set_index("variant").loc[order]
            ax.bar(range(len(order)), st["mean"], yerr=st["std"], capsize=2,
                   color=[VCOL[v] for v in order])
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels([NAMES[v] for v in order], rotation=30, ha="right", fontsize=7.5)
            ax.set_title(m.replace("_Avg", " (avg)"), fontsize=9)
            lo = (st["mean"] - st["std"].fillna(0)).min()
            ax.set_ylim(lo * 0.95 if lo > 0 else None)
        fig.suptitle("Ablation (mean ± std over seeds and held-out videos)", fontsize=10)
        _save(fig, out, "fig_ablation")

    # ---- Training curves ------------------------------------------------------------------
    curves, seen = {}, set()
    for r in runs:
        for d in sorted(Path(r).iterdir()) if Path(r).exists() else []:
            cfgp = d / "config.json"
            if d.name in seen or not (d / "train_log.csv").exists() or not cfgp.exists():
                continue
            seen.add(d.name)
            c = json.loads(cfgp.read_text())
            if c.get("tag") or c.get("train_device") != "galaxy" or c["variant"] not in ("esmrb", "ctx"):
                continue
            log = pd.read_csv(d / "train_log.csv", usecols=["episode", "quality", "e_norm", "rebuffer"])
            curves.setdefault(c["variant"], []).append(log)
    if curves:
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        for v, logs in curves.items():
            m = pd.concat(logs).groupby("episode").mean().rolling(500, min_periods=50).mean()
            for ax, col, title in zip(axes, ("quality", "e_norm", "rebuffer"),
                                      ("Mean VMAF per episode", "Normalised energy per chunk",
                                       "Rebuffering per episode (s)")):
                ax.plot(m.index, m[col], color=VCOL[v], label=NAMES[v], lw=1.2)
                ax.set_title(title, fontsize=9)
                ax.set_xlabel("episode")
        axes[0].legend(frameon=False, fontsize=7.5)
        _save(fig, out, "fig_training_curves")
