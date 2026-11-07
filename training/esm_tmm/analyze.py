"""Aggregate all runs (round 2: context-aware ESM-RB is the proposed method) + existing baseline logs into tables, statistics and figures.

python -m esm_tmm.analyze [--runs results/esm_tmm/runs] [--out results/esm_tmm/analysis]

Protocol
- Learned methods: per run, metrics are averaged over the 92 test traces of the held-out
  video; per seed the three folds (tos/bbb/doc held out) are averaged with equal weight;
  tables report mean ± std and the 95% t-interval over seeds.
- Rule-based baselines (no training) and the provided learned baselines are scored from
  the existing rep_6 logs with the same metric code (esm_tmm.metrics).
- Paired Wilcoxon signed-rank tests compare per-(video, trace) values averaged over seeds.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .data import DEVICES, VIDEOS, Paths
from .metrics import DeviceEnergy, read_log, session_metrics

REPO = Path(__file__).resolve().parents[2]
BASELINES = {"BB": "BOLA", "Bolae": "BOLA-E", "DD": "DynamicDash", "Dynamic": "Dynamic ABR",
             "Throughput": "Throughput", "ClassicMPC": "MPC", "Pensieve": "Pensieve*",
             "GreenABR": "GreenABR*", "Genet": "Genet*", "GreenABR-RB": "ESM-RB (ICWS ckpt)*",
             "NetLLM0": "NetLLM*", "NetLLM1": "NetLLM*", "NetLLM2": "NetLLM*"}
NAMES = {"esmrb": "ESM-RB", "ctx": "CA-ESM-RB (ours)", "ctx_nolook": "ours w/o lookahead",
         "look_only": "ours w/o throughput stats", "greenabr": "GreenABR (retrained)",
         "pensieve": "Pensieve (retrained)", "genet": "Genet (retrained)"}
RETRAINED = ("greenabr", "pensieve", "genet")
REUSED_V1 = ("esmrb", "ctx")   # unchanged methods, read from the v1 runs directory
TOL = 0.10                     # tolerance for the per-session violation rate
METRICS = ["QoE", "Efficiency_Avg", "Efficiency_G", "Efficiency_X", "Energy_Avg", "Energy_G", "Energy_X",
           "Rebuffer_Time", "Rebuffer_Count", "Smoothness_Change", "VMAF_mean", "VMAF_p10", "LowQ_ratio", "Data"]
HEADLINE = ["Efficiency_Avg", "QoE", "Energy_Avg", "Rebuffer_Time", "Smoothness_Change", "VMAF_mean", "Data"]


# ----------------------------------------------------------------------------- loading
def score_baselines(paths: Paths, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for video in VIDEOS:
        devices = {d: DeviceEnergy(paths.power_files, video, d) for d in DEVICES}
        folder = paths.baseline_logs / video
        if not folder.exists():
            continue
        for f in sorted(folder.iterdir()):
            parts = f.name.split("_", 3)
            if len(parts) < 4 or parts[1] not in BASELINES or parts[2] != "reward":
                continue
            m = session_metrics(read_log(f), devices)
            rows.append({"method": BASELINES[parts[1]], "test_video": video, "trace": parts[3], **m})
    df = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def load_runs(runs: Path, v1: Path | None = None) -> pd.DataFrame:
    frames = []
    for d in sorted(runs.iterdir()) if runs.exists() else []:
        if (d / "done.json").exists() and (d / "results.csv").exists():
            frames.append(pd.read_csv(d / "results.csv"))
    for d in sorted(v1.iterdir()) if (v1 and v1.exists() and v1 != runs) else []:
        if d.name.split("__")[0] in REUSED_V1 and (d / "done.json").exists() and (d / "results.csv").exists():
            frames.append(pd.read_csv(d / "results.csv"))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["tag"] = df["tag"].fillna("")
    df["budget_frac"] = df["budget_frac"].astype(str)
    return df


# ------------------------------------------------------------------------- statistics
def as_flag(series: pd.Series) -> pd.Series:
    """Booleans that may arrive as True/False, 'True'/'False' or 1.0/0.0 after concatenation."""
    mapped = series.map(lambda v: 1.0 if str(v).strip().lower() in ("true", "1", "1.0") else
                        (0.0 if str(v).strip().lower() in ("false", "0", "0.0") else np.nan))
    return mapped.astype(float)


def ci95(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def fmt(mean, std):
    if np.isnan(std):
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


def seed_level(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Average over traces per run, then over folds per seed (equal weight per video)."""
    per_run = df.groupby(keys + ["seed", "test_video"])[METRICS].mean().reset_index()
    return per_run.groupby(keys + ["seed"])[METRICS].mean().reset_index()


def summarize(seed_df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    out = []
    for k, g in seed_df.groupby(keys):
        row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        row["n_seeds"] = len(g)
        for m in METRICS:
            v = g[m].to_numpy()
            row[m] = v.mean(); row[m + "_std"] = v.std(ddof=1) if len(v) > 1 else np.nan
            row[m + "_ci95"] = ci95(v)
        out.append(row)
    return pd.DataFrame(out)


def baseline_summary(base: pd.DataFrame) -> pd.DataFrame:
    per_video = base.groupby(["method", "test_video"])[METRICS].mean().reset_index()
    s = per_video.groupby("method")[METRICS].mean().reset_index()
    for m in METRICS:
        s[m + "_std"] = np.nan
    s["n_seeds"] = 1
    return s


def wilcoxon_pairs(a: pd.DataFrame, b: pd.DataFrame, metric: str):
    """Paired test over (video, trace) of seed-averaged values."""
    ka = a.groupby(["test_video", "trace"])[metric].mean()
    kb = b.groupby(["test_video", "trace"])[metric].mean()
    idx = ka.index.intersection(kb.index)
    if len(idx) < 5:
        return np.nan, np.nan, len(idx)
    x, y = ka.loc[idx].to_numpy(), kb.loc[idx].to_numpy()
    if np.allclose(x, y):
        return 0.0, 1.0, len(idx)
    st, p = stats.wilcoxon(x, y)
    return float(np.median(x - y)), float(p), len(idx)


# ------------------------------------------------------------------------------ tables
def md_table(df: pd.DataFrame, label_col: str, metrics: list[str]) -> str:
    head = "| " + label_col + " | " + " | ".join(metrics) + " |\n|" + "---|" * (len(metrics) + 1) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = [fmt(r[m], r.get(m + "_std", np.nan)) for m in metrics]
        body += f"| {r[label_col]} | " + " | ".join(cells) + " |\n"
    return head + body


def df_md(df: pd.DataFrame) -> str:
    """Plain markdown table (avoids the optional `tabulate` dependency)."""
    def cell(v):
        if isinstance(v, float):
            return "nan" if np.isnan(v) else f"{v:.4g}"
        return str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(cell(v) for v in r) + " |\n" for r in df.itertuples(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(REPO / "results" / "esm_tmm" / "runs_v2"))
    ap.add_argument("--v1", default=str(REPO / "results" / "esm_tmm" / "runs"),
                    help="round-1 runs directory; the ESM-RB and context-aware runs are reused")
    ap.add_argument("--out", default=str(REPO / "results" / "esm_tmm" / "analysis_v2"))
    ap.add_argument("--repo", default=str(REPO))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    paths = Paths(Path(a.repo))
    base = score_baselines(paths, out / "baselines_per_trace.csv")
    df = load_runs(Path(a.runs), Path(a.v1))
    report = ["# CA-ESM-RB experiment report\n"]
    if df.empty:
        print("no finished runs yet"); return
    df.to_csv(out / "all_runs_per_trace.csv", index=False)

    galaxy = df.train_device == "galaxy"
    plain = df.tag == ""
    is_main = galaxy & plain

    # ---------------------------------------------------------------- A. main comparison
    main_df = df[is_main & df.variant.isin(["esmrb", "ctx", *RETRAINED])]
    if not main_df.empty:
        s = summarize(seed_level(main_df, ["variant"]), ["variant"])
        s["method"] = s.variant.map(NAMES)
        b = baseline_summary(base) if not base.empty else pd.DataFrame()
        table = pd.concat([s, b], ignore_index=True).sort_values("Efficiency_Avg", ascending=False)
        table.to_csv(out / "A_main_table.csv", index=False)
        report += ["## A. Overall comparison (held-out video, mean ± std over 5 seeds)\n",
                   "Learned methods marked * were trained by their authors (content may overlap the "
                   "test video); ours and ESM-RB never see the test video.\n",
                   md_table(table, "method", HEADLINE)]
        per_video = main_df.groupby(["variant", "seed", "test_video"])[METRICS].mean().reset_index()
        pv = per_video.groupby(["variant", "test_video"])[METRICS].agg(["mean", "std"])
        pv.to_csv(out / "A_per_video.csv")
        bv = base.groupby(["method", "test_video"])[METRICS].mean() if not base.empty else pd.DataFrame()
        bv.to_csv(out / "A_per_video_baselines.csv")
        pvt = per_video.groupby(["variant", "test_video"])[["QoE", "Energy_Avg", "Efficiency_Avg"]].mean().reset_index()
        if not base.empty:
            bvt = base.groupby(["method", "test_video"])[["QoE", "Energy_Avg", "Efficiency_Avg"]].mean().reset_index()
            pvt = pd.concat([pvt.assign(method=pvt.variant.map(NAMES)), bvt], ignore_index=True)
        report += ["\n### Per held-out video\n",
                   df_md(pvt[["method", "test_video", "QoE", "Energy_Avg", "Efficiency_Avg"]]
                         .sort_values(["test_video", "Efficiency_Avg"], ascending=[True, False])), "\n"]
        rows = []
        ours = main_df[main_df.variant == "ctx"]
        comps = {NAMES[v]: main_df[main_df.variant == v] for v in ("esmrb", *RETRAINED)
                 if (main_df.variant == v).any()}
        comps.update({m: base[base.method == m] for m in base.method.unique()} if not base.empty else {})
        for name, other in comps.items():
            for m in ("Efficiency_Avg", "QoE", "Energy_Avg", "Rebuffer_Time"):
                med, p, n = wilcoxon_pairs(ours, other, m)
                rows.append({"vs": name, "metric": m, "median_diff(ours-other)": med, "p_value": p, "n_pairs": n})
        wt = pd.DataFrame(rows); wt.to_csv(out / "A_wilcoxon.csv", index=False)
        report += ["\n### Paired Wilcoxon tests (ours vs …, over video × trace, seed-averaged)\n", df_md(wt), "\n"]

    # ------------------------------- A2. main comparison at the validation-selected weight
    sel_file = Path(a.runs).parent / "weight_selection.json"
    if sel_file.exists():
        sel = json.loads(sel_file.read_text())["selected"]
        tag_of = {0.5: "", 0.25: "ew0.25", 1.0: "ew1"}
        parts = [df[galaxy & (df.variant == v) & (df.tag == tag_of[float(w)])] for v, w in sel.items()]
        parts += [df[galaxy & (df.variant == v) & (df.tag == "")] for v in RETRAINED]
        sdf = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if not sdf.empty and sdf[sdf.variant.isin(list(sel))].groupby("variant").seed.nunique().min() >= 5:
            s = summarize(seed_level(sdf, ["variant"]), ["variant"])
            s["method"] = s.variant.map(NAMES) + s.variant.map(lambda v: f" | w={sel[v]}" if v in sel else "")
            b = baseline_summary(base) if not base.empty else pd.DataFrame()
            table = pd.concat([s, b], ignore_index=True).sort_values("Efficiency_Avg", ascending=False)
            table.to_csv(out / "A2_main_selected_weight.csv", index=False)
            report += ["\n## A2. Main comparison, energy weight of each learned method chosen on its "
                       "training videos (5 seeds)\n",
                       "Validation efficiency by weight: " + json.dumps(json.loads(sel_file.read_text())["summary"]) + "\n",
                       md_table(table, "method", HEADLINE)]
            pv = sdf.groupby(["variant", "seed", "test_video"])[["QoE", "Energy_Avg", "Efficiency_Avg"]].mean() \
                .groupby(["variant", "test_video"]).mean().reset_index()
            report += ["\n### A2 per held-out video\n", df_md(pv), "\n"]
            rows = []
            ours = sdf[sdf.variant == "ctx"]
            comps = {NAMES[v]: sdf[sdf.variant == v] for v in ("esmrb", *RETRAINED) if (sdf.variant == v).any()}
            comps.update({m: base[base.method == m] for m in base.method.unique()} if not base.empty else {})
            for name, other in comps.items():
                for m in ("Efficiency_Avg", "QoE", "Energy_Avg"):
                    med, p, n = wilcoxon_pairs(ours, other, m)
                    rows.append({"vs": name, "metric": m, "median_diff(ours-other)": med, "p_value": p})
            report += ["\n### A2 paired Wilcoxon tests\n", df_md(pd.DataFrame(rows)), "\n"]
        elif not sdf.empty:
            report += ["\n## A2. (waiting for seeds 3-4 of the selected weights)\n"]

    # -------------------------------------------------------------------- B. ablation
    abl = df[is_main & df.variant.isin(["esmrb", "ctx", "ctx_nolook", "look_only"])]
    if abl.variant.isin(["ctx_nolook", "look_only"]).any():
        s = summarize(seed_level(abl, ["variant"]), ["variant"]); s["method"] = s.variant.map(NAMES)
        s = s.set_index("variant").loc[[v for v in ("ctx", "ctx_nolook", "look_only", "esmrb")
                                        if v in set(s.variant)]].reset_index()
        s.to_csv(out / "B_ablation.csv", index=False)
        report += ["\n## B. Ablation of the context state\n", md_table(s, "method", HEADLINE)]

    # ---------------------------------------------------- C. energy-weight sweep (Pareto)
    ew = df[galaxy & df.variant.isin(["esmrb", "ctx"]) & df.tag.isin(["", "ew0.25", "ew1"])
            & (df.seed <= 2)]
    if (ew.tag != "").any():
        ew = ew.assign(energy_weight=ew.tag.map({"": 0.5, "ew0.25": 0.25, "ew1": 1.0}))
        s = summarize(seed_level(ew, ["variant", "energy_weight"]), ["variant", "energy_weight"])
        s["method"] = s.variant.map(NAMES) + " | w=" + s.energy_weight.astype(str)
        s = s.sort_values(["variant", "energy_weight"])
        s.to_csv(out / "C_energy_weight.csv", index=False)
        report += ["\n## C. Energy-weight sweep (seeds 0-2; w=0.5 is the default)\n",
                   md_table(s, "method", HEADLINE)]

    # ----------------------------------------------------------------- D. cross-device
    xd = df[df.variant.isin(["esmrb", "ctx"]) & plain]
    if (xd.train_device == "xcover").any():
        s = summarize(seed_level(xd, ["variant", "train_device"]), ["variant", "train_device"])
        s["method"] = s.variant.map(NAMES) + " | trained on " + s.train_device
        s.to_csv(out / "D_cross_device.csv", index=False)
        report += ["\n## D. Cross-device (energy of Galaxy S4 = G, XCover Pro = X)\n",
                   md_table(s, "method", ["QoE", "Energy_G", "Energy_X", "Efficiency_G", "Efficiency_X"])]

    # ------------------------------------------------- F. training protocol (random offset)
    prot = df[df.variant.isin(["esmrb", "ctx"]) & galaxy & df.tag.isin(["", "nooffset"])]
    if (prot.tag == "nooffset").any():
        prot = prot.assign(protocol=np.where(prot.tag == "nooffset", "fixed start (ICWS protocol)",
                                             "random start (ours)"))
        s = summarize(seed_level(prot, ["variant", "protocol"]), ["variant", "protocol"])
        s["method"] = s.variant.map(NAMES) + " | " + s.protocol
        s.to_csv(out / "F_protocol.csv", index=False)
        report += ["\n## F. Training protocol: fixed vs random content start\n", md_table(s, "method", HEADLINE)]

    # ------------------------------------------------- E. power-model generalisation
    pm_rows, seen = [], set()
    for r in (Path(a.runs), Path(a.v1)):
        for d in sorted(r.iterdir()) if r.exists() else []:
            f, c = d / "power_model.json", d / "config.json"
            if not (f.exists() and c.exists()):
                continue
            cfg = json.loads(c.read_text())
            key = (cfg["test_video"], cfg["train_device"], cfg["seed"])
            info = json.loads(f.read_text())
            if cfg.get("tag") or key in seen or "r2_and_within_chunk_corr" not in info:
                continue
            seen.add(key)
            for k, v in info["r2_and_within_chunk_corr"].items():
                video, dev, *metric = k.split("_", 2)
                role = "held-out video" if video == cfg["test_video"] else "training video"
                pm_rows.append({"fit_device": info["device"], "eval_device": dev, "role": role,
                                "metric": metric[0], "value": v, "seed": cfg["seed"]})
    if pm_rows:
        pm = pd.DataFrame(pm_rows).groupby(["fit_device", "eval_device", "role", "metric"]).value \
            .agg(["mean", "std"]).reset_index()
        pm.to_csv(out / "E_power_model.csv", index=False)
        report += ["\n## E. Power-model generalisation (within-chunk corr = how well level-to-level "
                   "energy differences are predicted)\n", df_md(pm)]

    (out / "report.md").write_text("\n".join(report))
    try:
        from .figures import make_figures
        make_figures(df, base, [Path(a.runs), Path(a.v1)], out)
    except Exception as e:  # figures are optional
        print("figure generation skipped:", e)
    print(f"analysis written to {out}")


if __name__ == "__main__":
    main()
