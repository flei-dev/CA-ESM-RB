import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
A, OUT = sys.argv[1], sys.argv[2]
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42})
BLUE, GREY, DARK = "#1f5fbf", "#9a9a9a", "#4a4a4a"
d = pd.read_csv(f"{A}/all_runs_per_trace.csv", low_memory=False); d["tag"] = d.tag.fillna("")
b = pd.read_csv(f"{A}/baselines_per_trace.csv")
g = d[d.train_device == "galaxy"]
M = ["QoE", "Energy_Avg", "Efficiency_Avg", "Rebuffer_Time", "Smoothness_Change", "VMAF_mean", "Data"]
def learned(v, tag):
    x = g[(g.variant == v) & (g.tag == tag)]
    s = x.groupby(["seed", "test_video"])[M].mean().groupby("seed").mean()
    return s.mean(), s.std(ddof=1), x
L = {"CA-ESM-RB (ours)": ("ctx", "ew1"), "ESM-RB": ("esmrb", ""), "GreenABR": ("greenabr", ""),
     "Pensieve": ("pensieve", ""), "Genet": ("genet", "")}
rows = []
for n, (v, t) in L.items():
    m, s, _ = learned(v, t); rows.append({"method": n, "kind": "learned", **{k: m[k] for k in M}, **{k + "_std": s[k] for k in M}})
RULE = {"MPC": "MPC", "BOLA": "BOLA", "BOLA-E": "BOLA-E", "DynamicDash": "DynamicDash", "Dynamic ABR": "Dynamic ABR",
        "Throughput": "Throughput", "NetLLM": "NetLLM*"}
for n, m in RULE.items():
    x = b[b.method == m].groupby("test_video")[M].mean().mean()
    rows.append({"method": n, "kind": "rule" if n != "NetLLM" else "llm", **{k: x[k] for k in M}})
T = pd.DataFrame(rows).sort_values("Efficiency_Avg", ascending=False)
T.to_csv(f"{OUT}/main_table_fair.csv", index=False)
# LaTeX table
with open(f"{OUT}/main_table_fair.tex", "w") as f:
    f.write("% Main comparison (held-out videos; learned methods: mean $\\pm$ std over 5 seeds)\n")
    f.write("\\begin{tabular}{lccccc}\n\\toprule\nMethod & Efficiency & QoE & Energy & Rebuffer (s) & Data (MB)\\\\\n\\midrule\n")
    for _, r in T.iterrows():
        pm = (lambda k: f"{r[k]:.2f} $\\pm$ {r[k+'_std']:.2f}" if r["kind"] == "learned" else f"{r[k]:.2f}")
        f.write(f"{r['method']} & {pm('Efficiency_Avg')} & {pm('QoE')} & {r['Energy_Avg']:,.0f} & {r['Rebuffer_Time']:.2f} & {r['Data']/1e6:.1f}\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# Fig 1: QoE vs energy
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for _, r in T.iterrows():
    if r.method.startswith("CA"):
        ax.errorbar(r.Energy_Avg, r.QoE, xerr=r.Energy_Avg_std, yerr=r.QoE_std, fmt="o", color=BLUE, ms=6, capsize=2, zorder=5)
    elif r.kind == "learned":
        ax.errorbar(r.Energy_Avg, r.QoE, xerr=r.Energy_Avg_std, yerr=r.QoE_std, fmt="o", color=GREY, ms=4.5, capsize=1.5)
    else:
        ax.plot(r.Energy_Avg, r.QoE, "o", mfc="white", mec=GREY, ms=4.5, mew=1.2)
off = {"CA-ESM-RB (ours)": (-6, 6, "right"), "ESM-RB": (5, -9, "left"), "Genet": (-5, 4, "right"), "GreenABR": (5, -9, "left"),
       "Pensieve": (0, -17, "center"), "NetLLM": (5, -9, "left"), "MPC": (5, 2, "left"), "BOLA-E": (-5, -9, "right"),
       "DynamicDash": (0, 7, "center"), "Dynamic ABR": (0, -12, "center"), "BOLA": (-5, -9, "right"), "Throughput": (5, 2, "left")}
for _, r in T.iterrows():
    dx, dy, ha = off[r.method]
    ax.annotate(r.method, (r.Energy_Avg, r.QoE), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=7,
                color=BLUE if r.method.startswith("CA") else DARK, fontweight="bold" if r.method.startswith("CA") else None)
ax.set_xlabel("Energy per session (mean of two phones)"); ax.set_ylabel("QoE")
ax.annotate("better", xy=(0.03, 0.99), xytext=(0.12, 0.88), xycoords="axes fraction", textcoords="axes fraction",
            fontsize=7, color=DARK, arrowprops=dict(arrowstyle="->", color=DARK, lw=0.8))
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=GREY, ms=4.5, label="learned (retrained, 5 seeds)"),
                   Line2D([], [], marker="o", ls="", mfc="white", mec=GREY, ms=4.5, label="rule-based / NetLLM")],
          frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
ax.grid(axis="y", color="#e5e5e5", lw=0.6)
fig.savefig(f"{OUT}/fig_qoe_energy.pdf"); fig.savefig(f"{OUT}/fig_qoe_energy.png", dpi=300); plt.close(fig)
# Fig 2: per-video efficiency
vids = ["tos", "bbb", "doc"]; meths = ["CA-ESM-RB (ours)", "ESM-RB", "GreenABR", "Pensieve", "Genet"]
pv = {}
for n in meths:
    v, t = L[n]; x = g[(g.variant == v) & (g.tag == t)]
    pv[n] = x.groupby(["seed", "test_video"]).Efficiency_Avg.mean().groupby("test_video").agg(["mean", "std"])
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.1))
for ax, vid in zip(axes, vids):
    ms = [pv[n].loc[vid, "mean"] for n in meths]; ss = [pv[n].loc[vid, "std"] for n in meths]
    ax.bar(range(5), ms, yerr=ss, capsize=2, color=[BLUE] + [GREY] * 4, width=0.7)
    ax.set_xticks(range(5)); ax.set_xticklabels(["CA-ESM-RB", "ESM-RB", "GreenABR", "Pensieve", "Genet"], rotation=30, ha="right", fontsize=7)
    ax.set_title(vid.upper(), fontsize=9); lo = min(m - s for m, s in zip(ms, ss)); ax.set_ylim(0, max(m + s for m, s in zip(ms, ss)) * 1.12)
    for i, m in enumerate(ms): ax.text(i, m + max(ms) * 0.03 + ss[i], f"{m:.2f}", ha="center", fontsize=6.5)
axes[0].set_ylabel("Efficiency (1000·QoE/E)")
fig.savefig(f"{OUT}/fig_per_video.pdf"); fig.savefig(f"{OUT}/fig_per_video.png", dpi=300); plt.close(fig)
# Fig 3: weight sweep
fig, ax = plt.subplots(figsize=(3.5, 2.5))
for v, col, lab in (("esmrb", GREY, "ESM-RB"), ("ctx", BLUE, "CA-ESM-RB (ours)")):
    pts = []
    for w, tag in ((0.25, "ew0.25"), (0.5, ""), (1.0, "ew1")):
        x = g[(g.variant == v) & (g.tag == tag) & (g.seed <= 2)]
        s = x.groupby(["seed", "test_video"])[["QoE", "Energy_Avg"]].mean().groupby("seed").mean()
        pts.append((w, s.Energy_Avg.mean(), s.QoE.mean(), s.Energy_Avg.std(ddof=1), s.QoE.std(ddof=1)))
    pts.sort(key=lambda p: p[1])
    ax.errorbar([p[1] for p in pts], [p[2] for p in pts], xerr=[p[3] for p in pts], yerr=[p[4] for p in pts],
                fmt="-o", color=col, ms=4, capsize=2, lw=1.4, label=lab)
    for p in pts: ax.annotate(f"w={p[0]:g}", (p[1], p[2]), textcoords="offset points",
                              xytext=((-4, 5) if (v == "ctx" and p[0] == 1.0) else (4, -11)), ha=("right" if (v == "ctx" and p[0] == 1.0) else "left"), fontsize=6.5, color=DARK)
ax.set_xlabel("Energy per session (mean of two phones)"); ax.set_ylabel("QoE"); ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.grid(axis="y", color="#e5e5e5", lw=0.6)
fig.savefig(f"{OUT}/fig_weight_sweep.pdf"); fig.savefig(f"{OUT}/fig_weight_sweep.png", dpi=300); plt.close(fig)
print(T[["method", "Efficiency_Avg", "QoE", "Energy_Avg"]].round(2).to_string())
