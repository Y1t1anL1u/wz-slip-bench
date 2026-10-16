"""Generate README / project-page figures from the numbers reported in the paper."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np, os

OUT = "figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold", "figure.dpi": 150})
C = {"knn": "#1f77b4", "ref": "#7f7f7f", "win": "#bcbd22", "combo": "#d62728",
     "sonnet": "#9467bd", "haiku": "#ff7f0e", "ok": "#2ca02c", "late": "#d62728"}

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.svg", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------ 1. pipeline overview
fig, ax = plt.subplots(figsize=(13.5, 3.4)); ax.set_xlim(0, 13.2); ax.set_ylim(0, 3.4); ax.axis("off")
boxes = [
    (0.2,  "Raw WZDx archive\nNC feed, 2022–2024\n6.74M entities", "#e8e8e8"),
    (2.85, "Label validation\n4 defect classes\n391 censored (3.5%)", "#fde0dd"),
    (5.5,  "Benchmark\nhard stratum n=1,989\nlookup + kNN baselines", "#e0ecf4"),
    (8.15, "LLM vs baselines\n217 paired items\nSonnet ≈ kNN (p=0.56)", "#e5d8f0"),
    (10.8, "Conformal alert gate\nFPR ≤ α verified\n~30% recall @ α=0.1", "#d9f0d3"),
]
for x, txt, col in boxes:
    ax.add_patch(FancyBboxPatch((x, 0.9), 2.35, 1.7, boxstyle="round,pad=0.05,rounding_size=0.15",
                                fc=col, ec="#555", lw=1.2))
    ax.text(x + 1.175, 1.75, txt, ha="center", va="center", fontsize=10, linespacing=1.4)
for x, _, _ in boxes[:-1]:
    ax.add_patch(FancyArrowPatch((x + 2.4, 1.75), (x + 2.8, 1.75), arrowstyle="-|>", mutation_scale=16, color="#555", lw=1.4))
ax.text(6.6, 0.35, "Every stage is reproducible from the released scripts; total LLM cost ≈ US$1.5",
        ha="center", fontsize=9.5, color="#444", style="italic")
ax.set_title("From a raw DOT feed to a guaranteed alert: pipeline overview", loc="left", fontsize=13)
save(fig, "01_pipeline")

# ------------------------------------------------------------------ 2. slip structure by window & type
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1.2, 1.5], "wspace": 0.35})
win = ["≤1 d", "2–7 d", "8–30 d", ">30 d"]; ontime = [84.4, 26.6, 24.0, 9.2]; p90 = [3, 37, 36, 160]; n = [3194, 1517, 620, 98]
ax = axes[0]; b = ax.bar(win, ontime, color=[C["ok"], C["late"], C["late"], C["late"]], alpha=.85)
for r, v, k, p in zip(b, ontime, n, p90):
    ax.text(r.get_x() + r.get_width() / 2, v + 2, f"{v:.0f}% on time\nP90 slip +{p} d\nn={k}", ha="center", fontsize=7.8, linespacing=1.25)
ax.set_ylim(0, 110); ax.set_ylabel("Ends within ±1 day of posted date (%)"); ax.set_xlabel("Posted window (posted end − first seen)")
ax.set_title("Posted window dominates slip")

ax = axes[1]
types = ["incident", "construction", "maintenance", "ramp", "bridge", "pipe/utility", "paving", "signal/sign", "guardrail", "culvert"]
ont = [87.7, 50.7, 43.1, 42.0, 24.7, 22.7, 15.7, 15.8, 34.7, 2.4]; p90t = [1, 27, 15, 36, 34, 42, 47, 56, 96, 56]
y = np.arange(len(types))
ax.barh(y, ont, color="#7fb3d5", label="on-time rate (%)"); ax.set_yticks(y); ax.set_yticklabels(types); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("Ends within ±1 day of posted date (%)")
for i, (o, p) in enumerate(zip(ont, p90t)): ax.text(o + 1.5, i, f"P90 +{p}d", va="center", fontsize=8.5, color="#333")
ax.set_title("Work type is the second feature; several types are bimodal")
fig.suptitle("Structure of end-date slip (5,429 validated NC work-zone records)", fontsize=13, fontweight="bold", y=1.02)
save(fig, "02_slip_structure")

# ------------------------------------------------------------------ 3. LLM vs baselines (paired, hard stratum)
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), gridspec_kw={"wspace": 0.3})
methods = ["kNN\n(TF-IDF, k=30)", "Haiku 4.5\n+ retrieval", "Sonnet 4.6\n+ retrieval"]
med = [3.9, 5.2, 3.8]; p90e = [30.9, 31.9, 29.8]; cols = [C["knn"], C["haiku"], C["sonnet"]]
ax = axes[0]; x = np.arange(3); w = 0.38
ax.bar(x - w / 2, med, w, color=cols, alpha=.9, label="median |error|"); ax.bar(x + w / 2, p90e, w, color=cols, alpha=.45, hatch="//", label="P90 |error|")
for i in range(3):
    ax.text(x[i] - w / 2, med[i] + .6, f"{med[i]}", ha="center", fontsize=9); ax.text(x[i] + w / 2, p90e[i] + .6, f"{p90e[i]}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(methods); ax.set_ylabel("Absolute error (days)"); ax.legend(frameon=False, fontsize=9)
ax.set_title("Point-estimate error (217 paired items)", fontsize=11)
ax.text(1, 37.5, "vs kNN:\np < 0.001", ha="center", fontsize=9, color=C["haiku"]); ax.text(2, 37.5, "vs kNN:\np = 0.56", ha="center", fontsize=9, color=C["sonnet"])
ax.set_ylim(0, 44); ax.legend(frameon=False, fontsize=9, loc="upper left")

ax = axes[1]
strata = ["on time\n(≤1 d)", "1–7 d", "7–30 d", ">30 d"]; s_knn = [1.9, 2.3, 6.7, 42.5]; s_son = [1.8, 2.4, 6.5, 36.7]; s_hai = [3.0, 3.7, 7.8, 41.7]
x = np.arange(4); w = 0.26
ax.bar(x - w, s_knn, w, color=C["knn"], label="kNN"); ax.bar(x, s_hai, w, color=C["haiku"], label="Haiku"); ax.bar(x + w, s_son, w, color=C["sonnet"], label="Sonnet")
ax.set_xticks(x); ax.set_xticklabels(strata); ax.set_ylabel("Median |error| (days)"); ax.set_xlabel("True slip"); ax.legend(frameon=False, fontsize=9)
ax.set_title("Median error by true slip", fontsize=11)
fig.suptitle("Do LLMs read more than a nearest neighbor?  On this feed, no.", fontsize=13, fontweight="bold", y=1.02)
save(fig, "03_llm_vs_knn")

# ------------------------------------------------------------------ 4. conformal gate: recall vs alpha
fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), gridspec_kw={"wspace": 0.28})
alphas = [0.05, 0.10, 0.15, 0.20, 0.30]
curves = {"combined (kNN + class + window)": ([16, 32, 44, 51, 64], C["combo"], "-"),
          "kNN neighbor median": ([19, 32, 38, 50, 63], C["knn"], "-"),
          "kNN neighbor P(slip>7d)": ([17, 28, 38, 45, 57], C["knn"], "--"),
          "reference class P(slip>7d)": ([9, 17, 24, 26, 41], C["ref"], "-"),
          "raw posted window": ([7, 11, 14, 16, 31], C["win"], "-")}
ax = axes[0]
for lab, (v, c, ls) in curves.items(): ax.plot(alphas, v, marker="o", color=c, ls=ls, label=lab)
ax.scatter([0.436], [77], marker="*", s=180, color="k", zorder=5); ax.annotate("uncalibrated rule\n(kNN frac ≥ 0.3):\n77% recall, 44% FPR", (0.436, 77), (0.33, 52), fontsize=8.5, ha="left", arrowprops=dict(arrowstyle="->", color="#333"))
ax.set_xlabel("False-alarm bound α  (realized FPR ≈ α, verified)"); ax.set_ylabel("Recall of true slips > 7 d (%)"); ax.set_xlim(0, 0.5); ax.set_ylim(0, 90)
ax.legend(frameon=False, fontsize=8.5, loc="upper left"); ax.set_title("Recall at a guaranteed false-alarm rate (n = 1,892)", fontsize=11)

ax = axes[1]
taus = ["τ = 7 d", "τ = 14 d", "τ = 30 d"]; base = [36.8, 23.5, 13.3]; prec = [66, 47, 34]; rec = [32, 29, 36]
x = np.arange(3); w = 0.3
ax.bar(x - w, base, w, color="#bbb", label="base rate (positives, %)"); ax.bar(x, prec, w, color=C["ok"], label="precision of alerts (%)"); ax.bar(x + w, rec, w, color=C["combo"], alpha=.8, label="recall (%)")
for i in range(3): ax.text(x[i], prec[i] + 1.5, f"{prec[i] / base[i]:.1f}×", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(taus); ax.set_ylim(0, 80); ax.legend(frameon=False, fontsize=8.5); ax.set_title("At α = 0.1: precision lift vs base rate", fontsize=11)
fig.suptitle("A conformal alert gate on real DOT data: the guarantee holds; recall is capped near 30%", fontsize=13, fontweight="bold", y=1.02)
save(fig, "04_conformal_gate")

# ------------------------------------------------------------------ 5. calibration size + LLM confidence
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.1), gridspec_kw={"wspace": 0.3})
ax = axes[0]; ncal = [120, 240, 360, 600]
ax.plot(ncal, [17.8, 11.4, 12.0, 12.6], "o-", color=C["knn"], label="worst-case FPR over 50 splits"); ax.plot(ncal, [8.4, 8.4, 8.5, 8.3], "s--", color=C["knn"], alpha=.6, label="mean FPR")
ax.axhline(10, color="k", lw=1, ls=":"); ax.text(610, 10.4, "α = 10%", fontsize=9, ha="right")
ax.set_xlabel("Non-slipping records in calibration set"); ax.set_ylabel("Realized false-alarm rate (%)"); ax.set_ylim(0, 20); ax.legend(frameon=False, fontsize=9)
ax.set_title("≈250 on-time records stabilize the gate", fontsize=11)

ax = axes[1]
scores = ["Sonnet\npoint pred.", "kNN\nmedian", "kNN\nP(slip)", "Sonnet\nself-conf."]; vals = [39, 34, 32, 17]; cols = [C["sonnet"], C["knn"], C["knn"], "#999"]
b = ax.bar(scores, vals, color=cols, alpha=.9)
for r, v in zip(b, vals): ax.text(r.get_x() + r.get_width() / 2, v + 1, f"{v}%", ha="center", fontsize=10)
ax.set_ylabel("Recall @ α = 0.1 (%)"); ax.set_ylim(0, 50); ax.set_title("Same 217 items: LLM confidence cannot rank records", fontsize=11)
ax.text(3, 21, "plateaus at 17%\nfor every α", ha="center", fontsize=8.5, color="#555")
fig.suptitle("What the gate needs, and what it cannot use", fontsize=13, fontweight="bold", y=1.02)
save(fig, "05_calibration_and_confidence")
print("done:", sorted(os.listdir(OUT)))
