"""Slide figures from artifacts/results.csv (static PNGs, light background)."""
import os

import matplotlib.pyplot as plt
import pandas as pd

OUT = os.path.join(os.path.dirname(__file__), "..", "artifacts")
FIG = os.path.join(OUT, "..", "talk", "final", "figures")
# reference categorical palette, fixed order (blue, orange, aqua), gray for context baselines
C1, C2, C3, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.size": 12, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.8, "figure.dpi": 160,
    "legend.frameon": False,
})


def _line(ax, x, y, color, label, ls="-"):
    # legend only: several series overlap exactly, so direct end-labels would collide
    ax.plot(x, y, ls, color=color, lw=2, marker="o", ms=6, label=label)


def success_vs_n(df):
    methods = [("lacam", "LaCAM (no learning)", GRAY, "--"),
               ("cspibt", "CS-PIBT every step", C2, "-"),
               ("random_lacam", "Random gate + LaCAM", C3, "-"),
               ("gate_lacam", "Learned gate + LaCAM", C1, "-")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m, lab, c, ls in methods:
        s = df[df.method == m].groupby("N").success.mean()
        _line(ax, list(s.index), list(s.values), c, lab, ls)
    ax.set_xlabel("Number of agents (32×32 map, 10% obstacles)")
    ax.set_ylabel("Success rate (20 test maps)")
    ax.set_ylim(0, 1.05)
    ax.set_xlim(right=ax.get_xlim()[1] * 1.28)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "success_vs_agents.png"))


def search_calls(df):
    """CS-PIBT calls per run (not per-step rate: runs that LaCAM finishes early have fewer steps)."""
    a = df[df.method == "cspibt"].groupby("N").n_cspibt.mean()
    b = df[df.method == "gate_lacam"].groupby("N").n_cspibt.mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = range(len(a))
    w = 0.38
    ax.bar([i - w / 2 for i in x], a.values, w, color=C2, label="CS-PIBT every step")
    ax.bar([i + w / 2 for i in x], b.values, w, color=C1, label="Learned gate + LaCAM")
    for i in x:
        ax.annotate(f"-{100 * (1 - b.values[i] / a.values[i]):.0f}%", (i + w / 2, b.values[i]),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK)
    ax.set_xticks(list(x), [str(n) for n in a.index])
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("CS-PIBT calls per map")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "search_calls.png"))


def quality(df):
    """Suboptimality on maps solved by every search-using method (from summary.csv)."""
    s = pd.read_csv(os.path.join(OUT, "summary.csv"))
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m, lab, c, ls in [("lacam", "LaCAM (no learning)", GRAY, "--"),
                          ("cspibt", "CS-PIBT every step", C2, "-"),
                          ("random_lacam", "Random gate + LaCAM", C3, "-"),
                          ("gate_lacam", "Learned gate + LaCAM", C1, "-")]:
        d = s[s.method == m].sort_values("N")
        _line(ax, list(d.N), list(d.subopt_common), c, lab, ls)
    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Flowtime / lower bound (lower is better)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "quality.png"))


def tradeoff(df):
    sw = df[df.method.isin(["gate_sweep", "random_sweep"])]
    agg = sw.groupby(["method", "param"]).agg(success=("success", "mean"),
                                              rate=("search_rate", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for m, lab, c in [("random_sweep", "Random gate", C2), ("gate_sweep", "Learned gate", C1)]:
        d = agg[agg.method == m].sort_values("rate")
        ax.plot(d.rate * 100, d.success, "-o", color=c, lw=2, ms=6, label=lab)
    ax.set_xlabel("Steps that call CS-PIBT (%)")
    ax.set_ylabel("Success rate (64–192 agents, 100 maps)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("Same compute, who solves more?  (no LaCAM fallback)", fontsize=11, color=MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "tradeoff.png"))


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    df = pd.read_csv(os.path.join(OUT, "results.csv"))
    search_calls(df)
    quality(df)
    tradeoff(df)
    print("saved to", FIG)
