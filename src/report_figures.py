"""Light-mode figures for the final report (figures/fig8-10). Numbers come from results/final_test.csv,
results/plain_formula.csv, results/new_datasets.csv, rounds 15-17 of results/iteration_log.csv and
results/driving_order_self_last.csv / driving_drop_one.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import ROOT

FIG = ROOT / "figures"
INK, MUT, GRID, CALM, WARM, GOLD, GREY = "#1F2933", "#5F6B7A", "#E3E7EB", "#00897B", "#C96A1B", "#C99A1E", "#9AA5B1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.spines.top": False, "axes.spines.right": False})


def first_vs_final():
    names = ["WESAD", "Stress +\nexercise", "Stress-\nPredict", "CAN-STRESS\n(daily life)", "AffectiveROAD\n(city vs highway)"]
    first = [92, 63, 62, 59, 54]
    final = [99.9, 91, 90, 67, 76]
    fig, ax = plt.subplots(figsize=(7.2, 3.1), dpi=220)
    x = range(len(names)); w = .36
    ax.bar([i - w/2 for i in x], first, w, color=GREY, label="First version (plain formula)")
    ax.bar([i + w/2 for i in x], final, w, color=CALM, label="Final BodySync")
    for i, (a, b) in enumerate(zip(first, final)):
        ax.text(i - w/2, a + 1.2, f"{a}", ha="center", fontsize=8.5, color=MUT)
        ax.text(i + w/2, b + 1.2, "99+" if b > 99 else f"{b}", ha="center", fontsize=8.5, color=INK, fontweight="bold")
    ax.axhline(50, color=MUT, lw=1, ls=(0, (4, 3))); ax.text(-.62, 51, "coin flip", fontsize=8, color=MUT, ha="left")
    ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=8.5); ax.set_ylim(40, 106); ax.set_ylabel("% of stressed/calm pairs\nranked correctly")
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2, bbox_to_anchor=(1, 1.13))
    fig.tight_layout(); fig.savefig(FIG / "fig8_first_vs_final.png", facecolor="white")


def steps():
    labels = ["Plain\nformula", "Trained\nmodel", "+ sweat\nbursts", "+ smooth-\ning", "+ fast/slow\nsweat", "+ change\n(10 min)", "+ self-\ntraining", "Final"]
    vals = [56.28, 69.44, 72.96, 74.14, 76.78, 79.92, 81.84, 81.84]
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=220)
    prev = 50
    for i, v in enumerate(vals):
        start = i in (0, len(vals) - 1)
        lo = 50 if start else prev
        col = GREY if i == 0 else INK if i == len(vals) - 1 else GOLD if i == 6 else CALM
        ax.bar(i, v - lo, .62, bottom=lo, color=col)
        ax.text(i, v + .6, f"{v:.0f}%" if start else f"+{v - prev:.1f}", ha="center", fontsize=8.5, color=INK, fontweight="bold")
        if not start and i > 0: ax.plot([i - 1 - .31, i - .31], [prev, prev], color=MUT, lw=.8, ls=":")
        prev = v
    ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8); ax.set_ylim(50, 88); ax.set_ylabel("five-study average (%)")
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "fig9_steps.png", facecolor="white")


def drop_one():
    names = ["Change over 10 min", "Smoothing", "Self-training", "Fast / slow sweat", "Sweat bursts"]
    drops = [3.0, 2.4, 1.9, 1.2, 0.3]
    fig, ax = plt.subplots(figsize=(7.2, 2.3), dpi=220)
    ax.barh(names[::-1], drops[::-1], color=[CALM, CALM, GOLD, CALM, CALM][::-1], height=.55)
    for i, d in enumerate(drops[::-1]): ax.text(d + .05, i, f"−{d:.1f}", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("points lost when removed from the finished system (five-study average)"); ax.set_xlim(0, 3.6)
    ax.grid(axis="x", color=GRID, lw=.8); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(FIG / "fig10_drop_one.png", facecolor="white")


if __name__ == "__main__":
    first_vs_final(); steps(); drop_one(); print("written to", FIG)
