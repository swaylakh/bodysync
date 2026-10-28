"""Exploration figures: what the data looks like and whether coupling differs by state."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare

from config import FIGURES, PROCESSED, RESULTS, STATE_COLORS, STATES
from features import COUPLING, LEVEL

sns.set_theme(style="whitegrid", font_scale=0.9)
LABELS = {
    "hr": "Heart rate (bpm)", "eda": "EDA (µS)", "temp": "Skin temp (°C)",
    "r_hr_eda": "HR–EDA", "r_hr_temp": "HR–temp", "r_eda_temp": "EDA–temp",
    "dr_hr_eda": "HR–EDA (changes)", "dr_hr_temp": "HR–temp (changes)", "dr_eda_temp": "EDA–temp (changes)",
}


def typical_day(clean: pd.DataFrame):
    # pick the day with the most complete data for a well-recorded person
    g = clean[clean.all3].copy()
    g["day"] = g.time.dt.normalize()
    best = g.groupby(["pid", "day"]).size().idxmax()
    d = clean[(clean.pid == best[0]) & (clean.time.dt.normalize() == best[1])]
    d = d.set_index("time").asfreq("min")

    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    for ax, s in zip(axes, ["hr", "eda", "temp"]):
        ax.plot(d.index, d[s], lw=0.8, color="#333")
        ax.set_ylabel(LABELS[s])
        for state, col in STATE_COLORS.items():
            m = (d.state == state).to_numpy()
            ax.fill_between(d.index, 0, 1, where=m, color=col, alpha=0.18, transform=ax.get_xaxis_transform(), lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in STATE_COLORS.values()]
    axes[0].legend(handles, STATE_COLORS.keys(), ncol=4, loc="upper right", fontsize=8)
    axes[0].set_title(f"A typical day: participant {best[0][-3:]}, {best[1].date()} (shading = state)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_typical_day.png", dpi=160)
    plt.close(fig)


def usable_hours(clean: pd.DataFrame):
    h = clean[clean.all3].groupby("pid").size().div(60).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar([p[-3:] for p in h.index], h.values, color="#3b5ba5")
    ax.set_ylabel("Usable hours (HR+EDA+temp)")
    ax.set_xlabel("Participant")
    ax.set_title(f"Usable data per person: total {h.sum():,.0f} h, top 2 people = {h.iloc[:2].sum() / h.sum():.0%}")
    plt.xticks(rotation=90)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_usable_hours.png", dpi=160)
    plt.close(fig)


def by_state(w: pd.DataFrame, cols: list[str], fname: str, title: str):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.flat, cols):
        sns.boxplot(data=w, x="state", y=c, order=STATES, palette=STATE_COLORS, hue="state",
                    showfliers=False, ax=ax, width=0.6)
        ax.set_title(LABELS.get(c, c))
        ax.set_xlabel("")
        ax.set_ylabel("")
        if c.startswith(("r_", "dr_")):
            ax.axhline(0, color="k", lw=0.6)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIGURES / fname, dpi=160)
    plt.close(fig)


def coupling_stats(w: pd.DataFrame) -> pd.DataFrame:
    # per-person medians so the 2 heavy participants don't dominate; Friedman test across states
    med = w.groupby(["pid", "state"])[LEVEL + COUPLING].median()
    n = w.groupby(["pid", "state"]).size()
    med = med[n >= 5]
    keep = med.reset_index().groupby("pid").state.nunique()
    med = med[med.index.get_level_values("pid").isin(keep[keep == 3].index)]
    rows = []
    for c in LEVEL + COUPLING:
        t = med[c].unstack()[STATES].dropna()
        stat = friedmanchisquare(*[t[s] for s in STATES])
        rows.append({"feature": c, "n_people": len(t),
                     **{f"median_{s}": t[s].median() for s in STATES}, "friedman_p": stat.pvalue})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "features_by_state.csv", index=False)
    return out


def main():
    clean = pd.read_parquet(PROCESSED / "clean.parquet")
    typical_day(clean)
    usable_hours(clean)
    w = pd.read_parquet(PROCESSED / "windows_30.parquet")
    w = w[w.labelled & w.eda_ok]
    by_state(w, LEVEL, "fig3_levels_by_state.png", "Plain signal values by state (30-min windows)")
    by_state(w, COUPLING, "fig4_coupling_by_state.png", "How signals move together, by state (Pearson r, 30-min windows)")
    print(coupling_stats(w).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
