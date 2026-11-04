"""ZU-PWD figures: what a day looks like, and whether "moving together" differs by state."""
import matplotlib.pyplot as plt
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
    # the day with the most complete data
    g = clean[clean.all3].assign(day=lambda d: d.time.dt.normalize())
    pid, day = g.groupby(["pid", "day"]).size().idxmax()
    d = clean[(clean.pid == pid) & (clean.time.dt.normalize() == day)].set_index("time").asfreq("min")

    fig, axes = plt.subplots(3, 1, figsize=(11, 6.5), sharex=True)
    for ax, s in zip(axes, ["hr", "eda", "temp"]):
        ax.plot(d.index, d[s], lw=0.8, color="#333")
        ax.set_ylabel(LABELS[s])
        for state, col in STATE_COLORS.items():
            ax.fill_between(d.index, 0, 1, where=(d.state == state).to_numpy(), color=col, alpha=0.18,
                            transform=ax.get_xaxis_transform(), lw=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.4) for c in STATE_COLORS.values()]
    axes[0].legend(handles, STATE_COLORS.keys(), ncol=4, loc="upper right", fontsize=8)
    axes[0].set_title(f"A typical day: participant {pid[-3:]}, {day.date()} (shading = state)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_typical_day.png", dpi=160)
    plt.close(fig)


def coupling_by_state(w: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.flat, COUPLING):
        sns.boxplot(data=w, x="state", y=c, order=STATES, palette=STATE_COLORS, hue="state",
                    showfliers=False, ax=ax, width=0.6)
        ax.set_title(LABELS[c])
        ax.set(xlabel="", ylabel="")
        ax.axhline(0, color="k", lw=0.6)
    fig.suptitle("How signals move together, by state (Pearson r, 30-min windows)")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_coupling_by_state.png", dpi=160)
    plt.close(fig)


def stats_by_state(w: pd.DataFrame) -> pd.DataFrame:
    # per-person medians so the 2 heavy participants don't dominate; Friedman test across states
    med = w.groupby(["pid", "state"])[LEVEL + COUPLING].median()[w.groupby(["pid", "state"]).size() >= 5]
    keep = med.reset_index().groupby("pid").state.nunique()
    med = med[med.index.get_level_values("pid").isin(keep[keep == 3].index)]
    rows = []
    for c in LEVEL + COUPLING:
        t = med[c].unstack()[STATES].dropna()
        rows.append({"feature": c, "n_people": len(t), **{f"median_{s}": t[s].median() for s in STATES},
                     "friedman_p": friedmanchisquare(*[t[s] for s in STATES]).pvalue})
    return pd.DataFrame(rows)


def main():
    typical_day(pd.read_parquet(PROCESSED / "clean.parquet"))
    w = pd.read_parquet(PROCESSED / "windows_30.parquet")
    w = w[w.labelled & w.eda_ok]
    coupling_by_state(w)
    out = stats_by_state(w)
    out.to_csv(RESULTS / "features_by_state.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
