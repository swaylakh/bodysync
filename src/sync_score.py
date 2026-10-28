"""The BodySync score: how in-sync is your body right now, compared with your own normal?

Every 30 minutes (sliding by 15):
1. Measure how heart rate, sweat (EDA) and skin temperature move together:
   Spearman r for the 3 pairs HR-EDA, HR-temp, EDA-temp.
2. Compare each r with what is normal for THIS person in THIS situation
   (asleep / resting / active / other): their own median and typical spread (MAD).
3. Average how far off the 3 pairs are (in units of typical spread) = desync.
4. Score = 100 when the body syncs exactly as usual, dropping 33 points per unit
   of desync, floored at 0. A "desync alert" is two low windows (< 50) in a row.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import FIGURES, PAIRS, PROCESSED, RESULTS, STATE_COLORS

PAIR_COLS = [f"s_{a}_{b}" for a, b in PAIRS]
MIN_BASELINE_WINDOWS = 10
POINTS_PER_UNIT = 100 / 3
ALERT_BELOW = 50
OUT = FIGURES / "sync_score"
OUT.mkdir(exist_ok=True)


def baselines(w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Median and spread of each pair's r per person x situation, falling back to the person's overall."""
    def mad(s):
        return (s - s.median()).abs().median() * 1.4826

    ctx = w.groupby(["pid", "context"])[PAIR_COLS]
    med, spr, n = ctx.median(), ctx.agg(mad), ctx.size()
    med, spr = med[n >= MIN_BASELINE_WINDOWS], spr[n >= MIN_BASELINE_WINDOWS]
    overall_med = w.groupby("pid")[PAIR_COLS].median()
    overall_spr = w.groupby("pid")[PAIR_COLS].agg(mad)
    return med, spr.clip(lower=0.05), overall_med, overall_spr.clip(lower=0.05)


def score(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["context"] = np.where(w.purity >= 0.7, w.state, "other")
    med, spr, omed, ospr = baselines(w)
    keys = pd.MultiIndex.from_frame(w[["pid", "context"]])
    m = med.reindex(keys).to_numpy(copy=True)
    s = spr.reindex(keys).to_numpy(copy=True)
    no_ctx = np.isnan(m).any(axis=1)                      # situation too rare for this person -> overall normal
    m[no_ctx] = omed.reindex(w.pid[no_ctx]).to_numpy()
    s[no_ctx] = ospr.reindex(w.pid[no_ctx]).to_numpy()
    off = np.abs(w[PAIR_COLS].to_numpy() - m) / s
    w["desync"] = np.nanmean(off, axis=1)
    for i, (a, b) in enumerate(PAIRS):
        w[f"off_{a}_{b}"] = off[:, i]
    w["sync_score"] = np.clip(100 - POINTS_PER_UNIT * w.desync, 0, 100)
    w = w.sort_values(["pid", "start"])
    low = w.sync_score < ALERT_BELOW
    prev_low = low.groupby(w.pid).shift(fill_value=False)
    gap_ok = w.groupby("pid").start.diff() == pd.Timedelta(minutes=15)
    w["alert"] = low & prev_low & gap_ok
    return w


def plot_person(w: pd.DataFrame, pid: str, days: int = 7):
    g = w[w.pid == pid]
    # show the week with the most scored windows
    week = g.start.dt.to_period("W").value_counts().index[0]
    g = g[g.start.dt.to_period("W") == week]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for ctx, col in STATE_COLORS.items():
        k = g[g.context == ctx]
        ax.scatter(k.start, k.sync_score, s=8, color=col, label=ctx)
    ax.plot(g.start, g.sync_score.rolling(4, min_periods=1).mean(), color="#222", lw=0.8)
    a = g[g.alert]
    ax.scatter(a.start, a.sync_score, s=40, facecolors="none", edgecolors="red", label="desync alert")
    ax.axhline(ALERT_BELOW, color="red", lw=0.6, ls="--")
    ax.set_ylim(0, 102)
    ax.set_ylabel("BodySync score")
    ax.set_title(f"Participant {pid[-3:]}: one week of BodySync scores (dot colour = situation)")
    ax.legend(ncol=6, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / f"participant_{pid[-3:]}.png", dpi=140)
    plt.close(fig)


def overview(w: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(w.sync_score, bins=40, color="#3b5ba5")
    axes[0].set_title("All scores")
    axes[0].set_xlabel("BodySync score")

    h = w.assign(hour=w.start.dt.hour).groupby("hour").sync_score.mean()
    axes[1].plot(h.index, h.values, marker="o", color="#3b5ba5")
    axes[1].set_title("Average score by hour of day")
    axes[1].set_xlabel("hour")

    p = w.groupby("pid").agg(score=("sync_score", "mean"), alerts_per_day=("alert", "sum"),
                             days=("start", lambda s: s.dt.date.nunique()))
    p["alerts_per_day"] /= p.days
    p = p.sort_values("score")
    axes[2].barh([i[-3:] for i in p.index], p.score, color="#5fa55a")
    axes[2].set_title("Average score per person")
    axes[2].set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(OUT / "overview.png", dpi=150)
    plt.close(fig)
    p.to_csv(RESULTS / "sync_score_per_person.csv")


def main():
    w = pd.read_parquet(PROCESSED / "windows_30.parquet")
    w = w[(w.coverage >= 0.75) & w.eda_ok].dropna(subset=PAIR_COLS)
    w = score(w)
    cols = ["pid", "start", "context", "sync_score", "desync", "alert", *PAIR_COLS, *[f"off_{a}_{b}" for a, b in PAIRS]]
    w[cols].to_csv(RESULTS / "sync_scores.csv", index=False)
    overview(w)
    for pid in w.pid.value_counts().index[:6]:
        plot_person(w, pid)

    trans = w.purity < 0.7
    print(f"{len(w)} scored windows, {w.pid.nunique()} people")
    print(f"median score {w.sync_score.median():.0f}; alerts {w.alert.sum()} ({w.alert.mean():.1%} of windows)")
    print(f"median score: single-situation {w[~trans].sync_score.median():.1f} vs changing situation {w[trans].sync_score.median():.1f}")
    print(w.groupby("context").sync_score.describe().round(1).to_string())


if __name__ == "__main__":
    main()
