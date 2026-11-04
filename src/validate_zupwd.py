"""BodySync on ZU-PWD (everyday life, no stress labels in the free copy).

Normal = each person's own windows in the same situation (asleep / resting / active /
other), as reported by the watch's sleep and activity detection.

Checks:
  - exercise: with situation-aware normals, active windows should look normal; with one
    normal for everything they would look "stressed"
  - sleep: is the next-day (08-22h) score lower after nights with more wake-ups
    (stage 102), more time awake or less sleep? Each person compared with themselves.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

import bodysync
from config import FIGURES, PROCESSED, RESULTS, STATE_COLORS, STATES

MIN_SLEEP_H = 3


def score(w: pd.DataFrame, situation_aware: bool = True) -> pd.DataFrame:
    ctx = np.where(w.purity >= 0.7, w.state, "other") if situation_aware else "all"
    return bodysync.compute(w, context=pd.Series(ctx, index=w.index), normal=pd.Series(True, index=w.index))


def nights(clean: pd.DataFrame) -> pd.DataFrame:
    c = clean[clean.sleep_stage.isin([101, 102])].copy()
    c["night"] = (c.time + pd.Timedelta(hours=6)).dt.normalize()   # 18:00 -> 18:00, labelled by the morning
    c["bout_start"] = (c.sleep_stage == 102) & (c.groupby("pid").sleep_stage.shift() != 102)
    n = c.groupby(["pid", "night"]).agg(sleep_min=("sleep_stage", lambda s: (s == 101).sum()),
                                        wake_min=("sleep_stage", lambda s: (s == 102).sum()),
                                        wakeups=("bout_start", "sum"))
    n["sleep_h"] = n.sleep_min / 60
    n = n[n.sleep_h >= MIN_SLEEP_H]
    n["wakeups_per_h"] = n.wakeups / n.sleep_h
    n["pct_awake"] = n.wake_min / (n.sleep_min + n.wake_min)
    return n.reset_index()


def sleep_check(s: pd.DataFrame) -> pd.DataFrame:
    n = nights(pd.read_parquet(PROCESSED / "clean.parquet"))
    day = s[s.start.dt.hour.between(8, 21)].assign(night=lambda d: d.start.dt.normalize())
    day = day.groupby(["pid", "night"]).agg(score=("bodysync", "mean"), k=("bodysync", "size"))
    m = n.merge(day[day.k >= 8].reset_index(), on=["pid", "night"])
    m = m[m.groupby("pid").pid.transform("size") >= 4]
    rows = []
    for measure, bad_is_low in [("wakeups_per_h", False), ("pct_awake", False), ("sleep_h", True)]:
        med = m.groupby("pid")[measure].transform("median")
        bad = m[measure] < med if bad_is_low else m[measure] > med
        per = m.assign(bad=bad).groupby(["pid", "bad"]).score.mean().unstack().dropna()
        rows.append({"bad_night_means": measure, "n_people": len(per), "score_after_bad": per[True].mean(),
                     "score_after_good": per[False].mean(), "people_lower_after_bad": int((per[True] < per[False]).sum()),
                     "p_value": wilcoxon(per[True], per[False]).pvalue})
    return pd.DataFrame(rows)


def plot_week(s: pd.DataFrame):
    pid = s.pid.value_counts().index[1]
    g = s[s.pid == pid]
    g = g[g.start.dt.to_period("W") == g.start.dt.to_period("W").value_counts().index[0]]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for ctx in [*STATES, "other"]:
        k = g[g.context == ctx]
        ax.scatter(k.start, k.bodysync, s=8, color=STATE_COLORS[ctx], label=ctx)
    ax.plot(g.start, g.bodysync.rolling(4, min_periods=1).mean(), color="#222", lw=0.8)
    lowest = g.loc[g.groupby(g.start.dt.date).bodysync.idxmin()].nsmallest(3, "bodysync")   # one per day, no overlap
    for _, r in lowest.iterrows():
        ax.annotate(r.main_reason, (r.start, r.bodysync), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7, color="#b00")
    ax.set_ylim(0, 102)
    ax.set_ylabel("BodySync score")
    ax.set_title(f"Participant {pid[-3:]}: one week of BodySync (100 = your normal for that situation)")
    ax.legend(ncol=5, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_example_week.png", dpi=140)
    plt.close(fig)


def main():
    w = pd.read_parquet(PROCESSED / "windows_30.parquet")
    w = w[w.coverage >= 0.75].reset_index(drop=True)
    s = score(w)
    naive = score(w, situation_aware=False)
    ex = pd.DataFrame({"situation": s.context, "situation_aware": s.bodysync,
                       "one_normal_for_all": naive.bodysync}).groupby("situation").median().round(1)
    ex.to_csv(RESULTS / "zupwd_exercise_check.csv")
    print("Median score by situation\n", ex.to_string())
    sl = sleep_check(s)
    sl.to_csv(RESULTS / "zupwd_sleep_check.csv", index=False)
    print("\nSleep check\n", sl.round(3).to_string(index=False))
    plot_week(s)


if __name__ == "__main__":
    main()
