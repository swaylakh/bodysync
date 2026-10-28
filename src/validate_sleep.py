"""Check the BodySync score against sleep quality (no self-reports needed).

The watch marks brief wake-ups during sleep (stage 102). For each person-night:
  - sleep hours, wake-ups per hour, % of the night awake
Then, within each person, compare the sync score of the following day (08:00-22:00)
after their worse nights vs their better nights. If the score means something,
days after bad nights should score lower.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from config import FIGURES, PROCESSED, RESULTS

MIN_SLEEP_H = 3
MIN_DAY_WINDOWS = 8


def nights(clean: pd.DataFrame) -> pd.DataFrame:
    c = clean[clean.sleep_stage.isin([101, 102])].copy()
    # a "night" runs 18:00 -> 18:00; label it by the morning's date
    c["night"] = (c.time + pd.Timedelta(hours=6)).dt.normalize()
    c["bout_start"] = (c.sleep_stage == 102) & (c.groupby("pid").sleep_stage.shift() != 102)
    n = c.groupby(["pid", "night"]).agg(
        sleep_min=("sleep_stage", lambda s: (s == 101).sum()),
        wake_min=("sleep_stage", lambda s: (s == 102).sum()),
        wakeups=("bout_start", "sum"),
    )
    n["sleep_h"] = n.sleep_min / 60
    n = n[n.sleep_h >= MIN_SLEEP_H]
    n["wakeups_per_h"] = n.wakeups / n.sleep_h
    n["pct_awake"] = n.wake_min / (n.sleep_min + n.wake_min)
    return n.reset_index()


def next_day_scores(scores: pd.DataFrame) -> pd.DataFrame:
    s = scores.copy()
    s["start"] = pd.to_datetime(s.start)
    s = s[s.start.dt.hour.between(8, 21)]
    s["night"] = s.start.dt.normalize()
    d = s.groupby(["pid", "night"]).agg(day_score=("sync_score", "mean"), n=("sync_score", "size"))
    return d[d.n >= MIN_DAY_WINDOWS].reset_index()


def main():
    clean = pd.read_parquet(PROCESSED / "clean.parquet")
    n = nights(clean)
    d = next_day_scores(pd.read_csv(RESULTS / "sync_scores.csv"))
    m = n.merge(d, on=["pid", "night"])
    m = m[m.groupby("pid").pid.transform("size") >= 4]      # need a few nights per person to compare
    print(f"{len(m)} night->day pairs from {m.pid.nunique()} people")

    rows = []
    for measure in ["wakeups_per_h", "pct_awake", "sleep_h"]:
        # within person: split each person's nights at their own median
        m["bad"] = m.groupby("pid")[measure].transform(
            lambda x: x < x.median() if measure == "sleep_h" else x > x.median())
        per = m.groupby(["pid", "bad"]).day_score.mean().unstack().dropna()
        diff = per[True] - per[False]
        # pooled within-person correlation
        z = m.groupby("pid")[[measure, "day_score"]].transform(lambda x: x - x.mean())
        rho = spearmanr(z[measure], z.day_score)
        rows.append({"sleep_measure": measure, "n_people": len(per),
                     "score_after_bad_nights": per[True].mean(), "score_after_good_nights": per[False].mean(),
                     "median_diff": diff.median(), "people_lower_after_bad": int((diff < 0).sum()),
                     "wilcoxon_p": wilcoxon(per[True], per[False]).pvalue,
                     "within_person_rho": rho.statistic, "rho_p": rho.pvalue})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "validation_sleep.csv", index=False)
    print(out.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 4))
    z = m.groupby("pid")[["wakeups_per_h", "day_score"]].transform(lambda x: x - x.mean())
    ax.scatter(z.wakeups_per_h, z.day_score, s=12, alpha=0.5, color="#3b5ba5")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("Wake-ups per hour last night (vs person's average)")
    ax.set_ylabel("Next-day sync score (vs person's average)")
    ax.set_title("Does a broken night lower next-day sync?")
    fig.tight_layout()
    fig.savefig(FIGURES / "sync_score" / "validation_sleep.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
