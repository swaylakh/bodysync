"""Compute the final BodySync score on both datasets and check whether it means anything.

WESAD (lab, real stress labels):
  - normal = each person's baseline condition
  - does the score drop under stress? vs baseline AND vs amusement (amusement was not
    used to set the normal, so it is the fair comparison)
  - within each person, how well does the score separate stress from non-stress windows (AUC)?
ZU-PWD (everyday life, no stress labels):
  - normal = all of the person's own windows in that situation
  - exercise check: with situation-aware normals, active windows should look normal;
    with one normal for everything they would look like "stress"
  - sleep check: is the next-day score lower after nights with more wake-ups?
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.metrics import roc_auc_score

import bodysync
from config import FIGURES, PROCESSED, RESULTS, STATE_COLORS, STATES
from validate_sleep import nights

OUT = FIGURES / "bodysync"
OUT.mkdir(exist_ok=True)


def wesad():
    w = pd.read_parquet(PROCESSED / "wesad_windows.parquet")
    w = bodysync.compute(w, context=pd.Series("lab", index=w.index), normal=w.state == "baseline")
    per = w.groupby(["pid", "state"]).bodysync.mean().unstack()
    rows = []
    for ref, cond in [("baseline", "stress"), ("amusement", "stress"), ("meditation", "stress"),
                      ("baseline", "amusement"), ("baseline", "meditation")]:
        t = per[[ref, cond]].dropna()
        rows.append({"comparison": f"{cond} vs {ref}", "n_people": len(t), "mean_ref": t[ref].mean(),
                     "mean_condition": t[cond].mean(), "people_lower": int((t[cond] < t[ref]).sum()),
                     "p_value": wilcoxon(t[cond], t[ref]).pvalue})
    auc = w.groupby("pid").apply(lambda g: roc_auc_score(g.state == "stress", -g.bodysync), include_groups=False)
    rows.append({"comparison": "within-person AUC, stress vs all other windows", "n_people": len(auc),
                 "mean_condition": auc.mean(), "people_lower": int((auc > 0.5).sum())})
    reasons = w[w.state == "stress"].main_reason.value_counts(normalize=True).round(2).to_dict()
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "bodysync_wesad.csv", index=False)
    print("WESAD\n", res.round(4).to_string(index=False), "\nmain reason during stress:", reasons)

    order = ["baseline", "amusement", "meditation", "stress"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for _, r in per[order].iterrows():
        ax.plot(order, r.values, color="#bbb", lw=0.8)
    ax.plot(order, per[order].mean().values, color="#e07b39", lw=2.5, marker="o", label="average")
    ax.set_ylim(0, 100)
    ax.set_ylabel("BodySync score")
    ax.set_title("WESAD: BodySync by condition (grey = one person)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "wesad.png", dpi=150)
    plt.close(fig)
    w.to_csv(RESULTS / "bodysync_wesad_windows.csv", index=False)


def zupwd():
    w = pd.read_parquet(PROCESSED / "windows_30.parquet")
    w = w[w.coverage >= 0.75].reset_index(drop=True)
    ctx = pd.Series(np.where(w.purity >= 0.7, w.state, "other"), index=w.index)
    aware = bodysync.compute(w, context=ctx, normal=pd.Series(True, index=w.index))
    naive = bodysync.compute(w, context=pd.Series("all", index=w.index), normal=pd.Series(True, index=w.index))

    # exercise check
    ex = pd.DataFrame({
        "situation": aware.context,
        "situation_aware": aware.bodysync,
        "one_normal_for_all": naive.bodysync,
    }).groupby("situation").median().round(1)
    ex.to_csv(RESULTS / "bodysync_zupwd_exercise_check.csv")
    print("\nZU-PWD median score by situation\n", ex.to_string())

    # sleep check: next-day (08-22h) mean score vs last night's wake-ups, within person
    n = nights(pd.read_parquet(PROCESSED / "clean.parquet"))
    day = aware[aware.start.dt.hour.between(8, 21)].assign(night=lambda d: d.start.dt.normalize())
    day = day.groupby(["pid", "night"]).agg(score=("bodysync", "mean"), k=("bodysync", "size"))
    m = n.merge(day[day.k >= 8].reset_index(), on=["pid", "night"])
    m = m[m.groupby("pid").pid.transform("size") >= 4]
    rows = []
    for measure in ("wakeups_per_h", "pct_awake", "sleep_h"):
        bad = m.groupby("pid")[measure].transform(lambda x: x < x.median() if measure == "sleep_h" else x > x.median())
        per = m.assign(bad=bad).groupby(["pid", "bad"]).score.mean().unstack().dropna()
        rows.append({"sleep_measure": measure, "n_people": len(per), "after_bad": per[True].mean(),
                     "after_good": per[False].mean(), "people_lower_after_bad": int((per[True] < per[False]).sum()),
                     "p_value": wilcoxon(per[True], per[False]).pvalue})
    sl = pd.DataFrame(rows)
    sl.to_csv(RESULTS / "bodysync_zupwd_sleep_check.csv", index=False)
    print("\nZU-PWD sleep check\n", sl.round(3).to_string(index=False))

    recovery_check(w, ctx)

    cols = ["pid", "start", "context", "bodysync", "distance", "main_reason", "hr_mean", "eda_mean", "temp_mean"]
    aware[cols].to_csv(RESULTS / "bodysync_zupwd_scores.csv", index=False)
    plot_week(aware)


def recovery_check(w: pd.DataFrame, ctx: pd.Series):
    """Nights when BodySync is low should also look like poor recovery on two markers the
    score never sees: lower HRV (RMSSD) and faster breathing during sleep. Both are only
    measured when the wrist is still, so they exist almost only at night. Compared within
    person. HR is tied to HRV, so the check is repeated with a score built without HR."""
    clean = pd.read_parquet(PROCESSED / "clean.parquet")
    sl = clean[clean.sleep_stage == 101].assign(night=lambda d: (d.time + pd.Timedelta(hours=6)).dt.normalize())
    markers = sl.groupby(["pid", "night"]).agg(rmssd=("rmssd", "median"), rr=("rr", "median"),
                                              n_rmssd=("rmssd", "count"), n_rr=("rr", "count"))
    markers = markers[(markers.n_rmssd >= 60) & (markers.n_rr >= 60)]

    rows = []
    for name, signals in [("full score", None), ("score without HR", ["log_eda", "temp_mean"])]:
        s = bodysync.compute(w, context=ctx, normal=pd.Series(True, index=w.index), signals=signals)
        s = s[s.context == "sleep"].assign(night=lambda d: (d.start + pd.Timedelta(hours=6)).dt.normalize())
        night = s.groupby(["pid", "night"]).bodysync.mean().rename("score")
        m = markers.join(night, how="inner").reset_index()
        m = m[m.groupby("pid").pid.transform("size") >= 5]
        z = m.groupby("pid")[["score", "rmssd", "rr"]].transform(lambda x: x - x.mean())
        for marker, expect in [("rmssd", "+"), ("rr", "-")]:
            rho = spearmanr(z.score, z[marker])
            per = m.groupby("pid").apply(lambda g: spearmanr(g.score, g[marker]).statistic, include_groups=False).dropna()
            agree = int((per > 0).sum() if expect == "+" else (per < 0).sum())
            rows.append({"score": name, "marker": {"rmssd": "HRV (RMSSD)", "rr": "breathing rate"}[marker],
                         "expected_sign": expect, "nights": len(m), "n_people": m.pid.nunique(),
                         "within_person_rho": rho.statistic, "p_value": rho.pvalue,
                         "people_in_expected_direction": agree,
                         "p_people": wilcoxon(per).pvalue if len(per) > 5 else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "bodysync_zupwd_recovery_check.csv", index=False)
    print("\nZU-PWD recovery check (night score vs night HRV / breathing)\n", res.round(4).to_string(index=False))


def plot_week(w: pd.DataFrame):
    pid = w.pid.value_counts().index[1]
    g = w[w.pid == pid]
    week = g.start.dt.to_period("W").value_counts().index[0]
    g = g[g.start.dt.to_period("W") == week]
    fig, ax = plt.subplots(figsize=(13, 3.6))
    for ctx in [*STATES, "other"]:
        k = g[g.context == ctx]
        ax.scatter(k.start, k.bodysync, s=8, color=STATE_COLORS[ctx], label=ctx)
    ax.plot(g.start, g.bodysync.rolling(4, min_periods=1).mean(), color="#222", lw=0.8)
    low = g.nsmallest(3, "bodysync")
    for _, r in low.iterrows():
        ax.annotate(r.main_reason, (r.start, r.bodysync), xytext=(0, -14), textcoords="offset points",
                    ha="center", fontsize=7, color="#b00")
    ax.set_ylim(0, 102)
    ax.set_ylabel("BodySync score")
    ax.set_title(f"Participant {pid[-3:]}: one week of BodySync (100 = exactly your normal for that situation)")
    ax.legend(ncol=5, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / f"week_participant_{pid[-3:]}.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    wesad()
    zupwd()
