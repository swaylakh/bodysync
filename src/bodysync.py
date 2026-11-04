"""BodySync score: how far is your body from YOUR OWN normal right now, in the stressed direction?

For each window:
1. Take average heart rate, sweat (EDA, on a log scale) and skin temperature.
2. Compare each with this person's normal for the SAME situation (e.g. asleep / resting /
   active), using their median and typical spread (MAD). Exercising is compared with your
   own exercising, so a workout doesn't look like stress.
3. Count only changes in the stress direction: heart rate up, sweat up, skin temp down.
   Being calmer than normal costs nothing.
4. distance = root-mean-square of the 3 "how many typical spreads away" numbers.
5. score = 100 / (1 + distance / 2)
     at your normal -> 100, 2 spreads away -> 50, 6 away -> 25. Never hits 0.
The signal that is furthest off is reported as the reason for a low score.
"""
import numpy as np
import pandas as pd

SIGNALS = {"hr_mean": "heart rate", "log_eda": "sweat", "temp_mean": "skin temp"}
STRESS_DIRECTION = {"hr_mean": 1, "log_eda": 1, "temp_mean": -1}
MIN_SPREAD = {"hr_mean": 2.0, "log_eda": 0.05, "temp_mean": 0.1}   # stops tiny spreads blowing up the score
HALF_SCORE_DISTANCE = 2.0
MIN_NORMAL_WINDOWS = 10


def mad(s: pd.Series) -> float:
    return (s - s.median()).abs().median() * 1.4826


def compute(w: pd.DataFrame, context: pd.Series, normal: pd.Series) -> pd.DataFrame:
    """w: windows with pid, hr_mean, eda_mean, temp_mean; context: situation per window;
    normal: True for windows that define each person's normal."""
    w = w.copy()
    w["log_eda"] = np.log(w.eda_mean.clip(lower=0) + 0.01)
    w["context"] = context.to_numpy()
    cols = list(SIGNALS)

    ref = w[normal.to_numpy()]
    by_ctx = ref.groupby(["pid", "context"])[cols]
    enough = by_ctx.size() >= MIN_NORMAL_WINDOWS
    med, spr = by_ctx.median()[enough], by_ctx.agg(mad)[enough]
    keys = pd.MultiIndex.from_frame(w[["pid", "context"]])
    m, s = med.reindex(keys).to_numpy(copy=True), spr.reindex(keys).to_numpy(copy=True)
    # situation too rare for this person -> fall back to their overall normal
    miss = np.isnan(m).any(axis=1)
    m[miss] = ref.groupby("pid")[cols].median().reindex(w.pid[miss]).to_numpy()
    s[miss] = ref.groupby("pid")[cols].agg(mad).reindex(w.pid[miss]).to_numpy()
    s = np.maximum(s, np.array([MIN_SPREAD[c] for c in cols]))

    z = (w[cols].to_numpy() - m) / s
    stress_z = np.clip(z * np.array([STRESS_DIRECTION[c] for c in cols]), 0, None)
    w["distance"] = np.sqrt(np.nanmean(stress_z ** 2, axis=1))
    w["bodysync"] = 100 / (1 + w.distance / HALF_SCORE_DISTANCE)
    worst = np.nanargmax(np.nan_to_num(stress_z, nan=0), axis=1)
    w["main_reason"] = [f"{SIGNALS[cols[i]]} {'high' if z[k, i] > 0 else 'low'}" for k, i in enumerate(worst)]
    return w
