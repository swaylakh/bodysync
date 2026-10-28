"""BodySync score (final version): how far is your body from YOUR OWN normal right now?

For each window (30 min in ZU-PWD, 2 min in WESAD):
1. Take average heart rate, sweat (EDA, on a log scale) and skin temperature.
2. Compare each with this person's normal for the SAME situation (asleep / resting /
   active / other), using their median and typical spread (MAD). Exercising is
   compared with your own exercising, so a workout doesn't look like stress.
3. distance = root-mean-square of the 3 "how many typical spreads away" numbers.
4. score = 100 / (1 + distance / 2)
     at your normal -> 100, 2 spreads away -> 50, 6 away -> 25. Never hits 0,
     so big reactions can still be told apart.
The signal that is furthest off is reported as the reason for a low score.

The first attempt (sync_score.py) measured how signals move together; it failed
validation, see README.
"""
import numpy as np
import pandas as pd

SIGNALS = {"hr_mean": "heart rate", "log_eda": "sweat", "temp_mean": "skin temp"}
MIN_SPREAD = {"hr_mean": 2.0, "log_eda": 0.05, "temp_mean": 0.1}   # stops tiny spreads blowing up the score
HALF_SCORE_DISTANCE = 2.0
MIN_NORMAL_WINDOWS = 10


def add_log_eda(w: pd.DataFrame) -> pd.DataFrame:
    w = w.copy()
    w["log_eda"] = np.log(w.eda_mean.clip(lower=0) + 0.01)
    return w


# Stress direction: heart rate up, sweat up, skin temperature down (vasoconstriction).
STRESS_DIRECTION = {"hr_mean": 1, "log_eda": 1, "temp_mean": -1}


def compute(w: pd.DataFrame, context: pd.Series, normal: pd.Series, signals: list[str] | None = None,
            directional: bool = False) -> pd.DataFrame:
    """w: windows with SIGNALS columns and pid; context: situation per window;
    normal: True for windows that define each person's normal; signals: subset of SIGNALS to use.
    directional: count only deviations in the stress direction (being calmer than normal costs nothing)."""
    w = add_log_eda(w) if "log_eda" not in w else w.copy()
    w["context"] = context.to_numpy()
    cols = signals or list(SIGNALS)

    def mad(s):
        return (s - s.median()).abs().median() * 1.4826

    ref = w[normal.to_numpy()]
    by_ctx = ref.groupby(["pid", "context"])[cols]
    med, spr, n = by_ctx.median(), by_ctx.agg(mad), by_ctx.size()
    med, spr = med[n >= MIN_NORMAL_WINDOWS], spr[n >= MIN_NORMAL_WINDOWS]
    keys = pd.MultiIndex.from_frame(w[["pid", "context"]])
    m, s = med.reindex(keys).to_numpy(copy=True), spr.reindex(keys).to_numpy(copy=True)
    # situation too rare for this person -> fall back to their overall normal
    miss = np.isnan(m).any(axis=1)
    m[miss] = ref.groupby("pid")[cols].median().reindex(w.pid[miss]).to_numpy()
    s[miss] = ref.groupby("pid")[cols].agg(mad).reindex(w.pid[miss]).to_numpy()
    s = np.maximum(s, np.array([MIN_SPREAD[c] for c in cols]))

    z = (w[cols].to_numpy() - m) / s
    for i, c in enumerate(cols):
        w[f"z_{c}"] = z[:, i]
    z_raw = z.copy()
    if directional:
        z = np.clip(z * np.array([STRESS_DIRECTION[c] for c in cols]), 0, None)
    w["distance"] = np.sqrt(np.nanmean(z ** 2, axis=1))
    w["bodysync"] = 100 / (1 + w.distance / HALF_SCORE_DISTANCE)
    worst = np.nanargmax(np.abs(np.nan_to_num(z, nan=0)), axis=1)
    w["main_reason"] = [f"{SIGNALS[cols[i]]} {'high' if z_raw[k, i] > 0 else 'low'}" for k, i in enumerate(worst)]
    return w
