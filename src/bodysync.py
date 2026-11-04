"""The two candidate scores, built the same way so they can be compared fairly.

Score B, BodySync (personal normal): how far are your LEVELS from your own normal right now?
  signals: average heart rate, sweat (EDA, log scale), skin temperature;
  only the stress direction counts (heart rate up, sweat up, skin temp down).
Score A, sync (the original idea): do your signals MOVE TOGETHER the way they usually do?
  signals: Pearson r of HR-EDA, HR-temp and EDA-temp inside the window;
  any change counts, in either direction.

Both, per window:
1. Compare each signal with this person's normal for the SAME situation (asleep / resting /
   active, or still / moving), using their median and typical spread (MAD). Exercise is
   compared with the person's own exercise, so a workout doesn't look like stress.
2. distance = root-mean-square of the "how many typical spreads away" numbers.
3. score = 100 / (1 + distance / 2): at your normal -> 100, 2 spreads away -> 50, 6 away -> 25.
The signal that is furthest off is reported as the reason for a low score.
"""
import numpy as np
import pandas as pd

LEVELS = {"hr_mean": "heart rate", "log_eda": "sweat", "temp_mean": "skin temp"}
STRESS_DIRECTION = {"hr_mean": 1, "log_eda": 1, "temp_mean": -1}
LEVEL_MIN_SPREAD = {"hr_mean": 2.0, "log_eda": 0.05, "temp_mean": 0.1}   # stops tiny spreads blowing up the score
SYNC = {"r_hr_eda": "HR-sweat sync", "r_hr_temp": "HR-temp sync", "r_eda_temp": "sweat-temp sync"}
SYNC_MIN_SPREAD = 0.05
HALF_SCORE_DISTANCE = 2.0
MIN_NORMAL_WINDOWS = 10


def mad(s: pd.Series) -> float:
    return (s - s.median()).abs().median() * 1.4826


def _score(w: pd.DataFrame, names: dict, min_spread: np.ndarray, direction: np.ndarray | None,
           context: pd.Series, normal: pd.Series, out: str) -> pd.DataFrame:
    cols = list(names)
    w = w.copy()
    w["context"] = context.to_numpy()
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
    s = np.maximum(s, min_spread)

    z = (w[cols].to_numpy() - m) / s
    off = np.clip(z * direction, 0, None) if direction is not None else np.abs(z)
    w[out] = 100 / (1 + np.sqrt(np.nanmean(off ** 2, axis=1)) / HALF_SCORE_DISTANCE)
    worst = np.nanargmax(np.nan_to_num(off, nan=0), axis=1)
    w[f"{out}_reason"] = [f"{names[cols[i]]} {'high' if z[k, i] > 0 else 'low'}" for k, i in enumerate(worst)]
    return w


def compute(w: pd.DataFrame, context: pd.Series, normal: pd.Series) -> pd.DataFrame:
    """Add score B (`bodysync`) and, if the window has the r_* columns, score A (`sync`)."""
    w = w.copy()
    w["log_eda"] = np.log(w.eda_mean.clip(lower=0) + 0.01)
    w = _score(w, LEVELS, np.array(list(LEVEL_MIN_SPREAD.values())), np.array(list(STRESS_DIRECTION.values())),
               context, normal, "bodysync")
    if set(SYNC) <= set(w.columns):
        w = _score(w, SYNC, np.full(len(SYNC), SYNC_MIN_SPREAD), None, context, normal, "sync")
    return w
