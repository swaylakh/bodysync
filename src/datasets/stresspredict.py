"""Stress-Predict dataset (Iqbal et al.) -> BodySync window table.  LAB dataset.

Source: https://github.com/italha-d/Stress-Predict-Dataset (zip of `main`),
extracted to data/external/stresspredict/Stress-Predict-Dataset-main/.
35 participants, Empatica E4, ~60-min lab protocol.

Phase boundaries come from Processed_data/Time_logs.xlsx (minute resolution,
Irish time = UTC in Feb/Mar 2022; times are written 12-hour without AM/PM, so
hours < 8 are taken as PM). The log's column headers are shifted relative to
the data; by duration and by matching the E4 button tags (agree within ~1 min)
the columns are, in chronological order:

    cols 6-7   baseline        (10 min, seated, questionnaires)  -> baseline, normal=True
    cols 8-9   Stroop test     (~5 min)                          -> stress  (task=stroop)
    cols 10-11 relax           (5 min)                           -> rest
    cols 12-13 interview       (~10 min)                         -> stress  (task=interview)
    cols 14-15 relax           (5 min)                           -> rest
    cols 16-17 hyperventilation(2 min)                           -> hyperventilation (physical, not psychological)
    cols 18-19 relax           (5 min)                           -> rest
    cols 20-21 questionnaire   (~5 min)                          -> questionnaire
    cols 22-23 relax           (5-10 min)                        -> rest
(Consent, cols 4-5, is left unlabelled.)  `task` holds the specific phase name.
S01 is irregular: its logged baseline (09:39-09:43) lies after Stroop+relax.

Windows: 2 min sliding by 30 s on a grid of epoch multiples of 30 s, 1-s
resolution; a window gets a phase's label if >= 90% of its seconds lie in that
phase, otherwise it is dropped. `start` is naive UTC.
Signals: HR.csv (1 Hz), EDA.csv / TEMP.csv (4 Hz), ACC.csv (32 Hz, 1/64 g).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "external", "stresspredict", "Stress-Predict-Dataset-main")
OUT = os.path.join(ROOT, "data", "processed", "ext_stresspredict_windows.parquet")
NAME = "stresspredict"

WIN, STEP, FRAC, MINCOV = 120, 30, 0.9, 0.75
PHASES = [  # (start col, label, task)
    (6, "baseline", "baseline"),
    (8, "stress", "stroop"),
    (10, "rest", "relax1"),
    (12, "stress", "interview"),
    (14, "rest", "relax2"),
    (16, "hyperventilation", "hyperventilation"),
    (18, "rest", "relax3"),
    (20, "questionnaire", "questionnaire"),
    (22, "rest", "relax4"),
]


def read_e4(path):
    raw = pd.read_csv(path, header=None)
    t0, fs = float(raw.iloc[0, 0]), float(raw.iloc[1, 0])
    return t0, fs, raw.iloc[2:].to_numpy(dtype=float)


def per_second(path, lo=None, hi=None, mag=False):
    """Return {epoch_second: (sum, sumsq, n)} arrays after validity masking."""
    t0, fs, x = read_e4(path)
    if mag:
        v = np.sqrt(((x / 64.0) ** 2).sum(axis=1))
    else:
        v = x[:, 0]
        bad = np.zeros(len(v), bool)
        if lo is not None:
            bad |= v < lo
        if hi is not None:
            bad |= v > hi
        v = np.where(bad, np.nan, v)
    sec = np.floor(t0 + np.arange(len(v)) / fs).astype(np.int64)
    ok = ~np.isnan(v)
    df = pd.DataFrame({"s": sec[ok], "v": v[ok]})
    df["v2"] = df["v"] ** 2
    return df.groupby("s").agg(sum=("v", "sum"), sq=("v2", "sum"), n=("v", "size"))


def phase_table(sid, row):
    date = pd.Timestamp(row[25])

    def ts(v):
        h = v.hour + (12 if v.hour < 8 else 0)
        return int((date + pd.Timedelta(hours=h, minutes=v.minute)).timestamp())

    return [(ts(row[c]), ts(row[c + 1]), lab, task) for c, lab, task in PHASES]


def subject(sid, phases):
    d = os.path.join(SRC, "Raw_data", sid)
    hr = per_second(os.path.join(d, "HR.csv"), 35, 200)
    eda = per_second(os.path.join(d, "EDA.csv"), 0, None)
    tmp = per_second(os.path.join(d, "TEMP.csv"), 20, 42)
    acc = per_second(os.path.join(d, "ACC.csv"), mag=True)

    s0 = min(x.index.min() for x in (hr, eda, tmp))
    s1 = max(x.index.max() for x in (hr, eda, tmp)) + 1
    grid = np.arange(s0, s1)
    n = len(grid)

    def col(df, c):
        return df[c].reindex(grid, fill_value=0).to_numpy(float)

    def csum(a):
        return np.concatenate([[0.0], np.cumsum(a)])

    # per-second means for HR/EDA/TEMP, then window mean of available seconds
    sig = {}
    have = np.ones(n, bool)
    for k, df in (("hr", hr), ("eda", eda), ("temp", tmp)):
        s, c = col(df, "sum"), col(df, "n")
        m = np.divide(s, c, out=np.zeros(n), where=c > 0)
        sig[k] = (csum(m), csum(c > 0))
        have &= c > 0
    cov_c = csum(have)
    a_sum, a_sq, a_n = csum(col(acc, "sum")), csum(col(acc, "sq")), csum(col(acc, "n"))

    # per-second phase membership
    lab_idx = np.full(n, -1)
    for i, (a, b, _, _) in enumerate(phases):
        lab_idx[(grid >= a) & (grid < b)] = i
    onehot = [csum(lab_idx == i) for i in range(len(phases))]

    first = int(np.ceil(s0 / STEP) * STEP)
    rows = []
    for ws in range(first, s1 - WIN + 1, STEP):
        i, j = ws - s0, ws - s0 + WIN
        frac = [(o[j] - o[i]) / WIN for o in onehot]
        k = int(np.argmax(frac))
        if frac[k] < FRAC:
            continue
        cov = (cov_c[j] - cov_c[i]) / WIN
        if cov < MINCOV:
            continue
        r = {"pid": f"{NAME}_{sid}", "start": pd.Timestamp(ws, unit="s"),
             "label": phases[k][2], "task": phases[k][3],
             "normal": phases[k][2] == "baseline", "coverage": cov}
        for key, (cs, cn) in sig.items():
            nn = cn[j] - cn[i]
            r[f"{key}_mean"] = (cs[j] - cs[i]) / nn if nn else np.nan
        an = a_n[j] - a_n[i]
        if an > 1:
            mu = (a_sum[j] - a_sum[i]) / an
            var = max((a_sq[j] - a_sq[i]) / an - mu * mu, 0.0) * an / (an - 1)
            r["move"] = np.sqrt(var)
        else:
            r["move"] = np.nan
        rows.append(r)
    return rows


def main():
    log = pd.read_excel(os.path.join(SRC, "Processed_data", "Time_logs.xlsx"), header=None).iloc[2:]
    rows = []
    for _, row in log.iterrows():
        sid = str(row[0]).strip()
        if not os.path.isdir(os.path.join(SRC, "Raw_data", sid)):
            print("missing", sid)
            continue
        rows += subject(sid, phase_table(sid, row))
    df = pd.DataFrame(rows)[["pid", "start", "label", "task", "normal", "hr_mean", "eda_mean",
                             "temp_mean", "move", "coverage"]]
    df["start"] = df["start"].astype("datetime64[ns]")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"{NAME}: {df.pid.nunique()} people, {len(df)} windows -> {OUT}")
    g = df.groupby("label")
    print(pd.DataFrame({"people": g.pid.nunique(), "windows": g.size(),
                        "hr": g.hr_mean.median(), "eda": g.eda_mean.median(),
                        "temp": g.temp_mean.median(), "move": g.move.median()}).round(3).to_string())
    print(df.groupby("task").agg(people=("pid", "nunique"), windows=("pid", "size"),
                                 hr=("hr_mean", "median"), eda=("eda_mean", "median")).round(2).to_string())


if __name__ == "__main__":
    sys.exit(main())
