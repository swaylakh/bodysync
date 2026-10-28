"""EmpaticaE4Stress (Campanella et al.) -> BodySync window table.  LAB dataset.

Source: Mendeley Data, DOI 10.17632/kb42z77m2g.2 (CC-BY-4.0), file
Data_29_subjects.zip, extracted to data/external/e4stress/Subjects/subject_XX/.
Paper: "PPG and EDA dataset collected with Empatica E4 for stress assessment",
Data in Brief 2024 (PMC10847510). 29 subjects, Empatica E4, ~34-44 min each.

Protocol timing (paper Fig. 1 + Sec. 4.3; no tags/time-log file exists; the
authors' MATLAB scripts that split the phases are not distributed). "To create
a baseline, the bracelet was turned on and three minutes of rest were
recorded", so phases are anchored at the recording start:

    0-3 min    rest (baseline)                          -> baseline, normal=True
    3-13       Task 1: Lego, no instructions (10 min)   -> stress (task=lego_no_instructions)
    13-15      rest                                     -> unknown (task=rest1)
    15-20      Task 2: Lego with instructions (5 min)   -> stress (task=lego_instructions)
    20-22      rest                                     -> unknown (task=rest2)
    22-25      Task 3: Lego large pieces + countdown    -> stress (task=lego_countdown)
    25-27      rest                                     -> unknown (task=rest3)
    27-?       Task 4: math (511 - 13 ...), NO time limit
    then       rest 2 min, Task 5: 1-min CV talk, rest 2 min
Everything after 27 min is NOT labelled (dropped): Task 4 has no fixed
duration and nothing proves the recording ends exactly at the final rest, so
math / presentation / last two rests cannot be placed reliably.

Accelerometer check: hand movement starts at ~3 min in nearly every subject
and the Lego tasks / breaks are visible, but actual boundaries lag the nominal
ones by ~0-60 s (up to ~2 min in a few subjects, e.g. 26-28). The 2-min
breaks fit exactly one 2-min window each, and in those windows wrist movement
(median SD 0.06 g) is as high as in the Lego tasks (0.07 g; baseline 0.01 g):
in only 3-9 of 29 subjects is a break window clearly calmer than its task.
So the nominal break windows mostly contain task time; they are labelled
`unknown` (task=rest1/2/3 kept) instead of `rest`. Consequence: this dataset
has baseline + stress only, no trustworthy later-rest condition.

File quirks: headers were partly stripped. EDA.csv never has a header (starts
at the recording start, 4 Hz); HR.csv has a header only for subjects 18-29
(otherwise 1 Hz starting 10 s after the start, as E4 HR always does); TEMP/ACC
have headers except subject_01, whose files have no timestamp at all (ACC keeps
only its "32,32,32" rate row) -> subject_01 gets start = NaT. In 20 of the 29
EDA files values >= 1 uS carry a locale thousands separator ("1.038.145" =
1.038145 uS, "10.156.459" = 10.156459); these are repaired on read (checked
against neighbouring samples and subject_01's unmangled export).

Windows: 2 min sliding by 30 s on a grid from the recording start, 1-s
resolution; a window gets a phase's label if >= 90% of its seconds lie in that
phase, otherwise it is dropped. `start` is naive UTC.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "external", "e4stress", "Subjects")
OUT = os.path.join(ROOT, "data", "processed", "ext_e4stress_windows.parquet")
NAME = "e4stress"

WIN, STEP, FRAC, MINCOV = 120, 30, 0.9, 0.75
FS = {"HR": 1.0, "EDA": 4.0, "TEMP": 4.0, "ACC": 32.0}
# The 2-min breaks are NOT trusted (see docstring): their windows are kept as
# `unknown` (task=rest1..3). Set True to label them `rest` anyway.
TRUST_NOMINAL_RESTS = False
REST = "rest" if TRUST_NOMINAL_RESTS else "unknown"
HR_DELAY = 10  # s; E4 HR.csv starts 10 s after the other signals
PHASES = [  # (start s, end s, label, task) relative to recording start
    (0, 180, "baseline", "baseline"),
    (180, 780, "stress", "lego_no_instructions"),
    (780, 900, REST, "rest1"),
    (900, 1200, "stress", "lego_instructions"),
    (1200, 1320, REST, "rest2"),
    (1320, 1500, "stress", "lego_countdown"),
    (1500, 1620, REST, "rest3"),
]


def read_e4(path, kind):
    """Return (t0 or None, offset_s, samples) handling stripped headers."""
    raw = pd.read_csv(path, header=None, skipinitialspace=True, dtype=str)
    # Locale damage: EDA values >= 1 were written with a "." thousands
    # separator, e.g. 1.038145 -> "1.038.145", 10.156459 -> "10.156.459".
    raw = raw.replace(r"^\s*(\d+)\.(\d{3})\.(\d{3})\s*$", r"\1.\2\3", regex=True)
    x = raw.to_numpy(dtype=float)
    if x[0, 0] > 1e8:  # full header: start time + rate
        t0, fs = x[0, 0], x[1, 0]
        assert fs == FS[kind], (path, fs)
        return t0, x[2:]
    if kind == "ACC" and (x[0] == 32).all():  # rate row only
        x = x[1:]
    return None, x


def per_second(x, fs, offset, lo=None, hi=None, mag=False):
    """Mean / SD per relative second -> DataFrame indexed by second."""
    if mag:
        v = np.sqrt(((x / 64.0) ** 2).sum(axis=1))
    else:
        v = x[:, 0].copy()
        if lo is not None:
            v[v < lo] = np.nan
        if hi is not None:
            v[v > hi] = np.nan
    sec = np.floor(offset + np.arange(len(v)) / fs).astype(np.int64)
    ok = ~np.isnan(v)
    df = pd.DataFrame({"s": sec[ok], "v": v[ok]})
    df["v2"] = df["v"] ** 2
    return df.groupby("s").agg(sum=("v", "sum"), sq=("v2", "sum"), n=("v", "size"))


def subject(d):
    sid = d[-2:]
    t_temp, temp = read_e4(os.path.join(SRC, d, "TEMP.csv"), "TEMP")
    t_acc, acc = read_e4(os.path.join(SRC, d, "ACC.csv"), "ACC")
    t_hr, hr = read_e4(os.path.join(SRC, d, "HR.csv"), "HR")
    _, eda = read_e4(os.path.join(SRC, d, "EDA.csv"), "EDA")
    t0 = t_temp if t_temp is not None else t_acc
    if t_acc is not None and t0 is not None:
        assert t_acc == t0
    hr_off = (t_hr - t0) if (t_hr is not None and t0 is not None) else HR_DELAY

    sig = {
        "hr": per_second(hr, 1.0, hr_off, 35, 200),
        "eda": per_second(eda, 4.0, 0, 0, None),
        "temp": per_second(temp, 4.0, 0, 20, 42),
        "acc": per_second(acc, 32.0, 0, mag=True),
    }
    dur = int(min(len(eda) / 4, len(temp) / 4, len(acc) / 32))
    n = dur + 1
    arr = {}
    for k, df in sig.items():
        df = df[(df.index >= 0) & (df.index < n)]
        a = np.zeros((n, 3))
        a[df.index.to_numpy()] = df[["sum", "sq", "n"]].to_numpy()
        arr[k] = a
    have = np.ones(n, bool)
    for k in ("hr", "eda", "temp"):
        have &= arr[k][:, 2] > 0

    phase = np.full(n, -1)
    for i, (a, b, _, _) in enumerate(PHASES):
        phase[a:min(b, n)] = i

    rows = []
    for s in range(0, dur - WIN + 1, STEP):
        ph = phase[s:s + WIN]
        best = np.bincount(ph[ph >= 0], minlength=len(PHASES)) if (ph >= 0).any() else None
        if best is None or best.max() < FRAC * WIN:
            continue
        p = PHASES[int(best.argmax())]
        sl = slice(s, s + WIN)
        out = {}
        for k in ("hr", "eda", "temp"):
            tot = arr[k][sl].sum(axis=0)
            out[k] = tot[0] / tot[2] if tot[2] else np.nan
        tot = arr["acc"][sl].sum(axis=0)
        move = np.sqrt(max(tot[1] / tot[2] - (tot[0] / tot[2]) ** 2, 0)) if tot[2] else np.nan
        rows.append({
            "pid": f"{NAME}_{sid}",
            "start": pd.to_datetime(t0 + s, unit="s") if t0 is not None else pd.NaT,
            "label": p[2], "task": p[3], "normal": p[2] == "baseline",
            "hr_mean": out["hr"], "eda_mean": out["eda"], "temp_mean": out["temp"],
            "move": move, "coverage": have[sl].mean(), "t_rel_s": s,
        })
    return rows


def main():
    rows = []
    for d in sorted(os.listdir(SRC)):
        if d.startswith("subject_"):
            rows += subject(d)
    w = pd.DataFrame(rows)
    w["start"] = pd.to_datetime(w["start"])
    w = w[w.coverage >= MINCOV].reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    w.to_parquet(OUT, index=False)

    print(f"wrote {OUT}")
    print(f"people: {w.pid.nunique()}, windows: {len(w)}")
    g = w.groupby("label")
    print(pd.DataFrame({"people": g.pid.nunique(), "windows": g.size()}).to_string())
    print(g[["hr_mean", "eda_mean", "temp_mean", "move"]].median().round(3).to_string())
    print(w.groupby("task").size().to_string())


if __name__ == "__main__":
    sys.exit(main())
