"""Wearable Device Dataset from Induced Stress and Structured Exercise Sessions
(Hongn et al., PhysioNet 1.0.1, CC-BY-4.0) -> BodySync window table.  LAB dataset.

Source: https://physionet.org/content/wearable-device-dataset/1.0.1/ (files taken
from the PhysioNet open-data S3 mirror, same paths as the zip), extracted to
data/external/stressexercise/Wearable_Dataset/{STRESS,AEROBIC,ANAEROBIC}/<id>/.
BVP.csv / IBI.csv were not downloaded (HR.csv is used).

People. subject-info.csv has ONE row per participant id with a column per
session (Stress / Aerobic / Anaerobic) -> the same id is the same person
across sessions, so pid = stressexercise_<id> (S01..S18 = protocol V1,
f01..f18 = protocol V2; S and f ids are different people). Normal (baseline
windows) is therefore pooled over that person's sessions. `session` column
keeps STRESS / AEROBIC / ANAEROBIC.

Phase mapping. Tags = E4 button presses (tags.csv). Mapping from the
protocol figures + axvspan code in the dataset's Wearable_Dataset.ipynb
(figure circle k = k-th tag; T[i] below = i-th tag, 0-based, s from start):

STRESS V1 (13 tags): baseline T0-T1 | stroop T2-T3 (stress) | rest1 T3-T4 |
    TMCT T4-T5 (stress) | rest2 T5-T6 | real opinion, opposite opinion,
    subtract-13 T6-T11 (stress; incl. the few-s self-rating gaps) | T12 = end.
STRESS V2 (9 tags): baseline = 270 s before T0 (no start tag; 4:30 as in the
    V2 exercise protocols) | TMCT T1-T2 (stress) | rest1 T2-T3 (relaxing
    video) | opinions T3-T6 (stress) | rest2 T6-T7 | subtract T7-T8 (stress).
AEROBIC V1 (12 tags): warm-up 0-T0 (unloaded pedalling, NOT a rest -> no
    baseline in V1 exercise sessions) | 60..110 rpm stages T0-T10 (aerobic) |
    cool-down T10-T11 (unknown) | final still rest T11-end (recovery).
AEROBIC V2 (8 tags): baseline 4:30 + warm-up 2:15 precede T0 with no tag in
    between -> baseline = [T0-135-270, T0-135-15] (estimated, task
    baseline_est) | 70..90/95 rpm T0-T5 (aerobic; f07 has T5<T4 -> T0-T4) |
    cool-down T5-T6 unknown | rest T6-T7 (recovery).
ANAEROBIC V1 (7 tags): warm-up 0-T0 unknown | sprints (T0,T1),(T2,T3),(T4,T5)
    | anaerobic = first sprint start .. last sprint end (sprints + the 4-min
    unloaded recoveries between them; sprints are only 30 s, so no 2-min
    window can be 90% sprint) | last cool-down unknown | T6-end recovery.
ANAEROBIC V2 (11 tags): baseline T0-270..T0 | warm-up T0-T1 unknown |
    sprints (T1,T2)..(T7,T8) -> anaerobic T1-T8 | cool-down T8-T9 unknown |
    rest T9-T10 recovery.
Post-exercise rest is labelled `recovery` (not `rest`) so it never enters the
calm group of multi_validate. `sprint_s` = seconds of actual sprint in a window.

Special cases (data_constraints.txt): STRESS S02 duplicated data -> cut at
1548 s (EDA/TEMP row 6195, ACC row 49545); STRESS f07 dock left on (HR/TEMP
invalid) -> excluded; STRESS f14_a = baseline only (last 270 s), f14_b = rest
of protocol (standard V2 rule); AEROBIC S03 (7 tags, stopped at 90 rpm):
aerobic T0-T6, rest unknown; S07 (9 tags, stopped at 95 rpm): aerobic T0-T7,
cool-down T7-T8, recovery T8-end; S11_a: aerobic T0-end; S11_b: aerobic
0-T0, rest unknown; ANAEROBIC S06 / S16_a / S16_b: sprints from tag pairs, an
odd last tag = rest start.

Windows: 2 min sliding 30 s from each file's start, 1-s resolution; label if
>= 90% of seconds in one phase, else dropped; coverage >= 0.75. start = naive UTC.
Run from project root: python src/datasets/stressexercise.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "external", "stressexercise", "Wearable_Dataset")
OUT = os.path.join(ROOT, "data", "processed", "ext_stressexercise_windows.parquet")
NAME = "stressexercise"
WIN, STEP, FRAC, MINCOV = 120, 30, 0.9, 0.75
V2_BASE = 270
EXCLUDE = {("STRESS", "f07")}
CUT = {("STRESS", "S02"): 1548}


def read_e4(path):
    """-> (start Timestamp, fs, samples ndarray)."""
    with open(path) as f:
        t0 = pd.Timestamp(f.readline().split(",")[0].strip())
    x = pd.read_csv(path, header=None, skiprows=1).to_numpy(dtype=float)
    return t0, x[0, 0], x[1:]


def read_tags(path, t0):
    try:
        t = pd.read_csv(path, header=None)[0]
    except pd.errors.EmptyDataError:
        return []
    return [(pd.Timestamp(x) - t0).total_seconds() for x in t]


def per_second(v, fs, offset, n):
    sec = np.floor(offset + np.arange(len(v)) / fs).astype(np.int64)
    ok = ~np.isnan(v) & (sec >= 0) & (sec < n)
    a = np.zeros((n, 3))
    np.add.at(a, (sec[ok], 0), v[ok])
    np.add.at(a, (sec[ok], 1), v[ok] ** 2)
    np.add.at(a, (sec[ok], 2), 1)
    return a


def phases(session, sid, T, end):
    """List of (start, end, label, task) in s; plus list of sprint intervals."""
    P, sprints = [], []
    v2 = sid.startswith("f")
    n = len(T)
    if session == "STRESS":
        if sid == "f14_a":
            P.append((end - V2_BASE, end, "baseline", "baseline"))
        elif not v2:
            P += [(T[0], T[1], "baseline", "baseline"), (T[2], T[3], "stress", "stroop"),
                  (T[3], T[4], "rest", "rest1"), (T[4], T[5], "stress", "tmct"),
                  (T[5], T[6], "rest", "rest2"), (T[6], T[11], "stress", "speech_math")]
        else:
            P += [(T[0] - V2_BASE, T[0], "baseline", "baseline"), (T[1], T[2], "stress", "tmct"),
                  (T[2], T[3], "rest", "rest1"), (T[3], T[6], "stress", "opinions"),
                  (T[6], T[7], "rest", "rest2"), (T[7], T[8], "stress", "subtract")]
    elif session == "AEROBIC":
        if v2:
            P.append((T[0] - 135 - V2_BASE, T[0] - 135 - 15, "baseline", "baseline_est"))
            P.append((T[0], T[5] if T[5] > T[4] else T[4], "aerobic", "cycling"))
            P.append((T[6], T[7], "recovery", "post_rest"))
        elif sid == "S11_a":
            P.append((T[0], end, "aerobic", "cycling"))
        elif sid == "S11_b":
            P.append((0, T[0], "aerobic", "cycling"))
        elif sid == "S03":
            P.append((T[0], T[6], "aerobic", "cycling"))
        elif sid == "S07":
            P += [(T[0], T[7], "aerobic", "cycling"), (T[8], end, "recovery", "post_rest")]
        else:
            assert n == 12, (session, sid, n)
            P += [(T[0], T[10], "aerobic", "cycling"), (T[11], end, "recovery", "post_rest")]
    else:  # ANAEROBIC
        if v2:
            assert n == 11, (session, sid, n)
            P.append((T[0] - V2_BASE, T[0], "baseline", "baseline"))
            sprints = [(T[i], T[i + 1]) for i in (1, 3, 5, 7)]
            P.append((T[1], T[8], "anaerobic", "sprint_intervals"))
            P.append((T[9], T[10], "recovery", "post_rest"))
        else:
            sprints = [(T[i], T[i + 1]) for i in range(0, n - 1, 2)]
            P.append((sprints[0][0], sprints[-1][1], "anaerobic", "sprint_intervals"))
            if n % 2 == 1:
                P.append((T[-1], end, "recovery", "post_rest"))
    return P, sprints


def session_rows(session, sid):
    d = os.path.join(SRC, session, sid)
    t0, _, eda = read_e4(os.path.join(d, "EDA.csv"))
    tt, _, temp = read_e4(os.path.join(d, "TEMP.csv"))
    th, _, hr = read_e4(os.path.join(d, "HR.csv"))
    ta, _, acc = read_e4(os.path.join(d, "ACC.csv"))
    off = lambda t: (t - t0).total_seconds()
    dur = int(min(len(eda) / 4, len(temp) / 4 + off(tt), len(acc) / 32 + off(ta)))
    dur = min(dur, CUT.get((session, sid), dur))
    n = dur
    h = hr[:, 0].copy(); h[(h < 35) | (h > 200)] = np.nan
    e = eda[:, 0].copy(); e[e < 0] = np.nan
    tp = temp[:, 0].copy(); tp[(tp < 20) | (tp > 42)] = np.nan
    mag = np.sqrt(((acc / 64.0) ** 2).sum(axis=1))
    arr = {"hr": per_second(h, 1.0, off(th), n), "eda": per_second(e, 4.0, 0, n),
           "temp": per_second(tp, 4.0, off(tt), n), "acc": per_second(mag, 32.0, off(ta), n)}
    have = (arr["hr"][:, 2] > 0) & (arr["eda"][:, 2] > 0) & (arr["temp"][:, 2] > 0)

    T = read_tags(os.path.join(d, "tags.csv"), t0)
    P, sprints = phases(session, sid, T, dur)
    phase = np.full(n, -1)
    for i, (a, b, _, _) in enumerate(P):
        a, b = max(int(round(a)), 0), min(int(round(b)), n)
        if b > a:
            phase[a:b] = i
    spr = np.zeros(n, bool)
    for a, b in sprints:
        spr[max(int(round(a)), 0):min(int(round(b)), n)] = True

    pid = f"{NAME}_{sid.split('_')[0]}"
    rows = []
    for s in range(0, n - WIN + 1, STEP):
        ph = phase[s:s + WIN]
        if not (ph >= 0).any():
            continue
        cnt = np.bincount(ph[ph >= 0], minlength=len(P))
        if cnt.max() < FRAC * WIN:
            continue
        p = P[int(cnt.argmax())]
        sl = slice(s, s + WIN)
        out = {}
        for k in ("hr", "eda", "temp"):
            tot = arr[k][sl].sum(axis=0)
            out[k] = tot[0] / tot[2] if tot[2] else np.nan
        tot = arr["acc"][sl].sum(axis=0)
        move = np.sqrt(max(tot[1] / tot[2] - (tot[0] / tot[2]) ** 2, 0)) if tot[2] else np.nan
        rows.append({
            "pid": pid, "start": t0 + pd.Timedelta(seconds=s), "label": p[2],
            "normal": p[2] == "baseline", "hr_mean": out["hr"], "eda_mean": out["eda"],
            "temp_mean": out["temp"], "move": move, "coverage": have[sl].mean(),
            "session": session, "file": sid, "task": p[3], "t_rel_s": s,
            "sprint_s": int(spr[sl].sum()),
        })
    return rows


def main():
    rows = []
    for session in ("STRESS", "AEROBIC", "ANAEROBIC"):
        for sid in sorted(os.listdir(os.path.join(SRC, session))):
            if (session, sid) in EXCLUDE:
                continue
            rows += session_rows(session, sid)
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
    print(w.groupby(["session", "label"]).agg(people=("pid", "nunique"), windows=("pid", "size")).to_string())
    print(w.groupby("task").size().to_string())


if __name__ == "__main__":
    sys.exit(main())
