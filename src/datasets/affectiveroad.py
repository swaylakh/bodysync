"""AffectiveROAD (El Haouij et al., ACM SAC 2018; MIT Media Lab) -> BodySync window table.
Real driving on real roads, Empatica E4 on both wrists.  Fixed-route protocol, so the LAB window format.

Source: https://www.media.mit.edu/groups/affective-computing/data/ (AffectiveROAD_Data.zip,
research use), extracted to data/external/affectiveroad/AffectiveROAD_Data/Database/E4/<k>-E4-Drv<k>/.

Route of every drive (Annot_E4_<wrist>.csv gives the segment borders as 4 Hz sample indices):
  rest (15 min, parked) | Z (car park) | city1 | highway | city2 | city2 back | highway back |
  city1 back | Z | rest (15 min, parked)
Labels (fixed before looking at any results):
  first rest            -> baseline (sets the person's normal)
  city segments         -> stress   (the high-stress part of the route by design, as in Healey & Picard 2005;
                                     the dataset's own subjective stress metric is checked against this below)
  highway segments      -> highway  (medium stress: not used as stress or calm in the main test)
  car park (Z)          -> unknown
  last rest             -> rest     (the later rest = the calm comparison, as in every lab dataset)
Wrist: left for every drive, except Drv2, whose left recording is split into two files -> right wrist.
People: Records.txt says Drv1/Drv8/Drv12 = driver D1, Drv2/Drv6 = D2, Drv10/Drv13 = D8; every other drive is
its own driver. pid = affectiveroad_<driver>, so a driver's drives are never split between training and test.
Windows: 2 min sliding 30 s, labelled if >= 90% in one segment; coverage >= 0.75. start = naive UTC.
Run from project root: python -E -P src/datasets/affectiveroad.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from richer import eda_bursts, eda_phasic_tonic, rmssd, slope_per_min  # noqa: E402
from stressexercise import corr, per_second  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB = os.path.join(ROOT, "data", "external", "affectiveroad", "AffectiveROAD_Data", "Database")
OUT = os.path.join(ROOT, "data", "processed", "ext_affectiveroad_windows.parquet")
NAME = "affectiveroad"
WIN, STEP, FRAC, MINCOV = 120, 30, 0.9, 0.75
DRIVER = {"Drv1": "D1", "Drv8": "D1", "Drv12": "D1", "Drv2": "D2", "Drv6": "D2", "Drv10": "D8", "Drv13": "D8"}
WRIST = {"Drv2": "Right"}
SEGMENTS = ["baseline", "unknown", "stress", "highway", "stress", "stress", "highway", "stress", "unknown", "rest"]
TASKS = ["rest_start", "park", "city1", "hwy", "city2", "city2_back", "hwy_back", "city1_back", "park_back", "rest_end"]


def read_e4(path):
    """Empatica CSV (first row = unix start time) -> (start Timestamp, fs, samples)."""
    with open(path) as f:
        t0 = pd.Timestamp(float(f.readline().split(",")[0]), unit="s")
    x = pd.read_csv(path, header=None, skiprows=1).to_numpy(dtype=float)
    return t0, x[0, 0], x[1:]


def read_ibi(path, t0):
    """IBI.csv -> (beat times in s from t0, IBI in s)."""
    with open(path) as f:
        ti = pd.Timestamp(float(f.readline().split(",")[0]), unit="s")
    x = pd.read_csv(path, header=None, skiprows=1).to_numpy(dtype=float).reshape(-1, 2)
    return (ti - t0).total_seconds() + x[:, 0], x[:, 1]


def annotations(wrist: str) -> dict:
    a = pd.read_csv(os.path.join(DB, "E4", f"Annot_E4_{wrist}.csv"), header=None, skiprows=1)
    return {r[0]: [(r[1 + 2 * i] / 4, r[2 + 2 * i] / 4) for i in range(10)] for r in a.itertuples(index=False)}


def drive_rows(drive: str, bounds: list) -> list:
    k = drive.removeprefix("Drv")
    d = os.path.join(DB, "E4", f"{k}-E4-{drive}", WRIST.get(drive, "Left"))
    t0, _, eda = read_e4(os.path.join(d, "EDA.csv"))
    tt, _, temp = read_e4(os.path.join(d, "TEMP.csv"))
    th, _, hr = read_e4(os.path.join(d, "HR.csv"))
    ta, _, acc = read_e4(os.path.join(d, "ACC.csv"))
    off = lambda t: (t - t0).total_seconds()
    n = int(min(len(eda) / 4, len(temp) / 4 + off(tt), len(acc) / 32 + off(ta)))
    h = hr[:, 0].copy(); h[(h < 35) | (h > 200)] = np.nan
    e = eda[:, 0].copy(); e[e < 0] = np.nan
    tp = temp[:, 0].copy(); tp[(tp < 20) | (tp > 42)] = np.nan
    e_time = np.arange(len(e)) / 4.0
    b_time, b_ibi = read_ibi(os.path.join(d, "IBI.csv"), t0)
    mag = np.sqrt(((acc / 64.0) ** 2).sum(axis=1))
    arr = {"hr": per_second(h, 1.0, off(th), n), "eda": per_second(e, 4.0, 0, n),
           "temp": per_second(tp, 4.0, off(tt), n), "acc": per_second(mag, 32.0, off(ta), n)}
    have = (arr["hr"][:, 2] > 0) & (arr["eda"][:, 2] > 0) & (arr["temp"][:, 2] > 0)
    sec = {k: np.where(arr[k][:, 2] > 0, arr[k][:, 0] / np.maximum(arr[k][:, 2], 1), np.nan)
           for k in ("hr", "eda", "temp")}
    phase = np.full(n, -1)
    for i, (a, b) in enumerate(bounds):
        a, b = max(int(round(a)), 0), min(int(round(b)), n)
        if b > a:
            phase[a:b] = i
    pid = f"{NAME}_{DRIVER.get(drive, drive)}"
    rows = []
    for s in range(0, n - WIN + 1, STEP):
        ph = phase[s:s + WIN]
        if not (ph >= 0).any():
            continue
        cnt = np.bincount(ph[ph >= 0], minlength=len(bounds))
        if cnt.max() < FRAC * WIN:
            continue
        i = int(cnt.argmax())
        sl = slice(s, s + WIN)
        out = {}
        for k in ("hr", "eda", "temp"):
            tot = arr[k][sl].sum(axis=0)
            out[k] = tot[0] / tot[2] if tot[2] else np.nan
        tot = arr["acc"][sl].sum(axis=0)
        move = np.sqrt(max(tot[1] / tot[2] - (tot[0] / tot[2]) ** 2, 0)) if tot[2] else np.nan
        for x, y in (("hr", "eda"), ("hr", "temp"), ("eda", "temp")):
            out[f"r_{x}_{y}"] = corr(sec[x][sl], sec[y][sl], WIN)
        e_win = e[(e_time >= s) & (e_time < s + WIN)]
        scr_rate, scr_amp = eda_bursts(e_win, 4)
        eda_phasic, eda_tonic_slope = eda_phasic_tonic(e_win, 4)
        t_win = np.arange(WIN, dtype=float)
        rows.append({
            "pid": pid, "start": t0 + pd.Timedelta(seconds=s), "label": SEGMENTS[i],
            "normal": SEGMENTS[i] == "baseline", "hr_mean": out["hr"], "eda_mean": out["eda"],
            "temp_mean": out["temp"], "move": move, "coverage": have[sl].mean(),
            "r_hr_eda": out["r_hr_eda"], "r_hr_temp": out["r_hr_temp"], "r_eda_temp": out["r_eda_temp"],
            "drive": drive, "task": TASKS[i], "t_rel_s": s,
            "scr_rate": scr_rate, "scr_amp": scr_amp,
            "rmssd_ms": rmssd(b_ibi[(b_time >= s) & (b_time < s + WIN)]),
            "eda_phasic": eda_phasic, "eda_tonic_slope": eda_tonic_slope,
            "hr_slope": slope_per_min(sec["hr"][sl], t_win),
            "temp_slope": slope_per_min(sec["temp"][sl], t_win),
        })
    return rows


def subjective_check(w: pd.DataFrame):
    """The dataset's own continuous stress rating (0-1, Subj_metric, 4 Hz from the first car-park start):
    mean per segment, to confirm that city > highway > car park."""
    a = pd.read_csv(os.path.join(DB, "Subj_metric", "Annot_Subjective_metric.csv"), header=None, skiprows=1)
    names = ["park", "city1", "hwy", "city2", "city2_back", "hwy_back", "city1_back", "park_back"]
    rows = []
    for r in a.itertuples(index=False):
        sm = pd.read_csv(os.path.join(DB, "Subj_metric", f"SM_{r[0]}.csv")).iloc[:, 0].to_numpy()
        for i, nm in enumerate(names):
            rows.append({"drive": r[0], "task": nm, "metric": np.nanmean(sm[int(r[1 + 2 * i]):int(r[2 + 2 * i])])})
    m = pd.DataFrame(rows).groupby("task").metric.mean().reindex(names)
    print("subjective stress metric by segment (mean over drives):")
    print(m.round(2).to_string())


def main():
    rows = []
    for wrist in ("Left", "Right"):
        for drive, bounds in annotations(wrist).items():
            if WRIST.get(drive, "Left") == wrist:
                rows += drive_rows(drive, bounds)
    w = pd.DataFrame(rows)
    w["start"] = pd.to_datetime(w["start"])
    w = w[w.coverage >= MINCOV].reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    w.to_parquet(OUT, index=False)
    print(f"wrote {OUT}")
    print(f"people: {w.pid.nunique()}, drives: {w.drive.nunique()}, windows: {len(w)}")
    g = w.groupby("label")
    print(pd.DataFrame({"people": g.pid.nunique(), "windows": g.size()}).to_string())
    print(g[["hr_mean", "eda_mean", "temp_mean", "move", "scr_rate"]].median().round(3).to_string())
    subjective_check(w)


if __name__ == "__main__":
    sys.exit(main())
