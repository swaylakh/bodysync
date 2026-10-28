"""ADARP loader (Sah et al. 2022, Zenodo 6640290, CC-BY-4.0).

11 people with alcohol use disorder wore an Empatica E4 for up to 14 days in
daily life (FREE-LIVING). Participants pressed the E4 event button when they
felt more stressed than usual; presses are in each session's ``tags.csv``.

Input:  data/external/adarp/Sensor Data/Part <id>C/<session>/{HR,EDA,TEMP,ACC,tags}.csv
        (extracted from "Sensor Data.zip"; BVP/IBI are not used)
Output: data/processed/ext_adarp_windows.parquet

Windows: 5-min windows sliding by 5 min, aligned to multiples of 300 s of unix
time. ``start`` is naive UTC (E4 timestamps are UTC). Sessions of one person
are merged on a 1-s grid (overlapping seconds are averaged).

Label rule (button tags only):
  * A tag is an "edge" tag if it lies within 60 s of its session's first or
    last sample (likely a power-on/off press artefact); otherwise "valid".
  * stress  = window [s, s+300) overlaps [t-15 min, t+15 min] of a valid tag.
  * calm    = window is >= 60 min away from EVERY tag (valid or edge).
  * unknown = everything else.
  normal = True for all windows (free-living; labels never define normal).

Phone-survey stress events ("Final Phone Survey Stress Events.xlsm") are NOT
used: they only give 1-hour recall bins (night bins reported next morning),
the local time zone is undocumented, and the authors' own "Is the tag valid?"
column shows many survey events have no matching tag. Mapping them would add
timing error rather than labels. Consequence: some "calm" windows may contain
stress that was reported in the survey but not tagged.

Cleaning: HR outside 35-200 bpm, TEMP outside 20-42 C and EDA < 0 set missing.
move = SD of the ACC vector magnitude (g) over the window's samples.
coverage = share of the 300 s having HR, EDA and TEMP; keep coverage >= 0.75.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "external" / "adarp" / "Sensor Data"
OUT = ROOT / "data" / "processed" / "ext_adarp_windows.parquet"
WIN = 300
STRESS_PAD = 15 * 60
CALM_GAP = 60 * 60
EDGE = 60


def read_e4(path):
    """Return (start unix s, fs Hz, values ndarray) for an Empatica CSV."""
    head = pd.read_csv(path, header=None, nrows=2)
    start, fs = float(head.iloc[0, 0]), float(head.iloc[1, 0])
    vals = pd.read_csv(path, header=None, skiprows=2).to_numpy(dtype=float)
    return start, fs, vals


def per_second(start, fs, x):
    """Per-second sum and count (x 1-D, NaN = missing). Returns secs, sum, n."""
    sec = np.floor(start + np.arange(len(x)) / fs).astype(np.int64)
    ok = ~np.isnan(x)
    s0 = sec[0]
    idx = sec - s0
    m = idx.max() + 1 if len(idx) else 0
    sm = np.bincount(idx[ok], weights=x[ok], minlength=m)
    n = np.bincount(idx[ok], minlength=m)
    return s0 + np.arange(m), sm, n


def load_session(d):
    """Per-second frame for one session, plus (first, last) unix time."""
    parts = {}
    span = []
    for name, lo, hi in (("HR", 35, 200), ("EDA", 0, np.inf), ("TEMP", 20, 42)):
        f = d / f"{name}.csv"
        if not f.exists() or f.stat().st_size == 0:
            return None, None
        try:
            st, fs, v = read_e4(f)
        except (pd.errors.EmptyDataError, IndexError):
            return None, None
        if len(v) == 0:
            return None, None
        x = v[:, 0]
        x = np.where((x >= lo) & (x <= hi), x, np.nan)
        secs, sm, n = per_second(st, fs, x)
        parts[name.lower() + "_s"] = pd.Series(sm, index=secs)
        parts[name.lower() + "_n"] = pd.Series(n, index=secs)
        span += [st, st + len(x) / fs]
    f = d / "ACC.csv"
    if f.exists() and f.stat().st_size > 0:
        try:
            st, fs, v = read_e4(f)
            mag = np.linalg.norm(v[:, :3] / 64.0, axis=1)
            secs, sm, n = per_second(st, fs, mag)
            _, sq, _ = per_second(st, fs, mag ** 2)
            parts["acc_s"] = pd.Series(sm, index=secs)
            parts["acc_q"] = pd.Series(sq, index=secs)
            parts["acc_n"] = pd.Series(n, index=secs)
        except (pd.errors.EmptyDataError, IndexError):
            pass
    df = pd.DataFrame(parts).fillna(0.0)
    return df, (min(span), max(span))


def read_tags(d, span):
    f = d / "tags.csv"
    if not f.exists() or f.stat().st_size == 0:
        return []
    t = pd.read_csv(f, header=None)[0].astype(float).tolist()
    out = []
    for x in t:
        edge = span is None or (x - span[0] < EDGE) or (span[1] - x < EDGE)
        out.append((x, edge))
    return out


def process_person(pdir):
    pid = "adarp_" + pdir.name.replace("Part ", "")
    frames, tags = [], []
    for d in sorted(p for p in pdir.iterdir() if p.is_dir()):
        df, span = load_session(d)
        tags += read_tags(d, span)
        if df is not None:
            frames.append(df)
    if not frames:
        return None, tags
    sec = pd.concat(frames).groupby(level=0).sum()  # overlapping sessions: pool sums/counts
    win = (sec.index.values // WIN) * WIN
    have = (sec["hr_n"] > 0) & (sec["eda_n"] > 0) & (sec["temp_n"] > 0)
    sec = sec.assign(win=win, full=have.astype(int))
    g = sec.groupby("win").sum()
    w = pd.DataFrame(index=g.index)
    w["hr_mean"] = g["hr_s"] / g["hr_n"].replace(0, np.nan)
    w["eda_mean"] = g["eda_s"] / g["eda_n"].replace(0, np.nan)
    w["temp_mean"] = g["temp_s"] / g["temp_n"].replace(0, np.nan)
    if "acc_n" in g:
        n = g["acc_n"].replace(0, np.nan)
        mu = g["acc_s"] / n
        var = (g["acc_q"] / n - mu ** 2).clip(lower=0) * n / (n - 1).replace(0, np.nan)
        w["move"] = np.sqrt(var)
    else:
        w["move"] = np.nan
    w["coverage"] = g["full"] / WIN
    w = w[w["coverage"] >= 0.75].copy()

    s = w.index.values.astype(float)
    e = s + WIN
    lab = np.full(len(w), "unknown", dtype=object)
    all_t = np.array([t for t, _ in tags]) if tags else np.array([])
    val_t = np.array([t for t, edge in tags if not edge]) if tags else np.array([])
    if len(all_t):
        # distance from window interval to nearest tag (0 if inside)
        dist = np.maximum(np.maximum(all_t[None, :] - e[:, None], s[:, None] - all_t[None, :]), 0).min(axis=1)
        calm = dist >= CALM_GAP
    else:
        calm = np.ones(len(w), bool)
    if len(val_t):
        stress = ((s[:, None] < val_t[None, :] + STRESS_PAD) & (e[:, None] > val_t[None, :] - STRESS_PAD)).any(axis=1)
    else:
        stress = np.zeros(len(w), bool)
    lab[calm] = "calm"
    lab[stress] = "stress"
    w["label"] = lab
    w["normal"] = True
    w["pid"] = pid
    w["start"] = pd.to_datetime(w.index.values, unit="s")
    w = w.reset_index(drop=True)
    return w[["pid", "start", "label", "normal", "hr_mean", "eda_mean", "temp_mean", "move", "coverage"]], tags


def main():
    out, ntag, nedge = [], 0, 0
    for pdir in sorted(p for p in RAW.iterdir() if p.is_dir()):
        w, tags = process_person(pdir)
        ntag += len(tags)
        nedge += sum(e for _, e in tags)
        if w is not None and len(w):
            out.append(w)
            print(f"{pdir.name}: {len(w)} windows, {len(tags)} tags, "
                  f"{w.label.value_counts().to_dict()}")
    df = pd.concat(out, ignore_index=True)
    df["pid"] = df["pid"].astype(str)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\ntags: {ntag} total, {nedge} edge (calm-exclusion only)")
    print(f"people: {df.pid.nunique()}  windows: {len(df)}")
    print(df.groupby("label").agg(people=("pid", "nunique"), windows=("pid", "size"),
                                  hr=("hr_mean", "median"), eda=("eda_mean", "median"),
                                  temp=("temp_mean", "median"), move=("move", "median")).round(3))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
