"""Nurse stress loader (Hosseini et al. 2022, Sci Data; Dryad 10.5061/dryad.5hqbzkh6f, CC0).

15 nurses wore an Empatica E4 during real hospital shifts (FREE-LIVING,
Apr-Dec 2020, Lafayette, Louisiana). The authors flagged candidate stress
events from the signals and asked the nurses to rate each one in a survey
(SurveyResults.xlsx: ID, date, Start time, End time, Stress level).
Stress level: 0 = no stress, 1 = medium, 2 = high, "na" = not answered.

Input:  data/external/nurse/zips/<ID>/<ID>_<unix>.zip  (inner zips, extracted
        from Stress_dataset.zip with 7z; CSVs are read straight from the inner
        zips with zipfile, BVP/IBI never read)
        data/external/nurse/SurveyResults.xlsx
Output: data/processed/ext_nurse_windows.parquet

Time zones: E4 timestamps are unix (UTC). Survey date + clock times are local
time, America/Chicago (CDT = UTC-5 until 2020-11-01, CST = UTC-6 after).
Checked empirically: with America/Chicago 305/358 survey intervals lie fully
inside a recording of the same nurse (322 overlap one); fixed offsets do worse
(UTC-5: 283, UTC-6: 198, UTC: 58), and the Nov/Dec rows only match after the
DST switch to UTC-6. The ~36 rows that do not overlap any recording are 0.1 h
to 40 days away (likely no data that day or a typo'd date); they create no
stress windows but still block "calm" near them. End < start = crosses
midnight (end + 1 day). ``start`` in the output is naive UTC.

Windows: 5 min, sliding by 5 min, aligned to multiples of 300 s of unix time.
Sessions of one person are pooled on a 1-s grid (overlaps averaged).

Label rule (survey intervals only; there are no button tags):
  * stress    = window [s, s+300) overlaps an interval rated 1 or 2.
  * no_stress = not stress, overlaps an interval rated 0, and overlaps no
                unanswered ("na") interval. The nurse explicitly denied stress
                but the detector flagged arousal there, so it is kept as its own
                condition and NOT used as calm.
  * calm      = >= 60 min from EVERY survey interval (any rating, incl. 0/na).
  * unknown   = everything else.
  normal = True for all windows (free-living; labels never define normal).

Cleaning: HR outside 35-200 bpm, TEMP outside 20-42 C and EDA < 0 set missing.
move = SD of the ACC vector magnitude (g) over the window's samples.
coverage = share of the 300 s having HR, EDA and TEMP; keep coverage >= 0.75.
"""
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "data" / "external" / "nurse"
ZIPS = EXT / "zips"
SURVEY = EXT / "SurveyResults.xlsx"
OUT = ROOT / "data" / "processed" / "ext_nurse_windows.parquet"
TZ = "America/Chicago"
WIN = 300
CALM_GAP = 60 * 60
COLS = ["pid", "start", "label", "normal", "hr_mean", "eda_mean", "temp_mean", "move", "coverage"]


def read_e4(zf, name):
    """(start unix s, fs, values) from an Empatica CSV inside a zip, or None."""
    if name not in zf.namelist():
        return None
    raw = zf.read(name)
    if not raw.strip():
        return None
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if len(df) < 3:
        return None
    return float(df.iloc[0, 0]), float(df.iloc[1, 0]), df.iloc[2:].to_numpy(dtype=float)


def per_second(start, fs, x):
    sec = np.floor(start + np.arange(len(x)) / fs).astype(np.int64)
    ok = ~np.isnan(x)
    idx = sec - sec[0]
    m = idx.max() + 1
    return (sec[0] + np.arange(m), np.bincount(idx[ok], weights=x[ok], minlength=m),
            np.bincount(idx[ok], minlength=m))


def load_session(path):
    parts = {}
    with zipfile.ZipFile(path) as zf:
        for name, lo, hi in (("HR", 35, 200), ("EDA", 0, np.inf), ("TEMP", 20, 42)):
            r = read_e4(zf, f"{name}.csv")
            if r is None:
                return None
            st, fs, v = r
            x = np.where((v[:, 0] >= lo) & (v[:, 0] <= hi), v[:, 0], np.nan)
            secs, sm, n = per_second(st, fs, x)
            parts[name.lower() + "_s"] = pd.Series(sm, index=secs)
            parts[name.lower() + "_n"] = pd.Series(n, index=secs)
        r = read_e4(zf, "ACC.csv")
        if r is not None:
            st, fs, v = r
            mag = np.linalg.norm(v[:, :3] / 64.0, axis=1)
            secs, sm, n = per_second(st, fs, mag)
            _, sq, _ = per_second(st, fs, mag ** 2)
            parts.update(acc_s=pd.Series(sm, index=secs), acc_q=pd.Series(sq, index=secs),
                         acc_n=pd.Series(n, index=secs))
    return pd.DataFrame(parts).fillna(0.0)


def load_survey():
    v = pd.read_excel(SURVEY)
    v["pid"] = v["ID"].astype(str).str.strip()
    td = lambda x: pd.to_timedelta(str(x))
    st = v["date"] + v["Start time"].map(td)
    en = v["date"] + v["End time"].map(td)
    en = en.where(en > st, en + pd.Timedelta(days=1))

    def to_unix(t):
        u = t.dt.tz_localize(TZ, ambiguous="NaT", nonexistent="shift_forward").dt.tz_convert("UTC")
        return (u.dt.tz_localize(None) - pd.Timestamp(0)) // pd.Timedelta("1s")

    lvl = pd.to_numeric(v["Stress level"], errors="coerce")
    return pd.DataFrame({"pid": v["pid"], "a": to_unix(st), "b": to_unix(en),
                         "level": lvl}).dropna(subset=["a", "b"])


def windows(pdir):
    frames = [f for f in (load_session(z) for z in sorted(pdir.glob("*.zip"))) if f is not None]
    if not frames:
        return None
    sec = pd.concat(frames).groupby(level=0).sum()
    have = (sec["hr_n"] > 0) & (sec["eda_n"] > 0) & (sec["temp_n"] > 0)
    sec = sec.assign(win=(sec.index.values // WIN) * WIN, full=have.astype(int))
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
    return w[w["coverage"] >= 0.75].copy()


def label(s, iv):
    """Labels for window starts s (unix) given this person's survey intervals."""
    e = s + WIN
    lab = np.full(len(s), "unknown", dtype=object)
    if iv.empty:
        lab[:] = "calm"
        return lab
    a, b, lv = iv["a"].to_numpy(float), iv["b"].to_numpy(float), iv["level"].to_numpy(float)
    ov = (s[:, None] < b[None, :]) & (e[:, None] > a[None, :])
    gap = np.maximum(np.maximum(a[None, :] - e[:, None], s[:, None] - b[None, :]), 0).min(axis=1)
    stress = (ov & (lv >= 1)[None, :]).any(axis=1)
    zero = (ov & (lv == 0)[None, :]).any(axis=1) & ~(ov & np.isnan(lv)[None, :]).any(axis=1)
    lab[gap >= CALM_GAP] = "calm"
    lab[zero & ~stress] = "no_stress"
    lab[stress] = "stress"
    return lab


def main():
    sv = load_survey()
    out = []
    for pdir in sorted(p for p in ZIPS.iterdir() if p.is_dir()):
        w = windows(pdir)
        if w is None or w.empty:
            continue
        iv = sv[sv["pid"] == pdir.name]
        s = w.index.values.astype(float)
        w["label"] = label(s, iv)
        w["normal"] = True
        w["pid"] = "nurse_" + pdir.name
        w["start"] = pd.to_datetime(w.index.values, unit="s")
        w = w.reset_index(drop=True)[COLS]
        out.append(w)
        print(f"{pdir.name}: {len(w)} windows, {len(iv)} survey rows, {w.label.value_counts().to_dict()}")
    df = pd.concat(out, ignore_index=True)
    df["pid"] = df["pid"].astype(str)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"\npeople: {df.pid.nunique()}  windows: {len(df)}")
    print(df.groupby("label").agg(people=("pid", "nunique"), windows=("pid", "size"),
                                  hr=("hr_mean", "median"), eda=("eda_mean", "median"),
                                  temp=("temp_mean", "median"), move=("move", "median")).round(3))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
