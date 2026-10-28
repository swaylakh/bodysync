"""CAN-STRESS loader (free-living, Empatica E4).

Source: Zenodo record 14842061 (CC-BY-4.0), file CAN-STRESS.zip, extracted into
data/external/canstress/CAN-STRESS/. It holds one inner zip per E4 session
(``<device>_<yymmdd-hhmmss UTC>__<session>.zip``) and ``logbook.xlsx`` with columns
ID, Cannabis (1 = user), E4_Sessions, Time (time of day only), Stress_Rating (0-9), Type.

Participant / timing
--------------------
* ``pid`` = ``canstress_<logbook ID>``; a session zip is mapped to a participant via
  the logbook's E4_Sessions number. Zips not referenced in the logbook are skipped
  (no participant, no ratings).
* Times are naive local time, America/Los_Angeles (study site in the US Pacific zone;
  verified: E4 button tags converted to this zone match logbook times to the minute).
* Logbook times carry no date. Each is placed on the session's start date; if that
  falls more than 30 min before the session start it is moved to the next day
  (recordings last ~24 h). Ratings outside the recording are kept for the
  per-person percentiles but match no windows.

Windows
-------
5-min windows sliding by 5 min from the session start. Per second: HR (HR.csv, 1 Hz),
EDA and TEMP (4 Hz, averaged per second). HR outside 35-200 bpm, TEMP outside 20-42 C
and EDA < 0 set to missing. ``coverage`` = share of the 300 s with all three valid;
windows with coverage < 0.75 dropped. ``move`` = SD of the ACC vector magnitude (g) over
the raw 32 Hz samples in the window.

Labels (ratings-based; replaces the SPEC "60 min away" calm rule)
----------------------------------------------------------------
Every logbook row with a Stress_Rating counts as a rating, whatever its Type
(Stress, Cannabis_Start/End, Exercise, Sleep). Per person, over all their ratings:
* high report: rating >= person's 75th percentile AND rating >= 5;
* low report:  rating <= person's 25th percentile OR rating <= 2.
A window "is near" a report if [start, start+5 min) overlaps [t-15 min, t+15 min].
* ``stress`` = near a high report and not near a low report;
* ``calm``   = near a low report and not near a high report;
* everything else (no nearby report, a mid rating, or conflicting reports) = ``unknown``.
Because ratings mark explicit calm moments, calm is defined by low ratings instead
of "60 min away from any stress report".

``normal`` = True for all windows (free-living). Extra column ``cannabis_user`` (bool).
"""
from __future__ import annotations

import io
import os
import re
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "data", "external", "canstress", "CAN-STRESS")
OUT = os.path.join(ROOT, "data", "processed", "ext_canstress_windows.parquet")
TZ = "America/Los_Angeles"
WIN = 300
NEAR = pd.Timedelta(minutes=15)


def _read_e4(zf: zipfile.ZipFile, name: str):
    raw = zf.read(name)
    head = pd.read_csv(io.BytesIO(raw), header=None, nrows=2)
    t0 = float(head.iloc[0, 0])
    fs = float(head.iloc[1, 0])
    data = pd.read_csv(io.BytesIO(raw), header=None, skiprows=2).to_numpy(dtype=float)
    return t0, fs, data


def _per_second(t0, fs, x, grid0, n):
    """Average a 1-D signal into 1-s bins on the grid starting at grid0 (unix s)."""
    out = np.full(n, np.nan)
    if len(x) == 0:
        return out
    sec = np.floor(t0 - grid0 + np.arange(len(x)) / fs).astype(np.int64)
    ok = (sec >= 0) & (sec < n) & np.isfinite(x)
    s = np.bincount(sec[ok], weights=x[ok], minlength=n)
    c = np.bincount(sec[ok], minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(c > 0, s / np.maximum(c, 1), np.nan)
    return out


def load_logbook():
    lb = pd.read_excel(os.path.join(SRC, "logbook.xlsx"))
    lb = lb.dropna(subset=["E4_Sessions"]).copy()
    lb["E4_Sessions"] = lb["E4_Sessions"].astype(int)
    lb["pid"] = "canstress_" + lb["ID"].astype(int).astype(str).str.zfill(2)
    return lb


def rating_times(rows: pd.DataFrame, start_local: pd.Timestamp) -> pd.DataFrame:
    rows = rows.dropna(subset=["Stress_Rating", "Time"]).copy()
    ts = []
    for t in rows["Time"]:
        tt = pd.Timestamp(str(t)) if not hasattr(t, "hour") else t
        d = start_local.normalize() + pd.Timedelta(hours=tt.hour, minutes=tt.minute, seconds=tt.second)
        if d < start_local - pd.Timedelta(minutes=30):
            d += pd.Timedelta(days=1)
        ts.append(d)
    rows["t"] = ts
    return rows


def session_windows(path: str) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        hr_t0, hr_fs, hr = _read_e4(zf, "HR.csv")
        eda_t0, eda_fs, eda = _read_e4(zf, "EDA.csv")
        tmp_t0, tmp_fs, tmp = _read_e4(zf, "TEMP.csv")
        acc_t0, acc_fs, acc = _read_e4(zf, "ACC.csv")
    hr, eda, tmp = hr[:, 0], eda[:, 0].copy(), tmp[:, 0].copy()
    hr = np.where((hr >= 35) & (hr <= 200), hr, np.nan)
    eda = np.where(eda >= 0, eda, np.nan)
    tmp = np.where((tmp >= 20) & (tmp <= 42), tmp, np.nan)

    grid0 = np.floor(min(eda_t0, tmp_t0, acc_t0))
    end = max(eda_t0 + len(eda) / eda_fs, tmp_t0 + len(tmp) / tmp_fs, hr_t0 + len(hr) / hr_fs)
    n = int(np.ceil(end - grid0))
    if n < WIN:
        return pd.DataFrame()
    H = _per_second(hr_t0, hr_fs, hr, grid0, n)
    E = _per_second(eda_t0, eda_fs, eda, grid0, n)
    T = _per_second(tmp_t0, tmp_fs, tmp, grid0, n)
    mag = np.sqrt((acc ** 2).sum(axis=1)) / 64.0
    acc_sec = np.floor(acc_t0 - grid0 + np.arange(len(mag)) / acc_fs).astype(np.int64)

    nw = n // WIN
    rows = []
    for w in range(nw):
        a, b = w * WIN, (w + 1) * WIN
        h, e, t = H[a:b], E[a:b], T[a:b]
        cov = float(np.mean(np.isfinite(h) & np.isfinite(e) & np.isfinite(t)))
        if cov < 0.75:
            continue
        i0, i1 = np.searchsorted(acc_sec, [a, b])
        m = mag[i0:i1]
        rows.append(dict(
            start_unix=grid0 + a,
            hr_mean=float(np.nanmean(h)), eda_mean=float(np.nanmean(e)),
            temp_mean=float(np.nanmean(t)),
            move=float(np.std(m)) if len(m) > 1 else np.nan,
            coverage=cov,
        ))
    return pd.DataFrame(rows)


def main():
    lb = load_logbook()
    zips = {int(re.search(r"__(\d+)\.zip$", f).group(1)): f
            for f in os.listdir(SRC) if f.endswith(".zip")}
    # per-person thresholds over all their ratings
    q = lb.dropna(subset=["Stress_Rating"]).groupby("pid")["Stress_Rating"].quantile([0.25, 0.75]).unstack()
    user = lb.groupby("pid")["Cannabis"].first().astype(bool)

    out = []
    skipped = sorted(set(zips) - set(lb["E4_Sessions"]))
    for sess, rows in lb.groupby("E4_Sessions"):
        if sess not in zips:
            continue
        pid = rows["pid"].iloc[0]
        w = session_windows(os.path.join(SRC, zips[sess]))
        if w.empty:
            continue
        w["start"] = (pd.to_datetime(w["start_unix"], unit="s", utc=True)
                      .dt.tz_convert(TZ).dt.tz_localize(None))
        start_local = w["start"].min()
        r = rating_times(rows, start_local)
        q25, q75 = q.loc[pid, 0.25], q.loc[pid, 0.75]
        hi = r.loc[(r.Stress_Rating >= q75) & (r.Stress_Rating >= 5), "t"]
        lo = r.loc[(r.Stress_Rating <= q25) | (r.Stress_Rating <= 2), "t"]

        def near(ts):
            m = np.zeros(len(w), bool)
            for t in ts:
                m |= (w["start"] < t + NEAR).to_numpy() & (w["start"] + pd.Timedelta(seconds=WIN) > t - NEAR).to_numpy()
            return m

        nh, nl = near(hi), near(lo)
        w["label"] = np.where(nh & ~nl, "stress", np.where(nl & ~nh, "calm", "unknown"))
        w["pid"] = pid
        w["session"] = int(sess)
        w["cannabis_user"] = bool(user[pid])
        out.append(w)

    df = pd.concat(out, ignore_index=True)
    df["normal"] = True
    df = df[["pid", "start", "label", "normal", "hr_mean", "eda_mean", "temp_mean",
             "move", "coverage", "cannabis_user", "session"]]
    df = df.sort_values(["pid", "start"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"saved {OUT}")
    print(f"skipped {len(skipped)} session zips not in logbook")
    print("people:", df.pid.nunique(), "| with stress:", df[df.label == "stress"].pid.nunique(),
          "| with calm:", df[df.label == "calm"].pid.nunique(),
          "| cannabis users:", df.groupby("pid").cannabis_user.first().sum())
    print(df.label.value_counts().to_string())
    print(df.groupby("label")[["hr_mean", "eda_mean", "temp_mean", "move"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
