"""WESAD (Schmidt et al. 2018): 15 people, Empatica E4 on the wrist, lab protocol of
baseline, stress (Trier Social Stress Test), amusement and meditation.

Input: data/wesad/WESAD.zip from https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx
The per-subject S*.pkl files hold the synchronised signals and labels. They are read with
an unpickler that only allows numpy arrays, so a pickle can't run code on load.

Signals are brought to 1 reading per second (EDA/temp 4 Hz, pulse 64 Hz -> heart rate
from pulse peaks), then cut into 2-min windows sliding by 30 s, per src/datasets/SPEC.md.
A window gets a condition if at least 90% of its seconds are in it. normal = baseline.
r_hr_eda, r_hr_temp, r_eda_temp: Pearson r of the two 1-s series inside each window.
scr_rate, scr_amp, rmssd_ms: sweat bursts from the raw 4 Hz EDA and HRV from pulse peaks (see richer.py).
"""
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

from richer import eda_bursts, rmssd

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "data" / "wesad" / "WESAD.zip"
OUT = ROOT / "data" / "processed" / "ext_wesad_windows.parquet"
CONDITIONS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
WIN, STEP = 120, 30          # seconds
LABEL_HZ, BVP_HZ, EDA_HZ, ACC_HZ = 700, 64, 4, 32


class SafeUnpickler(pickle.Unpickler):
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"),
               ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        return super().find_class(module.replace("numpy.core", "numpy._core"), name)


def per_second(x: np.ndarray, hz: int, n_sec: int) -> np.ndarray:
    return x[: n_sec * hz].reshape(n_sec, hz, *x.shape[1:]).mean(axis=1)


def load_subject(zf: zipfile.ZipFile, member: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-second signals, plus the raw 4 Hz EDA and the beat times (s) for the richer features."""
    with zf.open(member) as f:
        d = SafeUnpickler(f, encoding="latin1").load()
    wrist = d["signal"]["wrist"]
    eda, temp, bvp = wrist["EDA"].ravel(), wrist["TEMP"].ravel(), wrist["BVP"].ravel()
    acc = wrist["ACC"] / 64.0                                     # E4 units are 1/64 g
    n_sec = min(len(eda) // EDA_HZ, len(bvp) // BVP_HZ, len(acc) // ACC_HZ, len(d["label"]) // LABEL_HZ)

    b, a = butter(3, [0.7, 3.5], btype="band", fs=BVP_HZ)
    peaks, _ = find_peaks(filtfilt(b, a, bvp), distance=int(BVP_HZ * 0.33))
    hr = 60 / np.diff(peaks / BVP_HZ)
    ok = (hr > 40) & (hr < 180)
    hr_1hz = np.interp(np.arange(n_sec) + 0.5, peaks[1:][ok] / BVP_HZ, hr[ok])

    mag = np.linalg.norm(acc[: n_sec * ACC_HZ], axis=1).reshape(n_sec, ACC_HZ)
    lab = d["label"][: n_sec * LABEL_HZ].reshape(n_sec, LABEL_HZ)
    beats = peaks[1:][ok] / BVP_HZ
    return pd.DataFrame({
        "hr": hr_1hz, "eda": per_second(eda, EDA_HZ, n_sec), "temp": per_second(temp, EDA_HZ, n_sec),
        "mag_sd": mag.std(axis=1), "label": [np.bincount(r).argmax() for r in lab],
    }), eda, beats


def pearson(x: pd.Series, y: pd.Series) -> float:
    m = x.notna() & y.notna()
    if m.mean() < 0.5 or x[m].std() == 0 or y[m].std() == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def windows(sig: pd.DataFrame, eda: np.ndarray, beats: np.ndarray, pid: str) -> pd.DataFrame:
    rows, lab = [], sig.label.to_numpy()
    for start in range(0, len(sig) - WIN + 1, STEP):
        sl = slice(start, start + WIN)
        vals, counts = np.unique(lab[sl], return_counts=True)
        if counts.max() < 0.9 * WIN or vals[counts.argmax()] not in CONDITIONS:
            continue
        cond = CONDITIONS[vals[counts.argmax()]]
        part = sig.iloc[sl]
        scr_rate, scr_amp = eda_bursts(eda[start * EDA_HZ:(start + WIN) * EDA_HZ])
        in_win = (beats >= start) & (beats < start + WIN)
        ibi = np.diff(beats)[in_win[1:] & in_win[:-1]]
        rows.append({"pid": f"wesad_{pid}", "start": pd.Timestamp(0) + pd.Timedelta(seconds=start),
                     "label": cond, "normal": cond == "baseline", "hr_mean": part.hr.mean(),
                     "eda_mean": part.eda.mean(), "temp_mean": part.temp.mean(),
                     "move": part.mag_sd.mean(), "coverage": 1.0,
                     "r_hr_eda": pearson(part.hr, part.eda), "r_hr_temp": pearson(part.hr, part.temp),
                     "r_eda_temp": pearson(part.eda, part.temp),
                     "scr_rate": scr_rate, "scr_amp": scr_amp, "rmssd_ms": rmssd(ibi)})
    return pd.DataFrame(rows)


def main():
    out = []
    with zipfile.ZipFile(ZIP) as zf:
        for m in sorted(m for m in zf.namelist() if m.endswith(".pkl")):
            pid = Path(m).stem
            out.append(windows(*load_subject(zf, m), pid))
    w = pd.concat(out, ignore_index=True)
    w.to_parquet(OUT, index=False)
    print(f"{w.pid.nunique()} people, {w.label.value_counts().to_dict()}")
    print(w.groupby("label")[["hr_mean", "eda_mean", "temp_mean"]].median().round(2).to_string())


if __name__ == "__main__":
    main()
