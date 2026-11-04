"""Richer per-window features from the raw E4 signals, shared by all lab loaders.

- Sweat bursts (skin conductance responses, SCRs): small fast rises in EDA. Stress raises
  how often they happen and how big they are, even when the average EDA level barely moves.
    scr_rate: bursts per minute; scr_amp: mean burst size (uS, 0 if none)
- HRV: rmssd_ms = root mean square of successive differences between heartbeats (ms).
  Stress usually lowers it.

Fixed before looking at any results: EDA low-pass 1 Hz, a burst = a peak with prominence
>= 0.01 uS at least 1 s apart; beats outside 0.3-1.5 s are dropped; >= 20 beats needed.
"""
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

EDA_HZ = 4
MIN_PROMINENCE = 0.01   # uS
MIN_BEATS = 20


def eda_bursts(eda: np.ndarray, fs: int = EDA_HZ) -> tuple[float, float]:
    """eda: raw EDA samples of one window at `fs` Hz (NaNs allowed)."""
    x = eda[~np.isnan(eda)]
    if len(x) < 30 * fs:
        return np.nan, np.nan
    b, a = butter(2, 1.0, btype="low", fs=fs)
    x = filtfilt(b, a, x)
    peaks, props = find_peaks(x, prominence=MIN_PROMINENCE, distance=fs)
    minutes = len(x) / fs / 60
    return len(peaks) / minutes, float(props["prominences"].mean()) if len(peaks) else 0.0


def rmssd(ibi_s: np.ndarray) -> float:
    """ibi_s: inter-beat intervals in seconds inside one window."""
    ibi = ibi_s[(ibi_s > 0.3) & (ibi_s < 1.5)]
    if len(ibi) < MIN_BEATS:
        return np.nan
    return float(np.sqrt(np.mean(np.diff(ibi * 1000) ** 2)))
