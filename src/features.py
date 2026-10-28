"""Step 2: cut each person's timeline into sliding windows and compute features.

Per window:
- level features   ("plain signal values"): mean and SD of HR, EDA, temp
- personal levels: the same means relative to that person's own median / IQR
- coupling features ("moving together"): Pearson r for the 3 signal pairs,
  on the raw minute values (r_*) and on minute-to-minute changes (dr_*).
  The change-based version removes slow shared drifts, so it reflects
  signals reacting together rather than just trending together.
- coupling v2 (round 2, fixes chosen before looking at their results):
  * Spearman rank r (s_*): robust to EDA's 0.01 uS staircase steps
  * lagged cross-correlation (xmax_*, xlag_*): strongest r within +-10 min and the
    lag where it occurs, since temperature reacts to HR/EDA with a delay
  * personal coupling (*_pz): coupling relative to the person's own usual coupling,
    the same trick that helped the level features
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from config import (EDA_FLOOR, MIN_COVERAGE, MIN_EDA_ABOVE_FLOOR, MIN_PURITY,
                    PAIRS, PROCESSED, SIGNALS, WINDOWS)

LEVEL = [f"{s}_{a}" for s in SIGNALS for a in ("mean", "sd")]
PERSONAL = [f"{s}_mean_z" for s in SIGNALS]
COUPLING_RAW = [f"r_{a}_{b}" for a, b in PAIRS]
COUPLING_DIFF = [f"dr_{a}_{b}" for a, b in PAIRS]
COUPLING = COUPLING_RAW + COUPLING_DIFF
SPEARMAN = [f"s_{a}_{b}" for a, b in PAIRS]
LAGGED = [f"x{k}_{a}_{b}" for k in ("max", "lag") for a, b in PAIRS]
PERSONAL_COUPLING = [f"{c}_pz" for c in COUPLING + SPEARMAN]
COUPLING_V2 = SPEARMAN + LAGGED + PERSONAL_COUPLING
MAX_LAG = 10   # minutes


def pearson(x: np.ndarray, y: np.ndarray, min_n: int) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < min_n:
        return np.nan
    x, y = x[m], y[m]
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def spearman(x: np.ndarray, y: np.ndarray, min_n: int) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < min_n:
        return np.nan
    return pearson(rankdata(x[m]), rankdata(y[m]), min_n)


def lagged(x_full: np.ndarray, y_full: np.ndarray, start: int, length: int, min_n: int) -> tuple[float, float]:
    """Strongest Pearson r between x and y shifted by -MAX_LAG..+MAX_LAG minutes (positive = y follows x)."""
    best_r, best_k = np.nan, np.nan
    x = x_full[start:start + length]
    for k in range(-MAX_LAG, MAX_LAG + 1):
        if start + k < 0 or start + k + length > len(y_full):
            continue
        r = pearson(x, y_full[start + k:start + k + length], min_n)
        if not np.isnan(r) and (np.isnan(best_r) or abs(r) > abs(best_r)):
            best_r, best_k = r, k
    return best_r, best_k


def person_windows(g: pd.DataFrame, length: int, step: int) -> list[dict]:
    g = g.set_index("time").asfreq("min")       # continuous minute grid; gaps become NaN
    sig = {s: g[s].to_numpy(float) for s in SIGNALS}
    diff = {s: np.diff(v, prepend=np.nan) for s, v in sig.items()}
    state = g["state"].fillna("missing").to_numpy()
    floor = g["eda_floor"].astype(float).to_numpy()   # NaN where not recorded
    times = g.index

    rows = []
    for start in range(0, len(g) - length + 1, step):
        sl = slice(start, start + length)
        ok = ~np.isnan(sig["hr"][sl]) & ~np.isnan(sig["eda"][sl]) & ~np.isnan(sig["temp"][sl])
        coverage = ok.mean()
        if coverage < 0.5:
            continue
        labels, counts = np.unique(state[sl][ok], return_counts=True)
        dom = labels[counts.argmax()]
        row = {
            "start": times[start],
            "state": dom,
            "purity": counts.max() / ok.sum(),
            "coverage": coverage,
            "eda_above_floor": 1 - np.nanmean(floor[sl]),
        }
        for s in SIGNALS:
            v = sig[s][sl][ok]
            row[f"{s}_mean"] = v.mean()
            row[f"{s}_sd"] = v.std()
        min_n = int(0.5 * length)
        for a, b in PAIRS:
            row[f"r_{a}_{b}"] = pearson(sig[a][sl], sig[b][sl], min_n)
            row[f"dr_{a}_{b}"] = pearson(diff[a][sl], diff[b][sl], min_n)
            row[f"s_{a}_{b}"] = spearman(sig[a][sl], sig[b][sl], min_n)
            row[f"xmax_{a}_{b}"], row[f"xlag_{a}_{b}"] = lagged(sig[a], sig[b], start, length, min_n)
        rows.append(row)
    return rows


def build(length: int, step: int, clean: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, g in clean.groupby("pid"):
        rows = person_windows(g[["time", *SIGNALS, "state", "eda_floor"]], length, step)
        w = pd.DataFrame(rows)
        if w.empty:
            continue
        w.insert(0, "pid", pid)
        # personal baseline: each person's own median / IQR over all worn minutes (no labels used)
        for s in SIGNALS:
            q1, med, q3 = g[s].quantile([0.25, 0.5, 0.75])
            w[f"{s}_mean_z"] = (w[f"{s}_mean"] - med) / (q3 - q1 if q3 > q1 else 1)
        # same for coupling: this window's sync vs the person's usual sync (no labels used)
        for c in COUPLING + SPEARMAN:
            q1, med, q3 = w[c].quantile([0.25, 0.5, 0.75])
            w[f"{c}_pz"] = (w[c] - med) / (q3 - q1 if q3 > q1 else 1)
        out.append(w)
    w = pd.concat(out, ignore_index=True)
    w["labelled"] = (
        (w.coverage >= MIN_COVERAGE) & (w.purity >= MIN_PURITY) & w.state.isin(["sleep", "rest", "active"])
    )
    w["eda_ok"] = w.eda_above_floor >= MIN_EDA_ABOVE_FLOOR
    w["length_min"] = length
    return w


def main():
    clean = pd.read_parquet(PROCESSED / "clean.parquet")
    for length, step in WINDOWS.items():
        w = build(length, step, clean)
        w.to_parquet(PROCESSED / f"windows_{length}.parquet", index=False)
        lab = w[w.labelled & w.eda_ok]
        print(f"{length} min windows: {len(w)} total, {len(lab)} labelled with usable EDA "
              f"from {lab.pid.nunique()} people -> {lab.state.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
