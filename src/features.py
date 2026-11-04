"""Cut each ZU-PWD person's timeline into 30-min windows (sliding by 15) and compute features.

Per window:
- level features ("plain signal values"): mean and SD of HR, EDA, temp
- personal levels: the same means relative to that person's own median / IQR
- coupling features ("moving together"): Pearson r for the 3 signal pairs,
  on the raw minute values (r_*) and on minute-to-minute changes (dr_*).
  The change-based version removes slow shared drifts.
"""
import numpy as np
import pandas as pd

from config import MIN_COVERAGE, MIN_EDA_ABOVE_FLOOR, MIN_PURITY, PAIRS, PROCESSED, SIGNALS, STEP_MIN, WINDOW_MIN

LEVEL = [f"{s}_{a}" for s in SIGNALS for a in ("mean", "sd")]
PERSONAL = [f"{s}_mean_z" for s in SIGNALS]
COUPLING = [f"r_{a}_{b}" for a, b in PAIRS] + [f"dr_{a}_{b}" for a, b in PAIRS]


def pearson(x: np.ndarray, y: np.ndarray, min_n: int) -> float:
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < min_n:
        return np.nan
    x, y = x[m], y[m]
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return np.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def person_windows(g: pd.DataFrame) -> list[dict]:
    g = g.set_index("time").asfreq("min")       # continuous minute grid; gaps become NaN
    sig = {s: g[s].to_numpy(float) for s in SIGNALS}
    diff = {s: np.diff(v, prepend=np.nan) for s, v in sig.items()}
    state = g["state"].fillna("missing").to_numpy()
    floor = g["eda_floor"].astype(float).to_numpy()   # NaN where not recorded
    times = g.index

    rows = []
    for start in range(0, len(g) - WINDOW_MIN + 1, STEP_MIN):
        sl = slice(start, start + WINDOW_MIN)
        ok = ~np.isnan(sig["hr"][sl]) & ~np.isnan(sig["eda"][sl]) & ~np.isnan(sig["temp"][sl])
        coverage = ok.mean()
        if coverage < 0.5:
            continue
        labels, counts = np.unique(state[sl][ok], return_counts=True)
        row = {
            "start": times[start],
            "state": labels[counts.argmax()],
            "purity": counts.max() / ok.sum(),
            "coverage": coverage,
            "eda_above_floor": 1 - np.nanmean(floor[sl]),
        }
        for s in SIGNALS:
            v = sig[s][sl][ok]
            row[f"{s}_mean"] = v.mean()
            row[f"{s}_sd"] = v.std()
        min_n = WINDOW_MIN // 2
        for a, b in PAIRS:
            row[f"r_{a}_{b}"] = pearson(sig[a][sl], sig[b][sl], min_n)
            row[f"dr_{a}_{b}"] = pearson(diff[a][sl], diff[b][sl], min_n)
        rows.append(row)
    return rows


def build(clean: pd.DataFrame) -> pd.DataFrame:
    out = []
    for pid, g in clean.groupby("pid"):
        w = pd.DataFrame(person_windows(g[["time", *SIGNALS, "state", "eda_floor"]]))
        if w.empty:
            continue
        w.insert(0, "pid", pid)
        # personal baseline: each person's own median / IQR over all worn minutes (no labels used)
        for s in SIGNALS:
            q1, med, q3 = g[s].quantile([0.25, 0.5, 0.75])
            w[f"{s}_mean_z"] = (w[f"{s}_mean"] - med) / (q3 - q1 if q3 > q1 else 1)
        out.append(w)
    w = pd.concat(out, ignore_index=True)
    w["labelled"] = (w.coverage >= MIN_COVERAGE) & (w.purity >= MIN_PURITY) & w.state.isin(["sleep", "rest", "active"])
    w["eda_ok"] = w.eda_above_floor >= MIN_EDA_ABOVE_FLOOR
    return w


def main():
    w = build(pd.read_parquet(PROCESSED / "clean.parquet"))
    w.to_parquet(PROCESSED / "windows_30.parquet", index=False)
    lab = w[w.labelled & w.eda_ok]
    print(f"{len(w)} windows, {len(lab)} labelled with usable EDA from {lab.pid.nunique()} people "
          f"-> {lab.state.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
