"""Test the BodySync score on WESAD, which has real stress labels.

WESAD (Schmidt et al. 2018): 15 people wore an Empatica E4 on the wrist through a lab
protocol: baseline (sitting, reading), stress (Trier Social Stress Test: public speech
and mental arithmetic), amusement (funny videos) and meditation.

The wrist data is far more detailed than ZU-PWD: EDA and temperature 4x per second,
pulse (BVP) 64x per second. Everything is brought to 1 reading per second, then the
SAME sync score as for ZU-PWD is computed:
  2-minute windows (sliding 30 s), Spearman r for HR-EDA, HR-temp, EDA-temp,
  compared with the person's own normal = their baseline condition.

Questions:
  1. Does the sync score drop during stress?
  2. Does sync help a classifier tell stress from non-stress, on top of plain levels?
"""
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import wilcoxon

from config import FIGURES, PAIRS, PROCESSED, RESULTS, ROOT
from features import pearson, spearman
from model import loso, per_person, score

WESAD_ZIP = ROOT / "data" / "wesad" / "WESAD.zip"
CONDITIONS = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
WIN, STEP = 120, 30          # seconds
LABEL_HZ, BVP_HZ, EDA_HZ = 700, 64, 4


class SafeUnpickler(pickle.Unpickler):
    """Only lets numpy arrays through, so a pickle file can't run code on load."""
    ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
               ("numpy", "ndarray"), ("numpy", "dtype"),
               ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError(f"blocked {module}.{name}")
        return super().find_class(module.replace("numpy.core", "numpy._core"), name)


def load_subject(zf: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with zf.open(member) as f:
        d = SafeUnpickler(f, encoding="latin1").load()
    wrist = d["signal"]["wrist"]
    eda = wrist["EDA"].ravel()
    temp = wrist["TEMP"].ravel()
    bvp = wrist["BVP"].ravel()
    n_sec = min(len(eda) // EDA_HZ, len(bvp) // BVP_HZ, len(d["label"]) // LABEL_HZ)

    # heart rate from pulse peaks, then one value per second
    b, a = butter(3, [0.7, 3.5], btype="band", fs=BVP_HZ)
    clean = filtfilt(b, a, bvp)
    peaks, _ = find_peaks(clean, distance=int(BVP_HZ * 0.33))
    t_peak = peaks[1:] / BVP_HZ
    hr = 60 / np.diff(peaks / BVP_HZ)
    ok = (hr > 40) & (hr < 180)
    secs = np.arange(n_sec) + 0.5
    hr_1hz = np.interp(secs, t_peak[ok], hr[ok])

    def per_second(x, hz):
        return x[: n_sec * hz].reshape(n_sec, hz).mean(axis=1)

    lab = d["label"][: n_sec * LABEL_HZ].reshape(n_sec, LABEL_HZ)
    lab = np.array([np.bincount(r).argmax() for r in lab])
    return pd.DataFrame({"sec": np.arange(n_sec), "hr": hr_1hz, "eda": per_second(eda, EDA_HZ),
                         "temp": per_second(temp, EDA_HZ), "label": lab})


def windows(sig: pd.DataFrame, pid: str) -> pd.DataFrame:
    rows = []
    x = {s: sig[s].to_numpy(float) for s in ("hr", "eda", "temp")}
    lab = sig.label.to_numpy()
    for start in range(0, len(sig) - WIN + 1, STEP):
        sl = slice(start, start + WIN)
        vals, counts = np.unique(lab[sl], return_counts=True)
        if counts.max() < 0.9 * WIN or vals[counts.argmax()] not in CONDITIONS:
            continue
        row = {"pid": pid, "start": start, "state": CONDITIONS[vals[counts.argmax()]]}
        for s, v in x.items():
            row[f"{s}_mean"], row[f"{s}_sd"] = v[sl].mean(), v[sl].std()
        for a, b in PAIRS:
            row[f"s_{a}_{b}"] = spearman(x[a][sl], x[b][sl], WIN // 2)
            row[f"r_{a}_{b}"] = pearson(x[a][sl], x[b][sl], WIN // 2)
            row[f"dr_{a}_{b}"] = pearson(np.diff(x[a][sl]), np.diff(x[b][sl]), WIN // 2)
        rows.append(row)
    return pd.DataFrame(rows)


def build() -> pd.DataFrame:
    out = []
    with zipfile.ZipFile(WESAD_ZIP) as zf:
        members = sorted(m for m in zf.namelist() if m.endswith(".pkl"))
        for m in members:
            pid = m.split("/")[-1].replace(".pkl", "")
            w = windows(load_subject(zf, m), pid)
            print(f"{pid}: {w.state.value_counts().to_dict()}")
            out.append(w)
    w = pd.concat(out, ignore_index=True)
    w.to_parquet(PROCESSED / "wesad_windows.parquet", index=False)
    return w


def sync_score(w: pd.DataFrame) -> pd.DataFrame:
    """Same score as ZU-PWD: distance from the person's normal sync (their baseline), 100 = normal."""
    cols = [f"s_{a}_{b}" for a, b in PAIRS]
    w = w.copy()
    base = w[w.state == "baseline"].groupby("pid")[cols]
    med = base.median()
    mad = base.agg(lambda s: (s - s.median()).abs().median() * 1.4826).clip(lower=0.05)
    off = (w[cols] - med.reindex(w.pid).to_numpy()).abs() / mad.reindex(w.pid).to_numpy()
    w["sync_score"] = np.clip(100 - 100 / 3 * off.mean(axis=1), 0, 100)
    # the same idea on plain levels, for comparison
    lcols = ["hr_mean", "eda_mean", "temp_mean"]
    lb = w[w.state == "baseline"].groupby("pid")[lcols]
    lz = (w[lcols] - lb.median().reindex(w.pid).to_numpy()).abs() / lb.std().reindex(w.pid).to_numpy()
    w["level_score"] = np.clip(100 - 100 / 3 * lz.mean(axis=1), 0, 100)
    return w


def main():
    w = pd.read_parquet(PROCESSED / "wesad_windows.parquet") if (PROCESSED / "wesad_windows.parquet").exists() else build()
    w = sync_score(w)

    # 1. does the score drop during stress? (per-person means, paired tests)
    # Baseline windows define "normal", so any other condition drifts a little lower;
    # stress vs amusement is the fairer test, since neither was used to set the normal.
    per = w.groupby(["pid", "state"])[["sync_score", "level_score"]].mean().unstack()
    rows = []
    for sc in ("sync_score", "level_score"):
        for ref, cond in [("baseline", "stress"), ("baseline", "amusement"), ("baseline", "meditation"),
                          ("amusement", "stress")]:
            t = per[sc][[ref, cond]].dropna()
            rows.append({"score": sc, "comparison": f"{cond} vs {ref}", "n_people": len(t),
                         "mean_ref": t[ref].mean(), "mean_condition": t[cond].mean(),
                         "people_lower": int((t[cond] < t[ref]).sum()),
                         "p_value": wilcoxon(t[cond], t[ref]).pvalue})
    r1 = pd.DataFrame(rows)
    r1.to_csv(RESULTS / "wesad_score_by_condition.csv", index=False)
    print(r1.round(4).to_string(index=False))

    # 2. stress vs non-stress classification, same LOSO setup as ZU-PWD
    level = [f"{s}_{a}" for s in ("hr", "eda", "temp") for a in ("mean", "sd")]
    coup = [f"{k}_{a}_{b}" for k in ("r", "dr", "s") for a, b in PAIRS]
    c = w[w.state != "meditation"].assign(state=lambda d: np.where(d.state == "stress", "stress", "calm"))
    rows, pp = [], {}
    for name, cols in {"levels": level, "coupling": coup, "levels+coupling": level + coup,
                       "levels+sync_score": level + ["sync_score"]}.items():
        p = loso(c.reset_index(drop=True), cols, ["calm", "stress"], "xgb")
        rows.append({"features": name, **score(p)})
        pp[name] = per_person(p)
    r2 = pd.DataFrame(rows)
    wide = pd.DataFrame(pp).dropna()
    for name in ("levels+coupling", "levels+sync_score"):
        d = wide[name] - wide["levels"]
        r2.loc[r2.features == name, "per_person_median_gain"] = d.median()
        r2.loc[r2.features == name, "p_vs_levels"] = wilcoxon(wide[name], wide["levels"]).pvalue if (d != 0).any() else np.nan
    r2.to_csv(RESULTS / "wesad_classification.csv", index=False)
    print(r2.round(4).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    order = ["baseline", "stress", "amusement", "meditation"]
    for ax, sc, title in zip(axes, ["sync_score", "level_score"], ["BodySync score", "Plain-levels score"]):
        m = per[sc][order]
        for _, row in m.iterrows():
            ax.plot(order, row.values, color="#bbb", lw=0.8)
        ax.plot(order, m.mean().values, color="#e07b39", lw=2.5, marker="o", label="average")
        ax.set_title(f"WESAD: {title} by condition (one grey line per person)")
        ax.set_ylim(0, 100)
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "sync_score" / "wesad_validation.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
