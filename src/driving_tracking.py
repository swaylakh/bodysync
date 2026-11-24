"""Does BodySync follow drivers' stress MOMENT BY MOMENT on real roads? (AffectiveROAD; run once)

Protocol (written before running; the frozen system is used unchanged, nothing is tuned):
  - Stress rating: AffectiveROAD's own continuous subjective stress metric (0-1, 4 Hz, Subj_metric/SM_<drive>.csv),
    which the dataset authors built with each driver after the drive. It was never used to build BodySync.
  - Alignment: the metric's segment borders (Annot_Subjective_metric.csv) and the E4 segment borders
    (Annot_E4_<wrist>.csv) mark the same 8 route segments, so E4 time is mapped to metric time piecewise-linearly
    through those 16 borders. Each 2-min window gets the metric's mean over its span.
  - Windows: all driving windows (car park, city, highway; not the parked rests before/after).
  - Scores: (1) the frozen FINAL BodySync, trained only on the six original datasets (never saw this dataset);
            (2) FINAL trained on the other drivers; (3) the original formula + sweat bursts.
  - Test: per drive, Spearman correlation between the score (lower = stressed, so its negative) and the metric.
    Reported: mean r over the 13 drives, drives with r > 0, sign-test p.
  - Stricter: the same inside city segments only (no city-vs-highway contrast left; moment-to-moment only).
  - Shortcuts, same test: wrist movement, and minutes since the drive started.
Output: results/driving_tracking.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr

import feature_test
from config import RESULTS
from feature_test import BURSTS, score_variant

feature_test.LAB.append("affectiveroad")   # same setup as new_datasets.py (fixed-route protocol -> lab windows)
import final_eval  # noqa: E402
from final_eval import FINAL, prepare, run  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "datasets"))
import affectiveroad as ar  # noqa: E402

warnings.filterwarnings("ignore")
DRIVING = ["park", "city1", "hwy", "city2", "city2_back", "hwy_back", "city1_back", "park_back"]
CITY = ["city1", "city2", "city2_back", "city1_back"]


def metric_per_window(w: pd.DataFrame) -> pd.Series:
    sm_annot = pd.read_csv(os.path.join(ar.DB, "Subj_metric", "Annot_Subjective_metric.csv"), header=None, skiprows=1)
    sm_annot = {r[0]: np.array(r[1:17], float) / 4 for r in sm_annot.itertuples(index=False)}
    e4 = {**ar.annotations("Left"), **{d: b for d, b in ar.annotations("Right").items() if ar.WRIST.get(d) == "Right"}}
    out = pd.Series(np.nan, index=w.index)
    for drive, g in w.groupby("drive"):
        if drive not in sm_annot:
            continue
        sm = pd.read_csv(os.path.join(ar.DB, "Subj_metric", f"SM_{drive}.csv")).iloc[:, 0].to_numpy(float)
        e_b = np.array(e4[drive][1:9]).ravel()
        t_sm = np.arange(len(sm)) / 4
        for i, s in zip(g.index, g.t_rel_s):
            a, b = np.interp([s, s + ar.WIN], e_b, sm_annot[drive])
            m = (t_sm >= a) & (t_sm < b)
            out[i] = np.nanmean(sm[m]) if m.any() else np.nan
    return out


def tracking(w: pd.DataFrame, x: pd.Series, tasks: list[str]) -> dict:
    rs = []
    for _, g in w[w.task.isin(tasks) & x.notna() & w.metric.notna()].groupby("drive"):
        if len(g) >= 10 and g.metric.nunique() > 1:
            rs.append(spearmanr(-x[g.index], g.metric).statistic)
    rs = np.array(rs)
    return {"drives": len(rs), "mean_r": rs.mean(), "drives_r_above_0": int((rs > 0).sum()),
            "p_sign": binomtest(int((rs > 0).sum()), len(rs)).pvalue}


def main():
    allw, X, data = prepare()
    allw.loc[allw.dataset == "affectiveroad", "group"] = "dev"
    ar_rows = allw.dataset == "affectiveroad"
    w = allw[ar_rows].copy()
    w["metric"] = metric_per_window(w)
    print(f"windows with a stress rating: {w.metric.notna().sum()} of {len(w)}", flush=True)
    scores = {
        "FINAL BodySync, never saw this dataset": run({**FINAL, "other_datasets": True}, allw, X, "dev",
                                                       datasets=["affectiveroad"], return_scores=True)[1][ar_rows],
        "FINAL BodySync, trained on the other drivers": run(FINAL, allw, X, "dev", datasets=["affectiveroad"],
                                                             return_scores=True)[1][ar_rows],
        "original formula + sweat bursts": pd.Series(score_variant(data["affectiveroad"][0], "lab", BURSTS,
                                                                    {"scr_rate": 1, "scr_amp": 1}).v.to_numpy(), index=w.index),
        "shortcut: wrist movement (more = stressed)": -w["move"],
        "shortcut: minutes since start (later = stressed)": -(w.t_rel_s.astype(float)),
    }
    rows = []
    for name, x in scores.items():
        for comp, tasks in (("all driving", DRIVING), ("inside city only", CITY)):
            r = {"score": name, "comparison": comp, **tracking(w, x, tasks)}
            rows.append(r)
            print(f"{name:50s} {comp:17s} drives={r['drives']:2d}  mean r={r['mean_r']:+.2f}  "
                  f"r>0 in {r['drives_r_above_0']}/{r['drives']}  p={r['p_sign']:.3f}", flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "driving_tracking.csv", index=False)


if __name__ == "__main__":
    main()
