"""Do richer features help BodySync pick out lab stress? (fixed before looking at results)

On the four lab datasets, compare within-person AUC (same fair stress-vs-calm rule as
validate_datasets.py) for:
  B           the current BodySync score (HR, sweat level, skin temp)
  B + bursts  plus sweat-burst rate and size (more = stressed)
  B + bursts + HRV  plus RMSSD (lower = stressed)
  trained     XGBoost on the same "distance from your normal" inputs, each person tested on a
              model trained on the other people of that dataset (leave-one-subject-out)
  trained, other datasets   the same model trained only on the OTHER lab datasets, so the
              test dataset's protocol is never seen (leave-one-dataset-out)
Sweat bursts and RMSSD come from src/datasets/richer.py.
"""
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import bodysync
from config import PROCESSED, RESULTS, SEED
from validate_datasets import CALM, evaluate

LAB = ["wesad", "stressexercise", "stresspredict", "e4stress"]
BURSTS = {"scr_rate": "sweat bursts", "scr_amp": "burst size"}
HRV = {"rmssd_ms": "HRV"}
VARIANTS = {
    "B": ({}, {}),
    "B + bursts": (BURSTS, {"scr_rate": 1, "scr_amp": 1}),
    "B + bursts + HRV": ({**BURSTS, **HRV}, {"scr_rate": 1, "scr_amp": 1, "rmssd_ms": -1}),
}
MIN_SPREAD = {"scr_rate": 0.5, "scr_amp": 0.005, "rmssd_ms": 5.0}


def score_variant(w: pd.DataFrame, extra: dict, direction: dict, out: str) -> pd.DataFrame:
    names = {**bodysync.LEVELS, **extra}
    spread = np.array([bodysync.LEVEL_MIN_SPREAD.get(c, MIN_SPREAD.get(c)) for c in names])
    signs = np.array([{**bodysync.STRESS_DIRECTION, **direction}[c] for c in names])
    return bodysync._score(w, names, spread, signs, pd.Series("all", index=w.index), w["normal"].astype(bool), out)


def personal_z(w: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Each input as distance from the person's own baseline, in typical spreads (no labels used)."""
    ref = w[w["normal"]].groupby("pid")[cols]
    med, spr = ref.median(), ref.agg(bodysync.mad)
    z = (w[cols] - med.reindex(w.pid).to_numpy()) / np.maximum(spr.reindex(w.pid).to_numpy(), 1e-6)
    return z.add_prefix("z_")


def fit_predict(X: pd.DataFrame, w: pd.DataFrame, train: pd.Series, test: pd.Series) -> np.ndarray:
    tr = train & ((w.label == "stress") | w.label.isin(CALM | {"baseline"}))
    y = (w.label == "stress").astype(int)
    clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=SEED, verbosity=0)
    clf.fit(X[tr], y[tr], sample_weight=compute_sample_weight("balanced", y[tr]))
    return 100 * (1 - clf.predict_proba(X[test])[:, 1])     # same direction as the scores: lower = stressed


def trained(w: pd.DataFrame, X: pd.DataFrame) -> pd.Series:
    """Leave-one-subject-out within one dataset."""
    out = pd.Series(np.nan, index=w.index)
    for pid in w.pid.unique():
        out[w.pid == pid] = fit_predict(X, w, w.pid != pid, w.pid == pid)
    return out


def main():
    rows, data = [], {}
    cols = list(bodysync.LEVELS) + list(BURSTS) + list(HRV)
    for name in LAB:
        w = pd.read_parquet(PROCESSED / f"ext_{name}_windows.parquet")
        w = w[w.coverage >= 0.75].dropna(subset=["hr_mean", "eda_mean", "temp_mean"]).reset_index(drop=True)
        w["log_eda"] = np.log(w.eda_mean.clip(lower=0) + 0.01)
        data[name] = w.assign(dataset=name)
        for label, (extra, direction) in VARIANTS.items():
            s = score_variant(w, extra, direction, "v")
            rows.append({"variant": label, **evaluate(name, s, "lab", "v")})
        s = w.assign(v=trained(w, personal_z(w, cols)), v_reason="")
        rows.append({"variant": "trained (XGBoost)", **evaluate(name, s, "lab", "v")})

    allw = pd.concat(data.values(), ignore_index=True)
    X = pd.concat([personal_z(d, cols) for d in data.values()], ignore_index=True)
    for name in LAB:
        test = allw.dataset == name
        s = allw[test].assign(v=fit_predict(X, allw, ~test, test), v_reason="")
        rows.append({"variant": "trained, other datasets", **evaluate(name, s, "lab", "v")})
    res = pd.DataFrame(rows)[["variant", "dataset", "comparison", "n_people", "mean_auc", "people_auc_above_0.5",
                              "people_lower_in_stress", "p_value"]]
    res.to_csv(RESULTS / "feature_test.csv", index=False)
    print(res.round(3).to_string(index=False))
    print(res.pivot(index="variant", columns="dataset", values="mean_auc").round(2).to_string())


if __name__ == "__main__":
    main()
