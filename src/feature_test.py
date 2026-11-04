"""Do richer features and a trained model help BodySync pick out stress? (fixed before looking at results)

Variants, scored with the same fair stress-vs-calm rule as validate_datasets.py:
  B                 the current BodySync score (HR, sweat level, skin temp)
  B + bursts        plus sweat-burst rate and size (more = stressed)
  B + bursts + HRV  plus RMSSD (lower = stressed)
  trained           XGBoost on "distance from your normal" for all of the above
  trained + change  the same, plus how much each input changed over the last ~10 min and
                    movement relative to the person's usual, so a rise (stress onset) and
                    exercise can be told apart
Trained models are tested two ways:
  new people    leave-one-subject-out inside each dataset
  new dataset   trained on the 4 lab datasets only (minus the test one, if it is a lab set)
The everyday-life datasets are only ever test sets for the lab-trained model, so their
protocol and labels never shape it. Sweat bursts and RMSSD come from src/datasets/richer.py.
"""
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import bodysync
from config import PROCESSED, RESULTS, SEED
from validate_datasets import CALM, evaluate

LAB = ["wesad", "stressexercise", "stresspredict", "e4stress"]
LIFE = ["nurse", "adarp", "canstress"]
BURSTS = {"scr_rate": "sweat bursts", "scr_amp": "burst size"}
HRV = {"rmssd_ms": "HRV"}
VARIANTS = {
    "B": ({}, {}),
    "B + bursts": (BURSTS, {"scr_rate": 1, "scr_amp": 1}),
    "B + bursts + HRV": ({**BURSTS, **HRV}, {"scr_rate": 1, "scr_amp": 1, "rmssd_ms": -1}),
}
MIN_SPREAD = {"scr_rate": 0.5, "scr_amp": 0.005, "rmssd_ms": 5.0}
INPUTS = list(bodysync.LEVELS) + list(BURSTS) + list(HRV)
NON_STRESS = CALM | {"baseline", "no_stress"}
CHANGE_LAG = pd.Timedelta(minutes=10)


def load(name: str) -> tuple[pd.DataFrame, str]:
    w = pd.read_parquet(PROCESSED / f"ext_{name}_windows.parquet")
    w = w[w.coverage >= 0.75].dropna(subset=["hr_mean", "eda_mean", "temp_mean"]).reset_index(drop=True)
    w["log_eda"] = np.log(w.eda_mean.clip(lower=0) + 0.01)
    w["start"] = pd.to_datetime(w["start"]).astype("datetime64[ns]")
    for c in INPUTS:
        if c not in w:
            w[c] = np.nan
    return w.assign(dataset=name), "lab" if name in LAB else "free-living"


def context(w: pd.DataFrame, setting: str) -> pd.Series:
    """Free-living: still vs moving by the person's own median movement (as validate_datasets.py)."""
    if setting == "free-living" and w["move"].notna().any():
        return pd.Series(np.where(w["move"] > w.groupby("pid")["move"].transform("median"), "moving", "still"), index=w.index)
    return pd.Series("all", index=w.index)


def score_variant(w: pd.DataFrame, setting: str, extra: dict, direction: dict) -> pd.DataFrame:
    names = {**bodysync.LEVELS, **extra}
    spread = np.array([bodysync.LEVEL_MIN_SPREAD.get(c, MIN_SPREAD.get(c)) for c in names])
    signs = np.array([{**bodysync.STRESS_DIRECTION, **direction}[c] for c in names])
    return bodysync._score(w, names, spread, signs, context(w, setting), w["normal"].astype(bool), "v")


def features(w: pd.DataFrame, change: bool) -> pd.DataFrame:
    """Each input as distance from the person's own normal, in typical spreads (no labels used)."""
    ref = w[w["normal"]].groupby("pid")[INPUTS]
    med, spr = ref.median(), ref.agg(bodysync.mad)
    floor = np.array([{**bodysync.LEVEL_MIN_SPREAD, **MIN_SPREAD}[c] for c in INPUTS])   # same guard as the score
    z = (w[INPUTS] - med.reindex(w.pid).to_numpy()) / np.maximum(spr.reindex(w.pid).to_numpy(), floor)
    z = z.add_prefix("z_")
    if not change:
        return z
    # change over ~10 min: compare with the person's window that started ~10 min earlier
    cur = pd.concat([w[["pid", "start"]], z], axis=1).reset_index().dropna(subset=["start"]).sort_values("start")
    prev = cur.drop(columns="index").assign(start=lambda d: d.start + CHANGE_LAG)
    m = pd.merge_asof(cur, prev, on="start", by="pid", suffixes=("", "_prev"),
                      tolerance=pd.Timedelta(minutes=3), direction="nearest").set_index("index")
    for c in z.columns:
        z[f"d_{c}"] = m[c] - m[f"{c}_prev"]
    mv = w.groupby("pid")["move"]
    z["move_z"] = (w["move"] - mv.transform("median")) / (mv.transform(lambda x: x.quantile(0.75) - x.quantile(0.25)) + 1e-6)
    return z


def fit_predict(X: pd.DataFrame, w: pd.DataFrame, train: pd.Series, test: pd.Series) -> np.ndarray:
    tr = train & ((w.label == "stress") | w.label.isin(NON_STRESS))
    y = (w.label == "stress").astype(int)
    # each dataset counts equally in training, whatever its size
    weight = compute_sample_weight("balanced", y[tr]) / w.loc[tr].groupby("dataset").dataset.transform("size").to_numpy()
    clf = XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=SEED, verbosity=0)
    clf.fit(X[tr], y[tr], sample_weight=weight / weight.mean())
    return 100 * (1 - clf.predict_proba(X[test])[:, 1])     # same direction as the scores: lower = stressed


def main():
    data = {name: load(name) for name in LAB + LIFE}
    allw = pd.concat([w for w, _ in data.values()], ignore_index=True)
    X = {ch: pd.concat([features(w, ch) for w, _ in data.values()], ignore_index=True) for ch in (False, True)}
    rows = []
    for name, (w, setting) in data.items():
        for label, (extra, direction) in VARIANTS.items():
            rows.append({"variant": label, **evaluate(name, score_variant(w, setting, extra, direction), setting, "v")})
        in_set = allw.dataset == name
        lab_train = allw.dataset.isin(LAB) & ~in_set
        for ch, tag in [(False, "trained"), (True, "trained + change")]:
            new_ds = allw[in_set].assign(v=fit_predict(X[ch], allw, lab_train, in_set), v_reason="")
            rows.append({"variant": f"{tag}, new dataset", **evaluate(name, new_ds, setting, "v")})
            loso = pd.Series(np.nan, index=allw.index)
            for pid in allw.loc[in_set, "pid"].unique():
                te = allw.pid == pid
                loso[te] = fit_predict(X[ch], allw, in_set & ~te, te)
            rows.append({"variant": f"{tag}, new people", **evaluate(name, allw[in_set].assign(v=loso[in_set], v_reason=""), setting, "v")})
        print(name, "done")
    res = pd.DataFrame(rows)[["variant", "dataset", "comparison", "n_people", "mean_auc", "people_auc_above_0.5",
                              "people_lower_in_stress", "p_value"]]
    res.to_csv(RESULTS / "feature_test.csv", index=False)
    print(res.pivot(index="variant", columns="dataset", values="mean_auc")[LAB + LIFE].round(2).to_string())


if __name__ == "__main__":
    main()
