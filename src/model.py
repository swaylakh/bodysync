"""Step 3-4: train classifiers and test them on people they have never seen (LOSO).

Core question: do coupling ("moving together") features add anything over plain
signal levels when telling sleep / rest / active apart?

Every combination of task x window length x model x feature set is evaluated with
leave-one-subject-out cross-validation. Per-person scores are then compared between
feature sets with a paired Wilcoxon signed-rank test.
"""
import itertools

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import PROCESSED, RESULTS, SEED, STATES
from features import COUPLING, LEVEL, PERSONAL

FEATURE_SETS = {
    "levels": LEVEL,
    "levels+personal": LEVEL + PERSONAL,
    "coupling": COUPLING,
    "levels+coupling": LEVEL + COUPLING,
    "all": LEVEL + PERSONAL + COUPLING,
}
TASKS = {
    "3-class": STATES,
    "sleep-vs-rest": ["sleep", "rest"],   # the hard pair: both are still, so movement can't separate them
}
MIN_WINDOWS_PER_PERSON = 10


def make_model(name: str):
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                      n_jobs=-1, random_state=SEED)
    return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbosity=0)


def load(length: int, classes: list[str], eda_filter: bool = True) -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED / f"windows_{length}.parquet")
    w = w[w.labelled & w.state.isin(classes)]
    if eda_filter:
        w = w[w.eda_ok]
    n = w.groupby("pid").size()
    return w[w.pid.isin(n[n >= MIN_WINDOWS_PER_PERSON].index)].reset_index(drop=True)


def loso(w: pd.DataFrame, features: list[str], classes: list[str], model: str) -> pd.DataFrame:
    X = w[features].to_numpy(float)
    y = w.state.map({c: i for i, c in enumerate(classes)}).to_numpy()
    pred = np.full(len(w), -1)
    for pid in w.pid.unique():
        test = (w.pid == pid).to_numpy()
        clf = make_model(model)
        if model == "xgb":
            clf.fit(X[~test], y[~test], sample_weight=compute_sample_weight("balanced", y[~test]))
        else:
            clf.fit(X[~test], y[~test])
        pred[test] = clf.predict(X[test])
    return pd.DataFrame({"pid": w.pid, "start": w.start, "y": y, "pred": pred})


def score(p: pd.DataFrame) -> dict:
    return {
        "balanced_acc": balanced_accuracy_score(p.y, p.pred),
        "macro_f1": f1_score(p.y, p.pred, average="macro"),
    }


def per_person(p: pd.DataFrame) -> pd.Series:
    # balanced accuracy only makes sense for people who have more than one state
    return p.groupby("pid").apply(
        lambda g: balanced_accuracy_score(g.y, g.pred) if g.y.nunique() > 1 else np.nan,
        include_groups=False,
    )


def main():
    pooled, persons, preds = [], [], {}
    configs = [
        (task, length, model, eda_filter)
        for task, length, model in itertools.product(TASKS, (30, 60), ("rf", "xgb"))
        for eda_filter in (True,)
    ] + [("3-class", 30, "xgb", False), ("sleep-vs-rest", 30, "xgb", False)]   # sensitivity: keep low-EDA windows

    for task, length, model, eda_filter in configs:
        classes = TASKS[task]
        w = load(length, classes, eda_filter)
        for fs, cols in FEATURE_SETS.items():
            p = loso(w, cols, classes, model)
            key = dict(task=task, window_min=length, model=model, eda_filter=eda_filter, features=fs)
            pooled.append({**key, **score(p), "n_windows": len(w), "n_people": w.pid.nunique()})
            pp = per_person(p).rename("balanced_acc").reset_index()
            persons.append(pp.assign(**key))
            preds["|".join(map(str, key.values()))] = p
            print(f"{task:14s} {length:3d}m {model:3s} edaF={eda_filter!s:5s} {fs:16s} "
                  f"bal_acc={pooled[-1]['balanced_acc']:.3f} f1={pooled[-1]['macro_f1']:.3f}")

    pooled = pd.DataFrame(pooled)
    persons = pd.concat(persons, ignore_index=True)
    pooled.to_csv(RESULTS / "loso_pooled.csv", index=False)
    persons.to_csv(RESULTS / "loso_per_person.csv", index=False)
    pd.concat({k: v for k, v in preds.items()}, names=["config"]).reset_index(level=0).to_parquet(
        RESULTS / "loso_predictions.parquet", index=False)

    # paired tests across people: does adding coupling beat the matching level-only set?
    tests = []
    keys = ["task", "window_min", "model", "eda_filter"]
    for k, g in persons.groupby(keys):
        wide = g.pivot(index="pid", columns="features", values="balanced_acc").dropna()
        for a, b in [("levels+coupling", "levels"), ("all", "levels+personal"), ("coupling", "levels")]:
            d = wide[a] - wide[b]
            stat = wilcoxon(wide[a], wide[b]) if (d != 0).any() else None
            tests.append({**dict(zip(keys, k)), "comparison": f"{a} vs {b}", "n_people": len(d),
                          "median_diff": d.median(), "mean_diff": d.mean(),
                          "people_improved": int((d > 0).sum()),
                          "p_value": stat.pvalue if stat else np.nan})
    tests = pd.DataFrame(tests)
    tests.to_csv(RESULTS / "loso_paired_tests.csv", index=False)
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
