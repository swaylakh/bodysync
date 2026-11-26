"""Does "moving together" (coupling) add anything over plain signal values?

XGBoost tells sleep / rest / active apart from 30-min ZU-PWD windows, tested on people
it has never seen (leave-one-subject-out). Per-person balanced accuracy is compared
between feature sets with a paired Wilcoxon test. This is the test that made us drop
the original sync idea in favour of the personal-normal score.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import PROCESSED, RESULTS, SEED, STATES
from features import COUPLING, LEVEL, PERSONAL

FEATURE_SETS = {
    "coupling only": COUPLING,
    "levels": LEVEL,
    "levels + personal": LEVEL + PERSONAL,
    "levels + personal + coupling": LEVEL + PERSONAL + COUPLING,
}
TASKS = {"sleep / rest / active": STATES, "sleep vs rest": ["sleep", "rest"]}
COMPARISONS = [("levels + personal + coupling", "levels + personal"), ("coupling only", "levels")]
MIN_WINDOWS_PER_PERSON = 10


def load(classes: list[str]) -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED / "windows_30.parquet")
    w = w[w.labelled & w.eda_ok & w.state.isin(classes)]
    n = w.groupby("pid").size()
    return w[w.pid.isin(n[n >= MIN_WINDOWS_PER_PERSON].index)].reset_index(drop=True)


def loso(w: pd.DataFrame, cols: list[str], classes: list[str]) -> pd.DataFrame:
    X = w[cols].to_numpy(float)
    y = w.state.map({c: i for i, c in enumerate(classes)}).to_numpy()
    pred = np.empty(len(w), int)
    for pid in w.pid.unique():
        test = (w.pid == pid).to_numpy()
        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, n_jobs=-1, random_state=SEED, verbosity=0)
        clf.fit(X[~test], y[~test], sample_weight=compute_sample_weight("balanced", y[~test]))
        pred[test] = clf.predict(X[test])
    return pd.DataFrame({"pid": w.pid, "y": y, "pred": pred})


def main():
    pooled, tests = [], []
    for task, classes in TASKS.items():
        w = load(classes)
        per = {}
        for name, cols in FEATURE_SETS.items():
            p = loso(w, cols, classes)
            pooled.append({"task": task, "features": name, "balanced_acc": balanced_accuracy_score(p.y, p.pred),
                           "n_people": w.pid.nunique(), "n_windows": len(w)})
            # balanced accuracy only makes sense for people with more than one state
            per[name] = p.groupby("pid").apply(
                lambda g: balanced_accuracy_score(g.y, g.pred) if g.y.nunique() > 1 else np.nan,
                include_groups=False)
            print(f"{task:22s} {name:30s} {pooled[-1]['balanced_acc']:.3f}")
        per = pd.DataFrame(per).dropna()
        for a, b in COMPARISONS:
            d = per[a] - per[b]
            tests.append({"task": task, "comparison": f"{a} vs {b}", "n_people": len(d), "median_gain": d.median(),
                          "people_better": int((d > 0).sum()), "p_value": wilcoxon(per[a], per[b]).pvalue})
    pd.DataFrame(pooled).to_csv(RESULTS / "coupling_test.csv", index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(RESULTS / "coupling_test_paired.csv", index=False)
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
