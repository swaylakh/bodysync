"""Round 3: try many coupling variations, fairly.

Trying lots of variants and reporting the best one overstates the result, because
the best of many is partly luck. Nested cross-validation avoids that:

  outer loop: leave one person out (the final test person)
    inner loop: on the remaining people only, score every variant (5-fold by person)
                and pick the winner
    train the winner on the remaining people and test it once on the held-out person

The held-out person never influences which variant is chosen, so the nested score is
an honest estimate of "try many variants and keep the best" as a whole procedure.
"""
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from config import PAIRS, RESULTS
from features import COUPLING_DIFF, COUPLING_RAW, LAGGED, LEVEL, PERSONAL, PERSONAL_COUPLING, SPEARMAN
from model import TASKS, load, loso, make_model, per_person, score

STRENGTH = [f"abs_{c}" for c in COUPLING_RAW + SPEARMAN] + ["sync_index"]
BASE = LEVEL + PERSONAL
VARIANTS = {
    "none (levels+personal)": BASE,
    "+ pearson r": BASE + COUPLING_RAW,
    "+ change r": BASE + COUPLING_DIFF,
    "+ spearman r": BASE + SPEARMAN,
    "+ lagged r": BASE + LAGGED,
    "+ personal coupling": BASE + PERSONAL_COUPLING,
    "+ sync strength": BASE + STRENGTH,
    "+ all coupling": BASE + COUPLING_RAW + COUPLING_DIFF + SPEARMAN + LAGGED + PERSONAL_COUPLING + STRENGTH,
}


def add_strength(w: pd.DataFrame) -> pd.DataFrame:
    # strength of sync regardless of direction, and one overall BodySync-style index
    for c in COUPLING_RAW + SPEARMAN:
        w[f"abs_{c}"] = w[c].abs()
    w["sync_index"] = w[[f"abs_s_{a}_{b}" for a, b in PAIRS]].mean(axis=1)
    return w


def fit_predict(X_tr, y_tr, X_te):
    clf = make_model("xgb")
    clf.fit(X_tr, y_tr, sample_weight=compute_sample_weight("balanced", y_tr))
    return clf.predict(X_te)


def nested(w: pd.DataFrame, classes: list[str]) -> tuple[pd.DataFrame, list[str]]:
    y = w.state.map({c: i for i, c in enumerate(classes)}).to_numpy()
    X = {v: w[cols].to_numpy(float) for v, cols in VARIANTS.items()}
    pred, picks = np.full(len(w), -1), []
    for pid in w.pid.unique():
        test = (w.pid == pid).to_numpy()
        tr_idx = np.where(~test)[0]
        groups = w.pid.to_numpy()[tr_idx]
        inner = {}
        for v in VARIANTS:
            p = np.empty(len(tr_idx), int)
            for a, b in GroupKFold(n_splits=5).split(tr_idx, groups=groups):
                p[b] = fit_predict(X[v][tr_idx[a]], y[tr_idx[a]], X[v][tr_idx[b]])
            inner[v] = balanced_accuracy_score(y[tr_idx], p)
        best = max(inner, key=inner.get)
        picks.append(best)
        pred[test] = fit_predict(X[best][~test], y[~test], X[best][test])
        print(f"  test {pid[-3:]}: picked {best}")
    return pd.DataFrame({"pid": w.pid, "start": w.start, "y": y, "pred": pred}), picks


def main():
    pooled, tests, pick_rows = [], [], []
    for task in TASKS:
        classes = TASKS[task]
        w = add_strength(load(30, classes))

        # 1) every variant on its own (transparent, but the best of these is optimistic)
        pp = {}
        for v, cols in VARIANTS.items():
            p = loso(w, cols, classes, "xgb")
            pooled.append({"task": task, "method": v, **score(p)})
            pp[v] = per_person(p)
            print(f"{task:14s} {v:26s} bal_acc={pooled[-1]['balanced_acc']:.3f}")

        # 2) honest version: pick the variant inside each fold
        p, picks = nested(w, classes)
        pooled.append({"task": task, "method": "NESTED: best variant picked per fold", **score(p)})
        pp["nested"] = per_person(p)
        print(f"{task:14s} {'NESTED':26s} bal_acc={pooled[-1]['balanced_acc']:.3f}")
        pick_rows += [{"task": task, "picked": v} for v in picks]

        wide = pd.DataFrame(pp).dropna()
        base = "none (levels+personal)"
        for v in wide.columns.drop(base):
            d = wide[v] - wide[base]
            tests.append({"task": task, "method": v, "vs": base, "n_people": len(d),
                          "median_diff": d.median(), "people_improved": int((d > 0).sum()),
                          "p_value": wilcoxon(wide[v], wide[base]).pvalue if (d != 0).any() else np.nan})

    pd.DataFrame(pooled).to_csv(RESULTS / "v3_pooled.csv", index=False)
    tests = pd.DataFrame(tests)
    tests.to_csv(RESULTS / "v3_paired_tests.csv", index=False)
    picks = pd.DataFrame(pick_rows).value_counts().rename("folds").reset_index()
    picks.to_csv(RESULTS / "v3_nested_picks.csv", index=False)
    print(tests.round(4).to_string(index=False))
    print(picks.to_string(index=False))


if __name__ == "__main__":
    main()
