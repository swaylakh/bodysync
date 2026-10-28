"""Round 2: do the improved coupling features (Spearman, lagged, personal) rescue the sync idea?

The fixes and the tests below were fixed before running. Every result is reported,
and p-values are Holm-corrected across all the tests in this file.
"""
import itertools

import pandas as pd
from scipy.stats import wilcoxon

from config import RESULTS
from features import COUPLING, COUPLING_V2, LEVEL, PERSONAL
from model import TASKS, load, loso, per_person, score

FEATURE_SETS = {
    "levels+personal": LEVEL + PERSONAL,
    "coupling v1": COUPLING,
    "coupling v1+v2": COUPLING + COUPLING_V2,
    "levels+personal+coupling v1+v2": LEVEL + PERSONAL + COUPLING + COUPLING_V2,
}
COMPARISONS = [
    ("coupling v1+v2", "coupling v1"),                          # did the fixes make sync more informative?
    ("levels+personal+coupling v1+v2", "levels+personal"),      # does sync now add anything over the best baseline?
]


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    adj, running = {}, 0.0
    for i, idx in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[idx]))
        adj[idx] = running
    return pd.Series(adj)[p.index]


def main():
    pooled, tests = [], []
    for task, length in itertools.product(TASKS, (30, 60)):
        classes = TASKS[task]
        w = load(length, classes)
        pp = {}
        for fs, cols in FEATURE_SETS.items():
            p = loso(w, cols, classes, "xgb")
            pooled.append({"task": task, "window_min": length, "features": fs, **score(p)})
            pp[fs] = per_person(p)
            print(f"{task:14s} {length:3d}m {fs:32s} bal_acc={pooled[-1]['balanced_acc']:.3f}")
        wide = pd.DataFrame(pp).dropna()
        for a, b in COMPARISONS:
            d = wide[a] - wide[b]
            tests.append({"task": task, "window_min": length, "comparison": f"{a} vs {b}",
                          "n_people": len(d), "median_diff": d.median(), "people_improved": int((d > 0).sum()),
                          "p_value": wilcoxon(wide[a], wide[b]).pvalue})
    pooled, tests = pd.DataFrame(pooled), pd.DataFrame(tests)
    tests["p_holm"] = holm(tests.p_value)
    pooled.to_csv(RESULTS / "v2_loso_pooled.csv", index=False)
    tests.to_csv(RESULTS / "v2_paired_tests.csv", index=False)
    print(tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
