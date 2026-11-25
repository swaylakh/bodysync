"""Does self-training help on real driving? The frozen FINAL system with and without self-training,
on AffectiveROAD city vs highway (and city vs rest). Nothing else changes. Output: results/driving_selftrain.csv
"""
import pandas as pd

import feature_test
from config import RESULTS

feature_test.LAB.extend(["affectiveroad"])
from final_eval import FINAL, prepare, run  # noqa: E402

RELABEL = {"highway": "calm", "rest": "later_rest"}


def main():
    allw, X, _ = prepare()
    allw.loc[allw.dataset == "affectiveroad", "group"] = "dev"
    rows = []
    for st in (False, True):
        for setup, extra in [("new dataset (never saw it)", {"other_datasets": True}), ("new people", {})]:
            cfg = {**FINAL, **extra, "selftrain": st}
            for comp, relabel in [("city vs highway", RELABEL), ("city vs rest", None)]:
                r = run(cfg, allw, X, "dev", datasets=["affectiveroad"], eval_relabel=relabel).iloc[0]
                rows.append({"self_training": st, "setup": setup, "comparison": comp,
                             "auc": round(r.mean_auc, 3), "lower_in_stress": f"{r.people_lower_in_stress}/{r.n_people}"})
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "driving_selftrain.csv", index=False)


if __name__ == "__main__":
    main()
