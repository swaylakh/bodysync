"""Brand-new datasets, never used while building BodySync: the FROZEN final system, unchanged.

Protocol (written before running; run once):
  - System: final_eval.FINAL exactly as frozen; nothing is tuned on these datasets.
  - Every person in a new dataset is a fresh test person.
  - "new people":  each person is scored by a model trained on the OTHER people of the same dataset
                   (the same recipe the final system uses everywhere).
  - "new dataset": the model never sees this dataset at all: trained only on the DEV people of the
                   six original datasets.
  - Main comparison: stress vs the later rest, the same rule as every lab dataset.
  - AffectiveROAD (real driving) also gets a stricter comparison: city vs highway. Both are driving,
    so moving hands and a running engine cannot explain the difference.
  - References: the original formula, and the trained model from the start of the final push.
  - Leak check: FINAL with each person's labels shuffled (should be ~0.5).
Output: results/new_datasets.csv
"""
import pandas as pd

import feature_test
from config import RESULTS
from feature_test import BURSTS, score_variant
from validate_datasets import evaluate

NEW = ["affectiveroad"]
feature_test.LAB.extend(NEW)   # fixed-route protocol with a baseline rest -> lab window format

import final_eval  # noqa: E402  (after LAB is extended; final_eval shares the same list)
from final_eval import BASE, FINAL, prepare, run  # noqa: E402

STRICT = {"affectiveroad": ({"highway": "calm", "rest": "later_rest"}, "city vs highway (both driving)")}


def main():
    allw, X, data = prepare()
    allw.loc[allw.dataset.isin(NEW), "group"] = "dev"   # all of them are test people; "dev" lets them train each other
    out = []

    def add(system, res, comparison=None):
        res = res.assign(system=system)
        if comparison:
            res["comparison"] = comparison
        out.append(res)
        for r in res.itertuples():
            print(f"{system:55s} {r.dataset:14s} {r.comparison:32s} n={r.n_people:2d}  AUC {r.mean_auc:.3f}  "
                  f"lower in stress {r.people_lower_in_stress}/{r.n_people}")

    for name in NEW:
        w, setting = data[name]
        add("formula: BodySync + bursts", pd.DataFrame([evaluate(name, score_variant(w, setting, BURSTS, {"scr_rate": 1, "scr_amp": 1}), setting, "v")]))
        add("trained model (start of final push), new people", run(BASE, allw, X, "dev", datasets=[name]))
        add("FINAL BodySync, new people", run(FINAL, allw, X, "dev", datasets=[name]))
        add("FINAL BodySync, new dataset (never saw it)", run({**FINAL, "other_datasets": True}, allw, X, "dev", datasets=[name]))
        add("FINAL, shuffled labels (should be ~0.5)", run(FINAL, allw, X, "dev", shuffle=True, datasets=[name]))
        if name in STRICT:
            relabel, comp = STRICT[name]
            add("formula: BodySync + bursts", pd.DataFrame([evaluate(
                name, score_variant(w, setting, BURSTS, {"scr_rate": 1, "scr_amp": 1}).assign(
                    label=lambda d: d.label.replace(relabel)), setting, "v")]), comp)
            add("FINAL BodySync, new people", run(FINAL, allw, X, "dev", datasets=[name], eval_relabel=relabel), comp)
            add("FINAL BodySync, new dataset (never saw it)",
                run({**FINAL, "other_datasets": True}, allw, X, "dev", datasets=[name], eval_relabel=relabel), comp)
    pd.concat(out, ignore_index=True).to_csv(RESULTS / "new_datasets.csv", index=False)


if __name__ == "__main__":
    main()
