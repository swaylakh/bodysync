"""The very first version: the plain BodySync formula, WITHOUT sweat bursts, scored on the same people as
every other number in the final deck (DEV 60%, LOCKED TEST 40%, and AffectiveROAD city vs highway).
The formula has no training, so nothing is fitted. Output: results/plain_formula.csv
"""
import pandas as pd

import feature_test
from config import RESULTS
from feature_test import BURSTS, score_variant
from validate_datasets import evaluate

feature_test.LAB.extend(["affectiveroad"])
from final_eval import prepare  # noqa: E402

DATASETS = ["wesad", "stressexercise", "stresspredict", "canstress"]   # the four main studies in the deck

RELABEL = {"highway": "calm", "rest": "later_rest"}
SYSTEMS = {"plain formula": ({}, {}), "formula + sweat bursts": (BURSTS, {"scr_rate": 1, "scr_amp": 1})}


def main():
    allw, _, data = prepare()
    rows = []
    for system, (extra, direction) in SYSTEMS.items():
        for name in DATASETS:
            w, setting = data[name]
            s = score_variant(w, setting, extra, direction)
            grp = allw.loc[allw.dataset == name, "group"].to_numpy()
            for group in ("dev", "test"):
                r = evaluate(name, s[grp == group], setting, "v")
                rows.append({"system": system, "dataset": name, "people": group, "auc": round(r["mean_auc"], 3),
                             "lower_in_stress": f"{r['people_lower_in_stress']}/{r['n_people']}"})
                print(rows[-1], flush=True)
        w, setting = data["affectiveroad"]
        for comp, relabel in [("city vs highway", RELABEL), ("city vs rest", {})]:
            s = score_variant(w, setting, extra, direction).assign(label=lambda d: d.label.replace(relabel))
            r = evaluate("affectiveroad", s, setting, "v")
            rows.append({"system": system, "dataset": f"affectiveroad ({comp})", "people": "all (new)",
                         "auc": round(r["mean_auc"], 3), "lower_in_stress": f"{r['people_lower_in_stress']}/{r['n_people']}"})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "plain_formula.csv", index=False)


if __name__ == "__main__":
    main()
