"""Final held-out test on the nurse dataset (run once; nothing was tuned on it).

The nurses' survey intervals were chosen by the dataset authors' EDA-based detector, so
"stress vs calm" is partly built in (stress windows were picked for high sweat). The fair
test is stress vs no_stress: both were flagged by the detector, and only the nurse's own
rating differs. Both are reported; the fair one is the headline.
"""
import pandas as pd

import stress_model as sm
from config import RESULTS


def main():
    data = sm.build_with_labels({"no_stress"})
    train = data[data.dataset != sm.FINAL_TEST]
    te = data[data.dataset == sm.FINAL_TEST].copy()
    te["frozen_v2"] = -te.bodysync
    for name, model in sm.models().items():
        te[name] = sm.fit(model, train).predict_proba(te[sm.FEATURES].fillna(0).to_numpy())[:, 1]
    rows = []
    for test, against in [("FAIR: felt stress vs flagged-but-not-stressed", {"no_stress"}),
                          ("stress vs calm (inflated: intervals picked by an EDA detector)", {"calm"})]:
        sub = te[(te.label == "stress") | te.label.isin(against)]
        row = {"test": test}
        for name in ("frozen_v2", *sm.models()):
            auc, k, n = sm.within_person_auc(sub, name)
            row.update({f"{name}_auc": auc, f"{name}_people_above_0.5": k, "people": n})
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "stress_model_final_nurse.csv", index=False)
    print(res.round(3).T.to_string())


if __name__ == "__main__":
    main()
