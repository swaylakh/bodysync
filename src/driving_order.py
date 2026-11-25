"""Steps added one by one (self-training first, or last with --self-last) on real driving, so the progress chart can average all five studies.
Frozen system's steps added one by one; AffectiveROAD city vs highway, model never saw the dataset.
Output: results/driving_order.csv
"""
import sys

import pandas as pd

import feature_test
from config import RESULTS

feature_test.LAB.extend(["affectiveroad"])
from final_eval import BASE, prepare, run  # noqa: E402

RELABEL = {"highway": "calm", "rest": "later_rest"}


def main():
    allw, X, _ = prepare()
    allw.loc[allw.dataset == "affectiveroad", "group"] = "dev"
    last = len(sys.argv) > 1 and sys.argv[1] == "--self-last"
    steps = [("+ smoothing", {"smooth": True}), ("+ fast/slow sweat + trends", {"shape": True}),
             ("+ change over 10 min", {"change": True})]
    st = ("+ self-training", {"selftrain": True})
    order = [("base: trained, no sweat bursts", {"drop": ("z_scr", "d_z_scr")}), ("base: trained", {"drop": ()})] + (steps + [st] if last else [st] + steps)
    cfg, rows = {**BASE, "other_datasets": True}, []
    for name, step in order:
        cfg = {**cfg, **step}
        r = run(cfg, allw, X, "dev", datasets=["affectiveroad"], eval_relabel=RELABEL).iloc[0]
        rows.append({"step": name, "auc": round(r.mean_auc, 3), "lower_in_stress": f"{r.people_lower_in_stress}/{r.n_people}"})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / ("driving_order_self_last.csv" if last else "driving_order.csv"), index=False)


if __name__ == "__main__":
    main()
