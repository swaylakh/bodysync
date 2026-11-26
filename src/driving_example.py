"""One real drive for the 3D walkthrough: the frozen final system (never saw AffectiveROAD) scoring every window
of the drive whose city-vs-highway separation is clearest. Output: results/driving_example.json"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

import feature_test
from config import RESULTS

feature_test.LAB.extend(["affectiveroad"])
from final_eval import FINAL, prepare, run  # noqa: E402

RELABEL = {"highway": "calm", "rest": "later_rest"}


def main():
    allw, X, _ = prepare()
    allw.loc[allw.dataset == "affectiveroad", "group"] = "dev"
    res, score = run({**FINAL, "other_datasets": True}, allw, X, "dev", datasets=["affectiveroad"], eval_relabel=RELABEL, return_scores=True)
    print(res[["mean_auc", "people_lower_in_stress", "n_people"]])
    w = allw[allw.dataset == "affectiveroad"].assign(score=score).dropna(subset=["score"])
    best, best_auc = None, -1
    for drive, g in w.groupby("drive"):
        g = g[g.task.str.startswith(("city", "hwy"))]
        if g.task.str.startswith("city").sum() < 5 or g.task.str.startswith("hwy").sum() < 5:
            continue
        auc = roc_auc_score(g.task.str.startswith("city"), -g.score)
        print(drive, g.pid.iloc[0], round(auc, 3))
        if auc > best_auc:
            best, best_auc = drive, auc
    g = w[w.drive == best].sort_values("t_rel_s")
    out = {"drive": best, "auc_city_vs_highway": round(best_auc, 3),
           "windows": [{"t": round(float(t)/60, 2), "task": task, "score": round(float(s), 1)} for t, task, s in zip(g.t_rel_s, g.task, g.score)]}
    (RESULTS / "driving_example.json").write_text(json.dumps(out))
    seg = g.groupby("task", sort=False).agg(start=("t_rel_s", "min"), end=("t_rel_s", "max"), mean=("score", "mean"))
    print(best, best_auc); print((seg.assign(start=seg.start/60, end=seg.end/60)).round(1))


if __name__ == "__main__":
    main()
