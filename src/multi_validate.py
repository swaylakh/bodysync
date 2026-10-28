"""Run the same BodySync check on every Empatica dataset we could get.

Each dataset is converted by src/datasets/<name>.py into a window table (see
src/datasets/SPEC.md). Here every dataset is scored the same way:
  - normal = rows with normal=True (lab: the baseline rest; free-living: all windows)
  - free-living data is split into still / moving by each person's own movement,
    so moving is compared with the person's own moving (no sleep labels exist here)
  - per person: mean score in stress vs in calm/rest windows, and a within-person AUC
Lab datasets compare stress against LATER rest (not the baseline that set the normal),
so the comparison isn't rigged in the score's favour.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import bodysync
from config import FIGURES, PROCESSED, RESULTS

OUT = FIGURES / "bodysync"
CALM_LABELS = {"calm", "rest", "relax", "amusement", "meditation"}
MIN_WINDOWS = 3
# v1: any distance from normal; v2: only the stress direction counts (HR up, sweat up, temp down)
VERSIONS = {"v1 any change": False, "v2 stress direction": True}


def wesad_table() -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED / "wesad_windows.parquet")
    return pd.DataFrame({
        "pid": "wesad_" + w.pid, "start": pd.to_datetime(w.start, unit="s"), "label": w.state,
        "normal": w.state == "baseline", "hr_mean": w.hr_mean, "eda_mean": w.eda_mean,
        "temp_mean": w.temp_mean, "move": np.nan, "coverage": 1.0,
    })


def load_all() -> dict[str, tuple[pd.DataFrame, str]]:
    sets = {"wesad": (wesad_table(), "lab")}
    for f in sorted(PROCESSED.glob("ext_*_windows.parquet")):
        name = f.stem.removeprefix("ext_").removesuffix("_windows")
        w = pd.read_parquet(f)
        setting = "lab" if w["normal"].mean() < 0.999 else "free-living"
        sets[name] = (w, setting)
    return sets


def score_dataset(w: pd.DataFrame, setting: str, directional: bool = False) -> pd.DataFrame:
    w = w[w.coverage >= 0.75].dropna(subset=["hr_mean", "eda_mean", "temp_mean"]).reset_index(drop=True)
    if setting == "free-living" and w["move"].notna().any():
        thresh = w.groupby("pid")["move"].transform("median")
        ctx = pd.Series(np.where(w["move"] > thresh, "moving", "still"), index=w.index)
    else:
        ctx = pd.Series("all", index=w.index)
    return bodysync.compute(w, context=ctx, normal=w["normal"].astype(bool), directional=directional)


def evaluate(name: str, s: pd.DataFrame, setting: str) -> dict:
    calm = CALM_LABELS
    comparison = "stress vs calm/rest"
    if not s.label.isin(CALM_LABELS).any():
        # no later rest recorded: fall back to the baseline that set the normal (a weaker test)
        calm, comparison = {"baseline"}, "stress vs baseline (weaker: baseline set the normal)"
    s = s.assign(group=np.where(s.label == "stress", "stress",
                                np.where(s.label.isin(calm), "calm", "other")))
    s = s[s.group != "other"]
    per = s.groupby(["pid", "group"]).bodysync.agg(["mean", "size"]).unstack()
    ok = (per["size"] >= MIN_WINDOWS).all(axis=1)
    per = per["mean"][ok]
    aucs = s[s.pid.isin(per.index)].groupby("pid").apply(
        lambda g: roc_auc_score(g.group == "stress", -g.bodysync), include_groups=False)
    res = {"dataset": name, "setting": setting, "comparison": comparison, "n_people": len(per),
           "score_calm": per.calm.mean(), "score_stress": per.stress.mean(),
           "people_lower_in_stress": int((per.stress < per.calm).sum()),
           "p_value": wilcoxon(per.stress, per.calm).pvalue if len(per) >= 6 else np.nan,
           "mean_auc": aucs.mean(), "people_auc_above_0.5": int((aucs > 0.5).sum())}
    reasons = s[s.group == "stress"].main_reason.value_counts(normalize=True)
    res["top_reason_in_stress"] = f"{reasons.index[0]} ({reasons.iloc[0]:.0%})" if len(reasons) else ""
    return res


def exercise_check(name: str, s: pd.DataFrame) -> pd.DataFrame | None:
    if not s.label.isin(["aerobic", "anaerobic"]).any():
        return None
    return s.groupby("label").bodysync.agg(["mean", "median", "size"]).round(1).assign(dataset=name)


def main():
    rows, ex = [], []
    for name, (w, setting) in load_all().items():
        for version, directional in VERSIONS.items():
            s = score_dataset(w, setting, directional)
            try:
                rows.append({"version": version, **evaluate(name, s, setting)})
            except (KeyError, ValueError) as e:
                print(f"{name}: could not evaluate ({e})")
            e = exercise_check(name, s)
            if e is not None:
                ex.append(e.assign(version=version))
        print(f"{name}: {s.pid.nunique()} people, labels {s.label.value_counts().to_dict()}")
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "bodysync_all_datasets.csv", index=False)
    print(res.round(3).to_string(index=False))
    if ex:
        ex = pd.concat(ex)
        ex.to_csv(RESULTS / "bodysync_exercise_vs_stress.csv")
        print(ex.to_string())

    res = res[res.version == "v2 stress direction"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(res) + 1.5))
    y = np.arange(len(res))
    ax.hlines(y, res.score_stress, res.score_calm, color="#bbb", lw=3)
    ax.scatter(res.score_calm, y, color="#5fa55a", s=60, label="calm / rest", zorder=3)
    ax.scatter(res.score_stress, y, color="#e07b39", s=60, label="stress", zorder=3)
    ax.set_yticks(y, [f"{r.dataset} ({r.setting}, n={r.n_people}, AUC {r.mean_auc:.2f})" for r in res.itertuples()])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Average BodySync score")
    ax.set_title("BodySync: calm vs stress across Empatica datasets")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "all_datasets.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
