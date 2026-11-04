"""BodySync on every public Empatica stress dataset we could get.

Each dataset is first converted by src/datasets/<name>.py into a window table (see
src/datasets/SPEC.md). Every dataset is then scored the same way:
  - normal = rows with normal=True (lab: the baseline rest; free-living: all windows)
  - free-living data is split into still / moving by each person's own movement, so
    moving is compared with the person's own moving (these datasets have no sleep labels)
  - both scores (A sync, B BodySync; see bodysync.py) on the same windows
  - per person: mean score in stress vs calm windows, and a within-person AUC
    (how well a lower score picks out stress windows; 0.5 = chance)

What counts as "calm" is chosen so the test isn't rigged in the score's favour:
  - lab: LATER rest, amusement or meditation, never the baseline that set the normal
    (EmpaticaE4Stress has no later rest, so it falls back to baseline: a weaker test)
  - nurse: intervals the dataset's own EDA-based detector flagged where the nurse said
    "no stress". Calm periods far from any flag would be easier, because the stress
    intervals were picked for high sweat.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score

import bodysync
from config import FIGURES, PROCESSED, RESULTS

CALM = {"calm", "rest", "relax", "amusement", "meditation"}
CALM_OVERRIDE = {"nurse": ({"no_stress"}, "felt stress vs flagged but not stressed")}
MIN_WINDOWS = 3
SCORES = {"sync": "A sync", "bodysync": "B BodySync"}


def load_all() -> dict[str, tuple[pd.DataFrame, str]]:
    sets = {}
    for f in sorted(PROCESSED.glob("ext_*_windows.parquet")):
        w = pd.read_parquet(f)
        sets[f.stem.removeprefix("ext_").removesuffix("_windows")] = (
            w, "lab" if w["normal"].mean() < 0.999 else "free-living")
    return sets


def score(w: pd.DataFrame, setting: str) -> pd.DataFrame:
    w = w[w.coverage >= 0.75].dropna(subset=["hr_mean", "eda_mean", "temp_mean"]).reset_index(drop=True)
    if setting == "free-living" and w["move"].notna().any():
        ctx = np.where(w["move"] > w.groupby("pid")["move"].transform("median"), "moving", "still")
    else:
        ctx = "all"
    return bodysync.compute(w, context=pd.Series(ctx, index=w.index), normal=w["normal"].astype(bool))


def evaluate(name: str, s: pd.DataFrame, setting: str, col: str) -> dict:
    calm, comparison = CALM_OVERRIDE.get(name, (CALM, "stress vs calm/rest"))
    if not s.label.isin(calm).any():
        calm, comparison = {"baseline"}, "stress vs baseline (weaker: baseline set the normal)"
    s = s[(s.label == "stress") | s.label.isin(calm)].assign(stress=lambda d: d.label == "stress")
    s = s.dropna(subset=[col])
    per = s.groupby(["pid", "stress"])[col].agg(["mean", "size"]).unstack()
    per = per["mean"][(per["size"] >= MIN_WINDOWS).all(axis=1)]
    aucs = s[s.pid.isin(per.index)].groupby("pid").apply(
        lambda g: roc_auc_score(g.stress, -g[col]), include_groups=False)
    reasons = s[s.stress][f"{col}_reason"].value_counts(normalize=True)
    return {"score": SCORES.get(col, col), "dataset": name, "setting": setting, "comparison": comparison, "n_people": len(per),
            "score_calm": per[False].mean(), "score_stress": per[True].mean(),
            "people_lower_in_stress": int((per[True] < per[False]).sum()),
            "p_value": wilcoxon(per[True], per[False]).pvalue if len(per) >= 6 else np.nan,
            "mean_auc": aucs.mean(), "people_auc_above_0.5": int((aucs > 0.5).sum()),
            "top_reason_in_stress": f"{reasons.index[0]} ({reasons.iloc[0]:.0%})"}


def plot(res: pd.DataFrame):
    res = res[res.score == SCORES["bodysync"]].sort_values("mean_auc").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.5, 0.6 * len(res) + 1.5))
    y = np.arange(len(res))
    ax.hlines(y, res.score_stress, res.score_calm, color="#bbb", lw=3)
    ax.scatter(res.score_calm, y, color="#5fa55a", s=60, label="calm / rest", zorder=3)
    ax.scatter(res.score_stress, y, color="#e07b39", s=60, label="stress", zorder=3)
    weaker = res.comparison.str.startswith("stress vs baseline")
    ax.set_yticks(y, [f"{r.dataset}{' (weaker test)' if wk else ''}\n{r.setting}, n={r.n_people}, AUC {r.mean_auc:.2f}"
                      for r, wk in zip(res.itertuples(), weaker)])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Average BodySync score")
    ax.set_title("BodySync: calm vs stress across Empatica datasets")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_all_datasets.png", dpi=150)
    plt.close(fig)


def plot_a_vs_b(res: pd.DataFrame):
    auc = res.pivot(index="dataset", columns="score", values="mean_auc").sort_values(SCORES["bodysync"])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(auc) + 1.5))
    y = np.arange(len(auc))
    ax.barh(y + 0.2, auc[SCORES["sync"]], height=0.38, color="#9e9e9e", label="A: sync")
    ax.barh(y - 0.2, auc[SCORES["bodysync"]], height=0.38, color="#00a896", label="B: BodySync")
    ax.axvline(0.5, color="k", lw=0.8, ls="--")
    ax.set_yticks(y, auc.index)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("Within-person AUC, stress vs calm (0.5 = chance)")
    ax.set_title("Which score drops under stress?")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_sync_vs_bodysync.png", dpi=150)
    plt.close(fig)


def main():
    rows, ex = [], []
    for name, (w, setting) in load_all().items():
        s = score(w, setting)
        rows += [evaluate(name, s, setting, col) for col in SCORES]
        if s.label.isin(["aerobic", "anaerobic"]).any():
            ex.append(s.groupby("label")[list(SCORES)].median().round(1).rename(columns=SCORES))
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS / "datasets_stress_vs_calm.csv", index=False)
    print(res.round(3).to_string(index=False))
    if ex:
        ex = pd.concat(ex)
        ex.to_csv(RESULTS / "datasets_exercise_vs_stress.csv")
        print("\nMedian score by condition (exercise dataset)\n", ex.to_string())
    plot(res)
    plot_a_vs_b(res)


if __name__ == "__main__":
    main()
