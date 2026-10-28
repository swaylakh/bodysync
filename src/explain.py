"""Step 5: explain the models and build the "desync" score.

- SHAP: which features the XGBoost model relies on
- UMAP "body weather map": every 30-min window placed in 2D by its features
- Desync score: how far a window's coupling is from that person's usual coupling
  *in the same state* (activity-aware personal baseline). There are no stress
  labels in the free data, so the score is checked against two things we can see:
  windows the classifier got wrong, and mixed "transition" windows.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import umap
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from config import FIGURES, PROCESSED, RESULTS, SEED, STATE_COLORS, STATES
from explore import LABELS
from features import COUPLING, LEVEL, PERSONAL
from model import load, make_model

sns.set_theme(style="whitegrid", font_scale=0.9)
ALL = LEVEL + PERSONAL + COUPLING


def nice(c: str) -> str:
    base = c.replace("_mean_z", "").replace("_mean", "").replace("_sd", "")
    if c.startswith(("r_", "dr_")):
        return "sync " + LABELS[c]
    suffix = " (vs own normal)" if c.endswith("_z") else " SD" if c.endswith("_sd") else " mean"
    return LABELS[base].split(" (")[0] + suffix


def shap_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    rows = []
    for ax, (task, classes) in zip(axes, [("3-class", STATES), ("sleep-vs-rest", ["sleep", "rest"])]):
        w = load(30, classes)
        X = w[ALL]
        y = w.state.map({c: i for i, c in enumerate(classes)})
        clf = make_model("xgb").fit(X, y, sample_weight=compute_sample_weight("balanced", y))
        sv = shap.TreeExplainer(clf).shap_values(X)
        sv = np.abs(sv).mean(axis=0)
        imp = (sv.mean(axis=1) if sv.ndim == 2 else sv)
        imp = pd.Series(imp, index=ALL).sort_values()
        colors = ["#e07b39" if c in COUPLING else "#3b5ba5" for c in imp.index]
        ax.barh([nice(c) for c in imp.index], imp.values, color=colors)
        ax.set_title(f"{task}: mean |SHAP|")
        share = imp[COUPLING].sum() / imp.sum()
        ax.set_xlabel(f"orange = coupling features ({share:.0%} of total importance)")
        rows += [{"task": task, "feature": c, "mean_abs_shap": v} for c, v in imp.items()]
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_shap.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(RESULTS / "shap_importance.csv", index=False)


def body_weather_map():
    w = pd.read_parquet(PROCESSED / "windows_30.parquet")
    w = w[(w.coverage >= 0.75) & w.eda_ok].dropna(subset=ALL).reset_index(drop=True)
    w = w.sample(min(len(w), 15000), random_state=SEED)
    w["hour"] = w.start.dt.hour + w.start.dt.minute / 60
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (name, cols) in zip(axes, [("levels only", LEVEL + PERSONAL), ("coupling only", COUPLING), ("all features", ALL)]):
        emb = umap.UMAP(n_neighbors=30, min_dist=0.2, random_state=SEED).fit_transform(
            StandardScaler().fit_transform(w[cols]))
        for s in ["other", *STATES]:
            m = (w.state == s).to_numpy() if s != "other" else ~w.state.isin(STATES).to_numpy()
            ax.scatter(emb[m, 0], emb[m, 1], s=2, alpha=0.4, color=STATE_COLORS[s], label=s)
        ax.set_title(f"UMAP of 30-min windows: {name}")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(markerscale=6, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_body_weather_map.png", dpi=160)
    plt.close(fig)


def desync_scores(w: pd.DataFrame) -> pd.DataFrame:
    # personal, state-specific baseline: median and MAD of each coupling feature
    lab = w[w.labelled]
    base = lab.groupby(["pid", "state"])[COUPLING].median()
    spread = lab.groupby(["pid", "state"])[COUPLING].agg(lambda s: (s - s.median()).abs().median() * 1.4826)
    n = lab.groupby(["pid", "state"]).size()
    ok = n[n >= 10].index
    base, spread = base.loc[ok], spread.loc[ok].clip(lower=0.05)

    w = w[w.state.isin(STATES)].join(base, on=["pid", "state"], rsuffix="_base").join(
        spread, on=["pid", "state"], rsuffix="_spread")
    z = np.column_stack([(w[c] - w[f"{c}_base"]).abs() / w[f"{c}_spread"] for c in COUPLING])
    w["desync"] = np.nanmean(z, axis=1)
    # same idea on plain levels, to check desync is not just "unusual values"
    lbase = lab.groupby(["pid", "state"])[[f"{s}_mean" for s in ("hr", "eda", "temp")]].agg(["median", "std"])
    lz = []
    for s in ("hr", "eda", "temp"):
        m = w.join(lbase[(f"{s}_mean", "median")].rename("m"), on=["pid", "state"]).m
        sd = w.join(lbase[(f"{s}_mean", "std")].rename("sd"), on=["pid", "state"]).sd
        lz.append(((w[f"{s}_mean"] - m) / sd).abs())
    w["level_deviation"] = np.nanmean(np.column_stack(lz), axis=1)
    return w.dropna(subset=["desync"])


def desync_analysis():
    w = pd.read_parquet(PROCESSED / "windows_30.parquet")
    w = w[(w.coverage >= 0.75) & w.eda_ok]
    d = desync_scores(w)
    out = {}

    # (a) transition windows (mixed states) vs clean windows
    trans = d[d.purity < 0.7].desync
    clean = d[d.labelled].desync
    out["transition_vs_clean"] = dict(median_transition=trans.median(), median_clean=clean.median(),
                                      p=mannwhitneyu(trans, clean).pvalue, n_transition=len(trans))

    # (b) windows the 3-class levels-only model got wrong vs right
    p = pd.read_parquet(RESULTS / "loso_predictions.parquet")
    p = p[p.config == "3-class|30|xgb|True|levels"]
    m = d.merge(p[["pid", "start", "y", "pred"]], on=["pid", "start"])
    wrong, right = m[m.y != m.pred].desync, m[m.y == m.pred].desync
    out["misclassified_vs_correct"] = dict(median_wrong=wrong.median(), median_right=right.median(),
                                           p=mannwhitneyu(wrong, right).pvalue, n_wrong=len(wrong))
    rho = spearmanr(d.desync, d.level_deviation, nan_policy="omit")
    out["desync_vs_level_deviation_spearman"] = dict(rho=rho.statistic, p=rho.pvalue)
    pd.DataFrame(out).T.to_csv(RESULTS / "desync_validation.csv")
    print(pd.DataFrame(out).T.to_string())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(d, x="desync", hue=np.where(d.purity < 0.7, "transition", "single state"),
                 stat="density", common_norm=False, element="step", ax=axes[0], bins=50)
    axes[0].set_xlim(0, d.desync.quantile(0.99))
    axes[0].set_title("Desync: transition vs single-state windows")
    sns.histplot(m, x="desync", hue=np.where(m.y != m.pred, "model wrong", "model right"),
                 stat="density", common_norm=False, element="step", ax=axes[1], bins=50)
    axes[1].set_xlim(0, d.desync.quantile(0.99))
    axes[1].set_title("Desync: where the levels model fails")

    # body weather calendar for the best-recorded person: desync by day x hour
    pid = d.pid.value_counts().index[0]
    g = d[d.pid == pid].assign(day=lambda x: x.start.dt.date, hour=lambda x: x.start.dt.hour)
    cal = g.pivot_table(index="day", columns="hour", values="desync", aggfunc="mean")
    sns.heatmap(cal, cmap="magma_r", ax=axes[2], cbar_kws={"label": "desync"}, vmax=d.desync.quantile(0.95))
    axes[2].set_title(f"Body weather calendar: participant {pid[-3:]}")
    axes[2].set_yticks([])
    fig.tight_layout()
    fig.savefig(FIGURES / "fig7_desync.png", dpi=160)
    plt.close(fig)
    d.to_parquet(RESULTS / "desync_windows_30.parquet", index=False)


if __name__ == "__main__":
    shap_plot()
    body_weather_map()
    desync_analysis()
