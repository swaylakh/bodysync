"""BodySync stress detector: learn stress from data instead of a hand-made formula.

Features per window (all relative to the person's own normal, so they transfer across people):
  - how far heart rate, sweat and skin temp are from normal (z, as in bodysync.py)
  - how much they CHANGED over the last ~10 minutes (stress onset = rising HR/sweat)
  - movement relative to the person's usual movement, so exercise can be told apart
Labels: stress = 1; calm, rest, baseline, amusement, meditation, aerobic, anaerobic,
recovery = 0. Including exercise as 0 teaches the model that a working body is not a
stressed one.

Honest testing: leave-one-DATASET-out. The model is trained on all other datasets and
tested on the one it has never seen; the nurse dataset is never used for training and
is evaluated only once, at the end. Metric: within-person AUC (stress vs non-stress
windows of the same person), averaged over people, compared with the frozen v2 score.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import multi_validate as mv
from config import RESULTS, SEED

NON_STRESS = {"calm", "rest", "relax", "baseline", "amusement", "meditation", "aerobic", "anaerobic", "recovery"}
Z = ["z_hr_mean", "z_log_eda", "z_temp_mean"]
FEATURES = Z + [f"d_{c}" for c in Z] + ["move_z"]
FINAL_TEST = "nurse"
CHANGE_LAG = pd.Timedelta(minutes=10)


def add_features(s: pd.DataFrame) -> pd.DataFrame:
    s = s.sort_values(["pid", "start"]).copy()
    s["start"] = pd.to_datetime(s["start"]).astype("datetime64[ns]")
    # change over the last ~10 minutes: match each window with the one ~10 min earlier
    prev = s[["pid", "start", *Z]].copy()
    prev["start"] = prev["start"] + CHANGE_LAG
    prev = prev.dropna(subset=["start"]).sort_values("start")
    cur = s.dropna(subset=["start"]).sort_values("start")
    m = pd.merge_asof(cur, prev, on="start", by="pid", suffixes=("", "_prev"),
                      tolerance=pd.Timedelta(minutes=3), direction="nearest")
    for c in Z:
        m[f"d_{c}"] = m[c] - m[f"{c}_prev"]
    med = m.groupby("pid")["move"].transform("median")
    iqr = m.groupby("pid")["move"].transform(lambda x: x.quantile(0.75) - x.quantile(0.25))
    m["move_z"] = (m["move"] - med) / iqr.replace(0, np.nan)
    return m


def build() -> pd.DataFrame:
    return build_with_labels(set())


def build_with_labels(extra: set[str]) -> pd.DataFrame:
    """extra: additional labels kept as y=0 (only used by the final nurse test)."""
    parts = []
    for name, (w, setting) in mv.load_all().items():
        s = add_features(mv.score_dataset(w, setting, directional=True))
        s = s[(s.label == "stress") | s.label.isin(NON_STRESS | extra)]
        s["y"] = (s.label == "stress").astype(int)
        s["dataset"], s["setting"] = name, setting
        parts.append(s)
    return pd.concat(parts, ignore_index=True)


def models():
    return {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8,
                                 colsample_bytree=0.8, random_state=SEED, verbosity=0),
    }


CALM_EVAL = {"calm", "rest", "relax", "amusement", "meditation"}
EXERCISE = {"aerobic", "anaerobic"}


def eval_rows(te: pd.DataFrame, against: set[str]) -> pd.DataFrame:
    """Stress windows vs the given non-stress labels. Baseline is left out because it set the
    normal, except for e4stress, which has nothing else (weaker test, flagged in the README)."""
    if against is CALM_EVAL and not te.label.isin(CALM_EVAL).any():
        against = {"baseline"}
    return te[(te.label == "stress") | te.label.isin(against)]


def within_person_auc(df: pd.DataFrame, col: str) -> tuple[float, int, int]:
    if df.empty:
        return np.nan, 0, 0
    aucs = df.groupby("pid").apply(
        lambda g: roc_auc_score(g.y, g[col]) if g.y.nunique() == 2 and g.y.sum() >= 3 and (1 - g.y).sum() >= 3 else np.nan,
        include_groups=False).dropna()
    return aucs.mean(), int((aucs > 0.5).sum()), len(aucs)


def fit(model, train: pd.DataFrame):
    X = train[FEATURES].fillna(0).to_numpy()
    # every dataset gets equal weight, and stress/non-stress are balanced within it
    w = train.groupby("dataset").y.transform(lambda y: pd.Series(compute_sample_weight("balanced", y), index=y.index))
    w = w / train.groupby("dataset").y.transform("size")
    w = w / w.mean()                     # keep weights around 1 so trees can still split
    return model.fit(X, train.y, **({"logisticregression__sample_weight": w} if hasattr(model, "steps") else {"sample_weight": w}))


def evaluate(data: pd.DataFrame, test_sets: list[str]) -> pd.DataFrame:
    rows = []
    for test in test_sets:
        train = data[(data.dataset != test) & (data.dataset != FINAL_TEST)]
        te = data[data.dataset == test].copy()
        te["frozen_v2"] = -te.bodysync          # lower score = more stressed
        for name, model in models().items():
            te[name] = fit(model, train).predict_proba(te[FEATURES].fillna(0).to_numpy())[:, 1]
        calm, ex = eval_rows(te, CALM_EVAL), eval_rows(te, EXERCISE)
        row = {"test_dataset": test, "setting": te.setting.iloc[0], "people": within_person_auc(calm, "frozen_v2")[2]}
        for name in ("frozen_v2", *models()):
            auc, k, _ = within_person_auc(calm, name)
            row[f"{name}_auc"], row[f"{name}_people_above_0.5"] = auc, k
            if te.label.isin(EXERCISE).any():
                row[f"{name}_auc_vs_exercise"] = within_person_auc(ex, name)[0]
        rows.append(row)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})
    return pd.DataFrame(rows)


def main(final: bool = False):
    data = build()
    dev_sets = [d for d in data.dataset.unique() if d != FINAL_TEST]
    res = evaluate(data, [FINAL_TEST] if final else dev_sets)
    res.to_csv(RESULTS / ("stress_model_final_nurse.csv" if final else "stress_model_leave_one_dataset_out.csv"), index=False)
    if not final:
        coef = fit(models()["logistic"], data[data.dataset != FINAL_TEST]).named_steps["logisticregression"].coef_[0]
        print(pd.Series(coef, index=FEATURES).round(3).to_string())


if __name__ == "__main__":
    import sys
    main(final="--final" in sys.argv)
