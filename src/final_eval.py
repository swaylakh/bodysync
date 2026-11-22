"""Final push: iterate on DEV people, then test the frozen system once on LOCKED TEST people.

Protocol (fixed before any of these experiments ran):
  - In each of the 7 stress datasets, people are split at random (seed 42) into DEV (~60%) and
    LOCKED TEST (~40%), stratified so both groups contain people with stress windows.
    The split is saved in results/final_split.csv and committed before iterating.
  - Iteration (`--round`) only ever trains and scores DEV people; every idea tried is logged in
    results/iteration_log.csv, including the ones that don't help.
  - The frozen system (`--final`) is scored once on LOCKED TEST people: results/final_test.csv.
  - An idea is kept if it raises the DEV average AUC and drops no dataset by more than 0.03.

Two evaluation sets:
  - "all": every window of each person (leave-one-subject-out). Used for every idea except
    personal calibration.
  - "later": for personal calibration only. Calibration = a person's first 3 stress + first 3
    non-stress windows (in time order); later windows = after the last calibration window +
    CALIB_GAP (lab: no overlap, 2 min, since a stressor lasts ~10 min; everyday life: 30 min).
    Calibration is compared with the same model without it, on exactly these windows, and only
    on datasets where at least MIN_CALIB_PEOPLE people still have enough later windows.
  (Round 1 first scored every idea on "later" windows; that left 1 Stress + exercise person and
  4 CAN-STRESS people, so it was replaced by the two sets above. Both runs stay in the log.)
EmpaticaE4Stress is left out: its only calm windows are the baseline that sets the normal.
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from config import RESULTS, SEED
from feature_test import BURSTS, INPUTS, LAB, LIFE, NON_STRESS, SHAPE, features, load, score_variant
from validate_datasets import evaluate

SPLIT = RESULTS / "final_split.csv"
LOG = RESULTS / "iteration_log.csv"
TEST_SHARE = 0.4
DATASETS = [d for d in LAB + LIFE if d != "e4stress"]
CALIB_N = 3
CALIB_WEIGHT = 10
CALIB_GAP = {"lab": pd.Timedelta(minutes=2), "free-living": pd.Timedelta(minutes=30)}
SMOOTH = {"lab": "2min", "free-living": "15min"}
BASE = {"change": False, "calib": False, "smooth": False, "pooled": False, "shape": False,
        "context": False, "quality": False, "selftrain": False, "model": "xgb", "blend": False}
MODELS = {   # round 4 candidates; "xgb" is the model used in rounds 1-3
    "xgb": dict(n_estimators=200, max_depth=3, learning_rate=0.05),
    "xgb_shallow": dict(n_estimators=300, max_depth=2, learning_rate=0.05),
    "xgb_deep": dict(n_estimators=400, max_depth=5, learning_rate=0.03),
}
EDA_OFF = 0.01          # uS; mean sweat at or below this = sensor not on the skin
PSEUDO_SHARE = 0.1      # self-training: each person's most / least stressed-looking 10% of unlabelled windows
MIN_CALIB_PEOPLE = 5
# Frozen after round 4 (rounds 3 and 4 each gained < 0.01 on DEV, which ends iteration)
FINAL = {**BASE, "change": True, "smooth": True, "shape": True, "selftrain": True}


def make_split(data: dict) -> pd.DataFrame:
    """DEV / TEST per dataset, stratified by whether a person has any stress windows."""
    if SPLIT.exists():
        return pd.read_csv(SPLIT)
    rng = np.random.default_rng(SEED)
    rows = []
    for name, (w, _) in data.items():
        has_stress = w.groupby("pid").label.agg(lambda s: (s == "stress").any())
        for flag in (True, False):
            pids = np.array(sorted(has_stress[has_stress == flag].index))
            rng.shuffle(pids)
            n_test = int(round(TEST_SHARE * len(pids)))
            rows += [{"dataset": name, "pid": p, "group": "test" if i < n_test else "dev", "has_stress": flag}
                     for i, p in enumerate(pids)]
    split = pd.DataFrame(rows)
    split.to_csv(SPLIT, index=False)
    return split


def prepare() -> tuple[pd.DataFrame, dict, dict]:
    data = {name: load(name) for name in LAB + LIFE}
    split = make_split(data).set_index(["dataset", "pid"]).group
    allw = pd.concat([w for w, _ in data.values()], ignore_index=True)
    allw["group"] = split.reindex(pd.MultiIndex.from_frame(allw[["dataset", "pid"]])).to_numpy()
    allw["setting"] = np.where(allw.dataset.isin(LAB), "lab", "free-living")
    allw["y"] = (allw.label == "stress").astype(int)
    allw["labelled"] = (allw.label == "stress") | allw.label.isin(NON_STRESS)
    allw["calib"], allw["later"] = calibration(allw)
    allw["formula"] = pd.concat([score_variant(w, s, BURSTS, {"scr_rate": 1, "scr_amp": 1}).v for w, s in data.values()],
                                ignore_index=True).to_numpy()
    X = LazyX(data)
    return allw, X, data


class LazyX(dict):
    """Feature tables per (change, shape, context), built the first time a variant needs them."""
    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def __missing__(self, key):
        ch, sh, cx = key
        self[key] = pd.concat([features(w, ch, INPUTS + SHAPE if sh else INPUTS, cx) for w, _ in self.data.values()],
                              ignore_index=True)
        return self[key]


def calibration(allw: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Each person's first CALIB_N stress + CALIB_N non-stress windows, and the windows after them."""
    calib = np.zeros(len(allw), bool)
    later = np.zeros(len(allw), bool)
    for _, g in allw[allw.labelled].groupby(["dataset", "pid"]):
        g = g.sort_values("start")
        pick = pd.concat([g[g.y == 1].head(CALIB_N), g[g.y == 0].head(CALIB_N)])
        if (pick.y == 1).sum() < CALIB_N or (pick.y == 0).sum() < CALIB_N or pick.start.isna().any():
            continue
        calib[pick.index] = True
        person = (allw.dataset == g.dataset.iloc[0]) & (allw.pid == g.pid.iloc[0])
        later |= (person & (allw.start > pick.start.max() + CALIB_GAP[g.setting.iloc[0]])).to_numpy()
    return calib, later


def fit_score(X: pd.DataFrame, allw: pd.DataFrame, train: np.ndarray, extra: np.ndarray, test: np.ndarray,
              y_all: pd.Series | None = None, extra_weight: float = CALIB_WEIGHT, model: str = "xgb") -> np.ndarray:
    """XGBoost on train (+ extra rows, e.g. calibration, at extra_weight); returns a score, lower = stressed."""
    rows = train | extra
    y = (allw.y if y_all is None else y_all)[rows]
    weight = compute_sample_weight("balanced", y) / allw[rows].groupby("dataset").dataset.transform("size").to_numpy()
    weight = weight / weight.mean()
    weight[extra[rows]] *= extra_weight
    if model == "logistic":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[rows].fillna(0), y, logisticregression__sample_weight=weight)
        return 100 * (1 - clf.predict_proba(X[test].fillna(0))[:, 1])
    clf = XGBClassifier(**MODELS[model], subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=0)
    clf.fit(X[rows], y, sample_weight=weight)
    return 100 * (1 - clf.predict_proba(X[test])[:, 1])


def smooth(allw: pd.DataFrame, score: pd.Series) -> pd.Series:
    """Past-only rolling mean of the score per person (stress lasts minutes, not seconds)."""
    out = score.copy()
    for (_, _), g in allw[score.notna() & allw.start.notna()].groupby(["dataset", "pid"]):
        g = g.sort_values("start")
        roll = pd.Series(score[g.index].to_numpy(), index=g.start).rolling(SMOOTH[g.setting.iloc[0]]).mean()
        out[g.index] = roll.to_numpy()
    return out


def run(cfg: dict, allw: pd.DataFrame, X: dict, group: str, shuffle: bool = False,
        eval_set: str = "all", datasets: list[str] = DATASETS, quality_eval: bool = False) -> pd.DataFrame:
    """Each person in `group` is scored by a model trained on the other people (DEV only when testing)."""
    allw = allw.copy()
    if shuffle:   # credibility check: shuffle each person's labels, keeping their stress/calm counts
        rng = np.random.default_rng(SEED)
        for _, g in allw[allw.labelled].groupby(["dataset", "pid"]):
            y = rng.permutation(g.y.to_numpy())
            calm_labels = g.label[g.y == 0].to_numpy()
            calm_labels = calm_labels if len(calm_labels) else np.array(["calm"])
            # windows that turn calm take a calm label this person really has (e.g. no_stress for nurses)
            new = np.where(y == 1, "stress", rng.choice(calm_labels, size=len(y)))
            new[(y == 0) & (g.y.to_numpy() == 0)] = g.label[(y == 0) & (g.y.to_numpy() == 0)]
            allw.loc[g.index, "y"] = y
            allw.loc[g.index, "label"] = new
    Xc = X[(cfg["change"], cfg["shape"], cfg["context"])]
    eda_on = allw.eda_mean > EDA_OFF
    trainable = allw.labelled & (allw.group == "dev") & (eda_on if cfg["quality"] else True)
    score = pd.Series(np.nan, index=allw.index)
    for name in datasets:
        for pid in allw.loc[(allw.dataset == name) & (allw.group == group), "pid"].unique():
            me = ((allw.dataset == name) & (allw.pid == pid)).to_numpy()
            pool = allw.dataset == name if not cfg["pooled"] else allw.dataset.isin(DATASETS)
            train = (trainable & pool).to_numpy() & ~me
            extra = me & allw.calib.to_numpy() if cfg["calib"] else np.zeros(len(allw), bool)
            s = fit_score(Xc, allw, train, extra, me, model=cfg["model"])
            if cfg["selftrain"]:   # adapt to this person with their own unlabelled windows
                unl = np.where(me & ~allw.labelled.to_numpy())[0]
                if len(unl) >= 20:
                    su = pd.Series(s, index=np.where(me)[0])[unl]
                    k = max(1, int(PSEUDO_SHARE * len(unl)))
                    lo, hi = su.nsmallest(k).index, su.nlargest(k).index
                    pseudo = np.zeros(len(allw), bool)
                    pseudo[lo] = pseudo[hi] = True
                    y_ps = allw.y.copy()
                    y_ps.iloc[lo], y_ps.iloc[hi] = 1, 0
                    s = fit_score(Xc, allw, train | extra, pseudo, me, y_ps, extra_weight=1.0, model=cfg["model"])
            score[me] = s
    if cfg["blend"]:   # average with the hand-made BodySync formula (+ sweat bursts), same 0-100 scale
        score = 0.5 * score + 0.5 * allw["formula"]
    if cfg["smooth"]:
        score = smooth(allw, score)
    rows = []
    for name in datasets:
        m = (allw.dataset == name) & (allw.group == group)
        m &= allw.later if eval_set == "later" else ~allw.calib if cfg["calib"] else True
        if quality_eval:
            m &= eda_on
        s = allw[m].assign(v=score[m], v_reason="")
        rows.append(evaluate(name, s, allw.loc[m, "setting"].iloc[0], "v"))
    return pd.DataFrame(rows)


def calib_datasets(allw: pd.DataFrame, group: str) -> list[str]:
    """Datasets where enough people keep >= 3 stress and >= 3 non-stress windows after calibration."""
    d = allw[(allw.group == group) & allw.later & allw.labelled]
    ok = d.groupby(["dataset", "pid"]).y.agg(lambda y: (y.sum() >= 3) & ((1 - y).sum() >= 3))
    counts = ok.groupby("dataset").sum()
    return [n for n in DATASETS if counts.get(n, 0) >= MIN_CALIB_PEOPLE]


def formula(allw: pd.DataFrame, data: dict, group: str) -> pd.DataFrame:
    """Reference: the BodySync formula + sweat bursts, on all windows of the group's people."""
    rows = []
    for name in DATASETS:
        w, setting = data[name]
        s = score_variant(w, setting, BURSTS, {"scr_rate": 1, "scr_amp": 1})
        s = s[allw.loc[allw.dataset == name, "group"].to_numpy() == group]
        rows.append(evaluate(name, s, setting, "v"))
    return pd.DataFrame(rows)


def final_test(allw: pd.DataFrame, X: dict, data: dict):
    """Run once: the frozen system on LOCKED TEST people (models trained on DEV people only)."""
    out = []
    def add(name, res):
        out.append(res.assign(system=name))
        r = res.set_index("dataset").mean_auc.reindex(DATASETS)
        print(f"{name:45s}", " ".join(f"{v:.3f}" for v in r), f"| mean {r.mean():.3f}")
    print(" " * 45, " ".join(DATASETS))
    add("formula: BodySync + bursts", formula(allw, data, "test"))
    add("trained model (start of final push)", run(BASE, allw, X, "test"))
    add("FINAL BodySync", run(FINAL, allw, X, "test"))
    add("FINAL, shuffled labels (should be ~0.5)", run(FINAL, allw, X, "test", shuffle=True))
    cal = calib_datasets(allw, "test")
    print("calibration compared on:", cal)
    if cal:
        add("FINAL (later windows)", run(FINAL, allw, X, "test", eval_set="later", datasets=cal))
        add("FINAL + calibration (later windows)", run({**FINAL, "calib": True}, allw, X, "test", eval_set="later", datasets=cal))
    res = pd.concat(out, ignore_index=True)
    res.to_csv(RESULTS / "final_test.csv", index=False)


def log(round_: str, idea: str, res: pd.DataFrame):
    r = res.set_index("dataset").reindex(DATASETS)
    auc = r.mean_auc.round(3)
    entry = pd.DataFrame([{"round": round_, "idea": idea, **auc.to_dict(), "mean": round(auc.mean(), 3),
                           **{f"n_{d}": r.n_people.get(d) for d in DATASETS}}])
    header = not LOG.exists() or "n_wesad" not in LOG.read_text(encoding="utf-8").splitlines()[0]
    if header and LOG.exists():   # older rows had no per-dataset people counts; keep them in a side file
        LOG.rename(LOG.with_name("iteration_log_round1_later_windows.csv"))
    entry.to_csv(LOG, mode="a", header=header, index=False)
    pd.read_csv(LOG).drop_duplicates(subset=["round", "idea"], keep="last").to_csv(LOG, index=False)
    print(entry[["round", "idea", *DATASETS, "mean"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-only", action="store_true")
    ap.add_argument("--round", type=int)
    ap.add_argument("--final", action="store_true", help="score the frozen system once on LOCKED TEST people")
    args = ap.parse_args()
    if args.split_only:
        data = {name: load(name) for name in LAB + LIFE}
        split = make_split(data)
        print(split.groupby(["dataset", "group", "has_stress"]).size().unstack(fill_value=0).to_string())
        return
    allw, X, data = prepare()
    print(allw[allw.group == "dev"].groupby("dataset").agg(people=("pid", "nunique"), calib=("calib", "sum"), later=("later", "sum")).to_string())
    if args.final:
        final_test(allw, X, data)
        return
    if args.round == 1:
        log("1", "formula: BodySync + bursts (reference)", formula(allw, data, "dev"))
        variants = {"base: trained": {}, "+ change": {"change": True},
                    "+ smoothing": {"smooth": True}, "+ pooled": {"pooled": True}}
        for idea, change in variants.items():
            log("1", idea, run({**BASE, **change}, allw, X, "dev"))
        cal = calib_datasets(allw, "dev")
        print("calibration compared on:", cal)
        log("1-calib", "base: trained (later windows)", run(BASE, allw, X, "dev", eval_set="later", datasets=cal))
        log("1-calib", "+ calibration (later windows)", run({**BASE, "calib": True}, allw, X, "dev", eval_set="later", datasets=cal))
    if args.round == 11:   # stack the round-1 ideas that were kept
        kept = {"change": True, "smooth": True}
        log("1-stack", "change + smoothing", run({**BASE, **kept}, allw, X, "dev"))
        cal = calib_datasets(allw, "dev")
        log("1-stack-calib", "change + smoothing (later windows)", run({**BASE, **kept}, allw, X, "dev", eval_set="later", datasets=cal))
        log("1-stack-calib", "change + smoothing + calibration (later windows)",
            run({**BASE, **kept, "calib": True}, allw, X, "dev", eval_set="later", datasets=cal))
    if args.round == 2:   # round 2: fast/slow sweat + in-window trends, on top of the kept round-1 stack
        kept = {"change": True, "smooth": True}
        log("2", "round-1 stack (change + smoothing)", run({**BASE, **kept}, allw, X, "dev"))
        log("2", "+ fast/slow sweat + trends", run({**BASE, **kept, "shape": True}, allw, X, "dev"))
    if args.round == 3:   # round 3 on top of the kept stack; quality is compared on the same sensor-on windows
        kept = {"change": True, "smooth": True, "shape": True}
        log("3", "stack (sensor-on windows)", run({**BASE, **kept}, allw, X, "dev", quality_eval=True))
        log("3", "+ quality filter (sensor-on windows)", run({**BASE, **kept, "quality": True}, allw, X, "dev", quality_eval=True))
        log("3", "stack", run({**BASE, **kept}, allw, X, "dev"))
        log("3", "+ still/moving normals", run({**BASE, **kept, "context": True}, allw, X, "dev"))
        log("3", "+ self-training", run({**BASE, **kept, "selftrain": True}, allw, X, "dev"))
    if args.round == 4:   # round 4: the model itself, on top of the kept stack
        kept = {"change": True, "smooth": True, "shape": True, "selftrain": True}
        log("4", "stack (xgb)", run({**BASE, **kept}, allw, X, "dev"))
        for m in ("xgb_shallow", "xgb_deep", "logistic"):
            log("4", f"model: {m}", run({**BASE, **kept, "model": m}, allw, X, "dev"))
        log("4", "+ blend with BodySync formula", run({**BASE, **kept, "blend": True}, allw, X, "dev"))


if __name__ == "__main__":
    main()
