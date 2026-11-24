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
import os

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
EVENTS = RESULTS / "iteration_events.csv"   # round 8+: everyday results per stress event (see event_eval)
# CPU threads for XGBoost. Default = all, which reproduces the frozen results. Fewer threads (env
# BODYSYNC_THREADS=4) is quieter but shifts results slightly (round 8 ran with 4: DEV mean 0.747 vs 0.752).
N_JOBS = int(os.environ.get("BODYSYNC_THREADS", 0)) or None
TEST_SHARE = 0.4
DATASETS = [d for d in LAB + LIFE if d != "e4stress"]
CALIB_N = 3
CALIB_WEIGHT = 10
CALIB_GAP = {"lab": pd.Timedelta(minutes=2), "free-living": pd.Timedelta(minutes=30)}
SMOOTH = {"lab": "2min", "free-living": "15min"}
PSEUDO_SHARE = 0.1      # self-training: each person's most / least stressed-looking 10% of unlabelled windows
BASE = {"change": False, "calib": False, "smooth": False, "pooled": False, "shape": False,
        "context": False, "quality": False, "selftrain": False, "model": "xgb", "blend": False,
        "lookback": False, "pool_life": False, "life_model": None, "lab_helper": 0.0, "pseudo": PSEUDO_SHARE,
        "other_datasets": False, "drop": (), "selftrain_on": "unlabelled", "train_labels": None,
        "smooth_life": SMOOTH["free-living"], "still_only": None}
# smooth_life: smoothing window for everyday data. still_only: everyday data is scored (and trained) only on
#   windows where the person moves at most their own `still_only` quantile of movement (as watches do)
# train_labels: name of an allw column with other labels to TRAIN on (evaluation keeps `label`)
# drop: feature-name prefixes left out of the model (audit: e.g. "move_z", "d_" = change over 10 min)
# selftrain_on: "unlabelled" (frozen system) or "all" of the person's windows (audit check: in everyday data
#   the unlabelled windows sit near stress reports, so picking them uses the labels)
LOOKBACK = "15min"
MODELS = {   # round 4 candidates; "xgb" is the model used in rounds 1-3
    "xgb": dict(n_estimators=200, max_depth=3, learning_rate=0.05),
    "xgb_shallow": dict(n_estimators=300, max_depth=2, learning_rate=0.05),
    "xgb_deep": dict(n_estimators=400, max_depth=5, learning_rate=0.03),
}
EDA_OFF = 0.01          # uS; mean sweat at or below this = sensor not on the skin
MIN_CALIB_PEOPLE = 5
# Frozen after round 4 (rounds 3 and 4 each gained < 0.01 on DEV, which ends iteration)
FINAL = {**BASE, "change": True, "smooth": True, "shape": True, "selftrain": True}
# v2, frozen after round 7 (everyday-life push, rounds 5-7 on DEV). Its locked-test run is the
# SECOND look at the TEST people, which the report states.
FINAL_V2 = {**FINAL, "pool_life": True, "lab_helper": 0.3, "pseudo": 0.2, "life_model": "xgb+logistic"}


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
    """Feature tables per (change, shape, context, lookback), built the first time a variant needs them.
    lookback: in everyday-life datasets, each distance-from-normal input is averaged over the past
    LOOKBACK (everyday stress builds slowly); lab datasets are left as they are."""
    def __init__(self, data: dict):
        super().__init__()
        self.data = data

    def __missing__(self, key):
        ch, sh, cx, lb = key
        parts = []
        for w, setting in self.data.values():
            f = features(w, ch, INPUTS + SHAPE if sh else INPUTS, cx)
            if lb and setting == "free-living":
                z = [c for c in f.columns if c.startswith("z_")]
                for _, idx in w.dropna(subset=["start"]).groupby("pid").groups.items():
                    g = w.loc[idx].sort_values("start")
                    f.loc[g.index, z] = f.loc[g.index, z].set_axis(g.start).rolling(LOOKBACK).mean().to_numpy()
            parts.append(f)
        self[key] = pd.concat(parts, ignore_index=True)
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
              y_all: pd.Series | None = None, extra_weight: float = CALIB_WEIGHT, model: str = "xgb",
              lab_weight: float = 1.0) -> np.ndarray:
    """XGBoost on train (+ extra rows, e.g. calibration, at extra_weight); returns a score, lower = stressed."""
    rows = train | extra
    y = (allw.y if y_all is None else y_all)[rows]
    weight = compute_sample_weight("balanced", y) / allw[rows].groupby("dataset").dataset.transform("size").to_numpy()
    weight = weight / weight.mean()
    weight[extra[rows]] *= extra_weight
    weight[allw.dataset[rows].isin(LAB).to_numpy()] *= lab_weight
    if model == "xgb+logistic":   # average of the two models' scores
        return np.mean([fit_score(X, allw, train, extra, test, y_all, extra_weight, m, lab_weight)
                        for m in ("xgb", "logistic")], axis=0)
    if model == "logistic":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[rows].fillna(0), y, logisticregression__sample_weight=weight)
        return 100 * (1 - clf.predict_proba(X[test].fillna(0))[:, 1])
    clf = XGBClassifier(**MODELS[model], subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbosity=0, n_jobs=N_JOBS)
    clf.fit(X[rows], y, sample_weight=weight)
    return 100 * (1 - clf.predict_proba(X[test])[:, 1])


def smooth(allw: pd.DataFrame, score: pd.Series, life: str = SMOOTH["free-living"]) -> pd.Series:
    """Past-only rolling mean of the score per person (stress lasts minutes, not seconds)."""
    out = score.copy()
    for (_, _), g in allw[score.notna() & allw.start.notna()].groupby(["dataset", "pid"]):
        g = g.sort_values("start")
        roll = pd.Series(score[g.index].to_numpy(), index=g.start).rolling(life if g.setting.iloc[0] == "free-living" else SMOOTH["lab"]).mean()
        out[g.index] = roll.to_numpy()
    return out


def run(cfg: dict, allw: pd.DataFrame, X: dict, group: str, shuffle: bool | str = False,
        eval_set: str = "all", datasets: list[str] = DATASETS, quality_eval: bool = False,
        eval_relabel: dict | None = None, eval_labels: pd.Series | None = None,
        return_scores: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, pd.Series]:
    """Each person in `group` is scored by a model trained on the other people (DEV only when testing).
    shuffle: True = permute each person's labels; "circular" = rotate them in time (keeps stress blocks intact,
             a stricter leak check). eval_labels: replaces `label` at test time only (training unchanged)."""
    allw = allw.copy()
    eval_label = allw.label.copy()
    if cfg["train_labels"]:
        allw["label"] = allw[cfg["train_labels"]]
        allw["y"] = (allw.label == "stress").astype(int)
        allw["labelled"] = (allw.label == "stress") | allw.label.isin(NON_STRESS)
    if shuffle:   # credibility check: shuffle each person's labels, keeping their stress/calm counts
        rng = np.random.default_rng(SEED)
        for _, g in allw[allw.labelled].groupby(["dataset", "pid"]):
            if shuffle == "circular":
                g = g.sort_values("start")
                y = np.roll(g.y.to_numpy(), int(rng.integers(len(g) // 4, 3 * len(g) // 4 + 1)))
            else:
                y = rng.permutation(g.y.to_numpy())
            calm_labels = g.label[g.y == 0].to_numpy()
            calm_labels = calm_labels if len(calm_labels) else np.array(["calm"])
            # windows that turn calm take a calm label this person really has (e.g. no_stress for nurses)
            new = np.where(y == 1, "stress", rng.choice(calm_labels, size=len(y)))
            new[(y == 0) & (g.y.to_numpy() == 0)] = g.label[(y == 0) & (g.y.to_numpy() == 0)]
            allw.loc[g.index, "y"] = y
            allw.loc[g.index, "label"] = new
    Xc = X[(cfg["change"], cfg["shape"], cfg["context"], cfg["lookback"])]
    if cfg["drop"]:
        Xc = Xc.loc[:, ~Xc.columns.str.startswith(tuple(cfg["drop"]))]
    eda_on = allw.eda_mean > EDA_OFF
    still = np.ones(len(allw), bool)
    if cfg["still_only"]:
        q = allw.groupby(["dataset", "pid"]).move.transform(lambda x: x.quantile(cfg["still_only"]))
        still = ~((allw.setting == "free-living") & ~(allw.move <= q)).to_numpy()
    trainable = allw.labelled & (allw.group == "dev") & (eda_on if cfg["quality"] else True) & still
    score = pd.Series(np.nan, index=allw.index)
    for name in datasets:
        for pid in allw.loc[(allw.dataset == name) & (allw.group == group), "pid"].unique():
            me = ((allw.dataset == name) & (allw.pid == pid)).to_numpy()
            pool = allw.dataset == name if not cfg["pooled"] else allw.dataset.isin(DATASETS)
            life = name in LIFE
            if cfg["pool_life"] and life:   # everyday datasets learn from all everyday datasets
                pool = allw.dataset.isin(LIFE)
            if cfg["lab_helper"] and life:  # ... plus the lab datasets at a low weight
                pool = pool | allw.dataset.isin(LAB)
            if cfg["other_datasets"]:       # stricter test: never train on the test person's dataset
                pool = allw.dataset.isin(DATASETS) & (allw.dataset != name)
            model = (cfg["life_model"] or cfg["model"]) if life else cfg["model"]
            fit = dict(model=model, lab_weight=cfg["lab_helper"] if life else 1.0)
            train = (trainable & pool).to_numpy() & ~me
            extra = me & allw.calib.to_numpy() if cfg["calib"] else np.zeros(len(allw), bool)
            s = fit_score(Xc, allw, train, extra, me, **fit)
            if cfg["selftrain"]:   # adapt to this person with their own unlabelled windows
                unl = np.where(me & still & (True if cfg["selftrain_on"] == "all" else ~allw.labelled.to_numpy()))[0]
                if len(unl) >= 20:
                    su = pd.Series(s, index=np.where(me)[0])[unl]
                    k = max(1, int(cfg["pseudo"] * len(unl)))
                    lo, hi = su.nsmallest(k).index, su.nlargest(k).index
                    pseudo = np.zeros(len(allw), bool)
                    pseudo[lo] = pseudo[hi] = True
                    y_ps = allw.y.copy()
                    y_ps.iloc[lo], y_ps.iloc[hi] = 1, 0
                    s = fit_score(Xc, allw, train | extra, pseudo, me, y_ps, extra_weight=1.0, **fit)
            score[me] = s
    score[~still] = np.nan
    if cfg["blend"]:   # average with the hand-made BodySync formula (+ sweat bursts), same 0-100 scale
        score = 0.5 * score + 0.5 * allw["formula"]
    if cfg["smooth"]:
        score = smooth(allw, score, cfg["smooth_life"])
    rows = []
    for name in datasets:
        m = (allw.dataset == name) & (allw.group == group)
        m &= allw.later if eval_set == "later" else ~allw.calib if cfg["calib"] else True
        if quality_eval:
            m &= eda_on
        s = allw[m].assign(v=score[m], v_reason="")
        if cfg["train_labels"] and not shuffle:
            s["label"] = eval_label[m]
        if eval_labels is not None:
            s["label"] = eval_labels[m]
        if eval_relabel:   # a different comparison at test time only (training labels unchanged)
            s["label"] = s.label.replace(eval_relabel)
        rows.append(evaluate(name, s, allw.loc[m, "setting"].iloc[0], "v"))
    return (pd.DataFrame(rows), score) if return_scores else pd.DataFrame(rows)


def event_eval(allw: pd.DataFrame, score: pd.Series, group: str, labels: pd.Series | None = None,
               datasets: list[str] = LIFE) -> pd.DataFrame:
    """Everyday life, per stress EVENT instead of per window (more events than people, less noise).
    Event = a run of a person's stress windows with gaps < 30 min. Control = the same person's calm windows
    within 1 h of the event's clock time, on a different day. Hit = event's mean score below the control's.
    Fixed before running; reported as the share of events that are hits (0.5 = chance) and a sign-test p."""
    from scipy.stats import binomtest
    lab = allw.label if labels is None else labels
    calm_names = {"nurse": {"no_stress"}}
    rows = []
    for name in datasets:
        d = allw[(allw.dataset == name) & (allw.group == group) & score.notna() & allw.start.notna()]
        hits = []
        for _, g in d.sort_values("start").groupby("pid"):
            st = g[lab[g.index] == "stress"]
            calm = g[lab[g.index].isin(calm_names.get(name, NON_STRESS))]
            if st.empty or calm.empty:
                continue
            ev = (st.start.diff() > pd.Timedelta(minutes=30)).cumsum()
            ch = calm.start.dt.hour + calm.start.dt.minute / 60
            for _, e in st.groupby(ev):
                h = (e.start.dt.hour + e.start.dt.minute / 60).mean()
                diff = (ch - h).abs() % 24
                ctrl = calm[(np.minimum(diff, 24 - diff) <= 1) & (calm.start.dt.date != e.start.dt.date.iloc[0])]
                if len(ctrl) >= 3:
                    hits.append(score[e.index].mean() < score[ctrl.index].mean())
        n = len(hits)
        rows.append({"dataset": name, "events": n, "hit_rate": np.mean(hits) if n else np.nan,
                     "p_value": binomtest(int(np.sum(hits)), n).pvalue if n else np.nan})
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


def final_test_v2(allw: pd.DataFrame, X: dict):
    """Second look at the LOCKED TEST people, for v2 (everyday-life push). Models trained on DEV only."""
    out = []
    def add(name, res):
        out.append(res.assign(system=name))
        r = res.set_index("dataset").mean_auc.reindex(DATASETS)
        print(f"{name:52s}", " ".join(f"{v:.3f}" for v in r), f"| mean {r.mean():.3f}")
    print(" " * 52, " ".join(DATASETS))
    add("FINAL v2", run(FINAL_V2, allw, X, "test"))
    add("FINAL v2, shuffled labels (should be ~0.5)", run(FINAL_V2, allw, X, "test", shuffle=True))
    add("FINAL v2, trained on other datasets only", run({**FINAL_V2, "other_datasets": True}, allw, X, "test"))
    pd.concat(out, ignore_index=True).to_csv(RESULTS / "final_test_v2.csv", index=False)


def plot_final():
    """figures/fig6_final.png: locked-test AUC per dataset, original formula vs start vs final."""
    import matplotlib.pyplot as plt
    from config import FIGURES
    names = {"wesad": "WESAD", "stressexercise": "Stress +\nexercise", "stresspredict": "Stress-\nPredict",
             "canstress": "CAN-STRESS", "nurse": "Nurses", "adarp": "ADARP"}
    systems = {"formula: BodySync + bursts": ("Original formula", "#b8c2cc"),
               "trained model (start of final push)": ("Trained model (start)", "#6c8091"),
               "FINAL BodySync": ("Final BodySync", "#00a896")}
    r = pd.read_csv(RESULTS / "final_test.csv").pivot(index="dataset", columns="system", values="mean_auc").loc[list(names)]
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(names))
    for i, (key, (label, color)) in enumerate(systems.items()):
        ax.bar(x + (i - 1) * 0.27, r[key], width=0.26, color=color, label=label)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(x, list(names.values()))
    ax.set_ylim(0.3, 1.02)
    ax.set_ylabel("Picks out stress (0.5 = coin flip)")
    ax.set_title("Locked test people: how well each version picks out stress")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig6_final.png", dpi=160)
    plt.close(fig)


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
    ap.add_argument("--final-v2", action="store_true", help="second, disclosed look: v2 on LOCKED TEST people")
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
    if args.final_v2:
        final_test_v2(allw, X)
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
    if args.round == 5:   # round 5: everyday life; kept if the everyday average rises and no lab set drops > 0.03
        log("5", "final system", run(FINAL, allw, X, "dev"))
        log("5", "+ time-of-day normal", run({**FINAL, "context": "hour"}, allw, X, "dev"))
        log("5", "+ same-day normal", run({**FINAL, "context": "day"}, allw, X, "dev"))
        log("5", "+ 15-min lookback (everyday)", run({**FINAL, "lookback": True}, allw, X, "dev"))
        log("5", "+ everyday datasets pooled (everyday)", run({**FINAL, "pool_life": True}, allw, X, "dev"))
    if args.round == 6:   # round 6: more everyday ideas on top of round 5
        r5 = {**FINAL, "pool_life": True}
        log("6", "round-5 system", run(r5, allw, X, "dev"))
        log("6", "+ logistic model for everyday data", run({**r5, "life_model": "logistic"}, allw, X, "dev"))
        log("6", "+ lab data as a low-weight helper (0.3)", run({**r5, "lab_helper": 0.3}, allw, X, "dev"))
        log("6", "+ stronger self-training (20%)", run({**r5, "pseudo": 0.2}, allw, X, "dev"))
    if args.round == 7:   # round 7: stack the round-6 keepers, then average xgb + logistic for everyday data
        r6 = {**FINAL, "pool_life": True, "lab_helper": 0.3, "pseudo": 0.2}
        log("7", "round-6 stack (helper + stronger self-training)", run(r6, allw, X, "dev"))
        log("7", "+ xgb & logistic averaged (everyday)", run({**r6, "life_model": "xgb+logistic"}, allw, X, "dev"))
    if args.round == 8:   # round 8 (after the audit): everyday life, judged per window AND per event
        # keep rule (fixed before running): everyday mean AUC gain > 0.01 (seed noise SD of the mean = 0.002),
        # better in >= 2 of 3 everyday datasets, event hit rate not lower, no lab dataset down > 0.03
        import audit
        fair = allw.label.copy()
        fair[(allw.dataset == "nurse") & (allw.label == "calm")] = "unknown"   # nurse test = stress vs no_stress
        fair[allw.dataset == "adarp"] = audit.adarp_labels(allw)[allw.dataset == "adarp"]   # 15 min before a press
        allw["label_fair"] = fair
        variants = {"final system": FINAL,
                    "+ past-only normal (1 day)": {**FINAL, "context": "past1d"},
                    "+ past-only normal (3 days)": {**FINAL, "context": "past3d"},
                    "+ train on the tested contrast": {**FINAL, "train_labels": "label_fair"}}
        for idea, cfg in variants.items():
            res, score = run(cfg, allw, X, "dev", return_scores=True)
            log("8", idea, res)
            ev = event_eval(allw, score, "dev").assign(round="8", idea=idea)
            print(ev.round(3).to_string(index=False), flush=True)
            ev.to_csv(EVENTS, mode="a", header=not EVENTS.exists(), index=False)
    if args.round == 9:   # round 9: everyday life, same keep rule as round 8 (all threads, as the frozen system)
        variants = {"final system (all threads)": FINAL,
                    "+ 30-min smoothing (everyday)": {**FINAL, "smooth_life": "30min"},
                    "+ 60-min smoothing (everyday)": {**FINAL, "smooth_life": "60min"},
                    "+ score only when still (everyday, below own median movement)": {**FINAL, "still_only": 0.5}}
        for idea, cfg in variants.items():
            res, score = run(cfg, allw, X, "dev", return_scores=True)
            log("9", idea, res)
            ev = event_eval(allw, score, "dev").assign(round="9", idea=idea)
            print(ev.round(3).to_string(index=False), flush=True)
            ev.to_csv(EVENTS, mode="a", header=False, index=False)


if __name__ == "__main__":
    main()
