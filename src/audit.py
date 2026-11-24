"""Audit: is BodySync picking up stress, or shortcuts that come WITH stress in these studies?

Nothing here changes or tunes the frozen system (final_eval.FINAL). No decisions are made from these
numbers about which model to keep; they decide how each result should be described.
Checks (fixed before running), per group (DEV people, then LOCKED TEST people):
  raw shortcuts   a single raw column as the score, all people: wrist movement, minutes since the
                  session started, hour of day. Far below 0.5 also counts (it works read backwards).
  shortcut model  the FINAL pipeline (XGBoost + self-training + smoothing) fed ONLY movement, or
                  ONLY the clock (minutes since session start, hour of day). Knows nothing about the body.
  no shortcuts    FINAL without movement (move_z), and without movement + change-over-10-min (d_*).
  recovery        lab: leave out rest windows that start < 10 min after a stress window (they look
                  like "falling", i.e. recovery, by design of the protocol).
  CAN-STRESS      (a) calm only from low ratings on Type=Stress logbook rows (other low ratings are
                  cannabis / sleep / exercise entries); (b) stress and calm matched to within 1 h of
                  clock time, so evening vs daytime can't explain the result.
  ADARP           stress = only the 15 min BEFORE a button press, never a window holding a press
                  (pressing the button moves the wrist).
  self-training   adapt on ALL of a person's windows instead of the unlabelled ones only.
  circular shift  rotate each person's labels in time (keeps stress blocks intact): should be ~0.5.
  seeds (DEV)     FINAL and FINAL minus self-training with 5 XGBoost seeds: how big is seed noise
                  compared with the +0.01 gains used to pick ideas?
  ranges (TEST)   95% range of each locked-test result, resampling people (bootstrap).
Output: results/audit.csv
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import final_eval as fe
from config import RESULTS
from feature_test import LAB, NON_STRESS, load
from validate_datasets import CALM, evaluate, person_aucs

sys.path.insert(0, str(Path(__file__).parent / "datasets"))
import adarp  # noqa: E402
import canstress  # noqa: E402

warnings.filterwarnings("ignore")
KEY = (True, True, False, False)        # FINAL's feature table: change, shape, no context, no lookback
GAP = pd.Timedelta(minutes=10)
LAB_WIN = pd.Timedelta(minutes=2)
N_BOOT = 2000
rows = []


def add(check, group, res, **extra):
    for r in res.itertuples() if isinstance(res, pd.DataFrame) else res:
        r = r._asdict() if hasattr(r, "_asdict") else r
        rows.append({"check": check, "group": group, "dataset": r["dataset"], "n_people": r["n_people"],
                     "auc": round(r["mean_auc"], 3), **extra})
    df = pd.DataFrame(rows)
    d = df[(df.check == check) & (df.group == group)].set_index("dataset").auc
    print(f"{check:42s} {group:4s} " + "  ".join(f"{k}={v:.3f}" for k, v in d.items()), flush=True)


def ev_scores(allw, score, group, labels=None, datasets=fe.DATASETS):
    out = []
    for name in datasets:
        m = (allw.dataset == name) & (allw.group == group)
        s = allw[m].assign(v=score[m], v_reason="")
        if labels is not None:
            s["label"] = labels[m]
        try:
            out.append(evaluate(name, s, s.setting.iloc[0], "v"))
        except (KeyError, IndexError):   # no person left with >= 3 stress and >= 3 calm windows
            out.append({"dataset": name, "n_people": 0, "mean_auc": np.nan})
    return out


# ---------- shortcut columns ----------
def clock(allw):
    sess = allw.dataset + "|" + allw.pid + "|" + allw.get("session", pd.Series("", index=allw.index)).fillna("").astype(str)
    elapsed = (allw.start - allw.groupby(sess).start.transform("min")).dt.total_seconds() / 60
    return pd.DataFrame({"elapsed_min": elapsed, "hour": allw.start.dt.hour + allw.start.dt.minute / 60})


def raw_shortcuts():
    for name in fe.DATASETS + ["affectiveroad"]:
        w, _ = load(name)
        c = clock(w.assign(dataset=name))
        for col, x in [("move", w["move"]), ("elapsed", c.elapsed_min), ("hour", c.hour)]:
            r = evaluate(name, w.assign(v=-x, v_reason=""), "lab" if name in LAB else "free-living", "v")
            add(f"raw shortcut: {col}", "all", [r])


# ---------- relabelling ----------
def recovery_labels(allw):
    """Lab: rest windows starting < 10 min after the end of a stress window -> excluded."""
    lab = allw.label.copy()
    for _, g in allw[allw.dataset.isin(LAB)].dropna(subset=["start"]).groupby(["dataset", "pid"]):
        st = np.sort(g.start[g.label == "stress"].to_numpy())
        if not len(st):
            continue
        calm = g[g.label.isin(CALM)]
        i = np.searchsorted(st, calm.start.to_numpy(), side="right") - 1
        prev_end = np.where(i >= 0, st[np.clip(i, 0, None)] + LAB_WIN, np.datetime64("NaT"))
        gap = calm.start.to_numpy() - prev_end
        lab[calm.index[(i >= 0) & (gap < GAP)]] = "excluded"
    return lab


def canstress_labels(allw):
    """Re-derive CAN-STRESS labels with the loader's own rule; also the Type-restricted calm variant."""
    cw = allw[allw.dataset == "canstress"]
    lb = canstress.load_logbook()
    q = lb.dropna(subset=["Stress_Rating"]).groupby("pid")["Stress_Rating"].quantile([0.25, 0.75]).unstack()
    orig, typed = pd.Series("unknown", index=cw.index, dtype=object), pd.Series("unknown", index=cw.index, dtype=object)
    for sess, r in lb.groupby("E4_Sessions"):
        w = cw[cw.session.astype(str) == str(sess)]
        if w.empty:
            continue
        r = canstress.rating_times(r, w.start.min())
        q25, q75 = q.loc[r.pid.iloc[0], 0.25], q.loc[r.pid.iloc[0], 0.75]
        hi = (r.Stress_Rating >= q75) & (r.Stress_Rating >= 5)
        lo = (r.Stress_Rating <= q25) | (r.Stress_Rating <= 2)

        def near(ts):
            m = np.zeros(len(w), bool)
            for t in ts:
                m |= ((w.start < t + canstress.NEAR) & (w.start + pd.Timedelta(seconds=canstress.WIN) > t - canstress.NEAR)).to_numpy()
            return m
        nh, nl, nl_typed = near(r.t[hi]), near(r.t[lo]), near(r.t[lo & (r.Type == "Stress")])
        orig[w.index] = np.where(nh & ~nl, "stress", np.where(nl & ~nh, "calm", "unknown"))
        typed[w.index] = np.where(nh & ~nl, "stress", np.where(nl_typed & ~nh, "calm", "unknown"))
    agree = (orig == cw.label).mean()
    print(f"CAN-STRESS labels re-derived: {agree:.1%} agree with the loader", flush=True)
    return typed


def hour_matched(allw, labels, dataset):
    """Keep stress / calm windows only if the same person has the other kind within 1 h of clock time."""
    lab = labels.copy()
    d = allw[(allw.dataset == dataset) & labels.isin(["stress", "calm"])]
    h = d.start.dt.hour + d.start.dt.minute / 60
    def close(a, b):
        if not len(b):
            return np.zeros(len(a), bool)
        diff = np.abs(a[:, None] - b[None, :]) % 24
        return (np.minimum(diff, 24 - diff) <= 1).any(axis=1)

    for _, g in d.groupby("pid"):
        idx_s, idx_c = g.index[labels[g.index] == "stress"], g.index[labels[g.index] == "calm"]
        hs, hc = h[idx_s].to_numpy(), h[idx_c].to_numpy()
        lab[idx_s[~close(hs, hc)]] = "unknown"
        lab[idx_c[~close(hc, hs)]] = "unknown"
    return lab


def adarp_span(d):
    span = []
    for name in ("HR", "EDA", "TEMP"):
        f = d / f"{name}.csv"
        if not f.exists() or f.stat().st_size == 0:
            return None
        try:
            head = pd.read_csv(f, header=None, nrows=2)
            st, fs = float(head.iloc[0, 0]), float(head.iloc[1, 0])
        except (pd.errors.EmptyDataError, IndexError):
            return None
        with open(f, "rb") as fh:
            n = sum(1 for line in fh if line.strip()) - 2
        if n <= 0:
            return None
        span += [st, st + n / fs]
    return min(span), max(span)


def adarp_labels(allw):
    """Stress only from the 15 min before a valid press; windows holding any press -> unknown."""
    aw = allw[allw.dataset == "adarp"]
    s = (aw.start - pd.Timestamp("1970-01-01")).dt.total_seconds().to_numpy()
    e = s + adarp.WIN
    orig, pre = pd.Series("unknown", index=aw.index, dtype=object), aw.label.copy()
    for pdir in sorted(p for p in adarp.RAW.iterdir() if p.is_dir()):
        pid = "adarp_" + pdir.name.replace("Part ", "")
        tags = [t for d in sorted(p for p in pdir.iterdir() if p.is_dir()) for t in adarp.read_tags(d, adarp_span(d))]
        m = (aw.pid == pid).to_numpy()
        if not m.any() or not tags:
            continue
        all_t = np.array([t for t, _ in tags])
        val_t = np.array([t for t, edge in tags if not edge])
        ss, ee = s[m][:, None], e[m][:, None]
        stress = ((ss < val_t + adarp.STRESS_PAD) & (ee > val_t - adarp.STRESS_PAD)).any(axis=1) if len(val_t) else np.zeros(m.sum(), bool)
        dist = np.maximum(np.maximum(all_t - ee, ss - all_t), 0).min(axis=1)
        orig[aw.index[m]] = np.where(stress, "stress", np.where(dist >= adarp.CALM_GAP, "calm", "unknown"))
        before = ((ee <= val_t) & (ee > val_t - adarp.STRESS_PAD)).any(axis=1) if len(val_t) else np.zeros(m.sum(), bool)
        holds = ((ss <= all_t) & (all_t < ee)).any(axis=1)
        pre[aw.index[m]] = np.where(aw.label[m] == "stress", np.where(before & ~holds, "stress", "unknown"), aw.label[m])
    print(f"ADARP labels re-derived: {(orig == aw.label).mean():.1%} agree with the loader", flush=True)
    return pre


def bootstrap(allw, score, group):
    rng = np.random.default_rng(fe.SEED)
    out = []
    for name in fe.DATASETS:
        m = (allw.dataset == name) & (allw.group == group)
        aucs = person_aucs(name, allw[m].assign(v=score[m]), "v")[0].to_numpy()
        boot = rng.choice(aucs, size=(N_BOOT, len(aucs))).mean(axis=1)
        out.append({"dataset": name, "n_people": len(aucs), "mean_auc": aucs.mean(),
                    "lo": np.percentile(boot, 2.5), "hi": np.percentile(boot, 97.5)})
    return out


def main():
    raw_shortcuts()
    allw, X, _ = fe.prepare()
    allw["setting"] = np.where(allw.dataset.isin(LAB), "lab", "free-living")
    Xf = X[KEY]
    Xmove = {KEY: Xf[["move_z"]]}
    Xclock = {KEY: clock(allw)}
    recov = recovery_labels(allw)
    cs_typed = canstress_labels(allw)
    cs_hour = hour_matched(allw, allw.label, "canstress")
    cs_both = hour_matched(allw, cs_typed.reindex(allw.index).fillna(allw.label), "canstress")
    ad_pre = adarp_labels(allw)
    labelsets = {
        "eval: lab rest >10 min after stress": recov,
        "eval: CAN-STRESS calm from Type=Stress rows": cs_typed.reindex(allw.index).fillna(allw.label),
        "eval: CAN-STRESS hour-matched": cs_hour,
        "eval: CAN-STRESS Type=Stress + hour-matched": cs_both,
        "eval: ADARP 15 min before press only": ad_pre.reindex(allw.index).fillna(allw.label),
    }
    # how strong is the clock shortcut once CAN-STRESS / ADARP labels are cleaned?
    clk = clock(allw)
    for check, lab in labelsets.items():
        for col in ("hour", "elapsed_min"):
            add(f"raw {col} | {check}", "all", ev_scores(allw.assign(group="all"), -clk[col], "all", lab))
        add(f"raw move | {check}", "all", ev_scores(allw.assign(group="all"), -allw["move"], "all", lab))

    for group in ("dev", "test"):
        res, score = fe.run(fe.FINAL, allw, X, group, return_scores=True)
        add("FINAL", group, res)
        for check, lab in labelsets.items():
            add(f"FINAL | {check}", group, ev_scores(allw, score, group, lab))
        add("shortcut model: movement only", group, fe.run(fe.FINAL, allw, Xmove, group))
        add("shortcut model: clock only", group, fe.run(fe.FINAL, allw, Xclock, group))
        add("FINAL without movement", group, fe.run({**fe.FINAL, "drop": ("move_z",)}, allw, X, group))
        nos = {**fe.FINAL, "drop": ("move_z", "d_")}
        res_n, score_n = fe.run(nos, allw, X, group, return_scores=True)
        add("FINAL without movement + change", group, res_n)
        add("FINAL without movement + change | eval: lab rest >10 min after stress", group,
            ev_scores(allw, score_n, group, recov))
        add("FINAL, self-training on all windows", group, fe.run({**fe.FINAL, "selftrain_on": "all"}, allw, X, group))
        add("FINAL, labels rotated in time (should be ~0.5)", group, fe.run(fe.FINAL, allw, X, group, shuffle="circular"))
        if group == "test":
            for r in bootstrap(allw, score, group):
                add("FINAL 95% range (bootstrap people)", group, [r], lo=round(r["lo"], 3), hi=round(r["hi"], 3))
        pd.DataFrame(rows).to_csv(RESULTS / "audit.csv", index=False)

    no_self = {**fe.FINAL, "selftrain": False}
    for seed in range(42, 47):
        fe.SEED = seed
        add(f"seed {seed}: FINAL", "dev", fe.run(fe.FINAL, allw, X, "dev"))
        add(f"seed {seed}: FINAL without self-training", "dev", fe.run(no_self, allw, X, "dev"))
    fe.SEED = 42
    pd.DataFrame(rows).to_csv(RESULTS / "audit.csv", index=False)


if __name__ == "__main__":
    main()
