"""Turn results/*.csv into LaTeX tables in report/tables/, so the report never has hand-copied numbers,
and copy the figures into report/figures/ so the report folder can be uploaded to Overleaf as is.

Run after src/run_all.py:  python report/make_tables.py
"""
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "report" / "tables"
OUT.mkdir(exist_ok=True)

NAMES = {"wesad": "WESAD", "nurse": "Nurse stress", "stresspredict": "Stress-Predict",
         "stressexercise": "Stress + exercise", "e4stress": "EmpaticaE4Stress$^\\dagger$",
         "adarp": "ADARP", "canstress": "CAN-STRESS"}
ORDER = list(NAMES)


def p(x: float) -> str:
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def write(name: str, header: str, rows: list[str], cols: str):
    body = "\n".join(rows)
    (OUT / f"{name}.tex").write_text(
        f"\\begin{{tabular}}{{{cols}}}\n\\toprule\n{header} \\\\\n\\midrule\n{body}\n\\bottomrule\n\\end{{tabular}}\n",
        encoding="utf-8")


def coupling():
    acc = pd.read_csv(RES / "coupling_test.csv").pivot(index="features", columns="task", values="balanced_acc")
    order = ["coupling only", "levels", "levels + personal", "levels + personal + coupling"]
    rows = [f"{f.capitalize()} & {acc.loc[f, 'sleep / rest / active']:.3f} & {acc.loc[f, 'sleep vs rest']:.3f} \\\\"
            for f in order]
    write("coupling", "Features & Sleep / rest / active & Sleep vs rest", rows, "lcc")

    t = pd.read_csv(RES / "coupling_test_paired.csv")
    rows = [f"{r.task.capitalize()} & {r.comparison} & {r.median_gain:+.3f} & {r.people_better}/{r.n_people} & {p(r.p_value)} \\\\"
            for r in t.itertuples()]
    write("coupling_paired", "Task & Comparison & Median gain & People better & $p$", rows, "llccc")


def zupwd():
    ex = pd.read_csv(RES / "zupwd_exercise_check.csv").set_index("situation").loc[["sleep", "rest", "active", "other"]]
    rows = [f"{s.capitalize()} & {r.situation_aware:.1f} & {r.one_normal_for_all:.1f} \\\\" for s, r in ex.iterrows()]
    write("zupwd_exercise", "Situation & Situation-aware normal & One normal for all", rows, "lcc")

    sl = pd.read_csv(RES / "zupwd_sleep_check.csv")
    label = {"wakeups_per_h": "More wake-ups", "pct_awake": "More time awake", "sleep_h": "Less sleep"}
    score = {"sync": "A (sync)", "bodysync": "B (BodySync)"}
    rows = [f"{score[r.score]} & {label[r.bad_night_means]} & {r.score_after_bad:.1f} & {r.score_after_good:.1f} & "
            f"{r.people_lower_after_bad}/{r.n_people} & {p(r.p_value)} \\\\" for r in sl.itertuples()]
    write("zupwd_sleep", "Score & Bad night = & After bad & After good & People lower & $p$", rows, "llcccc")


def datasets():
    d = pd.read_csv(RES / "datasets_stress_vs_calm.csv")
    rows = []
    for ds in ORDER:
        a = d[(d.dataset == ds) & (d.score == "A sync")].iloc[0]
        b = d[(d.dataset == ds) & (d.score == "B BodySync")].iloc[0]
        auc_a, auc_b = f"{a.mean_auc:.2f}", f"{b.mean_auc:.2f}"
        if a.mean_auc > b.mean_auc:      # bold the better score of the two
            auc_a = f"\\textbf{{{auc_a}}}"
        else:
            auc_b = f"\\textbf{{{auc_b}}}"
        rows.append(f"{NAMES[ds]} & {b.setting} & {b.n_people} & {auc_a} & {a.people_lower_in_stress}/{a.n_people} & "
                    f"{p(a.p_value)} & {auc_b} & {b.people_lower_in_stress}/{b.n_people} & {p(b.p_value)} \\\\")
    write("datasets", " & & & \\multicolumn{3}{c}{A: sync} & \\multicolumn{3}{c}{B: BodySync} \\\\\n"
          "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\nDataset & Setting & $n$ & AUC & Lower & $p$ & AUC & Lower & $p$",
          rows, "llccccccc")

    ex = pd.read_csv(RES / "datasets_exercise_vs_stress.csv").set_index("label")
    ex = ex.loc[[c for c in ["baseline", "rest", "stress", "aerobic", "anaerobic", "recovery"] if c in ex.index]]
    rows = [f"{c.capitalize()} & {r['A sync']:.1f} & {r['B BodySync']:.1f} \\\\" for c, r in ex.iterrows()]
    write("exercise", "Condition & A: sync & B: BodySync", rows, "lcc")


def flatten():
    """report/overleaf.tex: main.tex with the tables and bibliography pasted in, so it is one file.
    Needs main.bbl, i.e. compile main.tex once first."""
    rep = ROOT / "report"
    bbl = rep / "main.bbl"
    if not bbl.exists():
        print("compile main.tex first to make main.bbl; overleaf.tex not written")
        return
    tex = (rep / "main.tex").read_text(encoding="utf-8")
    for f in sorted(OUT.glob("*.tex")):
        tex = tex.replace(f"\\input{{tables/{f.stem}}}", f.read_text(encoding="utf-8").strip())
    tex = tex.replace("\\bibliographystyle{unsrtnat}\n\\bibliography{refs}", bbl.read_text(encoding="utf-8").strip())
    (rep / "overleaf.tex").write_text(tex, encoding="utf-8")
    print("wrote", rep / "overleaf.tex")


if __name__ == "__main__":
    coupling()
    zupwd()
    datasets()
    shutil.copytree(ROOT / "figures", ROOT / "report" / "figures", dirs_exist_ok=True)
    print("tables and figures written to", ROOT / "report")
    flatten()
