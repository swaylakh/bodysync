"""Final push: iterate on DEV people, then test the frozen system once on LOCKED TEST people.

Protocol (fixed before any of these experiments ran):
  - In each of the 7 stress datasets, people are split at random (seed 42) into DEV (~60%) and
    LOCKED TEST (~40%), stratified so both groups contain people with stress windows.
    The split is saved in results/final_split.csv and committed before iterating.
  - Iteration (`--round`) only ever trains and scores DEV people; every idea tried is logged in
    results/iteration_log.csv, including the ones that don't help.
  - The frozen system (`--final`) is scored once on LOCKED TEST people: results/final_test.csv.
  - An idea is kept if it raises the DEV average AUC and drops no dataset by more than 0.03.

Every variant is scored on the same windows ("later" windows, see CALIB_GAP), so personal
calibration, which uses a person's earliest labelled windows, is compared fairly.
"""
import argparse

import numpy as np
import pandas as pd

from config import RESULTS, SEED
from feature_test import LAB, LIFE, load

SPLIT = RESULTS / "final_split.csv"
TEST_SHARE = 0.4


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-only", action="store_true")
    args = ap.parse_args()
    data = {name: load(name) for name in LAB + LIFE}
    split = make_split(data)
    print(split.groupby(["dataset", "group", "has_stress"]).size().unstack(fill_value=0).to_string())
    if args.split_only:
        return


if __name__ == "__main__":
    main()
