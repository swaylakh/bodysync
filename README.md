# BodySync

A personal stress score from a wristband. BodySync compares your heart rate, sweat (EDA) and skin temperature with **your own normal** and turns them into one 0–100 score: 100 at your normal, lower when your body reacts the way it does under stress.

Team: Swayam Lakhotia, Ananya Khaitan, Bhavya Bajaj. Start with `HANDOVER.md` for where everything is.

## Results (people the model never saw)

| Study | Setting | First version (plain formula) | Final BodySync |
|---|---|---|---|
| WESAD | lab | 0.92 | **0.99+** |
| Stress + exercise | lab | 0.63 | **0.91** |
| Stress-Predict | lab | 0.62 | **0.90** |
| CAN-STRESS | daily life | 0.59 | **0.67** |
| AffectiveROAD, city vs highway | real driving, a study it never saw | 0.54 | **0.76** |

The number is the within-person AUC: pick a stressed and a calm moment from the same person; how often does the stressed one score lower? 0.5 is a coin flip. The four main studies are scored on the 40% of people locked away before any building (`results/final_test.csv`); driving was added after the system was frozen (`results/new_datasets.csv`).

## How it works

1. **Your normal:** each person's median and typical spread of every signal.
2. **Stress signals measured against it:** distance from normal for heart rate, sweat and temperature, sweat bursts, heart-rate variability, fast and slow parts of the sweat signal, rising or falling trends, and change over the last 10 minutes.
3. **A trained XGBoost model**, which then tunes itself to each person (self-training on their own unlabelled time).
4. **A smoothed 0–100 score.**

The first version was a plain formula: a z-score from your own normal per signal, stress direction only, combined as score = 100 / (1 + distance / 2). It is in `src/bodysync.py`; the report works through an example.

## How it was tested

- **Locked split:** 40% of the people in every study were locked away first (`results/final_split.csv`).
- **Iteration on the other 60%:** every idea is logged in `results/iteration_log.csv` (17 rounds, including all that failed).
- **One look at the locked people** with the frozen system (`results/final_test.csv`).
- **Checks:** scrambled labels drop to a coin flip (0.51, in `results/final_test.csv`); wrist movement removed, rest periods left out and a clock-only model are in `results/audit.csv`.

## Code

| File | What it does |
|---|---|
| `src/datasets/*.py` | One loader per public dataset; each writes the common format in `src/datasets/SPEC.md` |
| `src/bodysync.py` | The plain-formula score (first version) |
| `src/feature_test.py` | Loading the converted datasets and building the features |
| `src/validate_datasets.py` | The evaluation (within-person AUC) |
| `src/final_eval.py` | The final system: locked split, rounds of iteration, locked test |
| `src/plain_formula.py` | The first version scored on the same people |
| `src/new_datasets.py` | The frozen system on AffectiveROAD (real driving) |
| `src/driving_selftrain.py`, `src/driving_order.py` | Self-training and step-by-step gains on driving |
| `src/driving_tracking.py` | Does it follow the drivers' moment-to-moment stress rating? |
| `src/driving_example.py` | One real drive, scored minute by minute (used in the 3D walkthrough) |
| `src/audit.py` | Shortcut checks, seed noise, 95% ranges |
| `src/report_figures.py` | Figures 8–10 of the report |

## Running it

```
pip install -r requirements.txt
python src/datasets/<name>.py          # once per downloaded dataset
python src/final_eval.py --round N     # one round of iteration on the 60% (DEV)
python src/final_eval.py --final       # the frozen system on the locked 40%
python src/new_datasets.py             # the frozen system on real driving
python src/audit.py                    # checks
python src/report_figures.py           # report figures
```

XGBoost uses all CPU threads, which reproduces the frozen numbers.

## Getting the data

The datasets are not included because of their size (about 7 GB) and licences.

| Dataset | Source | Put it in |
|---|---|---|
| WESAD | https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx | `data/wesad/WESAD.zip` |
| Stress + exercise | https://physionet.org/content/wearable-device-dataset/1.0.1/ | `data/external/stressexercise/` |
| Stress-Predict | https://github.com/italha-d/Stress-Predict-Dataset | `data/external/stresspredict/` |
| CAN-STRESS | https://zenodo.org/records/14842061 | `data/external/canstress/` |
| AffectiveROAD | https://www.media.mit.edu/groups/affective-computing/data/ | `data/external/affectiveroad/` |
| Nurse stress | https://doi.org/10.5061/dryad.5hqbzkh6f | `data/external/nurse/` |
| ADARP | https://zenodo.org/records/6640290 | `data/external/adarp/` |
| EmpaticaE4Stress | https://doi.org/10.17632/kb42z77m2g.2 | `data/external/e4stress/` |

Nurse stress and ADARP are loaded but left out of the results (only 4 test people each); EmpaticaE4Stress has no calm periods besides the baseline.

## Limitations

- Everyday life rests on 10 hidden people and driving on 9.
- In lab studies stress always comes at the same point; a clock alone scores high there. The driving test, where time alone scores 0.50, is the main evidence against this.
- Everyday stress is milder and rated a few times a day; calm ratings cluster in the evening.
- Each person's normal uses their whole recording; a real app would only have the past.
- It separates stressful stretches from calm ones but does not follow moment-to-moment stress.
- Self-training is a small, uneven gain (about +0.02 on average, +0.08 on driving).

## Earlier stages

`archive/` keeps the earlier work: the first "sync" idea and its everyday-data pipeline, the mid-term report, and the weekly decks. `archive/README_full_history.md` is the full technical log of every stage.
