# BodySync

A score out of 100 for how far your body is from **your own normal** right now, from wrist heart rate, sweat (EDA) and skin temperature. Built on free-living Empatica EmbracePlus data (ZU-PWD '25) and checked on lab stress data (WESAD).

The project started as a *sync* score: do the signals move together? That idea was tested thoroughly and did not work (sections below). The final score uses the signal levels instead.

## Final BodySync score

`src/bodysync.py` (the score), `src/bodysync_validate.py` (checks), `results/bodysync_*.csv`, `figures/bodysync/`

1. **Per window:** average heart rate, sweat (log scale) and skin temperature.
2. **Compare each with this person's normal in the same situation** (asleep / resting / active / other): median and typical spread (MAD). Minimum spreads are 2 bpm, 0.05 log-µS and 0.1 °C. A workout is compared with the person's own workouts, so it isn't mistaken for stress.
3. **Distance** = root-mean-square of the three "spreads away from normal" values.
4. **Score** = 100 / (1 + distance / 2). That gives 100 at normal, 50 at 2 spreads away and 25 at 6. It never reaches 0, so big reactions stay distinguishable.
5. **Reason:** the signal furthest from normal is reported with the score (for example "sweat high").
6. **Version 2 (current, `directional=True`):** only changes in the stress direction count. The checks in this section were run on version 1; the multi-dataset section below uses version 2.

### Check: does it catch stress? (WESAD, 15 people, real stress test)

Each person's normal comes from their baseline condition. The score is computed on 2-min windows.

| Condition | Mean score |
|---|---|
| Baseline | 69.8 |
| Amusement | 34.9 |
| Meditation | 28.6 |
| **Stress** | **20.1** |

- **Stress vs baseline:** lower in 15 of 15 people, p < .001.
- **Stress vs amusement:** lower in 13 of 15, p = .003. This is the fair comparison, because amusement wasn't used to set the normal.
- **Stress vs meditation:** lower in 12 of 15, p = .026.
- **Within each person, the score separates stress windows from all others with an average AUC of 0.89,** above chance for all 15 people.
- **Main reason given during stress:** sweat high 51%, skin temp low 36%, heart rate high 13%. That matches known stress physiology (sweating, colder hands from narrowing blood vessels, faster heart).
- **Caveat:** amusement and meditation also score well below baseline. Part of the drop is the body drifting over a 2-hour lab session, so the score means "different from your normal", not "stressed". Stress is simply the largest difference.

### Check: everyday life (ZU-PWD)

- **Exercise:** with situation-aware normals, the median score during activity is 68.0, similar to the other situations (69–70). With one normal for everything, it falls to 61.9. So the situation-aware version stops exercise from looking like stress. That was the main weakness of PWI.
- **Sleep:** the next-day score is not lower after nights with more wake-ups, more time awake or less sleep (12 people, all p ≥ .38).
- **Recovery:** night-time score was compared with night-time HRV (RMSSD) and breathing rate. The score never sees these markers, and they're only measured during sleep. Using 301 nights from 12 people, each compared with themselves:
  - **HRV:** no link (ρ = 0.08, p = .17).
  - **Breathing rate:** a faster-breathing night goes with a slightly lower score (ρ = −0.13, p = .02), but only 8 of 12 people go that way (p = .66). With HR taken out of the score the link disappears (ρ = 0.01). So it most likely just reflects heart rate and breathing rising together, not a real recovery signal.
  - Results: `results/bodysync_zupwd_recovery_check.csv`.
- **Stress in daily life can't be checked:** the self-reports are paywalled.

**In short:** the score works in the lab (it catches stress in every person) and treats exercise correctly in daily life. It hasn't yet been shown to track day-to-day wellbeing.

## Testing on every public Empatica dataset we could get

Loaders: `src/datasets/` (common format in `SPEC.md`). Scoring: `src/multi_validate.py`. Learned detector: `src/stress_model.py`. Final test: `src/nurse_final.py`. All of these datasets use the Empatica **E4**, the predecessor of the EmbracePlus in ZU-PWD; it measures the same three signals. No other public EmbracePlus dataset was found.

Citations are from OpenAlex, checked 2026-10-27, sorted by citations.

| Dataset | Paper | Citations | Setting | People | Stress label |
|---|---|---|---|---|---|
| WESAD | Schmidt et al. 2018, ICMI | **1,352** | lab | 15 | Trier Social Stress Test |
| Nurse stress | Hosseini et al. 2022, *Scientific Data* | **119** | hospital shifts | 15 | nurse-rated stress on detector-flagged intervals |
| Stress-Predict | Iqbal et al. 2023, *Neural Comput. Appl.* | 75 | lab | 35 | Stroop + interview |
| Stress + exercise | Hongn et al. 2025, *Scientific Data* | 37 | lab | 36 | arithmetic/emotional tasks; separate cycling sessions |
| EmpaticaE4Stress | 2024, *Data in Brief* | 26 | lab | 29 | Lego tasks (no later rest; weaker test) |
| ADARP | Sah et al. 2022, arXiv | 13 | free-living | 11 | button press when stressed |
| CAN-STRESS | 2025, BSN | 1 | free-living | 62 | self-rated stress 0–9 |
| *ZU-PWD (ours)* | Sharma et al. 2025, *IADIS IJCSIS* (PWI paper) | *0* | free-living | 27 | none in the free copy |

Not used:
- **K-EmoCon and AffectiveROAD:** access by request only.
- **UNIVERSE:** 20 GB.
- **STREX-VR and the 147-person EDA set:** EDA only.
- **Exam stress (PhysioNet):** no non-exam comparison.

### Version 2 of the score (frozen before the held-out tests)

Version 1 counted *any* change from normal, including being calmer than normal. After stress tasks, heart rate often drops below the starting level, so rest periods scored as badly as stress. Version 2 counts only changes in the stress direction: HR up, sweat up, skin temperature down. This change was fixed on the first five datasets. ADARP and the nurse data were then each tested once.

### Results

Within-person AUC: stress vs calm/rest windows of the same person, 0.5 = chance. Lab comparisons use *later* rest, not the baseline that set the normal.

| Dataset | Frozen score v2 | Learned, logistic | Learned, XGBoost |
|---|---|---|---|
| WESAD | **0.78** | 0.33 | 0.57 |
| Stress + exercise | 0.60 | **0.77** | 0.66 |
| Stress-Predict | 0.54 | 0.59 | **0.69** |
| CAN-STRESS (free-living) | 0.54 | **0.58** | 0.50 |
| ADARP (free-living, held out) | 0.45 | **0.59** | 0.50 |
| Nurse, fair: felt stress vs flagged-but-not-stressed (held out) | 0.52 | 0.54 | 0.53 |
| Nurse, stress vs calm (inflated, see below) | 0.60 | 0.60 | 0.61 |
| EmpaticaE4Stress (stress vs baseline only, weaker) | 0.91 | 0.55 | 0.74 |

Notes:
- **Learned models** use personal z-scores, 10-minute changes and personal movement. They are trained leave-one-dataset-out, with each dataset weighted equally, and the nurse data is never used for training.
- **Nurse "stress vs calm" is inflated:** the authors chose the survey intervals with an EDA-based detector, so stress windows were selected for high sweat. The fair test compares intervals the detector flagged where the nurse did or did not feel stressed.
- **Exercise vs stress** (Stress + exercise sessions): every method scores cycling as *more* stress-like than the stress tasks (AUC 0.11–0.27). Without a personal "exercising normal", which ZU-PWD does have, exercise looks like stress.

### Conclusion

**The score catches acute lab stress strongly on WESAD,** the field's standard benchmark, and weakly on the other lab sets. **On free-living stress (CAN-STRESS, ADARP, nurses) every method is close to chance,** including learned models. Average heart rate, sweat and temperature from a wrist overlap too much between felt-stress and non-stress moments in daily life.

This matches the cross-dataset generalisation problem reported in wearable stress research: within-dataset accuracy is high, but accuracy on a new dataset falls sharply. It is a property of the signals and labels, not of one formula. **BodySync should be presented as a personal "distance from your normal" score that detects acute physiological stress responses, not as a validated everyday stress detector.**

## Original question: does sync add anything?

**No, not in this data.** Coupling features on their own are close to chance. Adding them to plain signal values improves accuracy by under 1 percentage point, and the gain is not consistent. Round 2 fixed the known weaknesses of the coupling measure (rank correlation, time lags, personal baselines). That made coupling more informative on its own, but it still adds nothing significant on top of plain values.

A **personal baseline** (each person's values compared with their own normal) raises pooled accuracy by 2–7 points. Per person, though, that gain is significant in only 1 of 10 configurations, so it is a trend, not a proven result.

This matches Xu et al. (2026, *Sensors*), who also found that correlation features did not beat single signals. It conflicts with lab studies using ECG and breathing belts (Valenza 2013, Bartsch 2012). The likely reasons are the wristband's 1-minute resolution and an EDA signal that is coarse and often at the sensor floor (see Limitations).

## First attempt: a sync-based score (failed validation)

`src/sync_score.py` → `results/sync_scores.csv`, `figures/sync_score/`

Every 30 minutes (sliding by 15), for each person:
1. **Measure sync:** Spearman r for HR–EDA, HR–temp and EDA–temp.
2. **Compare with that person's own normal for the same situation** (asleep, resting, active or other): the median and typical spread (MAD) of each r.
3. **Desync** = the average distance from normal across the 3 pairs, in units of typical spread.
4. **Score:** 100 means syncing exactly as usual. It drops 33 points per unit of desync, with a floor of 0. Two windows in a row below 50 trigger a **desync alert**.

**Output:** 18,563 scored windows from 23 people. The median score is 77; 5% of windows score under 54; there are 24 alerts (0.1% of windows).

**Status: computed but not yet validated.**
- A person's average says nothing, by design: every person averages about 75 because each is compared with their own normal.
- Scores swing a lot from one window to the next.
- Windows where the situation is changing don't score lower than steady ones (77.1 vs 77.3).

Showing that a low score *means* something (stress, illness, poor sleep) needs an outcome to compare against. The ZU-PWD self-reports (stress, feelings, events) are behind the DataPort paywall and not available to us, so two free checks were run instead.

### Check 1: sleep quality (ZU-PWD)

`src/validate_sleep.py`, `results/validation_sleep.csv`. Stage 102 marks brief wake-ups during sleep. Using 190 night→next-day pairs from 10 people (each person compared with themselves), the next-day sync score is **not** lower after nights with more wake-ups, more time awake or less sleep (all p ≥ .10).

### Check 2: lab stress (WESAD)

`src/wesad.py`, `results/wesad_*.csv`, `figures/sync_score/wesad_validation.png`.
- **Data:** WESAD has 15 people wearing an Empatica E4 on the wrist through baseline, a stress test (TSST), amusement and meditation. It has much finer detail than ZU-PWD: everything is brought to 1 reading per second, with 2-min windows sliding by 30 s. The same sync score is used, with each person's normal set from their baseline condition.
- **The pickle files are loaded with an unpickler that only allows numpy arrays,** so loading them can't run code.

| | Baseline | Stress | Amusement | Meditation |
|---|---|---|---|---|
| Sync score (mean of people) | 74.0 | 71.3 | 71.5 | 69.7 |

- **Stress vs baseline:** lower in 7 of 15 people, p = .60.
- **Stress vs amusement:** p = .42.
- **For comparison:** the same score built on plain levels drops for all 15 people under stress (p < .001).
- **Classification (stress vs calm, LOSO):** levels alone 91.5%, coupling alone 58.7%, levels + coupling 91.8% (p = .14 vs levels alone), levels + sync score 91.6%.

**Conclusion:** the sync score doesn't detect stress even with high-resolution data, so the problem isn't only ZU-PWD's 1-minute resolution. Across both datasets and all checks, plain levels compared with a personal baseline carry the information. How signals move together adds little or nothing.

## Pipeline

```
python src/run_all.py        # ~1 h, mostly LOSO training
```

| Step | Script | What it does |
|---|---|---|
| 1 | `src/clean.py` | Keeps worn minutes (wear ≥ 50%); HR outside 35–200 bpm and temp outside 25–40 °C set to missing; labels each minute as **sleep** (stage 101), **rest** (awake + sedentary) or **active** (awake + MPA/VPA), with anything else marked other |
| 2 | `src/features.py` | Sliding windows of 30 min (step 15), 60 min (step 30) and 6 h (step 60). Per window: **levels** (mean, SD of HR / EDA / temp), **personal levels** (mean vs own median, scaled by IQR), **coupling** (Pearson r for HR–EDA, HR–temp and EDA–temp, on raw values and on minute-to-minute changes) |
| 3 | `src/explore.py` | Figures 1–4 and the feature-by-state statistics |
| 4 | `src/model.py` | Random Forest and XGBoost, leave-one-subject-out (LOSO), 5 feature sets × 2 tasks × 2 window lengths, plus paired Wilcoxon tests across people |
| 5 | `src/explain.py` | SHAP, UMAP "body weather map", desync score |
| 6 | `src/model_v2.py` | Round 2: improved coupling features (Spearman, lagged, personal), Holm-corrected tests |
| 7 | `src/model_v3.py` | Round 3: 8 coupling variants, plus nested CV that picks the best one fairly |

**When a window counts as labelled:** at least 75% of its minutes have all 3 signals, at least 70% of minutes are in one state, at least 50% of EDA minutes are above 0.05 µS, and the person has at least 10 windows. The 6-hour windows almost never contain a single state (only 6 active windows), so they are not used for classification.

## Data after cleaning

| | Rows |
|---|---|
| Raw minutes | 901,440 |
| Worn | 538,312 |
| Worn with HR + EDA + temp | 484,636 |
| Labelled 30-min windows | 8,203 from 23 people (4,865 sleep · 2,631 rest · 707 active) |
| Labelled 60-min windows | 3,593 from 22 people |

## Results

LOSO balanced accuracy, XGBoost, 30-min windows (`results/loso_pooled.csv`). Random Forest and 60-min windows give the same picture.

| Feature set | 3-class (chance 0.33) | Sleep vs rest (chance 0.50) |
|---|---|---|
| Coupling only | 0.372 | 0.562 |
| Levels | 0.807 | 0.831 |
| Levels + coupling | 0.810 | 0.839 |
| Levels + personal | 0.825 | 0.878 |
| All | **0.835** | **0.884** |

**Paired tests across people** (`results/loso_paired_tests.csv`, 20 comparisons):
- **Adding coupling:** the median gain per person is +0.0 to +1.4 points. It is significant at p < .05 in 7 of 20 comparisons, and none survive a Bonferroni correction.
- **Coupling alone:** worse than levels in every configuration (p < .001).
- **Personal baseline vs plain levels:** the median gain per person is about +2 points, and it is higher for 8–17 people per configuration. It is significant (p < .05) in only 1 of 10 configurations.
- **Sensitivity check:** keeping the low-EDA windows lowers every score by 1–5 points but doesn't change the ranking.

### Round 2: better coupling measures

`src/model_v2.py`, `results/v2_*.csv`. XGBoost only, since Random Forest gave the same picture in round 1. The three fixes were chosen before seeing any results:
- **Spearman rank r:** robust to EDA's 0.01 µS staircase steps.
- **Lagged cross-correlation:** the strongest r within ±10 min, plus the lag where it occurs, because skin temperature reacts with a delay.
- **Personal coupling:** coupling compared with the person's own usual coupling.

All 8 tests are Holm-corrected together.

| Task | Window | Coupling v1 | Coupling v1+v2 | Levels + personal | Levels + personal + all coupling |
|---|---|---|---|---|---|
| 3-class | 30 min | 0.372 | 0.432 | 0.825 | 0.839 |
| 3-class | 60 min | 0.383 | 0.399 | 0.824 | 0.827 |
| Sleep vs rest | 30 min | 0.562 | 0.599 | 0.878 | 0.887 |
| Sleep vs rest | 60 min | 0.532 | 0.562 | 0.891 | 0.899 |

- **Did the fixes make coupling alone better?** Yes for sleep vs rest at 30 min: +3 points per person, Holm p = .008. In the other 3 configurations the trend points the same way but isn't significant.
- **Does coupling now add to the best baseline?** No. The gain is +0.4 to +1.2 points per person, and no Holm-corrected p is below .21.

### Round 3: eight variants, picked fairly with nested CV

`src/model_v3.py`, `results/v3_*.csv`, 30-min windows, XGBoost. Each variant adds one kind of sync to the levels + personal baseline:
- Pearson r
- r of minute-to-minute changes
- Spearman r
- lagged r
- personal coupling
- sync strength (|r| plus a mean-|r| "sync index")
- all of these together

Nested CV picks the best variant using only the training people, then tests it once on the held-out person.

| Method | 3-class | Sleep vs rest |
|---|---|---|
| Baseline (levels + personal) | 0.825 | 0.878 |
| Best single variant (optimistic) | 0.839 (all coupling) | 0.888 (sync strength) |
| **Nested: best variant picked fairly** | **0.832** | **0.887** |

- **Direction is consistent:** 13 of the 14 variant scores are at or above the baseline. The inner loop also never picked "no sync": it chose personal coupling or all coupling in 34 of 40 folds.
- **But it is too small to confirm:** the gain is +0.7 to +1.0 points per person, and none of the 16 per-person tests reaches p < .05, even before correction. The smallest is p = .063.
- **Conclusion:** sync gives at most a small edge of about 1 point. Detecting an effect that small would need many more people than the ~20 here.

**Coupling barely differs by state** (`results/features_by_state.csv`, Friedman test on per-person medians, 13 people with all 3 states). All six level features differ (p ≤ .002). Of the six coupling features, only HR–EDA change-coupling does (p = .009).

**SHAP** (Fig. 5): coupling features make up 11–12% of total importance. Heart rate (mean, SD and relative to own normal) dominates.

**UMAP** (Fig. 6): with levels, sleep, rest and active separate into regions. With coupling only, the three states are fully mixed.

**Desync score** (Fig. 7, `results/desync_validation.csv`): the mean absolute deviation of a window's coupling from that person's usual coupling in the same state. There are no stress labels in the free data, so it was checked against what we can see:
- **Mixed (transition) windows score higher than single-state windows:** median 0.80 vs 0.75, p < 1e-35. Significant, but the effect is small.
- **Windows the classifier got wrong don't score higher than the ones it got right:** median 0.76 vs 0.75, p = 0.09. This does not support Farahani et al.'s (2026) idea of using coupling to flag classifier failures.
- **It is nearly unrelated to unusual signal levels** (Spearman ρ = 0.12). So it measures something different, but we have no outcome showing that something is meaningful.

## Limitations

- **Coarse EDA:** EDA is reported in steps of 0.01 µS and about 40% of minutes are below 0.05 µS. Over 30 minutes that often looks like a staircase, which makes correlations noisy.
- **1-minute data:** coupling in the literature uses beat-to-beat ECG or 4 Hz EDA. One-minute averages may smooth away the fast co-movement.
- **Labels from movement:** the states come from Empatica's own algorithms, which use the accelerometer, not ground truth. Accelerometer features were deliberately left out of the models.
- **No stress labels:** the self-report file is behind the IEEE DataPort paywall, so the desync score can't be checked against stress.
- **Uneven data:** participants 110 and 111 hold about half the windows. Per-person scores are reported for this reason.
- **Personal baselines:** each person's median and IQR are computed from all of their own data, including the test person in LOSO. No labels are used, but in a real deployment this needs a calibration period.

## Possible next steps

1. **Test in everyday life:** check the final score against real stress labels from daily life. The ZU-PWD self-reports (paywalled) would do, or another free-living dataset with stress labels, such as the nurse stress dataset (Hosseini et al. 2022, Empatica E4, real hospital shifts, publicly available).
2. **Correct for drift:** handle slow drift within a session, so that "different from normal" separates stress more clearly from amusement and meditation.
3. **Show it live:** build a simple dashboard of each person's score over time, with the reason for each dip.

## Getting the data

Datasets aren't in this repo because of their size and licences. Put each one where shown, then run its loader from the repo root.

| Dataset | Source | Put it in | Loader |
|---|---|---|---|
| ZU-PWD '25 | see `data/DATASET_SOURCE.md` | `data/final_merged_2024_to_2025.csv` | `python src/run_all.py` |
| WESAD | https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx | `data/wesad/WESAD.zip` | `python src/wesad.py` |
| Nurse stress | https://doi.org/10.5061/dryad.5hqbzkh6f (browser download) | `data/external/nurse/` | `python src/datasets/nurse.py` |
| ADARP | https://zenodo.org/records/6640290 | `data/external/adarp/` | `python src/datasets/adarp.py` |
| CAN-STRESS | https://zenodo.org/records/14842061 | `data/external/canstress/` | `python src/datasets/canstress.py` |
| Stress + exercise | https://physionet.org/content/wearable-device-dataset/1.0.1/ | `data/external/stressexercise/` | `python src/datasets/stressexercise.py` |
| Stress-Predict | https://github.com/italha-d/Stress-Predict-Dataset | `data/external/stresspredict/` | `python src/datasets/stresspredict.py` |
| EmpaticaE4Stress | https://doi.org/10.17632/kb42z77m2g.2 | `data/external/e4stress/` | `python src/datasets/e4stress.py` |

After the loaders have run:
- **All datasets:** `python src/multi_validate.py`
- **Learned detector, leave-one-dataset-out:** `python src/stress_model.py`
- **Held-out nurse test:** `python src/nurse_final.py`

Each loader's docstring explains its file layout and labelling rules.

## Layout

```
data/            datasets (not in git; see Getting the data)
docs/            paper summary, related work, slides
src/             pipeline code; src/datasets/ has one loader per external dataset
results/         summary CSVs (per-window outputs are regenerated, not committed)
figures/         fig1–fig7, bodysync/ (final score), sync_score/ (failed sync score)
data/wesad/      WESAD.zip (lab stress dataset, downloaded from the official site)
```
