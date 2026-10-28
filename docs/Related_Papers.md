# BodySync: Related Papers

Citation counts are from Semantic Scholar, checked 2026-09-25.

## Core references (dataset and direct competitors)

| Paper | Year | Venue | Link | Why it matters |
|---|---|---|---|---|
| Evangelista, Nazir, Bukhari et al. ZU-PWD '25 dataset | 2025 | IEEE DataPort | [10.21227/2tth-wj12](https://doi.org/10.21227/2tth-wj12) | Our dataset (free copy: [GitHub](https://github.com/SalmanResearchAI/Phyisiological-Wellness-Dataset)) |
| Sharma et al. Physiological Wellness Index (PWI) | 2025 | IADIS IJCSIS 20(2):19–48 | [PDF](https://www.iadisportal.org/ijcsis/papers/2025200202.pdf) | Existing score on this dataset; weighted average of levels |
| r3kind1e: Multimodal mood/stress forecasting on ZU-PWD | 2025 | GitHub | [repo](https://github.com/r3kind1e/Multimodal-Mood-Stress-Forecasting-with-Explainable-AI) | ML stress prediction on this data; chance-level (QWK ≤ 0) |
| Sun et al. Trust toward human vs AI-generated health info | 2026 | IJHCS 209:103714 | [arXiv:2512.12348](https://arxiv.org/abs/2512.12348) | Original motivating paper |

## Closest prior work on our method (from the novelty search)

| Paper | Year | Venue | Citations | Link | Overlap with BodySync |
|---|---|---|---|---|---|
| Gjoreski et al. Monitoring stress with a wrist device using context | 2017 | J Biomed Inform | 340 | [10.1016/j.jbi.2017.08.006](https://doi.org/10.1016/j.jbi.2017.08.006) | Empatica wrist stress in real life; activity context improved detection. Uses per-signal levels, not coupling |
| Thomas et al. ECG-based cardiopulmonary coupling for sleep | 2005 | Sleep 28(9) | 318 | [10.1093/sleep/28.9.1151](https://doi.org/10.1093/sleep/28.9.1151) | Coupling (heart × breathing) used to classify sleep states; ECG, not wrist |
| Bartsch et al. Phase transitions in physiologic coupling | 2012 | PNAS | 239 | [10.1073/pnas.1204568109](https://doi.org/10.1073/pnas.1204568109) | Coupling strength changes with sleep stage; lab recordings |
| Fonseca et al. Sleep stage classification with ECG and respiratory effort | 2015 | Physiol Meas 36:2027 | 176 | [10.1088/0967-3334/36/10/2027](https://doi.org/10.1088/0967-3334/36/10/2027) | Heart–breathing coupling features for sleep classification; ECG + breathing belt |
| Valenza, Lanata & Scilingo. Emotion recognition with cardiorespiratory coupling | 2013 | Physiol Meas 34(4):449 | 52 | [10.1088/0967-3334/34/4/449](https://doi.org/10.1088/0967-3334/34/4/449) | Coupling features beat level features for emotion; lab ECG + breathing belt |
| Kelava et al. Synchrony of multivariate psychophysiological variables | 2015 | Front Psychol 5:1507 | 7 | [10.3389/fpsyg.2014.01507](https://doi.org/10.3389/fpsyg.2014.01507) | Within-person ECG / EDA / respiration synchrony; descriptive only, lab |
| Xu, Cao, Barnett-Cowan et al. Consumer-grade wearables for pilot workload and stress (LOSO) | 2026 | Sensors | 2 (new) | [10.3390/s26123627](https://doi.org/10.3390/s26123627) | **Closest method.** EmbracePlus + chest strap; HR–EDA and EDA–temp correlation features; LightGBM / XGBoost / RF with LOSO. Combined features didn't beat EDA alone. Flight sessions, not free-living |

### Recent coupling work on wearables (new, so few citations yet)

| Paper | Year | Venue | Citations | Link | Overlap with BodySync |
|---|---|---|---|---|---|
| Farahani, Cao & Rahmani. "When Clean Signals Are Not Enough" | 2026 | ICCM (arXiv) | 0 (new) | [arXiv:2608.18397](https://arxiv.org/abs/2608.18397) | **Closest concept.** Same 3 pairs (EDA–HR, EDA–temp, HR–temp) on Empatica E4, deviation from personal baseline; used to predict where stress classifiers fail. Lab data only (WESAD) |
| Chen et al. Wrist activity × temperature coupling predicts disease | 2026 | Advanced Science | 0 (new) | [10.1002/advs.76217](https://doi.org/10.1002/advs.76217) | Free-living coupling, ~90k people (UK Biobank), but 2 signals, 24 h scale only |
| Long-term monitoring shows phase-lagged locomotor–autonomic coupling | 2025 | Sci Rep 15:32967 | 0 | [10.1038/s41598-025-16996-1](https://doi.org/10.1038/s41598-025-16996-1) | 30 days free-living; activity × heart rate only |
| Rosato et al. Heart rate and step rate lock during daily activity | 2024 | Front Physiol | 6 | [10.3389/fphys.2024.1394591](https://doi.org/10.3389/fphys.2024.1394591) | Fitbit, 5 months free-living; 2 signals |
| Ryan et al. Network physiology with Oura ring in COVID | 2024 | Front Netw Physiol | 2 | [10.3389/fnetp.2024.1211413](https://doi.org/10.3389/fnetp.2024.1211413) | Network framing on wearable data, but joint shifts in levels, not correlations |

## (a) Empatica / wrist datasets and stress detection

| Paper | Year | Venue | Citations | Link | Why it matters |
|---|---|---|---|---|---|
| Schmidt et al. WESAD | 2018 | ICMI | 1380 | [10.1145/3242969.3242985](https://doi.org/10.1145/3242969.3242985) | Benchmark Empatica E4 stress/affect dataset |
| Poh et al. Wearable sensor for long-term EDA | 2010 | IEEE TBME | 731 | [10.1109/TBME.2009.2038487](https://doi.org/10.1109/TBME.2009.2038487) | Founding wrist EDA paper (origin of Empatica) |
| Sano & Picard. Stress recognition with wearables and phones | 2013 | ACII | 541 | [10.1109/ACII.2013.117](https://doi.org/10.1109/ACII.2013.117) | Free-living stress from wrist EDA + phone |
| Reiss et al. Deep PPG (PPG-DaLiA) | 2019 | Sensors | 433 | [10.3390/s19143079](https://doi.org/10.3390/s19143079) | E4 heart rate during daily activities; motion artifacts |
| Gjoreski et al. Stress monitoring with a wrist device using context | 2017 | J Biomed Inform | 340 | [10.1016/j.jbi.2017.08.006](https://doi.org/10.1016/j.jbi.2017.08.006) | Real-life wrist stress with activity context (like our activity-aware baselines) |
| Smets et al. Daily-life stress with wearables | 2018 | npj Digit Med | 221 | [10.1038/s41746-018-0074-9](https://doi.org/10.1038/s41746-018-0074-9) | Large free-living wearable stress study |
| Regalia et al. Multimodal wrist seizure devices | 2019 | Epilepsy Res | 202 | [10.1016/j.eplepsyres.2019.02.007](https://doi.org/10.1016/j.eplepsyres.2019.02.007) | Clinical use of Empatica EDA + accelerometer |
| Onorati et al. Multicenter convulsive seizure detection | 2017 | Epilepsia | 194 | [10.1111/epi.13899](https://doi.org/10.1111/epi.13899) | Empatica Embrace validation |
| Park et al. K-EmoCon | 2020 | Sci Data | 193 | [10.1038/s41597-020-00630-y](https://doi.org/10.1038/s41597-020-00630-y) | E4 emotion dataset in natural conversations |
| Ollander et al. E4 vs stationary sensors for stress | 2016 | IEEE SMC | 112 | [10.1109/SMC.2016.7844917](https://doi.org/10.1109/SMC.2016.7844917) | E4 validity for stress detection |
| El Haouij et al. AffectiveROAD | 2018 | ACM SAC | 81 | [10.1145/3167132.3167395](https://doi.org/10.1145/3167132.3167395) | E4 on-road stress dataset |

## (b) Free-living / longitudinal wearable health

| Paper | Year | Venue | Citations | Link | Why it matters |
|---|---|---|---|---|---|
| Quer et al. DETECT: COVID from wearables | 2020 | Nat Med | 394 | [10.1038/s41591-020-1123-x](https://doi.org/10.1038/s41591-020-1123-x) | Personal-baseline deviations in free-living data |
| Mishra et al. Pre-symptomatic COVID from smartwatches | 2020 | Nat Biomed Eng | 379 | [10.1038/s41551-020-00640-6](https://doi.org/10.1038/s41551-020-00640-6) | Personal-baseline anomaly detection |
| Sano et al. Objective markers of students' wellbeing | 2018 | JMIR | 328 | [10.2196/jmir.9410](https://doi.org/10.2196/jmir.9410) | Longitudinal wrist EDA / temp / accelerometer |
| Radin et al. Fitbit flu surveillance | 2020 | Lancet Digit Health | 277 | [10.1016/S2589-7500(19)30222-5](https://doi.org/10.1016/S2589-7500(19)30222-5) | Population-scale wearable HR and sleep |
| Taylor et al. Personalized multitask mood/stress prediction | 2017 | IEEE TAFFC | 214 | [10.1109/TAFFC.2017.2784832](https://doi.org/10.1109/TAFFC.2017.2784832) | Personalized models on longitudinal wrist data |
| Alavi et al. Real-time COVID alerting from wearables | 2022 | Nat Med | n/a* | [10.1038/s41591-021-01593-2](https://doi.org/10.1038/s41591-021-01593-2) | Personal-baseline alerts at scale |
| Hosseini et al. Nurse stress dataset (E4) | 2022 | Sci Data | n/a* | [10.5061/dryad.5hqbzkh6f](https://doi.org/10.5061/dryad.5hqbzkh6f) | Real-world E4 stress data in hospital shifts |

*Semantic Scholar's count looks broken for these two. They are well cited in reality, so don't quote a number.

## (c) Physiological coupling / network physiology

| Paper | Year | Venue | Citations | Link | Why it matters |
|---|---|---|---|---|---|
| Lipsitz. Loss of "complexity" and aging | 1992 | JAMA | 1277 | [10.1001/jama.1992.03480130122036](https://doi.org/10.1001/jama.1992.03480130122036) | Theory: loss of coordination signals poor health |
| Bashan et al. Network physiology: topology and function | 2012 | Nat Commun | 675 | [10.1038/ncomms1705](https://doi.org/10.1038/ncomms1705) | Core citation: organ coupling changes with state |
| Bartsch et al. Network physiology: dynamic organ interactions | 2015 | PLoS ONE | 376 | [10.1371/journal.pone.0142143](https://doi.org/10.1371/journal.pone.0142143) | Coupling networks reorganize across states |
| Thomas et al. ECG cardiopulmonary coupling in sleep | 2005 | Sleep | 318 | [10.1093/sleep/28.9.1151](https://doi.org/10.1093/sleep/28.9.1151) | Coupling features used to classify sleep (closest to our ML task) |
| Bartsch et al. Phase transitions in physiologic coupling | 2012 | PNAS | 239 | [10.1073/pnas.1204568109](https://doi.org/10.1073/pnas.1204568109) | Coupling strength shifts with sleep stage |
| Goldberger, Peng & Lipsitz. What is physiologic complexity? | 2002 | Neurobiol Aging | 848 | [10.1016/s0197-4580(01)00266-4](https://doi.org/10.1016/s0197-4580(01)00266-4) | Complexity loss in aging and disease |

## (d) Reviews

| Paper | Year | Venue | Citations | Link | Why it matters |
|---|---|---|---|---|---|
| Can et al. Stress detection in daily life: a survey | 2019 | J Biomed Inform | 427 | [10.1016/j.jbi.2019.103139](https://doi.org/10.1016/j.jbi.2019.103139) | Review of real-world wearable stress detection |
| Schmidt et al. Wearable-based affect recognition review | 2019 | Sensors | 210 | [10.3390/s19194079](https://doi.org/10.3390/s19194079) | Standard review of wrist affect recognition |
| Chen et al. Pain and stress detection with wearables | 2021 | Sensors | 188 | [10.3390/s21041030](https://doi.org/10.3390/s21041030) | Recent EDA / wearable stress review |

## Top 5 for a related-work slide

1. WESAD (Schmidt 2018): the Empatica benchmark
2. Gjoreski 2017: wrist stress with activity context
3. Bashan 2012: network physiology, the science behind "moving together"
4. Thomas 2005: coupling features for sleep classification
5. Mishra 2020: personal baselines from smartwatches
