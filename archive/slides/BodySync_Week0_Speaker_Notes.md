# BodySync - Speaker Notes

## Slide 1 (Title)
Hi everyone, so our project is called BodySync. The idea is to detect when your body's different systems fall out of alignment, using data from the Empatica Embrace Plus wearable. We're working with a real-world dataset called ZU-PWD '25.

## Slide 2 (The Problem)
So the background here comes from this paper by Sun et al., published in 2026. They found that when people read health information, their body actually reacts differently depending on whether they think it came from AI or a human. Signals like skin sweat, heart rate variability, and temperature all changed. They got 73% accuracy predicting trust levels from these signals. But the big limitation is they looked at each signal on its own, in a lab, at one point in time.

## Slide 3 (What Is BodySync?)
So our approach is different. Instead of looking at heart rate alone or EDA alone, we ask -- are all these signals moving together the way they should? We call that the sync score. When your body is doing well, your signals are in agreement. When something is off -- stress, illness, bad sleep -- they start to disagree. That's a desync.

## Slide 4 (The Dataset)
We're using the ZU-PWD '25 dataset from Zayed University. 28 people wore the Empatica Embrace Plus for up to 15 months in their normal daily lives. That gives us about 900,000 rows of minute-by-minute data across 24 variables -- heart rate, HRV, EDA, temperature, breathing, activity, sleep, all of it. And importantly, this is real-world data, not lab data, which directly addresses the limitation the paper flagged.

## Slide 5 (Paper Connection)
This slide shows exactly how our work builds on the paper. They measured signals separately -- we look at relationships between them. They had one lab session -- we have 15 months. They used individual features for prediction -- we propose a combined coherence metric. And they specifically called for ecological, real-world validation -- that's exactly what this dataset gives us.

## Slide 6 (What We'll Do)
So the plan is four phases. First, explore and clean the data -- understand what we're working with. Then build the sync score itself -- rolling correlations between signal pairs, validated against known state changes like sleep versus activity. Third, we train ML models to predict when a desync is about to happen. And finally, we create visualizations -- body weather maps -- that show how each person's state evolves over time.

## Slide 7 (Tools & Methods)
On the technical side, we're using Python with standard ML libraries. The sync score is based on rolling Pearson correlations across sliding windows. We build personal baselines per person and per activity context. For classification, Random Forest and XGBoost with leave-one-subject-out cross-validation, same approach the paper used. And UMAP for dimensionality reduction to create those body weather maps.

## Slide 8 (Where This Can Go)
Looking ahead, this project has natural extensions. We can validate the sync score on labeled datasets like WESAD. We could build a real-time dashboard. We could even replicate the paper's trust experiment but using our sync score as the outcome instead of individual signals. The big picture is -- the paper showed your body reacts to how health info is presented. We're building the tool that catches those reactions from a wristband, in everyday life.
