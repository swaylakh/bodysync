# BodySync final presentation: presenter guide

## Who says what

| Who | Slides | Part |
|---|---|---|
| **Ananya** | 1–4 | **The idea.** What BodySync is, what we tried first, the new idea, how it works. |
| **Swayam** | 5–9 | **The method and results.** How the model learns, how we tested it, what we found. |
| **Bhavya** | 10–14 | **The real world.** The studies, real roads, one person's score, takeaways. |

Each person hands over with one line: Ananya → "Swayam will show how it learns and what we found." Swayam → "Bhavya will show the studies and how it does on real roads."

**Questions:** everyone answers their own part. Idea and formula → Ananya. Testing and results → Swayam. Studies and driving → Bhavya.

## The whole talk in three sentences

1. A wristband measures your heart rate, sweat and skin temperature, and BodySync compares them with **your own normal**, not with other people.
2. In lab stress tests it spots stress about **9 times out of 10**, on people it had never seen before.
3. On real roads it tells stressful city driving from calmer highway driving; everyday stress is still the hard part.

## Ananya: slides 1–4

**Slide 1, title.**
- BodySync: a personal stress score from your wristband.
- It uses three signals: heart rate, sweat and skin temperature.
- The watch shows what a user would see: a score through the day, dropping when the body shows stress.

**Slide 2, what we tried first.**
- Our first idea: is your body "in sync"? Do the signals rise and fall together the way they usually do?
- It didn't work. Telling sleep, rest and activity apart: 37%, where guessing gets 33%.
- Spotting lab stress: 46%, where a coin flip gets 50%.
- So we changed the idea.

**Slide 3, the new idea.**
- Compare each signal with **your own normal**.
- Under stress your heart beats faster, you sweat in small bursts you don't feel, and your hands cool as blood moves inward.
- Why your own normal: a heart rate of 85 can be normal for one person and stress for another.
- The score: 100 means your normal; lower means your body is reacting like it does under stress.

**Slide 4, how BodySync works.**
- Four steps: learn your normal, spot stress signs, learn your pattern, give a 0–100 score every few minutes.
- The first version was a plain formula. For each signal, how far you are from your normal in your usual spreads (a z-score), counting only the stressed direction.
- Combine the three: 100 at your normal, 50 when you're two spreads off.
- Sweat bursts: small, quick spikes in sweat; under stress they jump from about 2 to about 10 a minute.
- Hand over to Swayam.

## Swayam: slides 5–9

**Slide 5, how the model learns.**
- First, we split each study: the model learns from 60% of the people, and 40% stay hidden until the end.
- It never sees raw numbers like "heart rate 85", only how far you are from your own normal, so one model fits everyone.
- Then it tunes itself to you. It takes the 10% of your unmarked time that looks most stressed and the 10% that looks calmest, and practises on those, without your stress answers.
- Like a doctor who learns from many patients, then gets used to you.

**Slide 6, how we tested it.**
- Five public studies, about 160 people, all wearing the same Empatica wristband.
- The hidden 40% were scored once, at the very end.
- Check: with each person's stress labels scrambled, the score drops to a coin flip, so it's reading the body, not luck.

**Slide 7, lab stress.**
- On three lab studies, the first version scored 92%, 63% and 62%. The final version scores 99%+, 91% and 90%.
- What "90%" means: pick one stressed moment and one calm moment from the same person, and the stressed one scores lower 9 times out of 10.
- 32 of 34 hidden people scored lower during stress than when calm.

**Slide 8, all studies.**
- Grey is the first version, green is the final.
- Big gains in the lab, a smaller one in daily life (CAN-STRESS, 59% → 67%), and real driving goes from 54% to 76%.

**Slide 9, what each step added.** Walk left to right:
- Plain formula: 56%.
- A model that learns from examples: +13, the biggest jump.
- Sweat bursts +3.5, smoothing +1, fast and slow parts of sweat +2.6, change over the last 10 minutes +3.
- Self-training: +1.9 on average, and +8 on real driving.
- Final: 82% on the people we built it on, 85% on new people.
- Each bar averages five studies, driving included.
- Hand over to Bhavya.

## Bhavya: slides 10–14

**Slide 10, the studies behind the results.**
- Five public studies, all with the same Empatica wristband.
- WESAD is the field's standard benchmark, with over 1,300 citations.
- Three are lab studies, one is real driving, and CAN-STRESS is the largest daily-life study, 62 people.
- In each, 40% of the people were hidden for the final test.

**Slide 11, real roads.**
- After BodySync was finished, we tried it on a brand-new driving study; nothing was tuned on it.
- Nine people drove real routes: parked rest, then city traffic and highway, then parked rest.
- We compare city with highway. Both are driving, so arm movement can't explain a difference.
- 8 of 9 drivers scored lower in the city. A city moment scores lower than a highway moment 76% of the time.
- Self-training is what gets it there: 67% without it.

**Slide 12, one person.**
- One hidden person from Stress-Predict, scored by the final BodySync.
- The score drops during the colour-word test and the interview, and climbs back during rest.
- It already starts dropping just before the test begins.
- This is a clear example; on average, people are caught about 9 times in 10.

**Slide 13, takeaways.**
- Your own normal works; how the signals move together didn't.
- Lab stress: spotted about 9 times out of 10, on people it had never seen.
- Real world: a good start. It tells city from highway driving; everyday stress is still the hard part.

**Slide 14, thank you.**
- "Thank you. We're happy to take questions."

## Deeper detail, in case you're asked

You don't present these. Read your own part once so you can answer a follow-up.

### Ananya: the formula, the sync idea, the signals

**The formula, worked through.**

| Signal | Your normal | Your usual spread | Right now | z (stress direction only) |
|---|---|---|---|---|
| Heart rate | 70 bpm | 5 bpm | 80 bpm | (80 − 70) / 5 = **+2** |
| Sweat | your usual level | your usual spread | 2 spreads up | **+2** |
| Skin temperature | your usual level | your usual spread | normal | **0** |

- distance = √((2² + 2² + 0²) / 3) = √(8/3) ≈ **1.63**
- score = 100 / (1 + 1.63 / 2) = 100 / 1.82 ≈ **55**
- At your normal, distance = 0, so the score is 100. Two spreads off on everything gives 50; six spreads off gives 25. It never reaches 0.

**Follow-ups on the formula.**
- *"What exactly is the normal and the spread?"* Normal = your median. Spread = a robust version of the standard deviation (the median distance from the median, × 1.48). A few odd moments don't throw it off the way they would a standard deviation.
- *"Why only one direction?"* A heart rate below your normal, or warmer hands, isn't stress, so those count as 0.
- *"Why the ½?"* So that being two spreads off lands at exactly 50.
- *"Is sweat measured in raw units?"* On a log scale, because sweat varies hugely between people.
- *"Is there a minimum spread?"* Yes: 2 bpm for heart rate and 0.1 °C for temperature, so a very steady person doesn't get huge z-scores from tiny changes.

**The sync idea (slide 2).**
- We measured how strongly each pair of signals rises and falls together (heart rate–sweat, heart rate–temperature, sweat–temperature), in 30-minute windows.
- It was tested on our own dataset (ZU-PWD, 27 people, Empatica EmbracePlus) and on WESAD.
- 37% vs 33%: given only sync, a model told asleep, resting and active apart 37% of the time; guessing among three gets 33%.
- 46% vs 50%: on WESAD, a stressed moment scored lower than a calm one only 46% of the time.

**The signals.**
- *Sweat:* the watch passes a tiny current between two skin contacts. More sweat means more conductance.
- *Sweat bursts:* short spikes in that signal, about 2 a minute at rest and about 10 a minute under stress. You don't feel them.
- *Hands cool:* under stress blood moves away from the skin towards the core, so wrist temperature drops.

### Swayam: the model, the tests, the numbers

**What "90%" is exactly.** For each hidden person: every pair of (one stressed moment, one calm moment), and how often the stressed one scores lower. Then average over people. 50% = coin flip. The stats name is a within-person AUC.

**The model.**
- XGBoost: many small decision trees voting together.
- Its inputs are all distances from your own normal: heart rate, sweat, temperature, sweat bursts, heart-rate variability, how each changed over the last 10 minutes, the fast and slow parts of sweat, and whether heart rate and temperature are rising or falling.
- Hidden people are scored by a model trained only on the 60% group, never on them.

**Self-training, step by step.**
1. The trained model scores all of a person's unmarked time.
2. The 10% that look most stressed become "stress" examples, and the 10% that look calmest become "calm".
3. The model retrains once with those added, then scores that person.
- We tried weighting those examples ×3 and ×10: no gain.
- We tried retraining continuously using only past data, like a live app would: no gain over doing it once.

**The fairness checks.**
- *Scrambled labels:* shuffling which moments are "stress" within each person drops the average to 0.51, a coin flip.
- *Wrist movement removed from the model:* WESAD 99%+, Stress + exercise 91%, Stress-Predict 88%. So it isn't just detecting movement.
- *Rest right after a stressor left out:* 99%+, 91%, 90%. So it isn't just "before vs after".
- *Honest caveat:* in lab studies stress always happens at the same point in the session, so the clock alone also scores high (WESAD 97%, Stress-Predict 96%). BodySync never sees the clock, but lab data alone can't fully rule this out. That's why the driving test matters.

**How we iterated.** Every idea was tried on the 60% group and logged; 17 rounds in total. We kept an idea only if it raised the average and didn't drop any study by more than 3 points.
- Didn't help: pooling all datasets, separate normals for still vs moving, other model types, a normal built only from past days, and a few "are you stressed?" answers per person (84% → 83% on the hidden people).

**Remove one step from the finished system.**

| Step removed | Drop (five-study average) |
|---|---|
| Change over last 10 min | −3.0 |
| Smoothing | −2.4 |
| Self-training | −1.9 (−8 on driving) |
| Fast / slow sweat | −1.2 |
| Sweat bursts | −0.3 |

**Averages on the hidden 40% (four main studies):** first version 69%, final 87%.

### Bhavya: the studies, driving, the example person

**AffectiveROAD (driving).**
- From the MIT Media Lab: 9 drivers, 13 real drives, the same Empatica wristband.
- Each drive starts and ends with a 15-minute parked rest.
- The dataset's own stress rating confirms the city is the stressful part: 0.70–0.84 in the city, about 0.5 on the highway.
- Nothing was tuned on it. The finished system was run once.

**Driving checks, city vs highway.**
- Wrist movement alone scores 38%, and time since the start of the drive 50%. Neither explains the 76%.
- With scrambled labels: 55%.
- A model trained only on the other drivers gets 61%. The model that never saw any driving data gets 76%, thanks to self-training.
- City vs parked rest: 92%, all 9 drivers lower in the city. But wrist movement alone gets 90% there, so it's partly just moving vs sitting.
- *Honest caveat:* it doesn't follow the dataset's moment-to-moment stress rating within a drive (correlation about 0.1). It tells city from highway, not every small spike.

**CAN-STRESS (everyday life).**
- 62 people rating their own stress 0–9 during normal days; 10 of them were hidden for the test.
- First version 59%, final 67%.
- *Caveat:* the calm reports cluster in the evening, so time of day may explain part of it.

**The example person (slide 12).**
- Stress-Predict person S07, one of the hidden 40%.
- The score falls to about 42 during the colour-word test and about 15 during the interview, then recovers to 75–95 during rest.
- It's a clear example. Not every person looks this clean, which is why we quote the average (about 9 in 10).

**Why only five studies?** Two more (nurses at work, and ADARP) had only 4 hidden people each and scored near a coin flip. That's too few to judge, so we left them out. They're still in the code and the report's limitations.

**Citations.** Counted on OpenAlex in November 2026.

## If you're asked

- **"Did you just tweak it until it looked good?"** (Swayam) No. We hid 40% of the people at the start and only tested on them once, at the very end.
- **"Isn't the lab result too good?"** (Swayam) The lab stress tests are very intense, so they're easy to spot. Real life is harder, and we say so.
- **"Why is 85% higher than 82%?"** (Swayam) Luck of the draw: the hidden group happened to be a bit easier. It also shows it didn't just memorise the people it was built on.
- **"Does every step help in every study?"** (Swayam) On average yes, not everywhere. Sweat bursts help most on WESAD and driving, and cost a little on Stress-Predict and everyday life.
- **"Which step matters most?"** (Swayam) We removed one step at a time from the finished system. Change over 10 minutes matters most (about −3 without it), then smoothing (−2.4) and self-training (−1.9). No single step carries it; the gain is spread out.
- **"What is the formula?"** (Ananya) A z-score from your own normal for each signal, counting only the stressed direction, combined into one 0–100 score. Heart rate and sweat both 2 spreads up with normal temperature gives about 55.
- **"How do you know sleep from rest?"** (Ananya) The Empatica watch labels it; rest means awake and still.
- **"Isn't driving just movement?"** (Bhavya) That's why we compare city with highway: both are driving. City vs parked is even higher, 92%, but that's partly just moving vs sitting.
- **"Why is everyday life worse?"** (Bhavya) Everyday stress is milder, and people only report it a few times a day.
- **"What's next?"** (anyone) Testing it on people who wear it for weeks.
- **Don't know?** Say "We didn't test that," and don't guess.
