# Swayam: slides 5–9 (the method and results)

You take over from Ananya. Your job is to show **how it learns, that the test was fair, and how good it got.** You also lead the questions on testing and results.

## What to say, slide by slide

### Slide 5: how the model learns
> "Three things happen before BodySync gives anyone a score."

1. **Split first.** Before anything else, we split every study: the model learns from 60% of the people, and 40% are hidden until the very end.
2. **Your own normal.** The model never sees "heart rate 85". It only sees "15 beats above your own normal". That's why one model can fit everyone.
3. **Tune itself to you.** It scores every moment of your day, picks the 10% that look most stressed and the 10% that look calmest, and practises on those as extra examples. It never sees your real stress answers.

> "Like a doctor who learns stress from many patients, then gets used to you."

### Slide 6: how we tested it
- Five public studies, about 160 people, all wearing the same Empatica wristband.
- **A fair test:** we hid 40% of the people at the start and only tested on them once, at the very end.
- **Checked it isn't a fluke:** we mixed up which moments were marked as stressed, and BodySync dropped to a coin flip. So the real results come from the body, not luck.

### Slide 7: lab stress
> "In the lab, BodySync spots stress about 9 times out of 10."

| Lab study | First version | Final |
|---|---|---|
| WESAD | 92% | 99%+ |
| Stress + exercise | 63% | 91% |
| Stress-Predict | 62% | 90% |

- What 90% means: "Take one stressed moment and one calm moment from the same person. The stressed one scores lower 9 times out of 10."
- 32 of 34 hidden people scored lower during stress than when calm.
- "On the two harder studies, it went from about 60% to about 90%."

### Slide 8: all studies
> "Here's everything in one picture. Grey is our first version, green is the final."

- Big gains in the lab.
- A smaller gain in daily life: CAN-STRESS 59% → 67%.
- Real driving: 54% → 76%.

### Slide 9: what each step added
> "This is how we got from the first version to the final one, step by step."

Walk left to right:

| Step | Adds | Running total |
|---|---|---|
| Plain formula | | 56% |
| Trained model | +13.2 | 69% |
| + sweat bursts | +3.5 | 73% |
| + smoothing over a few minutes | +1.2 | 74% |
| + fast and slow parts of sweat | +2.6 | 77% |
| + change over the last 10 minutes | +3.1 | 80% |
| + self-training | +1.9 | 82% |
| **Final on new people** | | **85%** |

- "Letting a model learn from examples was the biggest jump."
- Point at the yellow box: "Self-training adds about 2 on average, and 8 on real driving."
- "Each bar is the average of five studies, and driving was never used to build anything."

**Hand over:**
> "Bhavya will show the studies and how it does on real roads."

## The details behind your slides

### What "90%" is exactly
For each hidden person:
1. Take every pair of (one stressed moment, one calm moment).
2. Count how often the stressed moment scores lower.
3. Average over all the people.

50% = coin flip, 100% = always right. The stats name is a **within-person AUC**. It's per person, so it can't be fooled by one person simply having higher numbers than another.

### The model
- **XGBoost:** many small decision trees voting together.
- Every input is a distance from your own normal:
  - heart rate, sweat, skin temperature
  - sweat bursts (rate and size)
  - heart-rate variability
  - how each changed over the last 10 minutes
  - the fast and slow parts of the sweat signal
  - whether heart rate and temperature are rising or falling
- The score is smoothed over the last few minutes, because stress lasts minutes, not seconds.
- Hidden people are scored by a model trained only on the 60% group, never on them.

### Self-training, step by step
1. The trained model scores all of a person's unmarked time.
2. The 10% that look most stressed become "stress" examples; the 10% that look calmest become "calm".
3. The model retrains once with those added, then scores that person.

What we tried that didn't help:
- Weighting those examples ×3 or ×10: no gain.
- Retraining continuously as new data comes in, using only the past like a live app: no gain over doing it once.

### The fairness checks

| Check | Result | What it rules out |
|---|---|---|
| Labels scrambled within each person | 51% (coin flip) | A bug or a leak |
| Wrist movement removed from the model | WESAD 99%+, Stress + exercise 91%, Stress-Predict 88% | "It's just detecting movement" |
| Rest right after a stressor left out | 99%+, 91%, 90% | "It's just before vs after" |
| Clock alone (time into the session) | WESAD 97%, Stress-Predict 96%, Stress + exercise 78% | Can't fully rule out (see below) |

**Honest caveat on timing:** in lab studies, stress always happens at the same point in the session, so the clock alone also scores high. BodySync never sees the clock, but lab data alone can't prove it isn't partly tracking time. That's exactly why the driving test (Bhavya) matters: there, time since the start of the drive only scores 50%.

### How we iterated
- Every idea was tried on the 60% group only, and logged: 17 rounds in total.
- We kept an idea only if it raised the average and didn't drop any study by more than 3 points.
- We stopped when two rounds in a row gained less than 1 point.
- The finished system was scored once on the hidden 40%.

What didn't help:
- Pooling all datasets together.
- Separate normals for still vs moving.
- Other model types.
- A normal built only from past days.
- A few "are you stressed?" answers per person (84% → 83% on the hidden people).

### Remove one step from the finished system

| Step removed | Drop (five-study average) |
|---|---|
| Change over last 10 min | −3.0 |
| Smoothing | −2.4 |
| Self-training | −1.9 (−8 on driving, almost 0 elsewhere) |
| Fast / slow sweat | −1.2 |
| Sweat bursts | −0.3 (other sweat features cover it) |

No single step carries the system; the gain is spread out.

### Hidden 40%, four main studies

| Study | Hidden people | First version | Final |
|---|---|---|---|
| WESAD | 6 | 92% | 99%+ |
| Stress + exercise | 14 | 63% | 91% |
| Stress-Predict | 14 | 62% | 90% |
| CAN-STRESS | 10 | 59% | 67% |
| **Average** | | **69%** | **87%** |

## Questions you might get

**"Did you just tweak it until it looked good?"**
No. We locked 40% of the people away before building anything, and only tested on them once, at the very end. Every idea we tried is logged, including the ones that failed.

**"Isn't the lab result too good?"**
Lab stress tests are very intense (a speech in front of judges), so they're easy to spot. Everyday life is harder, and we show that: 67%.

**"Why is 85% higher than 82%?"**
Luck of the draw: the hidden group happened to be slightly easier. It also shows it didn't just memorise the people it was built on.

**"Does every step help in every study?"**
On average yes, not everywhere. Sweat bursts help most on WESAD and driving, and cost a little on Stress-Predict and everyday life. Fast/slow sweat helps in the lab but hurts a bit on driving.

**"Which step matters most?"**
Removing one step at a time: change over 10 minutes matters most (−3), then smoothing (−2.4) and self-training (−1.9). No single step carries it.

**"Why does self-training show +1.9 here but +8 on driving?"**
It helps most when the model meets a new kind of data. On driving the model had never seen anything like it, so tuning to each person helped a lot. In the lab studies it already knew what stress looks like.

**"Why does the order of steps matter?"**
Steps overlap. Self-training adds about +1 if added first and about +0.3 if added last on the four main studies, because the other features already capture some of what it learns. We show it last because that's how it runs: on top of the finished model.

**"Why XGBoost?"**
It works well on small tables of features like ours, and we tried other model types; they didn't do better.

**"How many people is that?"**
About 160 across the five studies; 40% of each study's people were hidden for the final test.

**"Could it be overfitting?"**
The hidden 40% scored slightly higher than the group we built on (85% vs 82%), which is the opposite of what overfitting looks like.

**Don't know?** Say "We didn't test that," and don't guess.
