# BodySync final presentation: presenter guide

## Who says what

| Who | Slides | Part |
|---|---|---|
| **Ananya** | 1–5 | **The idea.** Title, what we tried first (sync), the new idea (your own normal), how BodySync works including the formula, how the model learns. |
| **Swayam** | 6–9 | **The results.** How we tested it fairly, lab stress, all studies in one chart, the step-by-step progress chart. |
| **Bhavya** | 10–14 | **The evidence and the close.** The studies we used, real roads, one person's score, takeaways, thank you. |

**Questions:** Swayam leads. Pass driving questions to Bhavya.

## The whole talk in three sentences

1. A wristband measures your heart rate, sweat and skin temperature, and BodySync compares them with **your own normal**, not with other people.
2. In lab stress tests it spots stress about **9 times out of 10**, on people it had never seen before.
3. On real roads it tells stressful city driving from calmer highway driving; everyday stress is still the hard part.

## Swayam's slides, line by line (the hardest part)

**Slide 6, how we tested it.**
- We split every study before building anything: 60% of people to build on, 40% hidden.
- The hidden 40% were scored once, at the very end.
- Leak check: with each person's stress labels scrambled, the score drops to a coin flip, so it's reading the body, not luck.

**Slide 7, lab stress.**
- On three lab studies, the first version (plain formula) scored 92%, 63% and 62%. The final version scores 99%+, 91% and 90%.
- What "90%" means: pick one stressed moment and one calm moment from the same person, and the stressed one scores lower 9 times out of 10.
- 32 of 34 hidden people scored lower during stress than when calm.

**Slide 8, all studies.**
- Grey is the first version, green is the final.
- Big gains in the lab, a smaller one in daily life (CAN-STRESS, 59% → 67%), and real driving goes from 54% to 76%.

**Slide 9, what each step added.** Walk left to right:
- Plain formula: 56%.
- A model that learns from examples: +13, the biggest jump.
- Sweat bursts: +3.5.
- Smoothing over a few minutes: +1.
- Fast and slow parts of sweat: +2.6.
- Change over the last 10 minutes: +3.
- Self-training, tuning to each person from their own unmarked data: +1.9 on average, and +8 on real driving.
- Final: 82% on the people we built it on, 85% on new people.
- Each bar averages five studies. Driving was never used to build anything, so its drivers are new in every bar.

## If you're asked

- **"Did you just tweak it until it looked good?"** No. We hid 40% of the people at the start and only tested on them once, at the very end.
- **"Isn't the lab result too good?"** The lab stress tests are very intense, so they're easy to spot. Real life is harder, and we say so.
- **"Why is everyday life worse?"** Everyday stress is milder, and people only report it a few times a day.
- **"What is the formula?"** A z-score from your own normal for heart rate, sweat and skin temperature, counting only the stressed direction, combined into one 0–100 score (slide 4). Example: heart rate and sweat both 2 spreads up, temperature normal, gives about 55.
- **"Why is the new-people score (85%) higher than 82%?"** Luck of the draw: the hidden group happened to be a bit easier. It also shows it didn't just memorise the people it was built on.
- **"Isn't driving just movement?"** That's why we compare city with highway: both are driving. City vs parked is even higher, 92%, but that's partly just moving vs sitting, so we don't lead with it.
- **"Does every step help in every study?"** On average yes, not everywhere. Sweat bursts help most on WESAD and driving, and cost a little on Stress-Predict and everyday life.
- **"What does self-training actually do?"** The model scores your unmarked time, takes the 10% that look most stressed and the 10% that look calmest, and practises on those. It never sees your real answers.
- **"What would a user see?"** A score from 0 to 100 through the day (slide 12).
- **"What's next?"** Testing it on people who wear it for weeks.
- **Don't know?** Say "We didn't test that," and don't guess.

## Don't say

- That it works perfectly in daily life.
- "100%" about anything.

## Before the talk

- [ ] Practise once together, out loud, in order: Ananya → Swayam → Bhavya
- [ ] Bring the slides as PowerPoint and PDF
- [ ] Use Presenter View so you can see the notes
