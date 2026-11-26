# BodySync final presentation: presenter guide

## Who says what

| Who | Slides | Part |
|---|---|---|
| **Ananya** | 1–4 | **The idea.** What BodySync is, what we tried first, the new idea, how it works. |
| **Swayam** | 5–9 | **The method and results.** How the model learns, how we tested it, what we found. |
| **Bhavya** | 10–15 | **The real world.** The studies, real roads, one person's score, takeaways, limitations. |

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

## Bhavya: slides 10–15

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

**Slide 14, limitations.**
- Small test groups: everyday life rests on 10 hidden people, driving on 9.
- Lab timing: stress always comes at the same point in a lab session, so a clock alone scores high there. The driving test is the answer.
- Everyday stress is milder and rated rarely; calm ratings cluster in the evening.
- Each person's normal uses their whole recording; a real app would only have the past.
- It tells stressful stretches apart, not every moment-to-moment spike. All data is from one wristband family (Empatica E4).

**Slide 15, thank you.**
- "Thank you. We're happy to take questions."

## Detailed notes per presenter

Each of us has a file with a fuller script, the details behind our slides, and the questions we might get:

- Ananya: [presenter_notes/ananya.md](presenter_notes/ananya.md)
- Swayam: [presenter_notes/swayam.md](presenter_notes/swayam.md)
- Bhavya: [presenter_notes/bhavya.md](presenter_notes/bhavya.md)

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
