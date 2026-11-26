# Bhavya: slides 10–15 (the real world)

You take over from Swayam and close the talk. Your job is to show **where the data came from, that it works on real roads, and what the score looks like for one person.** You answer questions on the studies and driving.

## What to say, slide by slide

### Slide 10: the studies behind the results
> "These are the five public studies behind our results, all using the same Empatica wristband."

| Study | Setting | People | Citations | How stress was marked |
|---|---|---|---|---|
| WESAD | Lab | 15 | 1,352 | Speech and mental maths in front of judges |
| AffectiveROAD | Real driving | 9 | 125 | City traffic vs highway |
| Stress-Predict | Lab | 35 | 113 | Colour-word test and an interview |
| Stress + exercise | Lab | 36 | 37 | Stress tasks, plus cycling |
| CAN-STRESS | Daily life | 62 | 1 | People rated their own stress, 0–9 |

- "WESAD is the field's standard benchmark, with over 1,300 citations."
- "CAN-STRESS is the largest daily-life study."
- "In each study, 40% of the people were hidden for the final test."

### Slide 11: real roads
> "After BodySync was finished, we tried it on a brand-new study it had never seen: real driving."

1. **The drives:** 9 people drove real routes wearing the same Empatica wristband.
2. **Two kinds of driving:** a parked rest, then busy city traffic (stressful) and the highway (calmer), then a parked rest.
3. **The question:** is the score lower in the city? "Both are driving, so arm movement can't fake it."

The answer:
- "8 of 9 drivers scored lower in the city."
- "Pick a city moment and a highway moment: the city one scores lower 76% of the time."
- "Self-training is what gets it there: 67% without it."

### Slide 12: one person
> "Here's what the score looks like for one person BodySync had never seen."

- At the calm start, the score sits at their normal, 100.
- "Look: it already starts dropping just before the test begins."
- It drops during the colour-word test, and much further during the interview.
- It climbs back towards normal during rest.
- "This is one of the clearer examples. On average, people are caught about 9 times in 10."

### Slide 13: takeaways
1. "Your own normal beats sync: comparing your signals with your own normal works; how they move together didn't."
2. "Lab stress: spotted 9 times out of 10, on people it had never seen."
3. "Real world: a good start. It tells city driving from highway driving; everyday stress is still the hard part."

### Slide 14: limitations
> "Before we finish, the limits."

- **Small test groups:** everyday life rests on 10 hidden people and driving on 9, so those numbers are uncertain.
- **Lab timing:** in the lab, stress always comes at the same point in the session, so a clock alone scores 78–97%. BodySync never sees the clock; the driving test, where time alone scores 50%, is our answer.
- **Everyday life is hard:** milder stress, rated a few times a day; calm ratings cluster in the evening.
- **The normal uses all the data:** each person's normal comes from their whole recording. A real app would only have the past. (A past-only normal was tried and didn't help.)
- **Stretches, not every spike:** it tells stressful stretches from calm ones but hardly follows the drivers' moment-to-moment rating (correlation about 0.1).
- **One wristband:** all stress data is from the Empatica E4; other devices are untested.

### Slide 15: thank you
> "Thank you. We're happy to take questions."

## The details behind your slides

### AffectiveROAD (driving)
- From the MIT Media Lab (El Haouij et al., 2018).
- 9 drivers, 13 real drives, the same Empatica wristband as every other study.
- Each drive starts and ends with a 15-minute parked rest.
- The dataset has its own stress rating, and it confirms the city is the stressful part: about 0.70–0.84 in the city, about 0.5 on the highway.
- Nothing was tuned on it. The finished system was run once.

### Driving checks, city vs highway

| Check | Score | What it means |
|---|---|---|
| BodySync, never saw driving data | **76%** (8 of 9 drivers) | The result on the slide |
| Without self-training | 67% (7 of 9) | Self-training is what gets it there |
| First version (plain formula) | 54% (5 of 9) | |
| Wrist movement alone | 38% | Movement doesn't explain it |
| Time since start of the drive alone | 50% | Timing doesn't explain it |
| Labels scrambled | 55% | Close to a coin flip, so it's not a fluke |
| Model trained only on the other drivers | 61% | Less data to learn from than all the other studies |

### City vs parked rest (not on the slide)
- BodySync: **92%**, all 9 drivers lower in the city. First version: 70%.
- But wrist movement alone already scores 90% there, because it's driving vs sitting still.
- That's why we lead with city vs highway.

### Honest caveat on driving
BodySync doesn't follow the dataset's moment-to-moment stress rating within a drive (correlation about 0.1). It tells city from highway, not every small spike.

### CAN-STRESS (everyday life)
- 62 people rating their own stress from 0 to 9 during normal days.
- 10 of them were hidden for the final test.
- First version 59%, final 67%.
- **Caveat:** the calm reports cluster in the evening, so time of day may explain part of the result. Too few people report stress and calm at matching hours to separate the two.
- Why it's harder: everyday stress is milder than a lab test, and people only report it a few times a day.

### The example person (slide 12)
- Stress-Predict person S07, one of the hidden 40%.
- Calm start: 100. Score falls to about **42** during the colour-word test and about **15** during the interview, then recovers to **75–95** during rest.
- Stress-Predict's protocol: calm baseline, colour-word (Stroop) test, rest, a job-style interview, rest, a 2-minute fast-breathing exercise, rest, a questionnaire, rest. Only the colour-word test and the interview count as stress; the breathing exercise is physical, so it isn't counted either way. On the chart it sits inside "Rest and recovery".
- Not every person looks this clean, which is why we quote the average (about 9 in 10).

### Why only five studies?
Two more, nurses at work in a hospital and ADARP (stress in recovery from alcohol use), had only 4 hidden people each and scored near a coin flip. That's too few people to judge, so we left them out. They're still in the code and in the report's limitations.

### Citations
Counted on OpenAlex in November 2026.

## Questions you might get

**"Isn't driving just movement?"**
That's why we compare city with highway: both are driving. Wrist movement alone scores 38% there, worse than a coin flip. City vs parked is even higher (92%), but that's partly just moving vs sitting, so we don't lead with it.

**"How do you know the city is actually stressful?"**
The dataset has its own stress rating, and it's much higher in the city (0.70–0.84) than on the highway (about 0.5).

**"Why is everyday life worse?"**
Everyday stress is milder, and people only report it a few times a day, so there's far less to learn from. It still improved from 59% to 67%.

**"Why is there a dip before the test on slide 12?"**
The score is from the dataset's own timeline: the drop starts in the last minutes of the calm period, just before the colour-word test. Most likely the person was getting ready for, or anticipating, the test.

**"Is that person cherry-picked?"**
Yes, it's a clear example, and we say so on the slide. The average across all hidden people is about 9 in 10.

**"Why only 9 drivers?"**
That's the size of the public dataset. It's small, but it's real roads, and nothing was tuned on it.

**"Why did you leave out the nurse and ADARP studies?"**
Only 4 hidden people each, and they scored near a coin flip. Too few to say anything either way. They're in the report's limitations.

**"What's next?"**
Testing it on people who wear it for weeks, and building the normal only from past data, like a real app would.

**"Could this be a product?"**
It runs on signals that consumer wristbands already measure. The next step would be a longer real-world test.

**Don't know?** Say "We didn't test that," and don't guess.
