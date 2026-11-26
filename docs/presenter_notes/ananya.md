# Ananya: slides 1–4 (the idea)

You open the talk. Your job is to make everyone understand the idea before any numbers: **BodySync compares your body with your own normal.**

## What to say, slide by slide

### Slide 1: title
> "Hi, we're Ananya, Swayam and Bhavya. This is BodySync: a personal stress score from your wristband."

- The wristband measures three things: heart rate, sweat and skin temperature.
- Point at the watch: "This is what a user would see. A score through the day. Right now it's 64 and dropping, because the body is showing stress signs: sweat bursts are up."

### Slide 2: what we tried first
> "Our first idea was 'is your body in sync?'"

- The guess: stress throws heart rate, sweat and temperature out of step with each other.
- What we found: they stayed in step whether people were stressed or calm.
- The numbers:
  - Telling sleep, rest and activity apart: 37%, where pure guessing among three gets 33%.
  - Spotting lab stress: 46%, where a coin flip gets 50%.
- "So sync didn't work, and we changed the idea."

### Slide 3: the new idea
> "Instead of asking whether your signals move together, we ask how far each one is from your own normal."

- Heart rate beats faster under stress.
- You sweat in small bursts you don't even feel.
- Your hands cool as blood moves inward.
- Why your own normal: "A heart rate of 85 can be completely normal for one person and stress for another."
- The score: 100 means you're at your normal. Lower means your body is reacting like it does under stress. A new score every few minutes.

### Slide 4: how BodySync works
> "BodySync works in four steps."

1. It learns your normal heart rate, sweat and temperature.
2. It spots stress signs: sweat bursts, a rising heart rate, quick changes.
3. It learns what stress looks like, then tunes itself to you.
4. It gives you a score from 0 to 100, updated every few minutes.

Then the box at the bottom:
> "Our very first version was just a formula. For each signal, how far are you from your normal, measured in your usual spreads. That's basically a z-score. We combine the three into one score: 100 at your normal, 50 when you're two spreads off."

- Sweat bursts (footnote): small, quick spikes in sweat; under stress they jump from about 2 to about 10 a minute.

**Hand over:**
> "Swayam will show how the model learns and what we found."

## The formula in detail

On the slide:

- z = (now − your normal) / your spread, for each signal
- score = 100 / (1 + ½ · √(average of z²))

### Worked example

| Signal | Your normal | Your spread | Right now | z |
|---|---|---|---|---|
| Heart rate | 70 bpm | 5 bpm | 80 bpm | (80 − 70) / 5 = **+2** |
| Sweat | your usual level | your usual spread | 2 spreads above | **+2** |
| Skin temperature | your usual level | your usual spread | at normal | **0** |

1. Square each z: 4, 4, 0.
2. Average them: 8 / 3 ≈ 2.67.
3. Square root: √2.67 ≈ **1.63**. This is the "distance".
4. Score = 100 / (1 + 1.63 / 2) = 100 / 1.82 ≈ **55**.

### Quick reference

| Distance from your normal | Score |
|---|---|
| 0 (exactly normal) | 100 |
| 1 spread | 67 |
| 2 spreads | 50 |
| 4 spreads | 33 |
| 6 spreads | 25 |

It never reaches 0; it keeps sinking the further you drift.

### Second example: one direction only

Heart rate 2 spreads **below** normal, sweat normal, temperature 1 spread **up** (warmer hands):
- Heart rate down isn't stress → counts as 0.
- Warmer hands aren't stress → counts as 0.
- Distance 0 → score **100**. Not stressed.

## Questions you might get

**"What exactly is the normal and the spread?"**
Normal is the median of your calm readings. The spread is a robust version of the standard deviation: the median distance from the median, times 1.48 so it's on the same scale as a standard deviation. A few odd moments (running for a bus, the sensor slipping) would inflate a normal standard deviation; this barely moves.

**"So it's a z-score?"**
Yes. A robust z-score from your own normal, only counting the stressed direction, combined across the three signals and turned into 0–100.

**"Why only one direction?"**
A heart rate below your normal, or warmer hands, isn't stress. Counting them would make relaxing look like stress.

**"Why the ½ in the formula?"**
So that being two spreads off lands at exactly 50. It sets the scale.

**"Why average the squares instead of a plain average?"**
It's the root-mean-square: bigger jumps count a bit more. One signal far off matters more than three signals slightly off.

**"Is sweat in raw units?"**
On a log scale. Sweat levels vary hugely between people, and the log keeps it comparable.

**"What if someone is super steady, with a tiny spread?"**
There's a minimum spread: 2 bpm for heart rate and 0.1 °C for temperature. Otherwise tiny wobbles would look like huge z-scores.

**"Is the formula what you use now?"**
No. The final BodySync replaces it with a model that learns (Swayam's part). The formula is the starting point we compare against: it scored 56% on average, and the final system 82%.

**"How did the sync idea work exactly?"**
For every 30 minutes, we measured how strongly each pair of signals rises and falls together: heart rate with sweat, heart rate with temperature, and sweat with temperature. We compared that with the person's usual sync. Low sync was supposed to mean stress.

**"Where do sleep, rest and active come from?"**
The Empatica watch labels them itself. Rest means awake and still; active means awake and moving.

**"What data was the sync test on?"**
An everyday-life dataset of 27 people wearing an Empatica wristband for days, and the WESAD lab study for stress.

**"How does the watch measure sweat?"**
It passes a tiny current between two contacts on your skin. More sweat means the skin conducts better. That's called EDA (electrodermal activity).

**"What's a sweat burst?"**
A short spike in that sweat signal. About 2 a minute at rest, about 10 a minute under stress. You don't feel them.

**"Why do hands cool under stress?"**
Under stress, blood moves away from the skin towards the core, so the wrist gets slightly cooler.

**Don't know?** Say "We didn't test that," and don't guess.
