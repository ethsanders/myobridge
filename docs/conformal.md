# Knowing when it doesn't know

A gesture decoder always names a gesture, even when the person is halfway
between two, the armband has slipped, or an electrode has come off. For a
prosthesis, a game or a robot arm, a confident wrong answer is worse than "not
sure". `myobridge.conformal` gives each window a *set* of gestures instead of
one, with a promise attached: the true gesture is in the set 90% of the time
(or whatever rate you pick). One gesture in the set means the decoder is sure.
Two or three mean it is torn between them, and an application can hold its
last output instead of guessing.

That promise comes from split conformal prediction, which is standard and
works on any classifier. The catch is that the promise only holds when new data
look like the calibration data, and EMG breaks that every time the armband is
put back on. The rest of this page is about keeping the promise anyway: what
goes wrong across days and people, and a recipe that holds on 91 later days.

## Using it

```python
from myobridge import ConformalGestures, GesturePipeline
from myobridge.classifier import adapt
from myobridge.conformal import temperature_for

# Once, from held-out recordings the model did not train on:
temperature = temperature_for(model, X_held_out_a, y_held_out_a)
sets = ConformalGestures.calibrate(
    model, X_held_out_b, y_held_out_b, alpha=0.1, temperature=temperature
)

# Each day: a second per gesture to recalibrate, and keep the same thresholds
today = adapt(model, X_today, y_today, mean_prior=20, covariance_weight=0.5)
pipeline = GesturePipeline.from_model(today, n_channels=4, conformal=sets)

for p in pipeline.process_block(block):
    if p.sure:
        act_on(p.gesture)        # the set holds exactly one gesture
    else:
        print("not sure:", p.gesture_set)
```

`sets.save(path)` and `ConformalGestures.load(path, model)` store the
thresholds next to a model. When the quality monitor drops a channel, the
pipeline keeps the same thresholds on the smaller model.

## What goes wrong

Everything here is measured by `myobridge bench conformal` and stored in
`results/conformal.json`. The MultiDay splits are the same as on the
[results page](results.md#4-drift-across-days-multiday): days 1-10 train the
model, days 11-20 fit the temperature, days 21-30 set the threshold, and the
promise is checked on each of days 31-121. Numbers in brackets are 95%
bootstrap intervals over days.

**The textbook version breaks across days.** Calibrate the day 1-10 model on
days 11-30 with the usual score (one minus the probability of the true
gesture), promise 90%, and on later days the true gesture is in the set only
**74.6%** [73.0, 76.2] of the time. The model has drifted, and the threshold
describes a model that no longer exists.

**Recalibrating each day is not enough on its own.** With `adapt()` and a
second per gesture on each test day, the same textbook threshold gives
**85.7%**. The adapted model is much better, but its probabilities are
nearly always 0 or 1: an LDA with 60 features is very sure of itself. The
threshold then sits on one of those extremes and the sets can't get any
bigger when they need to.

**Calibrating on the day itself doesn't work either.** Setting each day's
threshold from that day's own recalibration windows gives 79.1%, because the
start of a hold is easier than the rest of it.

## A recipe that holds

1. Recalibrate the model each day with `adapt()` and about a second per
   gesture ([Vidovic et al. 2016](how-it-works.md)).
2. Soften its probabilities with one temperature, fitted on adapted held-out
   days (`temperature_for`). This is temperature scaling (Guo et al. 2017);
   here it divides the scores by about 10.
3. Use the randomized adaptive prediction sets score (APS, Romano, Sesia &
   Candès 2020) and take its threshold from other adapted held-out days
   (`ConformalGestures.calibrate`).
4. Keep that threshold. Nothing is fitted on the test days apart from the
   `adapt()` step itself.

| Promised | Delivered on days 31-121 | Average set size | One-gesture sets | Right when one gesture |
|---|---|---|---|---|
| 95% | 93.9% [93.6, 94.1] | 1.95 | 43.4% | 98.2% |
| **90%** | **89.1% [88.7, 89.4]** | **1.43** | **61.1%** | **97.0%** |
| 80% | 79.9% [79.4, 80.3] | 1.07 | 72.0% | 94.6% |

At 90%, every one of the 91 test days lands within 5 points of the promise,
and the first, middle and last thirds of the test days deliver 88.9%,
88.5% and 89.8%, so the threshold isn't going stale over three months. The
least-covered gesture still gets 81.6%; the `mondrian=True` option sets a
threshold per gesture and lifts that to 86.1%. The adapted decoder's plain
accuracy on these days is 89.0%. The recipe keeps that accuracy and adds a
way to tell which decisions to trust: when the set has one gesture, which is
61% of the time, it is right 97% of the time.

![Coverage on each MultiDay test day, and promised vs delivered coverage](figures/conformal.png)

Each step matters. Without step 1 (temperature and APS on the old model) the
90% promise delivers 84.1%. Without steps 2 and 3 it delivers 85.7%, as
above, and with the temperature but the textbook score, 86.7%. Swapping which held-out days fit the temperature and
which set the threshold gives 89.8%, so the split isn't a lucky one. The APS
score has a random share; over five seeds the 90% row ranges from
88.9% to 89.8%.

## The same recipe on other data

**A new sitting (3DC).** 16 people (7-22; 1-6 chose settings), threshold
from each person's blocked out-of-fold scores before a 5 minute break,
checked after it. At 90%: 89.6% [87.4, 91.1], 1.27 gestures per set on
average, 69.6% one-gesture sets that are right 97.3% of the time. At 95% and
80% it delivers 94.1% and 80.6%. Without the temperature, the textbook score
gives 86.2% and APS 88.7%.

**A new person (3DC).** Leave one person out; their threshold comes from the
other people.

- With no calibration from them, the decoder is right 39.5% of the time,
  and the sets show it: the promise holds (89.8%) by including 5.8 of the 11
  gestures on average. It knows it doesn't know.
- With 2 seconds per gesture and `adapt()`: 89.3% [85.4, 92.4], 2.2 gestures
  per set, 42.3% one-gesture sets that are right 91.0% of the time.
- The textbook score on the adapted model gives up at 90% and 95%: its
  threshold rounds to 1, so every set holds all 11 gestures.

**An electrode fails (3DC).** Each of the four evenly spaced channels fails
in turn on each of the 16 people, for the whole session after the break,
with the simulated faults from the [signal quality page](signal-quality.md).
At 90%:

| Fault | No quality monitor | With the monitor |
|---|---|---|
| Electrode comes off | 22.8% [19.7, 25.9] | 89.5% [88.5, 90.4] |
| Clipping | 17.2% | 87.2% |
| Mains hum | 89.6% | 89.5% |
| Motion | 89.6% | 89.6% |

The first row is the important one. Without the monitor, 79.2% of the sets
still hold a single gesture, and those are right 24.5% of the time. Conformal
sets do not notice a broken electrode: the scores look as sure as ever. The
monitor has to catch the fault and drop the channel; after that the sets keep
their promise with the thresholds they already had.

## Caveats

- **Across days this is measured, not guaranteed.** The guarantee in
  conformal prediction assumes new windows are exchangeable with the
  calibration ones. Days are not, which is the point of this page. The recipe
  holds on one person over 91 days and on 16 people over a break. A new
  person, armband or task should be checked again.
- **One person for drift.** MultiDay is one person, so the across-day result
  is one person's.
- **The end of a hold is harder.** In MultiDay, the last second of each hold
  is 26.8% of the test windows, and the adapted decoder is 96.7% accurate
  without it, against 89.0% with it. With those windows removed from the
  test days only, the 90% sets over-deliver (94.1%); removed from every
  step, the recipe delivers 89.7%. Either way the promise isn't broken, but
  where the hard windows fall matters, and live use will have its own
  pattern.
- **Coverage is per window, not per decision.** The pipeline's majority vote
  smooths the gesture it reports, but `gesture_set` belongs to the newest
  window.
- **Cost.** Two things to fit, one temperature and one threshold, both
  instant. Prediction adds a sort of 11 numbers per window.

## Related work

Rejecting uncertain EMG decisions is not new: Scheme, Hudgins & Englehart
(IEEE TBME 2013) reject on low confidence, and later work learns rejection or
uncertainty estimates. Those approaches choose a confidence cut-off by hand
and do not say how often the right gesture will be kept. Conformal prediction
(Vovk, Gammerman & Shafer 2005), APS (Romano, Sesia & Candès 2020) and
adaptive conformal inference (Gibbs & Candès 2021) are standard; the latter
is in the module as `AdaptiveAlpha`, and in the benchmark it underdelivers
(85.3% at 90%) because the only feedback available on a test day comes from
the easier start of each hold. As far as I found, conformal gesture sets for
EMG, and whether their promise survives taking the armband off and putting it
back on, haven't been studied before.
