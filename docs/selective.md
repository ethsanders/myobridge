# Acting only when sure

Gesture sets ([previous page](conformal.md)) promise that the right gesture
is in the set. Many applications want something simpler: one gesture or
nothing, and a promise about how often that one gesture is right. "When it
names a gesture, it is right 97% of the time; otherwise it says nothing."
`myobridge.selective` does this. It picks a confidence cut-off on held-out
recordings, answers above it and stays quiet below it.

Picking that cut-off by eye is easy and breaks quietly: a cut-off where the
held-out windows happen to be 97% right is just as likely to be a little too
loose as a little too strict. `SelectiveGestures` picks it with Learn then
Test (Angelopoulos, Bates, Candès, Jordan & Lei 2021). On new windows like
the held-out ones, accuracy when answering is then at least the target, with
probability 1 - delta over the held-out windows. This page checks whether
that promise survives new days and new people. Across days it falls a
little short; on new people, far short.

## Using it

```python
from myobridge import SelectiveGestures
from myobridge.classifier import adapt
from myobridge.conformal import softmax, temperature_for
from myobridge.selective import grid_from

# Once, from held-out recordings the model did not train on:
temperature = temperature_for(model, X_held_out_a, y_held_out_a)
grid = grid_from(softmax(model.decision_function(X_held_out_a), temperature))
sure = SelectiveGestures.calibrate(
    model, X_held_out_b, y_held_out_b,
    target=0.97, delta=0.1, temperature=temperature, grid=grid,
)
print(sure.cutoff, sure.answer_rate)   # cutoff None: it will never answer

# Each day: a second per gesture to recalibrate, and keep the same cut-off
today = sure.with_model(
    adapt(model, X_today, y_today, mean_prior=20, covariance_weight=0.5)
)
for gesture in today.predict(X_live):
    if gesture is not None:
        act_on(gesture)
```

`answer_rate` is the share of held-out windows the cut-off answers, so it
says how often to expect an answer. `accuracy_bound` is a 1 - delta lower
bound (Clopper-Pearson) on accuracy at that cut-off. `sure.save(path)` and
`SelectiveGestures.load(path, model)` store it next to a model.

## How the cut-off is picked

Learn then Test treats each candidate cut-off as a test. For a cut-off that
answers n held-out windows and gets k of them wrong, the exact binomial
p-value is the chance of k or fewer errors if the true error rate were 3%.
A small p-value means the cut-off is safe. The candidates are tested so that
the chance of wrongly calling *any* unsafe cut-off safe is at most delta,
and the loosest safe one is kept. If none is safe, `cutoff` is None and
`predict()` never answers.

Three ways to run the tests:

- `"fallback"` (the default; Wiens 2003). Tests from strict to loose; each
  test passes its share of delta on when it succeeds, and a failure doesn't
  stop the rest.
- `"fixed_sequence"`. Strict to loose, stopping at the first failure. It
  uses all of delta at every step, but it stops at once if the strictest
  cut-offs answer too few windows to prove anything. With no errors, a 97%
  promise at delta 0.1 needs 76 windows.
- `"bonferroni"`. Each cut-off gets delta divided by the number of
  cut-offs. The most cautious.

The candidates must be fixed without looking at the windows that pick the
cut-off. `default_grid()` is fixed in advance: 241 cut-offs from 1 - 10⁻¹²
to 0. `grid_from()` builds cut-offs that answer 1%, 2%, ..., 100% of some
*other* held-out windows, such as the ones that fitted the temperature.

Three scores can be compared with the cut-off: the top probability
(`"max_prob"`), the gap between the top two (`"margin"`), and whether the
APS gesture set from the previous page holds one gesture (`"aps_singleton"`;
the cut-off is then the APS threshold).

## Across days (MultiDay)

Everything here is measured by `myobridge bench selective` and stored in
`results/selective.json`. The splits are the ones on the
[previous page](conformal.md): days 1-10 train the model, days 11-20 fit the
temperature (and the grid), days 21-30 pick the cut-off, and the promise is
checked on each of days 31-121. Each test day is recalibrated with `adapt()`
and one second per gesture. Numbers in brackets are 95% bootstrap intervals
over days. "Within 2 points" counts days no more than 2 points below the
target. Delta is 0.1 throughout. Procedures and scores were chosen on
days 11-30 only.

The naive rule is the loosest cut-off at which days 21-30 are as accurate as
the target. "Learn then Test" is the default fallback procedure on a grid
from days 11-20. On these days it picks the same cut-offs as fixed-sequence
on that grid, which the benchmark chose on validation days.

| Target | Rule | Right when answering | Answers | Days at target | Within 2 points | Worst day |
|---|---|---|---|---|---|---|
| 97% | Naive | 95.6% [95.1, 96.1] | 84.3% | 20.9% | 75.8% | 82.6% |
| 97% | **Learn then Test** | **96.0% [95.5, 96.4]** | **82.3%** | **35.2%** | **80.2%** | **84.1%** |
| 97% | Learn then Test, Bonferroni | 96.8% [96.3, 97.1] | 77.0% | 58.2% | 87.9% | 87.0% |
| 95% | Naive | 93.0% [92.4, 93.6] | 92.7% | 23.1% | 60.4% | 78.5% |
| 95% | **Learn then Test** | **93.8% [93.1, 94.3]** | **90.9%** | **46.2%** | **70.3%** | **79.7%** |
| 95% | Learn then Test, Bonferroni | 95.0% [94.4, 95.5] | 86.9% | 65.9% | 84.6% | 81.6% |

**The promise is not kept across days.** Learn then Test is 0.4 to 0.8
points more accurate than the naive rule and meets the target on more days,
but at 97% it delivers 96.0%, and the interval does not reach 97%. On days
21-30 the same cut-off answered 86.2% of windows and was 97.4% right. Later
days are harder, and no held-out rule can see that coming. Bonferroni, the
most cautious, comes closest: 96.8% at 97%, at the cost of answering 5.3
points less often. It still meets 97% on only 58.2% of days. At 95% it
delivers 95.0% on average (94.99% before rounding, so just short), and
meets the target on 65.9% of days.

**Recalibrating each day matters most.** Without `adapt()`, the model is
right 73.5% of the time on the test days rather than 89.0%. With the same
recipe the 97% rule then delivers 93.2% [92.0, 94.3] and answers 50.1% of
windows (naive: 92.4%, answering 52.7%). It meets 97% on 26.4% of days.

**The temperature lets it answer more.** The adapted model's probabilities
are nearly always 0 or 1. Without the temperature, Learn then Test at 97%
(fixed-sequence on the fixed grid) delivers 96.0% [95.6, 96.3] and answers
77.2% of windows; with it, 82.3% at the same accuracy. The temperature here
divides the scores by about 10 (9.94). One trap: with the temperature,
fixed-sequence on the fixed grid never answers on any of the 91 days. Its
strictest cut-offs answer almost nothing, the first test fails, and it
stops. That is why fallback is the default.

**Which score.** The top probability answers most often at both targets:

| Score | Right when answering (97%) | Answers (97%) | Answers (95%) |
|---|---|---|---|
| **Top probability** | **96.0%** | **82.3%** | **90.9%** |
| Top-two margin | 96.0% | 79.9% | 89.9% |
| One-gesture APS set | 96.2% | 66.8% | 72.3% |

## A new sitting and a new person (3DC)

**A new sitting.** 16 people (7-22; 1-6 chose procedures), cut-off from each
person's blocked out-of-fold scores before a 5 minute break, checked after
it. The decoder is right 92.2% of the time. With the temperature (7.30) and
Learn then Test at 97%: 96.3% [94.5, 97.7] when answering, answering 83.0%
of windows; 43.8% of people get 97% or more and the worst gets 86.6%. The
naive rule gives 96.0% [94.1, 97.5] and answers 87.5%. Bonferroni gives
96.8% [95.2, 98.1], answers 76.1%, and 68.8% of people get 97%. At 95%,
Learn then Test delivers 94.9% [92.8, 96.5] and answers 91.7%. Here the
top-two margin answers a little more often than the top probability (84.0%
at 97%, 96.2% right).

**A new person.** Leave one person out; their cut-off comes from the other
people.

- With 2 seconds per gesture and `adapt()`, the decoder is right 77.3% of
  the time. At 95%, Learn then Test (fixed-sequence on the fixed grid,
  chosen on people 1-6) delivers 93.9% [88.5, 98.1] but answers only 20.4%
  of windows. 75.0% of people get 95% or more; the worst gets 66.0%.
- **At 97% it fails.** It answers 4.2% of windows and those are right 76.8%
  [58.6, 93.2] of the time. The worst person gets none right, and 2 people
  never get an answer. The naive rule does no better (82.7%).
- Without the temperature nobody ever gets an answer: the adapted model's
  probabilities are so close to 1 that no cut-off reaches 95% on the other
  people.
- With no calibration from them, the decoder is right 39.5% of the time,
  and every rule answers 3% to 4% of windows. Four people never get an
  answer.

## Caveats

- **Across days this is measured, not guaranteed.** The promise assumes new
  windows are exchangeable with the held-out ones. Days and people are not,
  and on MultiDay and new 3DC people the promise is broken, by a point or so
  across days and by much more on new people.
- **Windows are not independent.** Windows overlap and come in holds of the
  same gesture. The binomial p-values treat them as independent, so the
  promise is optimistic even on held-out windows from the same day.
- **One person for drift.** MultiDay is one person, so the across-day result
  is one person's.
- **The promise is per window, not per decision.** It covers single windows
  before the pipeline's majority vote.
- **Cost.** One cut-off, fitted instantly. Prediction adds a softmax and a
  comparison per window.

## Related work

Answering only when confident is old: Chow (1970) gave the optimal reject
rule, and Scheme, Hudgins & Englehart (IEEE TBME 2013) reject low-confidence
EMG decisions. Those cut-offs are set by hand or by the naive rule above.
Learn then Test (Angelopoulos et al. 2021; section 5 of Angelopoulos & Bates'
gentle introduction covers selective classification) turns the choice into
multiple testing with a finite-sample promise. The fallback procedure is
Wiens (2003), a graphical procedure in the sense of Bretz et al. (2009).
