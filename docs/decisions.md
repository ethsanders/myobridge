# Acting on it

The [previous page](conformal.md) is about windows: is the true gesture in
the set, 90% of the time? A user doesn't feel windows. A prosthesis or a
game acts on *decisions*, and what the user notices is how often it does the
wrong thing, how long the right thing takes to appear, and how often it
just holds. This page asks whether any of the uncertainty the decoder knows
about can be turned into fewer wrong actions, without making it slower.

On MultiDay it can, and the simplest rule does best. Act on a window's
gesture only when its temperature-scaled probability is at least 0.7, and
otherwise keep doing what you were doing. That makes **14.0** wrong actions
per minute instead of **25.6** for the pipeline's 5-window vote, on 91 later
days, with no measurable slowdown. Acting only on one-gesture conformal sets helps
less (22.2). On 3DC the gains are smaller, and the best rule there is a
different one.

## Using it

`myobridge.gating` has four decision policies. Each takes one window at a
time and returns a gesture to act on, or `None` to hold:

- `VotePolicy(k)`: the majority of the last k windows. This is what the
  pipeline reports as `Prediction.gesture`. It never holds.
- `ConfidenceGate(cutoff, k)`: the same vote, held while the newest window's
  top probability is below `cutoff`.
- `SetGate(n, m)`: act on a gesture when at least m of the last n conformal
  sets are exactly that gesture.
- `EvidenceGate(threshold, memory)`: add up each gesture's log-probability
  over recent windows, with older ones fading, and act when one gesture is
  likely enough. It is a leaky version of the sequential probability ratio
  test.

With `latch=True` a policy does not hold with "nothing". It repeats its last
action instead, so it only changes what it does when it is sure of the new
gesture.

```python
from myobridge import ConfidenceGate, GesturePipeline

pipeline = GesturePipeline.from_model(
    today, n_channels=4, conformal=sets,
    policy=ConfidenceGate(cutoff=0.7, k=1, latch=True),
)
for p in pipeline.process_block(block):
    if p.action is not None:
        act_on(p.action)
```

The policy sees the conformal calibration's temperature-scaled
probabilities, so the cut-off depends on the temperature. The 0.7 above was
chosen with the MultiDay recipe, whose temperature is about 10. Without a
policy, `p.action` is the voted gesture, as before. The pipeline's default
behaviour hasn't changed.

## How it's measured

`myobridge bench decision` writes `results/decision.json`. Every recording
is one held gesture. A day's (or a person's) recordings are played back to
back in a random order, with no gesture following itself, window by window
through one policy. Each hold therefore starts with the previous gesture
still in the policy's memory, as when a user changes gesture. Played one at
a time from a fresh policy, every vote size looks equally fast, because the
first window of a MultiDay hold is nearly always right.

For each day or person:

- **Wrong actions per minute**: how many times a wrong gesture starts, per
  minute. A run of the same wrong gesture counts once.
- **Right action**: the share of the time spent acting on the true gesture.
- **Time to the right action**: from a hold's first window to its first
  correct decision.

The rates are scored from 2.5 s into each hold, like the other benchmarks,
and also from the hold's first window. The MultiDay sets come from the
conformal page's recipe: adapt each day with 1 s per gesture, temperature
from days 11-20, threshold from days 21-30.

Every policy setting is chosen on validation data only: MultiDay days 11-30
and 3DC people 1-6. The rule is to match the 5-window vote. A setting must
be no slower to the right action, and spend no less time on the right
action, than the vote on validation. Among the settings that qualify, the
one with the fewest wrong actions per minute is picked. 230 settings were
tried. Brackets are 95% bootstrap intervals over test days or people. A
policy counts as a win when its wrong actions per minute are lower with the
interval below zero, and neither of the other two measures is worse with
its interval entirely on the bad side.

## MultiDay: 91 later days

Days 31-121, from 2.5 s into each hold, settings chosen on days 11-30:

| | Wrong actions per minute | Change | Right action | Time to the right action |
|---|---|---|---|---|
| 5-window vote (the pipeline today) | 25.6 [24.3, 27.0] | | 91.7% | 0.16 s |
| **Confidence 0.7, latched** | **14.0 [13.2, 14.8]** | **−11.6 [−12.8, −10.6]** | **94.2%** | **0.17 s** (+8 ms [−4, +21]) |
| Evidence, latched | 19.4 | −6.3 [−7.2, −5.3] | 92.3% | 0.14 s |
| One-gesture 80% sets, latched | 22.2 | −3.4 [−4.5, −2.2] | 92.2% | 0.12 s |
| 9-window vote, for comparison | 14.2 | −11.4 [−12.4, −10.5] | 94.2% | 0.23 s |

All three latched policies win by the rule above. The confidence cut-off
wins by the most. It gets the 9-window vote's rate of wrong actions at
about the 5-window vote's speed. From the hold's first window, it makes
16.7 wrong actions per minute against 28.8.

Holding with "nothing" instead of latching costs too much. With the same
0.7 cut-off, the gate makes 15.8 wrong actions per minute, but it only acts
78.9% of the time, and on the right gesture 76.2% of the time. Of the
policies that hold this way, only one evidence setting met the rule above,
and it acts 99.5% of the time: −1.9 [−2.7, −1.1].

Conformal sets don't beat plain confidence here.

**Other speeds.** The same rule, matched to other vote sizes on days 11-30,
gives the curve below (days 31-121). At each of these speeds, the
confidence cut-off makes fewer wrong actions than the vote it was matched
to. On the test days it is also a little slower than that vote, by 8 to
53 ms, though it was no slower on the validation days. Matched to a
15-window vote, no cut-off qualified.

| Matched to | Vote: wrong per minute | Time | Confidence, latched: wrong per minute | Time |
|---|---|---|---|---|
| 3-window vote | 32.9 | 0.12 s | 17.1 (cut-off 0.65) | 0.14 s |
| 5-window vote | 25.6 | 0.16 s | 14.0 (0.7) | 0.17 s |
| 7-window vote | 18.3 | 0.19 s | 10.7 (0.75, 3-window vote) | 0.23 s |
| 9-window vote | 14.2 | 0.23 s | 8.6 (0.8) | 0.28 s |
| 11-window vote | 11.2 | 0.26 s | 8.0 (0.8, 5-window vote) | 0.30 s |

## 3DC: after a break, and a new person

Settings chosen on people 1-6, reported on 7-22.

**After a 5 minute break** (each person's own model and thresholds from the
session before it). The vote makes 12.4 [9.0, 15.9] wrong actions per
minute. Here the choice made on people 1-6 is the evidence policy (memory
0.7, threshold 0.5), latched: 10.0, a change of −2.4 [−3.8, −1.2]. It wins,
but by little. One-gesture sets change it by −1.1 [−3.5, +1.1], which is
not a clear gain. No confidence cut-off met the rule on people 1-6. The
MultiDay choice, applied unchanged, does better: 5.8, a change of
−6.7 [−9.1, −4.3]. Its time to the right action changes by
+28 ms [−44, +133], and 3.0% of holds never get the right action, against
1.9% for the vote.

**A new person** (leave one out, 2 s per gesture through `adapt()`). The
vote makes 33.2 [26.1, 40.8] wrong actions per minute. The choice made on
people 1-6 is the evidence policy (memory 0.5, threshold 0.5), latched:
27.9, a change of −5.2 [−10.0, −1.1]. The MultiDay cut-off doesn't carry
over. The fitted temperature for this recipe is 74.8, so the probabilities
are much flatter. With the 0.7 cut-off, the gate takes 0.46 s longer to
reach the right action, and 13.5% of holds never reach it.

## Caveats

- **One person for the main result.** The large gain is on MultiDay, one
  person over 91 days. On 3DC the policy chosen there is a different one,
  and it gains less.
- **The cut-off belongs to its temperature.** It is a probability after
  temperature scaling. When the recipe or its temperature changes, choose
  the cut-off again on held-out data.
- **Gesture changes are idealised.** Holds are joined end to end, so a
  change of gesture is instant and the next window is already the new
  gesture. A real change has windows that are half one gesture and half the
  other. The order of holds is random, with one seed.
- **A latch never stops by itself.** It keeps doing the last thing until it
  is sure of a new gesture, and rest counts as a gesture. If doing nothing
  is safer than doing the last thing, as with some prostheses, use the
  unlatched gate and accept that it holds more often.
- **The time to the right action includes training windows on MultiDay.**
  Each day's model is adapted on [1.0, 2.0) s of the same holds. A first
  correct action after 1.0 s is 5.2% of holds with the cut-off and 2.0%
  with the vote. Those decisions are on windows the model was adapted on.
  The scores from 2.5 s don't have this problem, and neither does 3DC.
- **The pipeline starts sooner than the bench.** Live, the first window ends
  200 ms into the stream. The bench skips the first 250 ms of each
  recording while the filters settle, as the other benchmarks do.

## Related work

Majority voting over windows is the usual post-processing in EMG control
(Englehart & Hudgins 2003). Rejecting low-confidence decisions is also
established: Scheme, Hudgins & Englehart (IEEE TBME 2013) hold the output
when the classifier is unsure.
Sequential tests that accumulate evidence go back to Wald (1945). The
contribution here is the measurement. It counts decisions rather than
windows, on held-out days, with every threshold chosen on validation data,
and compares the policies at matched speed.
