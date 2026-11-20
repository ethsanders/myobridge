# Your own 90% promise

The [benchmark](conformal.md) shows gesture sets that keep a 90% promise on
91 later days of someone else's arm. This guide runs the same recipe on your
armband: record a short session on a few days, fit the promise, then check
it on a later day with a score that counts every decision.

You need the armband streaming (see [first sessions on
hardware](hardware-bringup.md)). Every command also takes `--sim` to try the
steps with the simulator first.

## 1. Record a session on each of a few days

Each session holds every gesture for 5 seconds. Take the armband off between
days, so each session has its own electrode placement. That placement change
is what the promise has to survive.

```bash
myobridge calibrate --gestures rest,fist,open,point,pinch,flex \
    --seconds 5 --out day1-model.npz --session day1.npz
```

Do this on three different days (`day2.npz`, `day3.npz`). Two days work too,
but they are a weaker test, as step 2 explains.

## 2. Fit the promise

```bash
myobridge fit day1.npz day2.npz day3.npz --out model.npz
```

Give the sessions oldest first. `fit` trains the decoder on all but the last
two. It fits the temperature on the second-to-last session and sets the
threshold on the last. Each of those two later sessions is first
recalibrated the way you will recalibrate every day: `adapt()` with one
second of each gesture. That way the threshold is set on the kind of model
you will actually use. `fit` prints which session did what, and writes
`model.npz` with `model.sets.json` beside it.

With only two sessions, the second does both jobs: the earlier half of each
hold fits the temperature and the later half sets the threshold. Both halves
share one placement, so this tests a new day less well.

## 3. On any later day

```bash
myobridge sets --model model.npz --prompt --web
```

The app first asks you to hold each gesture for 2 seconds. It uses the
second one of those to adapt the decoder to today's placement, and keeps the
temperature and threshold as they were. That is the recipe.

Then it shows the decoder's answer for every 50 ms window:

- **One gesture**: the decoder is sure.
- **"not sure: fist or open"**: the right one should be among these.

With `--prompt`, it asks for one gesture at a time, 4 seconds each, in a new
order every round. It counts how often the set held the gesture it asked
for, next to the promise:

```text
hold pinch   2.3 s  |  not sure: pinch or point  |  promised 90%, delivered 89.6% of 412 windows, one gesture 64%
```

The first 1.5 seconds after each prompt aren't scored, because you're still
moving into the gesture. `--web` also opens the same view in large type at
<http://127.0.0.1:8765/>, for a second screen or a camera. Stop with Ctrl-C
or `--duration`. The last line gives the final score.

## What the number means

The promise is about windows like the scored ones: on a later day, after the
one-second-per-gesture recalibration, holding a gesture you trained. A few
hundred windows give a rough number. Each window sits next to many others
like it, so the count moves more than its size suggests. Several sessions
tell you more than one long one.

The benchmark measures two ways to break the promise. One is skipping the
daily recalibration ([what goes wrong](conformal.md#what-goes-wrong)), which
`--recalibrate 0` lets you try. The other is an electrode that comes off
with no quality monitor ([an electrode
fails](conformal.md#the-same-recipe-on-other-data)). `myobridge sets` runs
the monitor, using today's rest hold as each channel's noise floor, and
drops an electrode that goes flat or clips. `--no-monitor` turns that off,
so you can watch the sets miss.
