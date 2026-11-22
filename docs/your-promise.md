# Gesture sets on your own armband

The [benchmark](conformal.md) shows gesture sets that keep a 90% promise on
91 later days of another person's recordings. This guide runs the same
recipe on your armband. You record a short session on each of a few days,
fit the sets, and then check them on a later day with a score that counts
every decision.

You need the armband streaming (see [first sessions on
hardware](hardware-bringup.md)). Every command also takes `--sim`, so you
can try the steps on the simulator first.

## 1. Record a session on each of a few days

Each session holds every gesture for 5 seconds. Take the armband off between
days, so that each session has its own electrode placement. The promise has
to hold across these placement changes.

```bash
myobridge calibrate --gestures rest,fist,open,point,pinch,flex \
    --seconds 5 --out day1-model.npz --session day1.npz
```

Record on three different days (`day2.npz`, `day3.npz`). Two days also work,
but they make a weaker test, as step 2 explains.

## 2. Fit the promise

```bash
myobridge fit day1.npz day2.npz day3.npz --out model.npz
```

Give the sessions oldest first. `fit` trains the decoder on all but the last
two sessions. It fits the temperature on the second-to-last session and sets
the threshold on the last. Each of those two sessions is first recalibrated
the way you will recalibrate every day: `adapt()` with one second of each
gesture. The threshold is therefore set on the same kind of model you will
use. `fit` prints which session was used for each step, and writes
`model.npz` with `model.sets.json` beside it.

With only two sessions, the second session does both jobs: the earlier half
of each hold fits the temperature, and the later half sets the threshold.
Both halves share one placement, so this is a weaker test of a new day.

## 3. On any later day

```bash
myobridge sets --model model.npz --prompt --web
```

The app first asks you to hold each gesture for 2 seconds. It uses the
second of those two seconds to adapt the decoder to today's placement, and
keeps the temperature and threshold unchanged, as the recipe requires.

It then shows the decoder's output for every 50 ms window:

- **One gesture**: the decoder is sure.
- **"not sure: fist or open"**: the right gesture should be one of these.

With `--prompt`, the app asks for one gesture at a time, 4 seconds each, in a
new order every round. It counts how often the set contained the requested
gesture and prints that next to the promise:

```text
hold pinch   2.3 s  |  not sure: pinch or point  |  promised 90%, delivered 89.6% of 412 windows, one gesture 64%
```

The first 1.5 seconds after each prompt are not scored, because you are
still moving into the gesture. `--web` also opens the same view in large
type at <http://127.0.0.1:8765/>, for a second screen or a camera. Stop with
Ctrl-C or `--duration`. The last line gives the final score.

`--gate` shows what an application would do. It acts on the newest window's
top gesture only when that gesture's probability is 0.7 or more, and
otherwise keeps its last action ([why](decisions.md)). With `--prompt` it
also counts how often that action was right and the number of wrong actions
per minute. Because it keeps its last action, it acts on almost every
window; the gate changes how often that action is wrong. The 0.7 was chosen
on the benchmark, and the right cut-off depends on your calibration's
temperature, so try others, for example `--gate 0.8`, and compare the wrong
actions per minute.

## Interpreting the score

Three sessions give a looser promise than the benchmark's thirty days. When
the same procedure was run on 28 different stretches of three days and
checked on months of later days, the 90% sets delivered 88.5% on average,
and single days went as low as 78.0%
([details](conformal.md#against-simpler-rules)). The benchmark's tighter
numbers come from setting the temperature and the threshold on ten days
each.

The promise applies to windows like the scored ones: on a later day, after
the one-second-per-gesture recalibration, while you hold a gesture you
trained. A few hundred windows give only a rough estimate. Each window
closely resembles the windows around it, so the count varies more than its
size suggests. Several sessions tell you more than one long session.

The benchmark measures two ways to break the promise. One is skipping the
daily recalibration ([what goes wrong](conformal.md#what-goes-wrong)), which
you can try with `--recalibrate 0`. The other is an electrode that comes off
while no quality monitor is running ([an electrode
fails](conformal.md#the-same-recipe-on-other-data)). `myobridge sets` runs
the monitor, using today's rest hold as each channel's noise floor, and
drops an electrode that goes flat or clips. `--no-monitor` turns the monitor
off, so you can watch the sets miss.
