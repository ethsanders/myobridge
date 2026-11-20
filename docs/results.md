# Results

Every number on this page is read from `results/*.json`, which
`myobridge bench` writes along with the dataset commit and settings that
produced it. `tests/test_docs.py` checks that the headline numbers here and
in the README still match those files. Brackets are 95% bootstrap intervals
of the mean over days or people; "points" are percentage points. All tasks
use 11 gestures, so chance is 9%.

How the data are split, and why it can't leak, is in
[datasets.md](datasets.md).

## Setup shared by every experiment

- **Signal chain:** 4th-order Butterworth bandpass 20-450 Hz, notch at the
  mains frequency, 200 ms windows every 50 ms, the first 250 ms of each
  recording skipped while the filters settle.
- **Features:** 15 per channel, the `td-ar-psd` set: log mean absolute
  value, log RMS, log waveform length, zero crossings and slope sign changes
  (noise thresholds from training-split rest recordings), four
  autoregressive coefficients, and the six TD-PSD spectral-moment features
  of Al-Timemy et al. (2016). Section 3 shows how this set was chosen.
- **Classifier:** linear discriminant analysis (`myobridge.classifier`), the
  same code that runs live, with a small ridge scaled to each feature's own
  variance so the model doesn't care what units a feature is in.

## 1. Within one sitting (3DC)

Per person: train on the 4 cycles before the 5 minute break, test on the 4
after. The armband stays on.

| | Accuracy |
|---|---|
| 22 people, 10 channels | **92.4%** [90.3, 94.4] |
| worst / best person | 77.4% / 98.5% |

![Confusion matrix and accuracy by channel count](figures/threedc_confusion.png)

**Window length** trades latency for accuracy, but not by much:

| Window | 100 ms | 150 ms | 200 ms | 250 ms | 300 ms |
|---|---|---|---|---|---|
| Accuracy | 90.3% | 91.6% | 92.4% | 92.9% | 93.3% |

## 2. How many channels? (3DC)

The same split, using only *k* of the 10 channels, for every one of the
C(10, *k*) subsets.

| Channels | Mean over all subsets | Fixed choice |
|---|---|---|
| 1 | 50.1% | |
| 2 | 68.7% | |
| 3 | 78.0% | |
| **4** | **83.2%** | **85.0%** [82.8, 87.4] (channels 0, 2, 5, 7) |
| 6 | 88.4% | |
| 8 | 90.9% | |
| 10 | 92.4% | |

The fixed choice, four channels spread evenly around the forearm, was set
before running anything, so it is an unbiased estimate for a 4-sensor band.
The best 4-channel subset reaches 86.1%, but it was picked after seeing test
accuracy, so that number is optimistic and isn't used anywhere.

**Reading:** a 4-sensor band gives up about 7 points against a 10-channel
armband on 11 gestures. Spreading the sensors evenly beats an average
placement by about 2 points.

## 3. Which features, and which classifier? (3DC, MultiDay)

**Features.** The original decoder used only the five time-domain features.
Adding autoregressive and spectral-moment features was decided on
validation data only: 3DC people 1-6 and MultiDay days 11-30. The tables
show validation first, then the reported people and days, which were
looked at only after the choice.

| 3DC, within one sitting | `td` (5 per channel) | `td-ar-psd` (15) | Difference |
|---|---|---|---|
| Validation, people 1-6, 10 channels | 91.6% | 92.9% | |
| Validation, people 1-6, even 4 | 81.5% | 83.1% | |
| Reported, people 7-22, 10 channels | 91.2% | 92.2% | +1.0 [0.6, 1.4] |
| Reported, people 7-22, even 4 | 83.6% | 85.7% | +2.1 [1.4, 2.7] |

| MultiDay, base model from days 1-10 | `td` | `td-ar-psd` | Difference |
|---|---|---|---|
| Validation days 11-30, no recalibration | 87.0% | 89.5% | |
| Validation days 11-30, `adapt` with 1 s | 90.7% | 92.0% | |
| Test days 31-121, no recalibration | 70.4% | 73.5% | +3.1 [2.5, 3.7] |
| Test days 31-121, `adapt` with 1 s | 87.1% | 89.0% | +1.9 [1.6, 2.3] |

The gain is largest where the data are hardest: fewer channels and later
days. The 3DC rows for people 7-22 differ from section 1, which averages
all 22 people.

One disclosure: the per-feature ridge was added in the same change. It
fixed a real problem (with mixed feature scales, one fixed ridge
over-regularised some features and under-regularised others), but on
MultiDay it also moved the old `td` features' no-recalibration test score
from 72.8% to 70.4%, so part of the gap in that row comes from the ridge
treating `td` slightly worse. With `adapt`, the comparison that matters in
practice, both feature sets gain.

**Classifier.** On the same features and people 7-22, LDA against three
standard alternatives from scikit-learn, with settings fixed before the run
and not tuned: an RBF SVM (standardised features, C = 10), a random forest
(200 trees) and a one-hidden-layer MLP (128 units, standardised features).

| | LDA | SVM (RBF) | Random forest | MLP |
|---|---|---|---|---|
| Accuracy | **92.2%** | 89.8% | 85.8% | 89.9% |
| LDA minus this | | +2.4 [1.4, 3.2] | +6.4 [4.7, 8.4] | +2.3 [1.5, 3.0] |

**Reading:** with a few minutes of data per person, LDA isn't leaving
accuracy on the table, and it is the only one of the four that trains in
milliseconds, runs in microseconds and has a closed-form recalibration
(section 4). Tuned SVMs and MLPs may close the gap; untuned, they don't.

## 4. Drift across days (MultiDay)

One person, electrodes re-applied each day, 121 day indices. Days 1-10 train
the base model, days 11-30 choose recalibration settings, days 31-121 are
reported.

**How fast a model goes stale.** Train on one whole day, test on a later one:

| Days later | 1 | 2 | 5 | 10 | 20 | 30 | 50 | 90 |
|---|---|---|---|---|---|---|---|---|
| Accuracy | 86.5% | 86.0% | 85.4% | 84.1% | 82.4% | 79.7% | 76.7% | 73.8% |

More training days help (70.5% from 1 day, 73.5% from 10, 82.1% from 30,
tested on days 31-121), but they don't stop the decay.

**Recalibration.** On each test day, a few seconds per gesture of new
labelled data (from 1.0 s into each hold; test windows start at 2.5 s).
Four ways to use it:

- `none`: the base model from days 1-10, unchanged
- `new_only`: a fresh model from the new data alone
- `pooled`: retrain on days 1-10 plus the new data
- `adapt`: `myobridge.classifier.adapt`, which pulls each class mean toward
  the new data's mean and blends in some of its covariance, with two
  settings chosen on days 11-30. This is the supervised adaptation of
  Vidovic et al. (*IEEE TNSRE* 24(9):961-970, 2016); myobridge only states
  the mean's weight as a number of windows

| New data per gesture | none | new_only | pooled | **adapt** | adapt − next best |
|---|---|---|---|---|---|
| 0.25 s | 73.5% | 61.9% | 73.6% | **82.2%** | +8.6 [7.9, 9.3] vs pooled |
| 0.5 s | 73.5% | 78.4% | 74.3% | **85.2%** | +6.8 [5.4, 8.1] vs new_only |
| 1.0 s | 73.5% | 85.8% | 75.6% | **89.0%** | +3.2 [2.5, 3.9] vs new_only |

Intervals on the differences are paired: each method is scored on the same
91 days. For reference, a model trained only on 1.5 s per gesture from the
test day itself, with no older data, scores 87.9%.

![Decay across days and recalibration](figures/multiday_recalibration.png)

**Reading:**

- Pooling barely helps: ten days of old data outvote a second of new data.
- A fresh model needs enough data to estimate its covariance; at 0.25 s per
  gesture it is worse than doing nothing.
- `adapt` keeps the old covariance, which electrode shifts change little, and
  moves the means, which they change a lot. It wins at every length, and with
  1 s per gesture it beats a same-day model trained on more data.

**Rest alone isn't enough.** A daily calibration that asked only for rest
would be much easier for a user. Moving every class mean by the shift that
rest shows recovers 3.1 points [2.5, 3.7] with 1 s of rest, against 15.5
for `adapt()` with every gesture. Adding two more gestures (key grip and
index point, chosen on days 11-30) and giving those their own new means
reaches 82.4%, and one shift estimated from all 11 gestures only 80.0%. One
shared shift explains 21.3% [20.1, 22.5] of how the class means move from
day to day; most of the drift is specific to each gesture. Numbers from
`myobridge bench partial_calibration`, in `results/partial_calibration.json`.

**Where the errors are.** Most of the adapted model's mistakes come at the
end of a hold. The last second of each hold is 26.8% of the test windows;
without it, the adapted model scores 96.7% and the base model 77.4%. The
headline numbers keep those windows, because a live user's holds end too.

## 5. A new person (3DC)

Leave one person out: train on the other 21, test on the held-out person's
4 cycles after the break. Calibration data come from their first cycle, from 1.0 s
into each 5 s hold. Settings for `adapt` are chosen on people 1-6; people
7-22 are reported.

| New data per gesture | none | new_only | pooled | **adapt** | adapt − new_only |
|---|---|---|---|---|---|
| 0.5 s | 39.5% | 65.2% | 39.7% | **65.3%** | +0.1 [−3.4, 3.7] |
| 1.0 s | 39.5% | 70.7% | 40.1% | **72.0%** | +1.3 [−0.6, 3.1] |
| 2.0 s | 39.5% | 75.8% | 40.7% | **77.3%** | +1.5 [0.0, 3.0] |

![Channels and cross-subject results](figures/threedc_channels_cross_subject.png)

**Reading:** other people's data barely transfer (39.5% with no calibration),
and as a prior they add little: the settings chosen on validation people
trust the new person's data almost entirely. Across days, the old model is a
good starting point; across people it isn't. Every new user needs a short
calibration.

## 6. Aliasing (MultiDay + model)

Summarised here, in full in [aliasing.md](aliasing.md).

| | |
|---|---|
| Median share of EMG power above 512 Hz | 0.04-0.17% per channel |
| Same-day accuracy at emulated 1024 Hz, steep anti-alias filter | 88.1% |
| ... with no anti-alias filter at all | 88.1% |
| Folded white noise at 1 kHz behind a 1st-order roll-off at 498 Hz | +2.9 dB |
| ... at 2 kHz with the firmware's 47-tap halfband decimator | +1.0 dB |

![Folded noise by filter order and sample rate](figures/aliasing_noise.png)

## 7. Signal quality (3DC)

Summarised here, in full in [signal-quality.md](signal-quality.md).

Thresholds are set for about 1% false alarms per check on clean data from
people 1-6; people 7-22 are reported.

| | |
|---|---|
| Clean channel reports raising any check | 1.3% |
| Lifted electrode (signal x0.03) caught | 98.7%, median 1.0 s after onset |
| Mains hum at the EMG's power caught | 84.7% |

One channel broken for the whole test session (16 people x 10 channels):

| Fault | No fault | No monitor | **Monitor** | Always drop the broken channel |
|---|---|---|---|---|
| Electrode off | 92.2% | 24.4% | **91.2%** [90.4, 92.0] | 91.6% |
| Clipping | 92.2% | 17.9% | **88.7%** [87.8, 89.5] | 91.6% |
| Mains hum | 92.2% | 92.2% | **91.9%** | 91.6% |
| Motion | 92.2% | 92.2% | **91.9%** | 91.6% |

**Reading:** a dead or clipping channel wrecks the classifier and the monitor
gets nearly all of it back. Hum and motion don't hurt, because the filters
remove them, so the decoder only drops a channel for flatline or clipping.
That policy was chosen after a first run; signal-quality.md reports both.

![Fault detection and fault tolerance](figures/quality.png)

## 8. Knowing when it doesn't know (MultiDay, 3DC)

Summarised here, in full in [conformal.md](conformal.md). Instead of one
gesture, the decoder gives a set that should hold the true gesture 90% of the
time. The question is whether that promise survives new days, a break, new
people and broken electrodes. Thresholds are always set on held-out days or
people.

| At a 90% promise | Delivered | Gestures per set | Right when the set has one |
|---|---|---|---|
| MultiDay days 31-121, textbook threshold, old model | 74.6% [73.0, 76.2] | 1.03 | 74.4% |
| ... adapted each day, recipe (temperature + APS) | 89.1% [88.7, 89.4] | 1.43 | 97.0% |
| 3DC after the break, recipe | 89.6% [87.4, 91.1] | 1.27 | 97.3% |
| 3DC new person, 2 s per gesture, recipe | 89.3% [85.4, 92.4] | 2.18 | 91.0% |
| 3DC electrode off, with the quality monitor | 89.5% [88.5, 90.4] | 1.34 | 96.7% |
| ... without the monitor | 22.8% [19.7, 25.9] | 1.06 | 24.5% |

**Reading:** the textbook calibration breaks as soon as the armband is put
back on. Recalibrating with `adapt()`, softening the LDA's probabilities with
one temperature and using the APS score keeps the promise on all 91 test
days, within 5 points on every one. A broken electrode breaks it again,
silently, which is why the quality monitor runs first.

## 9. Speed

Measured by `myobridge bench latency` on the machine that produced the
results file (its CPU is recorded there), with the live code path: 4
channels, 20-sample packets, 200 ms windows every 50 ms, 60 features, a
5-window vote.

| | |
|---|---|
| Processing per 20 ms packet (filter, features, LDA, vote) | 117 µs median, 550 µs p99 |
| Share of real time used | 1.3% |
| `adapt()` recalibration, 11 gestures | 0.33 ms |

Processing is not where the delay is. The time from starting a gesture to
the smoothed output switching is set by the settings: up to 20 ms to fill a
packet, about 100 ms until the window is mostly the new gesture, and about
100 ms for the majority vote, so roughly 220 ms before network and muscle
onset. That is an estimate from the settings, not a measurement; the knobs
are the window length, the step and the vote size.

## Reproduce

```bash
pip install -e ".[dev]"
myobridge fetch
myobridge bench                     # everything, then the figures
myobridge bench multiday aliasing   # or any subset
```

The benchmarks are deterministic: the same code and dataset commits give
the same JSON.
