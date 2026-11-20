# Datasets

myobridge's numbers come from two public surface-EMG datasets. Neither is
redistributed here: `myobridge fetch` clones each one from the LibEMG
project's GitHub mirror and checks out a pinned commit, so everyone runs on
byte-identical files.

```bash
myobridge fetch            # both, into ~/.cache/myobridge (or $MYOBRIDGE_DATA, or --data DIR)
myobridge bench            # every experiment; writes results/*.json and docs/figures/*.png
myobridge bench --quick    # a few subjects/days, ~1 minute; writes *.quick.json (gitignored)
```

About 4 GB on disk once both are cloned. Text files are parsed once and
cached as `.npy` next to the data (`.myobridge-cache/`, ignored by git).

## 3DC: many people, one sitting

| | |
|---|---|
| Source | [github.com/libemg/3DCDataset](https://github.com/libemg/3DCDataset), commit `1af90582` |
| People | 22 |
| Channels | 10 bipolar channels around the forearm (3DC armband) |
| Sampling | 1000 Hz, after an analog anti-aliasing low-pass at 460 Hz |
| Mains | 60 Hz (recorded in Québec) |
| Gestures | 11: neutral, radial and ulnar deviation, wrist flexion and extension, supination, pronation, power grip, open hand, chuck grip, pinch grip |
| Protocol | 5 s per gesture; 4 cycles ("train"), a 5 minute break, 4 more cycles ("test"). The armband stays on throughout. |

Why it's here: it has the same sample rate as myobridge's ESP32, enough
people to test what happens with someone new, and more channels than
myobridge so we can ask what using only 4 costs.

The recordings were made for U. Côté-Allard, G. Gagnon-Turcotte, F.
Laviolette and B. Gosselin, "A low-cost, wireless, 3-D-printed custom armband
for sEMG hand gesture recognition", *Sensors* 19(12):2811, 2019
([doi:10.3390/s19122811](https://doi.org/10.3390/s19122811)), which
introduces the 3DC armband and describes "a new offline dataset featuring 22
able-bodied participants performing eleven hand/wrist gestures". Cite that
paper. The LibEMG mirror's README instead names Côté-Allard et al., *IEEE
TNSRE* 27(4):760-771, 2019, a transfer-learning paper whose experiments use
the Myo armband, and it also lists the dataset's later use in *Frontiers in
Bioengineering and Biotechnology* 8:158, 2020.

## MultiDay: one person, many days

| | |
|---|---|
| Source | [github.com/LibEMG/MultiDay](https://github.com/LibEMG/MultiDay), commit `2b11c859` |
| People | 1 |
| Days | 121 day indices, electrodes re-applied each day |
| Channels | 4 |
| Sampling | 2048 Hz |
| Mains | 50 Hz (recorded in Germany; the spectra show a 50 Hz line) |
| Gestures | 11: no motion, wrist extension and flexion, wrist adduction and abduction, supination, pronation, hand open, hand closed, key grip, index point |
| Protocol | about 6 s per gesture per day |

Why it's here: day-to-day drift is the main thing that breaks a wearable
gesture model, and this is the longest public record of it I know of. It also
has 4 channels like myobridge, and its 2048 Hz rate lets us emulate what a
1 kHz system with a poor anti-aliasing filter would have recorded
([aliasing.md](aliasing.md)).

The 4th channel is about 1000x smaller in amplitude than the other three on
every day, and the mirror doesn't say why. Everything downstream is
per-channel and scale-free (log amplitude features, per-channel thresholds),
so it is used as is.

Cite: P. Kaufmann, K. Englehart and M. Platzner, "Fluctuating EMG signals:
Investigating long-term effects of pattern matching algorithms", *IEEE EMBC*
2010, pp. 6357-6360.

## How the splits avoid leaking

The easiest way to get an impressive EMG number is to let test data leak into
training. Neighbouring windows overlap and are nearly identical, so a random
split of windows mostly tests memory. Every split here is by **recording or
time**, and every choice is made without the reported data:

- **Whole recordings or whole days** go to one side of a split, never both.
- **Calibration vs test inside one day** (MultiDay): calibration windows come
  from 1.0 s up to at most 2.0 s into each hold, test windows start at 2.5 s,
  so no window overlaps both.
- **Feature thresholds** (the noise floor for zero crossings and slope
  changes) are estimated from the training data of each split only.
- **Recalibration settings** are chosen on days 11-30 (MultiDay) or people
  1-6 (3DC), then frozen. Reported numbers are days 31-121 and people 7-22.
- **Quality-monitor thresholds** are chosen on people 1-6 of 3DC; false-alarm
  and detection rates are reported on people 7-22.
- **Uncertainty**: 95% intervals are bootstrap intervals over days or people
  (2,000 resamples, seed 0). Comparisons between methods are *paired*: the
  interval is on the per-day or per-person difference, which is far tighter
  than comparing two separate intervals.

Tests check the two rules that are easiest to break by accident: that
calibration and test windows never share a sample
(`tests/test_evaluation.py`), and that scrambling every test recording leaves
the training features, thresholds included, bit-for-bit unchanged
(`tests/test_bench.py`).

## Not used, and why

Larger datasets such as the Ninapro databases need an account or a licence
click-through before download. I kept to datasets anyone can fetch with one
command, so every number in this repo can be checked by anyone.
