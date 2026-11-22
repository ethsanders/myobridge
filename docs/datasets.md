# Datasets

myobridge's numbers come from two public surface-EMG datasets. Neither is
redistributed here. `myobridge fetch` clones each one from the LibEMG
project's GitHub mirror and checks out a pinned commit, so every run uses
byte-identical files.

```bash
myobridge fetch            # both, into ~/.cache/myobridge (or $MYOBRIDGE_DATA, or --data DIR)
myobridge bench            # every experiment; writes results/*.json and docs/figures/*.png
myobridge bench --quick    # a few subjects/days, ~1 minute; writes *.quick.json (gitignored)
```

The two datasets take about 4 GB on disk once cloned. Text files are parsed
once and cached as `.npy` next to the data (`.myobridge-cache/`, ignored by
git).

## 3DC

| | |
|---|---|
| Source | [github.com/libemg/3DCDataset](https://github.com/libemg/3DCDataset), commit `1af90582` |
| People | 22 |
| Channels | 10 bipolar channels around the forearm (3DC armband) |
| Sampling | 1000 Hz, after an analog anti-aliasing low-pass at 460 Hz |
| Mains | 60 Hz (recorded in Québec) |
| Gestures | 11: neutral, radial and ulnar deviation, wrist flexion and extension, supination, pronation, power grip, open hand, chuck grip, pinch grip |
| Protocol | 5 s per gesture; 4 cycles ("train"), a 5 minute break, 4 more cycles ("test"). The armband stays on throughout. |

I use 3DC because it has the same sample rate as myobridge's ESP32 and
enough people to test performance on someone new. It also has more
channels than myobridge, which makes it possible to measure the cost of
using only 4.

The recordings were made for U. Côté-Allard, G. Gagnon-Turcotte, F.
Laviolette and B. Gosselin, "A low-cost, wireless, 3-D-printed custom armband
for sEMG hand gesture recognition", *Sensors* 19(12):2811, 2019
([doi:10.3390/s19122811](https://doi.org/10.3390/s19122811)). That paper
introduces the 3DC armband and describes "a new offline dataset featuring 22
able-bodied participants performing eleven hand/wrist gestures". Cite that
paper. The LibEMG mirror's README instead names Côté-Allard et al., *IEEE
TNSRE* 27(4):760-771, 2019, a transfer-learning paper whose experiments use
the Myo armband. The README also lists the dataset's later use in
*Frontiers in Bioengineering and Biotechnology* 8:158, 2020.

## MultiDay

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

I use MultiDay because day-to-day drift is the main cause of failure in a
wearable gesture model, and this is the longest public record of it I know
of. Like myobridge it has 4 channels. Its 2048 Hz rate also allows an
emulation of what a 1 kHz system with a poor anti-aliasing filter would
have recorded ([aliasing.md](aliasing.md)).

On every day, the 4th channel is about 1000x smaller in amplitude than the
other three, and the mirror does not say why. The processing after loading
is per-channel and scale-free (log amplitude features, per-channel
thresholds), so the channel is used as is.

Cite P. Kaufmann, K. Englehart and M. Platzner, "Fluctuating EMG signals:
Investigating long-term effects of pattern matching algorithms", *IEEE EMBC*
2010, pp. 6357-6360.

## Train/test splits

Test data leaking into training is a common way to inflate EMG results.
Neighbouring windows overlap and are nearly identical, so a random split of
windows mostly tests memorisation. Every split here is by recording or by
time, and every choice is made without the reported data:

- Whole recordings or whole days go to one side of a split, never both.
- Within one MultiDay day, calibration windows come from 1.0 s up to at most
  2.0 s into each hold and test windows start at 2.5 s, so no window falls
  in both.
- Feature thresholds (the noise floor for zero crossings and slope changes)
  are estimated from the training data of each split only.
- Recalibration settings are chosen on days 11-30 (MultiDay) or people 1-6
  (3DC), then frozen. Reported numbers are days 31-121 and people 7-22.
- Quality-monitor thresholds are chosen on people 1-6 of 3DC. False-alarm
  and detection rates are reported on people 7-22.
- The 95% intervals are bootstrap intervals over days or people (2,000
  resamples, seed 0). Comparisons between methods are *paired*: the
  interval is on the per-day or per-person difference, which is far tighter
  than comparing two separate intervals.

Tests check the two rules that are easiest to break by accident. One checks
that calibration and test windows never share a sample
(`tests/test_evaluation.py`). The other checks that scrambling every test
recording leaves the training features, thresholds included, bit-for-bit
unchanged (`tests/test_bench.py`).

## Datasets not used

Larger datasets such as the Ninapro databases need an account or a licence
click-through before download. I kept to datasets that can be fetched with
one command, so that anyone can check the numbers in this repo.
