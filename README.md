# myobridge

[![CI](https://github.com/ethsanders/myobridge/actions/workflows/ci.yml/badge.svg)](https://github.com/ethsanders/myobridge/actions/workflows/ci.yml)
[![Docs](https://github.com/ethsanders/myobridge/actions/workflows/docs.yml/badge.svg)](https://ethsanders.github.io/myobridge/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/ethsanders/myobridge/blob/main/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](https://github.com/ethsanders/myobridge/blob/main/LICENSE)

myobridge reads the electrical activity of forearm muscles and turns it into
hand gestures, live. An ESP32 streams four EMG sensors over WiFi, and a
Python library filters the signal, decodes the gesture and passes it on to a
game engine or a recorder. It also measures how well that works on public
datasets, including the things that usually go wrong: models going stale
from one day to the next, new users, and electrodes coming loose.

![A real forearm recording decoded live; halfway through, an electrode is pulled off and the decoder carries on without it](https://raw.githubusercontent.com/ethsanders/myobridge/main/docs/figures/demo.gif)

**[Try the live demo](https://ethsanders.github.io/myobridge/demo/)** in
your browser (real recordings, nothing to install) ·
[Documentation](https://ethsanders.github.io/myobridge/) ·
[Results](https://ethsanders.github.io/myobridge/results/) ·
[Build the armband](https://ethsanders.github.io/myobridge/hardware/)

## Quick start

```bash
pip install "myobridge @ git+https://github.com/ethsanders/myobridge"
```

```python
from myobridge import EmgSimulator, GesturePipeline, calibrate

arm = EmgSimulator(seed=0)  # or the real armband, via myobridge.Receiver
model, _, _ = calibrate(arm.record, ["rest", "fist", "open"], seconds=5, fs=arm.fs)
pipeline = GesturePipeline.from_model(model, n_channels=4)

arm.set_gesture("fist")
for prediction in pipeline.process_block(arm.generate(500)):
    print(prediction.gesture, round(prediction.confidence, 2))
```

With the armband, or the simulator standing in for it:

```bash
myobridge simulate                    # skip this with the real armband
myobridge listen                      # live signal, channel health and packet loss
myobridge calibrate --out model.npz   # hold each gesture when asked
myobridge forward --model model.npz   # gestures out as JSON over UDP
```

[Getting started](https://ethsanders.github.io/myobridge/getting-started/)
goes further, including decoding a real person from a public dataset.

## How well it works

Measured on two public datasets: 3DC (22 people, 10 electrodes) and MultiDay
(one person over 121 days, 4 electrodes), with 11 gestures, so chance is 9%.
Every setting was chosen on separate validation people or days, and the
brackets are 95% confidence intervals.

| Situation | Accuracy |
|---|---|
| Same person, same sitting, 10 electrodes | 92.4% [90.3, 94.4] |
| Same, with 4 electrodes like myobridge | 85.0% [82.8, 87.4] |
| A model from days 1-10, used on days 31-121 | 73.5% [71.8, 75.1] |
| ... after 1 second per gesture of that day's data, with `adapt()` | 89.0% [88.2, 89.7] |
| A new person, no calibration | 39.5% |
| ... with 2 seconds per gesture from them | 77.3% |
| One electrode comes off, no quality monitor | 24.4% [22.6, 26.3] |
| ... with the monitor dropping it | 91.2% [90.4, 92.0] |

What stands out:

- **Day-to-day drift is the main problem, and it's cheap to fix.** A
  second per gesture blended into the old model is worth
  +15.5 points [14.3, 16.7], and does better than retraining on the new
  data alone.
- **New people need their own calibration.** Starting from 21 other
  people's model adds 1.5 points [0.0, 3.0] over using the new
  person's 2 seconds per gesture alone.
- **A loose electrode is easy to catch.** The monitor notices within about
  a second and switches to a model without that electrode, computed
  instantly from the full one.
- **It's fast.** Processing a 20 ms packet takes 101 µs (median,
  534 µs p99), 1.0% of real time.
- **1 kHz sampling is enough for the EMG.** Wideband amplifier noise does
  fold back, +2.9 dB at 1 kHz behind the sensor's gentle filter,
  which the firmware's optional 2x oversampling cuts to +1.0 dB.

The [results page](https://ethsanders.github.io/myobridge/results/) has
every experiment, including feature sets and other classifiers. To
reproduce all of it, `myobridge fetch && myobridge bench` (about 4 GB and
an hour); a test checks that the numbers in the docs match the output.

## What's inside

- `firmware/`: the ESP32 sketch. A hardware timer drives sampling, and a
  queue hands packets to the WiFi sender, so a slow send never disturbs the
  sample clock. Packets carry a sequence number and a CRC-16.
- `myobridge`, the Python package: streaming filters, 15 features per
  electrode, linear discriminant analysis with one-step recalibration, a
  signal-quality monitor, replay of recorded data, and output as UDP JSON
  or [Lab Streaming Layer](https://labstreaminglayer.org).
- `myobridge.bench`: the experiments behind every number, on dataset
  versions pinned to a commit.
- `docs/demo/`: the browser demo, which runs the package itself via Pyodide.

## Limitations

- **Not yet run on my own armband.** The firmware compiles and its logic is
  tested, but every accuracy above comes from public datasets recorded with
  other hardware. [Bring-up](https://ethsanders.github.io/myobridge/hardware-bringup/)
  is next.
- **Drift is measured on one person**, because MultiDay is the best public
  record of day-to-day change I found.
- **Faults are simulated** on clean recordings, not recorded from real
  failures.

## Safety

This is a hobby project, not a medical device. Power the armband from a
battery only whenever electrodes are on skin, and never while it is plugged
into anything mains-powered, including a charging laptop.

## Contributing, citing, license

Issues and pull requests are welcome; [CONTRIBUTING.md](https://github.com/ethsanders/myobridge/blob/main/CONTRIBUTING.md)
has the setup. Hardware reports from your own builds are especially useful.
If you use myobridge in research, please cite it with the metadata in
[CITATION.cff](https://github.com/ethsanders/myobridge/blob/main/CITATION.cff).

MIT licensed. The datasets belong to their authors: 3DC by Côté-Allard et
al. and MultiDay by Kaufmann, Englehart and Platzner, both mirrored by the
[LibEMG](https://github.com/LibEMG) project. myobridge downloads them and
doesn't redistribute them; citations are in
[Datasets](https://ethsanders.github.io/myobridge/datasets/).
