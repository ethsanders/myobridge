# Getting started

This page takes you from nothing installed to decoding gestures: first from
a simulated armband, then from a real person's recordings, then live.

## Install

myobridge needs Python 3.10 or newer, NumPy and SciPy.

```bash
pip install "myobridge @ git+https://github.com/ethsanders/myobridge"
```

For the benchmarks and figures, clone the repository and install the extras:

```bash
git clone https://github.com/ethsanders/myobridge
cd myobridge
pip install -e ".[dev]"
```

## Decode a simulated arm

No data or hardware needed. The simulator stands in for the armband: it
produces the same 4-channel, 1 kHz stream of ADC counts.

```python
--8<-- "examples/decode_simulated.py"
```

```text
trained on 460 windows of 60 features each
holding   fist: decoded fist in 37 of 37 decisions
holding   rest: decoded rest in 34 of 40 decisions
holding extend: decoded extend in 35 of 40 decisions
```

Two things to notice:

- `calibrate` takes any function that records a gesture for some seconds.
  Swap the simulator for the UDP receiver and the same code calibrates on a
  real arm.
- `GesturePipeline.from_model` builds the live pipeline with the sample
  rate, window and thresholds the model was trained with, so training and
  live use can't drift apart. A model saved with `model.save("model.npz")`
  keeps them too.

The simulator is clean, band-limited noise, so it only proves the plumbing.
For a real test, use real muscles.

## Decode a real forearm

The [3DC dataset](datasets.md) has 22 people doing 11 hand and wrist
gestures with a 10-electrode armband. This script calibrates on one
person's first two passes through the gestures, then streams a later pass
through the live pipeline, 20 samples at a time, and pulls an electrode off
half-way:

```bash
myobridge fetch 3dc      # once; clones the dataset at a pinned commit (about 1.3 GB)
python examples/replay_3dc.py --person 7
```

```python
--8<-- "examples/replay_3dc.py"
```

The [live demo](demo/index.html) runs this same code in your browser, with a
menu to break any electrode.

## Run it live

With the ESP32 armband (see [Build the armband](hardware.md)), or the
simulator standing in for it over UDP:

```bash
myobridge simulate --port 5005 --drop 0.01       # skip this with the real armband
myobridge listen --port 5005                     # live RMS bars, channel health, link stats
myobridge calibrate --seconds 5 --out model.npz  # hold each gesture when asked
myobridge forward --model model.npz --to 127.0.0.1:5006 --verbose
```

`forward` sends one JSON message per decision to the address you give, for a
game engine to read. The message format is in
[Send gestures to a game](protocol.md).

Add `--lsl` to also publish on [Lab Streaming Layer](https://labstreaminglayer.org),
which recorders like LabRecorder and most neuroscience tools can read: a
`myobridge-emg` stream with the raw samples and a `myobridge-gestures`
stream with one marker per decision. It needs `pip install "myobridge[lsl]"`.

Outside North America, put `--mains 50` before the subcommand, for example
`myobridge --mains 50 listen`.

## Reproduce the results

Every number in [Results](results.md) comes from one command:

```bash
myobridge fetch      # both datasets at pinned commits, about 4 GB
myobridge bench      # writes results/*.json and docs/figures/*.png, about 25 minutes
pytest tests/test_docs.py
```

The last line checks that every number quoted in the docs matches the
fresh results.
