# Getting started

This page covers installing myobridge and running the decoder on simulated
data, on recorded data from a public dataset, and on a live armband.

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

This example needs no data or hardware. The simulator stands in for the
armband and produces the same 4-channel, 1 kHz stream of ADC counts.

```python
--8<-- "examples/decode_simulated.py"
```

```text
trained on 460 windows of 60 features each
holding   fist: decoded fist in 37 of 37 decisions
holding   rest: decoded rest in 34 of 40 decisions
holding extend: decoded extend in 35 of 40 decisions
```

`calibrate` takes any function that records a gesture for a given number
of seconds. If the simulator is replaced with the UDP receiver, the same code
calibrates on a real arm.

`GesturePipeline.from_model` builds the live pipeline with the sample rate,
window and thresholds the model was trained with, so the training and live
settings cannot diverge. A model saved with `model.save("model.npz")` stores
these settings as well.

The simulator produces clean, band-limited noise, so this example only shows
that the code runs from end to end. It says nothing about accuracy on real
muscles, which the next section tests.

## Decode a real forearm

The [3DC dataset](datasets.md) has 22 people doing 11 hand and wrist
gestures with a 10-electrode armband. This script calibrates on one
person's first two passes through the gestures, then streams a later pass
through the live pipeline, 20 samples at a time. Half-way through, it
simulates an electrode coming off.

```bash
myobridge fetch 3dc      # once; clones the dataset at a pinned commit (about 1.3 GB)
python examples/replay_3dc.py --person 7
```

```python
--8<-- "examples/replay_3dc.py"
```

The [live demo](demo/index.html) runs the same code in the browser and has a
menu for disconnecting any electrode.

## Run it live

These commands work with the ESP32 armband (see
[Build the armband](hardware.md)) or with the simulator sending packets over
UDP in its place.

```bash
myobridge simulate --port 5005 --drop 0.01       # skip this with the real armband
myobridge listen --port 5005                     # live RMS bars, channel health, link stats
myobridge calibrate --seconds 5 --out model.npz  # hold each gesture when asked
myobridge forward --model model.npz --to 127.0.0.1:5006 --verbose
```

`forward` sends one JSON message per decision to the given address, where a
game engine can read it. The message format is described in
[Send gestures to a game](protocol.md).

With `--lsl`, `forward` also publishes two streams on
[Lab Streaming Layer](https://labstreaminglayer.org), which LabRecorder and
most neuroscience tools can read. The `myobridge-emg` stream carries the raw
samples, and the `myobridge-gestures` stream carries one marker per decision.
This option needs `pip install "myobridge[lsl]"`.

Outside North America, put `--mains 50` before the subcommand, for example
`myobridge --mains 50 listen`.

## Reproduce the results

The numbers in [Results](results.md) are produced by `myobridge bench`:

```bash
myobridge fetch      # both datasets at pinned commits, about 4 GB
myobridge bench      # writes results/*.json and docs/figures/*.png, about 25 minutes
pytest tests/test_docs.py
```

The last command checks that the numbers quoted in the docs match the new
results.
