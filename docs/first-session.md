# The first recorded session

Every accuracy in the README comes from public datasets recorded with other
hardware. This page is the protocol for the first sessions on myobridge's own
armband, and `myobridge report` turns them into evidence in one command: raw
traces, the noise floor, mains hum, the quality monitor at work, gestures
decoded within a sitting and across days, and how the WiFi link behaved.

Do the [bring-up checklist](hardware-bringup.md) first. This page assumes
the link works and every channel reads `ok`.

## Try it with the simulator

The whole flow runs without hardware, so you can see the report before the
first real session. The numbers mean nothing: the simulator is not EMG, and
its mains hum is strong on purpose.

```bash
myobridge calibrate --sim --gestures rest,fist,open --out sim-model.npz --session sim1.npz
myobridge calibrate --sim --seed 1 --gestures rest,fist,open --out sim-model.npz --session sim2.npz
myobridge report sim1.npz sim2.npz --out sim-report
```

## Before you start

- **Battery only.** The ESP32 and sensors run from a battery whenever
  electrodes touch skin. Unplug the USB cable before the electrodes go on.
- **Check the link.** Run `myobridge listen` (`--mains 50` in Europe).
  Loss should be under 1%, and every channel `ok` at rest and in a fist.
- **Mark the skin** around each electrode with a pen, so later days can put
  them back in the same spots. Write down which muscle each channel sits on.
- **Take a photo** of the armband on the arm. The README has none yet.
- **Find the sensor's gain.** The MyoWare's gain from the electrodes to its
  RAW output isn't written down in this repository. Look it up in the
  datasheet for your board revision. The report needs it for microvolts;
  without it, you get ADC counts and millivolts at the ADC pin.

## Day 1: what to record, in order

About 15 minutes once the electrodes are on. Sit with the forearm resting on
a table.

**1. Gestures.** Rest first, then each gesture held for 5 seconds:

```bash
myobridge calibrate --gestures rest,fist,open,point,pinch,flex \
    --seconds 5 --out day1-model.npz --session day1.npz
```

Relax fully during `rest`: it sets each channel's noise floor and the
reference for the flatline check. Hold each gesture firmly but not at full
strength, the same way every time.

**2. Two deliberate faults.** A second session with 10 second holds:

```bash
myobridge calibrate --gestures rest,lifted,mains \
    --seconds 10 --out checks-model.npz --session day1-checks.npz
```

- `rest`: relaxed, as before.
- `lifted`: before pressing Enter, peel one electrode of channel 0 off the
  skin. Then hold a gentle fist. Press it back on afterwards.
- `mains`: relaxed, with a plugged-in lamp cord laid across the sensor
  cables. Nothing mains-powered touches the circuit; the armband stays on
  its battery.

The report knows these two names. It keeps them out of the gesture results
and shows which checks the quality monitor raised on each channel. A pass
is a flag on channel 0 during `lifted` and none on the others, and
`line_noise` on the channels near the cord during `mains`. The model this
writes isn't used; delete `checks-model.npz`.

**3. Free use.** Three minutes of normal movement: open and close the hand,
type, reach for things, rest now and then.

```bash
myobridge record --seconds 180 --out day1-free.npz
```

This shows how often the monitor raises a false alarm in real use, and
measures the link over minutes rather than seconds.

**4. The report.**

```bash
myobridge report day1.npz day1-checks.npz day1-free.npz --gain G --out report-day1
```

Replace `G` with the datasheet gain, or leave `--gain` out.

## Days 2 and 3

Take the armband off between days and put it back on the marks. On each
day, run step 1 only, with `day2` or `day3` in the file names. Leave at
least a day between sessions. Then:

```bash
myobridge report day1.npz day2.npz day3.npz day1-checks.npz day1-free.npz \
    --gain G --out report
myobridge fit day1.npz day2.npz day3.npz --out model.npz
```

Give the gesture sessions oldest first: the across-day result trains on all
but the last and tests on the last. The `fit` step is
[your own 90% promise](your-promise.md), from the same sessions.

## What the report gives back

`report/report.md` with its figures beside it, and `report/report.json` with
every number. The Markdown has:

- **Recordings**: rate, channels, mains frequency, every hold's length, when
  it was recorded, and whether link stats were kept.
- **Signal chain**: how counts become millivolts at the ADC and microvolts at
  the electrodes, and every assumption behind it. If a number in the chain
  is wrong, `--gain` and `--adc-full-scale` fix it.
- **Raw signal**: half a second of every channel at rest and during the
  strongest gesture.
- **Noise floor at rest**: 20-450 Hz RMS per channel, in counts, millivolts
  and microvolts.
- **Mains hum, clipping and flatline**: hum as a share of the band at rest,
  the share of samples at the ADC's rail, and how often the flatline check
  fired during gestures (it should be near 0).
- **Deliberate faults** and **Free use**: what the quality monitor flagged.
- **Gestures**: the spectrum of each channel at rest and in gestures, RMS per
  gesture and channel, and the accuracy of LDA on the library's own
  features. Within a sitting, the first 2.5 seconds of every hold train and
  the rest tests. With several sessions, also each session decoded by a
  model from the others, and the last day decoded by a model from the
  earlier days, before and after `adapt()` with one second per gesture.
- **Link**: packets lost, how many gaps and the longest, CRC errors, late
  and duplicate packets, restarts, and jitter. The two clocks aren't shared,
  so jitter is measured, the delay itself isn't.

`myobridge report --model model.npz` also scores a saved model on every
session, with its gesture sets if `fit` made it.

## Which numbers go in the README

The report ends with a few sentences under "For the README". These are the
ones to quote, with the date and the number of sessions:

- the rest noise floor (in microvolts once the gain is known) and the mains
  hum at rest
- accuracy within one sitting, with the number of gestures and chance
- the across-day result, before and after `adapt()`: the hardware version of
  the MultiDay result
- packet loss and the 99th-percentile extra delay

Put them in the "How well it works" table as rows marked "own armband", add
the photo and the raw-trace figure, and replace "Not yet run on my own
armband" under Limitations. They are measured on one arm with fewer gestures
than the datasets, so they don't compare one-to-one with the 11-gesture
numbers. Copy the noise floor into the
[bring-up results log](hardware-bringup.md#results-log) too.
