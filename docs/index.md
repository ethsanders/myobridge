# myobridge

**It tells you when it doesn't know, and that promise holds across 91 days.**
A decoded gesture comes with a set that holds the right one 90% of the time,
and on one person over months, with the electrodes put back on every day, it
delivers 89.1% ([how](conformal.md)).

![One recorded day: the decoder's sets, the textbook recipe, and an electrode coming off](figures/sets.gif)

myobridge turns the electrical activity of forearm muscles into hand
gestures, live. It covers the whole path: firmware that streams four EMG
sensors from an ESP32 over WiFi, a Python library that cleans the signal and
decodes gestures in about a tenth of a millisecond per packet, and
benchmarks on public datasets that say how well it works and where it
breaks.

![A real recording decoded live: one electrode comes off and the decoder carries on without it](figures/demo.gif)

*Person 7 of the 3DC dataset, four electrodes, decoded by the same code that
runs live. Halfway through, one electrode is pulled off in software; the
quality monitor drops it and the decoder keeps going on the other three.*

<div class="grid cards" markdown>

- **[Try the live demo](demo/index.html)**

    Watch the promise hold, break and recover on real recordings, then break
    an electrode yourself. Nothing to install.

- **[Get started](getting-started.md)**

    Install the library and decode a simulated arm, a real one from a
    public dataset, or your own.

- **[Build the armband](hardware.md)**

    Four MyoWare sensors, an ESP32 and a battery. Wiring, firmware and the
    first checks.

- **[See the results](results.md)**

    Accuracy across days, people and electrode counts, what one second of
    recalibration buys, and what a loose electrode costs.

</div>

## What's in it

- **Firmware** for the ESP32: a timer-driven sampler, a queue to the WiFi
  sender so slow sends never disturb the sample clock, and optional 2x
  oversampling with a halfband filter.
- **A streaming pipeline**: band-pass and mains notch filters that carry
  their state between packets, 15 features per channel, LDA, and majority
  voting.
- **Recalibration** with `adapt()`: one second per gesture from a new day
  recovers most of what a model loses to drift.
- **Knowing when it doesn't know**: gesture sets that hold the right
  gesture a promised share of the time, and keep that promise across days
  after recalibration ([details](conformal.md)).
- **Decision policies** that act only when a window is confident enough and
  otherwise keep the last action, measured in wrong actions per minute
  ([details](decisions.md)).
- **A signal-quality monitor** that spots a lifted or clipping electrode
  within about a second and switches to a model without it.
- **Benchmarks** on two public datasets, with splits by recording, day and
  person, settings chosen on validation data, and every number in the docs
  checked against the results by the test suite.

```python
from myobridge import EmgSimulator, GesturePipeline, calibrate

arm = EmgSimulator(seed=0)
model, _, _ = calibrate(arm.record, ["rest", "fist", "open"], seconds=5, fs=arm.fs)
pipeline = GesturePipeline.from_model(model, n_channels=4)
for prediction in pipeline.process_block(arm.generate(20)):
    print(prediction.gesture, prediction.confidence)
```

## Status

The library, firmware and benchmarks are tested and working. Every accuracy
number so far comes from public datasets recorded with other hardware; the
first sessions on the myobridge armband itself are next, following
[the bring-up checklist](hardware-bringup.md).
