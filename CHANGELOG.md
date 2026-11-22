# Changelog

All notable changes to myobridge. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `myobridge sets --gate [CUTOFF]`: shows the confidence gate's action next to
  the sets and, with `--prompt`, scores it live: how often it acted, how
  often it was right, and wrong actions per minute. In code:
  `LiveSets(policy=...)` and `Score.act`.
- `myobridge report`: sessions from the armband in, a hardware report out
  (report.md, report.json and figures). Raw traces, the rest noise floor
  in counts, millivolts at the ADC and, with `--gain`, microvolts at the
  electrodes, mains hum, clipping, what the quality monitor flagged
  (including a deliberately lifted electrode and a mains lead near the
  cables), spectra, RMS per gesture, time-blocked and across-session
  accuracy with and without `adapt()`, and link quality. In code:
  `myobridge.report`. Guide: "The first recorded session".
- Link stats are kept: `calibrate --session` saves the receiver's packet
  loss, gaps, CRC errors and jitter in the session (`Session.link`), and
  `record` saves them in its .npz. Older files still load.
  `Receiver.link_summary()`, `Receiver.arrivals` (`ArrivalLog`),
  `Receiver.invalid`, and `LinkStats.gaps` and `longest_gap`.
- `sessions.adapt_to()`: the decoder half of `recalibrate()`.
- `myobridge.gating`: decision policies that say, window by window, whether
  to act and on which gesture. `VotePolicy` (the pipeline's vote),
  `ConfidenceGate`, `SetGate` (conformal sets) and `EvidenceGate` (leaky
  evidence accumulation), each with an optional latch that keeps the last
  action instead of holding with nothing. `GesturePipeline(policy=...)` adds
  the decision to every `Prediction` as `prediction.action`; without a
  policy nothing changes.
- `myobridge bench decision`: wrong actions per minute, time to the right
  action and time spent acting, on holds played back to back, with every
  setting chosen on validation data. On MultiDay, `ConfidenceGate(0.7, k=1,
  latch=True)` makes 14.0 wrong actions per minute against 25.6 for the
  5-window vote, at the same speed. On 3DC the gain is smaller. Guide:
  "Acting on it".
- `myobridge.selective`: answer only when sure, with an accuracy promise
  ("when it names a gesture, it's right 97% of the time").
  `SelectiveGestures.calibrate()` picks a confidence cut-off with Learn then
  Test, so the promise holds with probability 1 - delta on data like the
  held-out windows; `predict()` returns the gesture or None. Scores: top
  probability, top-two margin, or a one-gesture APS set. `grid_from()`
  builds the cut-offs to test from other held-out windows.
- `myobridge bench selective`: whether that promise holds on 91 later days
  (MultiDay), after a break and on new people (3DC), against the naive
  cut-off. Guide: "Acting only when sure".
- The recipe on your own armband. `myobridge calibrate --session day1.npz`
  keeps the raw recordings. `myobridge fit day1.npz day2.npz day3.npz` trains
  on the early sessions and fits the temperature and threshold on the later
  ones. `myobridge sets` recalibrates for today with 2 s per gesture and shows
  live sets; `--prompt` asks for gestures and scores the sets against the
  promise, and `--web` shows it all in large type in a browser. In code:
  `Session`, `fit_sets()`, `recalibrate()` and `myobridge.live_sets`. Guide:
  "Your own 90% promise".
- `myobridge.conformal`: gesture *sets* with a coverage promise ("one of
  these, 90% of the time"), so an application can tell a sure decision from
  a guess. `ConformalGestures.calibrate()`, `fit_temperature()`,
  `blocked_scores()`, `set_metrics()` and `AdaptiveAlpha`.
  `GesturePipeline(conformal=...)` adds the set to every `Prediction`
  (`prediction.gesture_set`, `prediction.sure`), and keeps working when the
  monitor drops a channel.
- `myobridge bench conformal`: whether that promise holds across 91 later
  days (MultiDay), a 5 minute break and new people (3DC), and electrode
  faults, with a recipe that does.
- The demo opens on the promise: four MultiDay days played back with live
  gesture sets and a running score, the textbook recipe and a lifted
  electrode a click away. `myobridge bench demo` writes what it plays
  (results/demo.json: decoded sets only, no EMG, 30 KB gzipped). The live
  3DC part now shows sets too, with a switch for the quality monitor.
- `Replay(alpha=...)` gives every window a conformal set and counts how the
  sets do; `Replay.drop_flagged()` turns the monitor's channel dropping off
  and on.
- `bench conformal` also measures what `GesturePipeline` does when an
  electrode fails (keeps the full model's threshold): 90.2% at 90% with an
  electrode off.
- `myobridge bench partial_calibration`: whether a day can be recalibrated
  from rest alone, or rest plus a gesture or two (mostly it can't).
- A richer default feature set, `td-ar-psd`: the Hudgins time-domain
  features plus 4th-order autoregressive coefficients and the TD-PSD
  spectral moments, 15 per channel. The old 5-feature set is still
  available as `feature_set="td"`, and models saved before this release load
  as `td`.
- `Replay`: streams recorded EMG through the live pipeline and injects
  faults on the fly. It powers the new browser demo, which runs the library
  itself in the page via Pyodide.
- `GesturePipeline(monitor=...)`: the pipeline can watch signal quality and
  stop using a channel that goes flat or clips, until it recovers.
- `LDAClassifier.subset()`: the model you'd get by retraining without some
  features, computed instantly.
- `TrainingSettings` and `GesturePipeline.from_model()`: a calibrated model
  remembers its sample rate, window and thresholds, and a saved file keeps
  them.
- `myobridge.faults`: the simulated electrode faults, shared by the demo and
  the benchmarks.
- Top-level imports for the public API (`from myobridge import ...`).
- A documentation site with guides and an API reference, and runnable
  examples in `examples/`.
- Benchmarks comparing feature sets and classifiers (LDA, SVM, random
  forest, MLP).
- Lab Streaming Layer output: `myobridge forward --lsl` and
  `myobridge.LslOutlets` publish the raw EMG and the decoded gestures
  (`pip install "myobridge[lsl]"`).
- `Receiver.record(timeout_s=...)` and `Receiver.drain()`;
  `myobridge --version`; `--mains` can follow the subcommand.

### Changed

- The LDA ridge is now relative to each feature's own variance, so the
  model doesn't depend on feature units.
- Streaming windowing is vectorised.
- `myobridge forward` sets the decoder up exactly as the model was trained,
  and refuses, with a way out, a device running at a different sample rate.
- The CLI reports problems in plain words instead of tracebacks, and says
  when it is waiting for packets.

### Fixed

- The 3DC citation now points to the paper that introduced the dataset
  (Côté-Allard et al., *Sensors* 2019). The docs credit Vidovic et al.
  (2016) for the method behind `adapt()` and Zhang & Huang (2015) for
  `subset()`.
- Packets are validated field by field when encoded and decoded.
- A device restart is told apart from a late packet by its timestamp, and
  duplicates are counted.
- The receiver no longer shares its port with another process and closes
  its socket on every path.
- `EmgFilter` handles empty blocks and caps its band below Nyquist, so
  slower devices work; so does the simulator.
- The firmware counts failed UDP sends; the halfband filter comments now
  match the filter.

## [0.1.0] - 2026-11-19

First public version: ESP32 firmware streaming 4 channels over UDP, packet
protocol with CRC-16 and loss tracking, streaming filters, time-domain
features, a numpy LDA with `adapt()` recalibration, a per-channel signal
quality monitor, pinned loaders for the 3DC and MultiDay datasets, and the
benchmarks behind every number in the docs.

[Unreleased]: https://github.com/ethsanders/myobridge/compare/06d11b5...HEAD
[0.1.0]: https://github.com/ethsanders/myobridge/commit/06d11b5
