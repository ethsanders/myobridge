# Changelog

All notable changes to myobridge. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

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
