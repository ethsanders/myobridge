# NEXT: `myobridge report` (branch `report`)

## Done

- `myobridge report SESSION.npz [...] [--model M] [--gain G] [--adc-full-scale V] [--rail R] [--no-figures] --out DIR`
  writes report.md, report.json and traces/spectra/gesture_rms/confusion PNGs (`src/myobridge/report.py`).
- Link stats: `Receiver.arrivals` (`ArrivalLog`), `Receiver.invalid`, `Receiver.link_summary()`,
  `LinkStats.gaps`/`longest_gap`; saved by `calibrate --session` (`Session.link`, meta JSON) and `record`.
  Old session and record files still load.
- `sessions.adapt_to()` (decoder half of `recalibrate`).
- docs/first-session.md (in nav), docs/api/report.md, CHANGELOG, README Limitations line,
  pointer in hardware-bringup.md.
- tests/test_report.py and CLI tests in tests/test_cli.py.
- Fixed a pre-existing E501 in classifier.py from 9512cee so `ruff check` passes.
- ruff check, ruff format --check, mypy src, mkdocs build --strict: pass.
  Full pytest: 375 passed, 1 skipped, 1 failed: `test_bench.py::test_conformal_quick`,
  which fails at the base commit 9512cee too (conformal_bench emits `max_prob_rejection_nested`;
  that file belongs to another session).

## Left

- The MyoWare RAW gain has no default (not in the repo); the owner passes `--gain` from the datasheet.
- Real hardware sessions, then copy the "For the README" lines into README.
- Delete this file before merging.
