# Decisions work: status

## Done
- `src/myobridge/gating.py`: VotePolicy, ConfidenceGate, SetGate, EvidenceGate,
  optional `latch`; `GesturePipeline(policy=...)` -> `Prediction.action`.
- `src/myobridge/bench/decision_bench.py`, registered as `decision`.
  Holds are played back to back per day/person, so every hold begins as a
  gesture change. Played one at a time, every vote size looked equally fast
  on validation (first MultiDay window nearly always right).
- Full run: `results/decision.json` (230 settings, chosen on MultiDay days
  11-30 / 3DC persons 1-6, matched to the 5-window vote's speed and time on
  the right gesture).
- Tests: tests/test_gating.py, decision tests in tests/test_bench.py,
  test_decisions in tests/test_docs.py.
- Docs: docs/decisions.md ("Acting on it"), nav, API entry, results.md
  section 9, CHANGELOG, index.md bullet.
- ruff check, ruff format, mypy, mkdocs build --strict passed.

## Left
- Confirm the whole pytest suite passes (was running at this commit).
- Final commit without "WIP:".

## Exploratory numbers (validation days only, before the chaining fix
## was added the vote sizes all looked equally fast)
- MultiDay days 11-30, chained holds: vote k5 19.3 wrong/min from 2.5 s,
  ttc 0.116 s; ConfidenceGate(0.7, k=1, latch) 11.3 wrong/min, ttc 0.101 s.
- 3DC persons 1-6 within sitting: vote k5 12.8 wrong/min; latched gates ~10.
