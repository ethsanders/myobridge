# Selective gestures: where this branch stands

## Done

- `src/myobridge/selective.py`: `SelectiveGestures` (Learn then Test with
  binomial p-values; fixed-sequence, fallback (default) and Bonferroni;
  scores max_prob, margin, aps_singleton; `default_grid()`, `grid_from()`;
  save/load). Exported from `myobridge`.
- `src/myobridge/bench/selective_bench.py`, registered as `selective`.
- Tests: `tests/test_selective.py`, `test_selective_quick` in
  `tests/test_bench.py`, `tests/test_public_api.py`.
- API docs entry, mkdocs nav entry, CHANGELOG entry.
- Full run done; results in `results/selective.json`.
- `classifier.py`: docstring reflow only, to fix an existing ruff E501.

## Left

- Write `docs/selective.md` (the nav already points to it, so
  `mkdocs build --strict` fails until it exists), and add a docs-number
  test to `tests/test_docs.py`.
- Run ruff, ruff format --check, mypy, the full pytest and
  mkdocs --strict. Rerun `test_selective_quick`, which has not run since the
  last `_rule_for` change.
- Final commit and report.

## Validation-only numbers (MultiDay days 11-30)

Mean answer rate over max_prob and margin at both targets:

| Procedure | Mean answer rate |
|---|---|
| fixed_sequence, a-priori grid | 0.437 |
| fallback | 0.873 |
| bonferroni | 0.837 |
| fixed_sequence, held-out grid | 0.881 |
| fallback, held-out grid | 0.877 |

The procedure chosen was fixed_sequence on the held-out grid. With a
temperature, fixed_sequence on the a-priori grid never answers, which is why
fallback is the library default.

On 3DC persons 4-6, fallback on the held-out grid was chosen.
