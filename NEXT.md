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
- Guide `docs/selective.md`, with its numbers checked by `tests/test_docs.py`.
- `classifier.py`: docstring reflow only, to fix an existing ruff E501.
- `test_conformal_quick` already failed at 9512cee: it did not allow the
  `max_prob_rejection_nested` key that `conformal_bench.py` writes. The test
  now allows it; `conformal_bench.py` is unchanged.

## Left

- Nothing in the brief. `docs/selective.md` and `test_selective` in
  `tests/test_docs.py` are written. ruff, ruff format --check, mypy and
  mkdocs --strict pass. The full pytest passes (377 passed, 1 skipped,
  about 15 min); `test_selective_quick` takes 87 s.
- Delete this file before merging.

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
