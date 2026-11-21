"""Answering only when sure: the cut-off, its promise, and its storage."""

from __future__ import annotations

import json

import numpy as np
import pytest

from myobridge import ConformalGestures, LDAClassifier, SelectiveGestures
from myobridge.selective import (
    METHODS,
    SCORES,
    accept_matrix,
    binomial_pvalues,
    clopper_pearson_lower,
    default_grid,
    grid_from,
    learn_then_test,
)


class Fixed:
    """A model whose decision scores are the rows it is given."""

    def __init__(self, k: int = 3):
        self._classes = np.array([f"g{i}" for i in range(k)])

    @property
    def classes(self):
        return self._classes

    def decision_function(self, X):
        return np.asarray(X, dtype=np.float64)


def calibrated(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Windows whose top probability s ~ U(1/3, 1) is right with probability s.

    The accuracy of the answers at a cut-off c >= 1/3 is then exactly
    (1 + c) / 2, so whether a cut-off keeps a promise can be checked exactly.
    """
    s = rng.uniform(1 / 3, 1, n)
    p = np.column_stack([s, (1 - s) / 2, (1 - s) / 2])
    right = rng.random(n) < s
    y = np.where(right, 0, rng.integers(1, 3, n))
    return np.log(p), y, s


# fixed before seeing any data, strict to loose, where calibrated() has windows
GRID = np.linspace(0.95, 0.35, 25)


def true_accuracy(cutoff: float | None) -> float:
    if cutoff is None:
        return 1.0  # never answering never answers wrongly
    return (1 + max(cutoff, 1 / 3)) / 2


# ------------------------------------------------------------- the promise


@pytest.mark.parametrize("method", METHODS)
def test_promise_holds_on_exchangeable_data(method):
    """Over many held-out draws, the chosen cut-off misses the target at most
    delta of the time (plus Monte Carlo slack), while the naive cut-off misses
    it much more often."""
    rng = np.random.default_rng(0)
    target, delta, draws = 0.9, 0.1, 300
    grid = GRID
    misses, naive_misses, rates = 0, 0, []
    for _ in range(draws):
        d, y, _ = calibrated(1500, rng)
        sel = SelectiveGestures.from_scores(Fixed(), d, y, target, delta, method=method, grid=grid)
        misses += true_accuracy(sel.cutoff) < target
        rates.append(sel.answer_rate)
        # naive: the loosest grid cut-off whose held-out answers were right enough
        acc = accept_matrix(np.exp(d), grid)
        hit = (acc & (y == 0)[:, None]).sum(0)
        ok = np.nonzero(hit >= target * acc.sum(0))[0]
        naive_misses += true_accuracy(float(grid[ok[-1]])) < target
    assert misses / draws <= delta + 3 * np.sqrt(delta * (1 - delta) / draws)
    assert naive_misses / draws > 0.3
    # the best possible rule answers on s >= 0.8, 30% of the windows
    assert 0.1 < np.mean(rates) < 0.3


def test_margin_promise_holds_too():
    rng = np.random.default_rng(1)
    misses = 0
    for _ in range(200):
        d, y, _ = calibrated(1500, rng)
        margins = np.linspace(0.9, 0.0, 25)
        sel = SelectiveGestures.from_scores(Fixed(), d, y, 0.9, 0.1, score="margin", grid=margins)
        # margin = (3s - 1) / 2, so a margin cut-off m is the s cut-off (2m + 1) / 3
        cut = None if sel.cutoff is None else (2 * sel.cutoff + 1) / 3
        misses += true_accuracy(cut) < 0.9
    assert misses / 200 <= 0.1 + 3 * np.sqrt(0.09 / 200)


# --------------------------------------------------------------- monotonicity


def test_stricter_promises_give_stricter_cutoffs():
    d, y, _ = calibrated(3000, np.random.default_rng(2))
    for method in METHODS:
        cuts, rates = [], []
        for target in (0.7, 0.8, 0.85, 0.9):
            sel = SelectiveGestures.from_scores(
                Fixed(), d, y, target, 0.1, method=method, grid=GRID
            )
            cuts.append(sel.cutoff)
            rates.append(sel.answer_rate)
        assert all(c is not None for c in cuts)
        assert cuts == sorted(cuts), method
        assert rates == sorted(rates, reverse=True), method
        loose = SelectiveGestures.from_scores(Fixed(), d, y, 0.9, 0.3, method=method, grid=GRID)
        tight = SelectiveGestures.from_scores(Fixed(), d, y, 0.9, 0.01, method=method, grid=GRID)
        assert loose.cutoff <= tight.cutoff


def test_fallback_survives_a_grid_that_starts_too_strict():
    d, y, _ = calibrated(5000, np.random.default_rng(8))
    grid = np.r_[1 - 1e-9, GRID]  # nothing is ever that sure
    fixed = SelectiveGestures.from_scores(Fixed(), d, y, 0.9, method="fixed_sequence", grid=grid)
    fallback = SelectiveGestures.from_scores(Fixed(), d, y, 0.9, method="fallback", grid=grid)
    assert fixed.abstains
    assert not fallback.abstains and fallback.cutoff >= 0.8 - 0.03


def test_grid_from_other_windows():
    d, _, s = calibrated(2000, np.random.default_rng(9))
    g = grid_from(np.exp(d))
    assert np.all(np.diff(g) < 0)
    assert np.mean(s >= g[0]) == pytest.approx(0.01, abs=0.002)
    assert np.mean(s >= g[49]) == pytest.approx(0.5, abs=0.002)
    assert np.all(s >= g[-1])
    ties = grid_from(np.array([[1.0, 0.0]] * 10 + [[0.6, 0.4]] * 10), "margin", [0.1, 0.5, 1.0])
    assert ties.tolist() == pytest.approx([1.0, 0.2])
    with pytest.raises(ValueError):
        grid_from(np.exp(d), "aps_singleton")
    with pytest.raises(ValueError):
        grid_from(np.exp(d), rates=[0.0, 0.5])


def test_answers_nest_as_the_cutoff_loosens():
    d, _, _ = calibrated(500, np.random.default_rng(3))
    grid = default_grid(5)
    for score in ("max_prob", "margin"):
        acc = accept_matrix(np.exp(d), grid, score)
        assert np.all(acc[:, 1:] >= acc[:, :-1])
        assert acc[:, -1].all()  # a cut-off of 0 answers everything


def test_grid_runs_strict_to_loose():
    g = default_grid(10, 1e-6)
    assert len(g) == 61
    assert g[0] == pytest.approx(1 - 1e-6) and g[-1] == 0.0
    assert np.all(np.diff(g) < 0)
    with pytest.raises(ValueError):
        default_grid(0)


# ----------------------------------------------------------- building blocks


def test_binomial_pvalues():
    p = binomial_pvalues([0, 10, 100], [0, 0, 1], 0.05)
    assert p[0] == 1.0
    assert p[1] == pytest.approx(0.95**10)
    assert p[2] == pytest.approx(0.95**100 + 100 * 0.05 * 0.95**99)


def test_learn_then_test_procedures():
    p = np.array([0.5, 0.01, 0.01, 0.2, 0.001])
    # fixed sequence stops at the strictest cut-off, which can't be certified
    assert learn_then_test(p, 0.1, "fixed_sequence") is None
    assert learn_then_test(p[1:], 0.1, "fixed_sequence") == 1
    # fallback: 0.02 each, plus the level carried from a rejected predecessor
    # (0.01 <= 0.02 at 1; 0.01 <= 0.04 at 2; 0.2 > 0.06 at 3; 0.001 <= 0.02 at 4)
    assert learn_then_test(p, 0.1, "fallback") == 4
    assert learn_then_test(p, 0.1, "bonferroni") == 4
    assert learn_then_test([0.03, 0.03], 0.1, "bonferroni") == 1
    assert learn_then_test([0.03, 0.06], 0.1, "bonferroni") == 0
    assert learn_then_test([], 0.1) is None
    with pytest.raises(ValueError):
        learn_then_test(p, 0.1, "holm")
    with pytest.raises(ValueError):
        learn_then_test(p, 1.5)


def test_clopper_pearson_lower():
    assert clopper_pearson_lower(0, 10, 0.1) == 0.0
    assert clopper_pearson_lower(10, 10, 0.1) == pytest.approx(0.1 ** (1 / 10))
    assert clopper_pearson_lower(90, 100, 0.1) < 0.9


def test_from_scores_matches_the_building_blocks():
    d, y, _ = calibrated(3000, np.random.default_rng(4))
    grid = GRID
    sel = SelectiveGestures.from_scores(Fixed(), d, y, 0.9, 0.1, grid=grid)
    assert sel.method == "fallback"
    acc = accept_matrix(np.exp(d), grid)
    n, e = acc.sum(0), (acc & (y != 0)[:, None]).sum(0)
    j = learn_then_test(binomial_pvalues(n, e, 0.1), 0.1, "fallback")
    assert sel.cutoff == grid[j]
    assert sel.answer_rate == n[j] / len(y)
    assert sel.calibration_accuracy == (n[j] - e[j]) / n[j]
    assert sel.calibration_accuracy >= sel.accuracy_bound
    assert sel.n_calibration == 3000


def test_aps_singleton_answers_when_the_set_has_one_gesture():
    d, y, _ = calibrated(400, np.random.default_rng(5))
    tau = 0.8
    sets = ConformalGestures(Fixed(), np.full(3, tau), 0.1, score="aps", seed=7)
    one = sets.predict_sets(d).sum(axis=1) == 1
    assert 0 < one.mean() < 1

    def fresh():
        return SelectiveGestures(Fixed(), tau, 0.9, score="aps_singleton", seed=7)

    assert np.array_equal(fresh().accept(d), one)
    got = fresh().predict(d)
    assert all((g == "g0") if o else g is None for g, o in zip(got, one, strict=True))


# ------------------------------------------------------------ the wrapper


def test_never_answers_when_nothing_can_be_certified():
    d = np.log(np.array([[0.8, 0.1, 0.1]] * 50))
    y = np.ones(50, dtype=int)  # always wrong
    sel = SelectiveGestures.from_scores(Fixed(), d, y, 0.95)
    assert sel.abstains and sel.cutoff is None
    assert sel.answer_rate == 0.0 and np.isnan(sel.calibration_accuracy)
    assert sel.predict(d[:3], unsure="?") == ["?"] * 3
    assert not sel.accept(d).any()
    # too few windows to certify anything, even when every one is right
    few = SelectiveGestures.from_scores(Fixed(), d[:5], np.zeros(5, dtype=int), 0.95)
    assert few.abstains


LOOSE = np.linspace(0.99, 0.0, 34)  # a short grid for a few hundred windows


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(6)
    X = np.vstack([rng.normal(c, 0.7, size=(60, 2)) for c in range(3)])
    y = np.repeat(["a", "b", "c"], 60)
    model = LDAClassifier().fit(X, y)
    Xc = np.vstack([rng.normal(c, 0.7, size=(200, 2)) for c in range(3)])
    yc = np.repeat(["a", "b", "c"], 200)
    for score in SCORES:
        sel = SelectiveGestures.calibrate(
            model, Xc, yc, 0.8, 0.1, temperature=2.0, score=score, grid=LOOSE
        )
        assert not sel.abstains
        path = sel.save(tmp_path / f"{score}.json")
        back = SelectiveGestures.load(path, model)
        assert back.to_dict() == sel.to_dict()
        assert back.predict(Xc) == sel.with_model(model).predict(Xc)
    gone = SelectiveGestures(model, None, 0.97)
    gone.save(tmp_path / "none.json")
    assert json.loads((tmp_path / "none.json").read_text())["cutoff"] is None
    assert SelectiveGestures.load(tmp_path / "none.json", model).abstains
    other = LDAClassifier().fit(X, np.repeat(["a", "b", "x"], 60))
    with pytest.raises(ValueError):
        SelectiveGestures.load(path, other)
    with pytest.raises(ValueError):
        sel.with_model(other)


def test_answers_are_the_top_gesture_and_unsure_otherwise():
    rng = np.random.default_rng(7)
    X = np.vstack([rng.normal(c, 0.6, size=(300, 2)) for c in range(3)])
    y = np.repeat(["a", "b", "c"], 300)
    model = LDAClassifier().fit(X[::2], y[::2])
    sel = SelectiveGestures.calibrate(model, X[1::2], y[1::2], 0.9, grid=LOOSE)
    pred = sel.predict(X)
    named = np.array([p is not None for p in pred])
    assert np.array_equal(named, sel.accept(X))
    assert 0 < named.mean() < 1
    top = model.predict(X)
    assert all(p == t for p, t in zip(pred, top, strict=True) if p is not None)
    assert np.mean(np.array(pred)[named] == y[named]) > 0.9


def test_bad_arguments():
    with pytest.raises(ValueError):
        SelectiveGestures(Fixed(), 0.5, 0.9, score="entropy")
    with pytest.raises(ValueError):
        SelectiveGestures(Fixed(), 0.5, 0.9, method="holm")
    with pytest.raises(ValueError):
        SelectiveGestures(Fixed(), 0.5, 1.0)
    with pytest.raises(ValueError):
        SelectiveGestures.from_scores(Fixed(), np.zeros((3, 3)), [0, 1], 0.9)
    with pytest.raises(ValueError):
        accept_matrix(np.ones((2, 3)) / 3, [0.5], "entropy")
    assert SelectiveGestures(Fixed(), float("inf"), 0.9).abstains
