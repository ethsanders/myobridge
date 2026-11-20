"""Conformal gesture sets: the promise on exchangeable data, and the plumbing."""

from __future__ import annotations

import numpy as np
import pytest

from myobridge.classifier import LDAClassifier, adapt
from myobridge.conformal import (
    AdaptiveAlpha,
    ConformalGestures,
    blocked_scores,
    conformal_quantile,
    fit_temperature,
    nonconformity,
    set_metrics,
    softmax,
    temperature_for,
)


def blobs(n_per_class, rng, k=4, d=6, spread=1.6):
    """Overlapping Gaussian classes, so sets are sometimes bigger than one."""
    centers = np.random.default_rng(99).normal(0, spread, (k, d))
    X = np.vstack([rng.normal(c, 1.0, (n_per_class, d)) for c in centers])
    y = np.repeat(np.array([f"g{i}" for i in range(k)]), n_per_class)
    return X, y


@pytest.fixture(scope="module")
def model():
    X, y = blobs(400, np.random.default_rng(0))
    return LDAClassifier().fit(X, y)


# ------------------------------------------------------------ building blocks


def test_softmax_matches_lda_probabilities(model):
    X, _ = blobs(5, np.random.default_rng(1))
    np.testing.assert_allclose(softmax(model.decision_function(X)), model.predict_proba(X))
    p = softmax(model.decision_function(X), 7.0)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        softmax([[1.0, 2.0]], 0.0)


def test_softmax_is_safe_with_huge_scores():
    p = softmax([[1e6, 0.0, -1e6]])
    assert np.all(np.isfinite(p)) and p[0, 0] == pytest.approx(1.0)


def test_fit_temperature_recovers_a_known_temperature():
    rng = np.random.default_rng(2)
    logits = rng.normal(0, 2.0, (20000, 5))
    y = np.array([rng.choice(5, p=softmax(row[None])[0]) for row in logits])
    assert fit_temperature(logits * 8.0, y) == pytest.approx(8.0, rel=0.05)
    with pytest.raises(ValueError):
        fit_temperature(np.zeros((3, 2)), np.zeros(2, dtype=int))


def test_conformal_quantile_by_hand():
    s = np.arange(1, 10, dtype=float)  # n = 9
    assert conformal_quantile(s, 0.1) == 9.0  # ceil(10 * 0.9) = 9th smallest
    assert conformal_quantile(s, 0.5) == 5.0  # ceil(10 * 0.5) = 5th
    assert conformal_quantile(s, 0.05) == float("inf")  # needs the 10th of 9
    for bad in (0.0, 1.0, -0.1):
        with pytest.raises(ValueError):
            conformal_quantile(s, bad)


def test_lac_and_aps_scores():
    p = np.array([[0.6, 0.3, 0.1]])
    np.testing.assert_allclose(nonconformity(p, "lac"), [[0.4, 0.7, 0.9]])

    class Fixed:  # u = 0.5 for every row
        def random(self, shape):
            return np.full(shape, 0.5)

    aps = nonconformity(p, "aps", Fixed())  # type: ignore[arg-type]
    np.testing.assert_allclose(aps, [[0.3, 0.75, 0.95]])  # mass above + half of own
    with pytest.raises(ValueError):
        nonconformity(p, "nope")


def test_set_metrics_by_hand():
    sets = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 1]], dtype=bool)
    y = np.array([0, 1, 2, 0])
    m = set_metrics(sets, y)
    assert m["coverage"] == 0.5
    assert m["mean_set_size"] == 1.0
    assert m["singleton_rate"] == 0.5
    assert m["empty_rate"] == 0.25
    assert m["accuracy_when_singleton"] == 0.5


# ------------------------------------------------------------ the guarantee


@pytest.mark.parametrize("score", ["lac", "aps"])
@pytest.mark.parametrize("alpha", [0.05, 0.2])
def test_coverage_holds_on_exchangeable_data(model, score, alpha):
    """Averaged over many calibration draws, coverage is at least 1 - alpha and,
    for APS (randomized), not much more: 1 - alpha + 1/(n + 1) at most."""
    rng = np.random.default_rng(3)
    n_cal = 199
    covs = []
    for _ in range(300):
        Xc, yc = blobs(n_cal // 4 + 1, rng)
        keep = rng.permutation(len(yc))[:n_cal]
        Xt, yt = blobs(50, rng)
        cg = ConformalGestures.calibrate(model, Xc[keep], yc[keep], alpha, score=score, seed=None)
        sets = cg.predict_sets(Xt)
        covs.append(np.mean(sets[np.arange(len(yt)), np.searchsorted(cg.classes, yt)]))
    mean, se = float(np.mean(covs)), float(np.std(covs) / np.sqrt(len(covs)))
    assert mean >= 1 - alpha - 3 * se
    if score == "aps":
        assert mean <= 1 - alpha + 1 / (n_cal + 1) + 3 * se


def test_mondrian_covers_every_gesture(model):
    rng = np.random.default_rng(4)
    hits = np.zeros(4)
    counts = np.zeros(4)
    for i in range(80):  # fixed seeds, so the test can't flake
        Xc, yc = blobs(100, rng)
        Xt, yt = blobs(200, rng)
        cg = ConformalGestures.calibrate(model, Xc, yc, 0.1, mondrian=True, seed=i)
        sets = cg.predict_sets(Xt)
        y = np.searchsorted(cg.classes, yt)
        np.add.at(hits, y, sets[np.arange(len(y)), y])
        np.add.at(counts, y, 1)
    assert np.all(hits / counts >= 0.89)
    assert len(set(cg.thresholds.tolist())) > 1


def test_temperature_does_not_break_the_guarantee(model):
    rng = np.random.default_rng(5)
    Xc, yc = blobs(500, rng)
    Xt, yt = blobs(2000, rng)
    cg = ConformalGestures.calibrate(model, Xc, yc, 0.1, temperature=12.0)
    sets = cg.predict_sets(Xt)
    assert set_metrics(sets, np.searchsorted(cg.classes, yt))["coverage"] >= 0.88


def test_temperature_for_matches_fit_temperature_on_labels(model):
    rng = np.random.default_rng(9)
    X, y = blobs(300, rng)
    by_hand = fit_temperature(model.decision_function(X), np.searchsorted(model.classes, y))
    assert temperature_for(model, X, y) == pytest.approx(by_hand)
    with pytest.raises(ValueError, match="doesn't know"):
        temperature_for(model, X[:3], ["nope"] * 3)


# ------------------------------------------------------------ the wrapper


def test_predict_names_one_gesture_or_unsure(model):
    rng = np.random.default_rng(6)
    Xc, yc = blobs(200, rng)
    cg = ConformalGestures.calibrate(model, Xc, yc, 0.05)
    Xt, _ = blobs(100, rng)
    sets = ConformalGestures.calibrate(model, Xc, yc, 0.05).predict_sets(Xt)
    out = cg.predict(Xt, unsure="?")
    for row, label in zip(sets, out, strict=True):
        if row.sum() == 1:
            assert label == cg.classes[row.argmax()]
        else:
            assert label == "?"
    assert "?" in out and any(label != "?" for label in out)
    assert cg.coverage == pytest.approx(0.95)


def test_seed_makes_sets_repeatable(model):
    rng = np.random.default_rng(7)
    Xc, yc = blobs(100, rng)
    Xt, _ = blobs(50, rng)
    a = ConformalGestures.calibrate(model, Xc, yc, 0.1, seed=3).predict_sets(Xt)
    b = ConformalGestures.calibrate(model, Xc, yc, 0.1, seed=3).predict_sets(Xt)
    np.testing.assert_array_equal(a, b)


def test_too_little_calibration_data_gives_every_gesture(model):
    Xc, yc = blobs(2, np.random.default_rng(8))  # 8 rows: can't promise 95%
    cg = ConformalGestures.calibrate(model, Xc, yc, 0.05)
    assert np.all(cg.predict_sets(Xc))


def test_with_model_keeps_thresholds_after_adapt(model):
    rng = np.random.default_rng(9)
    Xc, yc = blobs(100, rng)
    cg = ConformalGestures.calibrate(model, Xc, yc, 0.1, temperature=3.0)
    moved = adapt(model, Xc + 0.5, yc)
    cg2 = cg.with_model(moved)
    assert cg2.model is moved
    np.testing.assert_array_equal(cg2.thresholds, cg.thresholds)
    assert cg2.temperature == 3.0
    other = LDAClassifier().fit(Xc[:, :3], np.where(yc == "g0", "a", "b"))
    with pytest.raises(ValueError):
        cg.with_model(other)


def test_calibrate_rejects_unknown_labels_and_bad_settings(model):
    X, _ = blobs(5, np.random.default_rng(10))
    with pytest.raises(ValueError):
        ConformalGestures.calibrate(model, X, ["zzz"] * len(X))
    with pytest.raises(ValueError):
        ConformalGestures(model, np.zeros(4), 0.1, score="nope")
    with pytest.raises(ValueError):
        ConformalGestures(model, np.zeros(3), 0.1)


def test_save_and_load_round_trip(model, tmp_path):
    rng = np.random.default_rng(11)
    Xc, yc = blobs(3, rng)  # too few for 95%: thresholds are inf
    cg = ConformalGestures.calibrate(model, Xc, yc, 0.05, temperature=2.5, mondrian=True)
    assert np.all(np.isinf(cg.thresholds))
    path = cg.save(tmp_path / "conformal.json")
    back = ConformalGestures.load(path, model)
    np.testing.assert_array_equal(back.thresholds, cg.thresholds)
    assert (back.alpha, back.temperature, back.score) == (0.05, 2.5, "aps")
    other = LDAClassifier().fit(Xc, np.where(yc == "g0", "a", "b"))
    with pytest.raises(ValueError):
        ConformalGestures.load(path, other)


def test_blocked_scores_match_fitting_by_hand():
    rng = np.random.default_rng(12)
    X, y = blobs(30, rng)
    blocks = np.tile(np.repeat([0, 1, 2], 10), 4)
    decision, yi, classes = blocked_scores(lambda A, b: LDAClassifier().fit(A, b), X, y, blocks)
    held = blocks == 1
    by_hand = LDAClassifier().fit(X[~held], y[~held]).decision_function(X[held])
    np.testing.assert_allclose(decision[held], by_hand)
    np.testing.assert_array_equal(classes[yi], y)
    with pytest.raises(ValueError):
        blocked_scores(lambda A, b: LDAClassifier().fit(A, b), X, y, np.zeros(len(y)))
    with pytest.raises(ValueError):
        blocked_scores(lambda A, b: LDAClassifier().fit(A, b), X, y, blocks[:-1])


def test_blocked_scores_refuse_a_block_holding_a_whole_class():
    X, y = blobs(10, np.random.default_rng(13))
    blocks = np.where(y == "g0", 1, np.arange(len(y)) % 2 * 2)  # g0 lives only in block 1
    with pytest.raises(ValueError, match="no training data"):
        blocked_scores(lambda A, b: LDAClassifier().fit(A, b), X, y, blocks)


# ------------------------------------------------------------ adaptive alpha


def test_adaptive_alpha_moves_toward_the_target():
    aci = AdaptiveAlpha(target=0.1, gamma=0.1)
    assert aci.update(0.3) < 0.1  # missed too often: ask for bigger sets
    assert AdaptiveAlpha(0.1, 0.1).update(0.0) > 0.1
    aci = AdaptiveAlpha(0.1, 10.0)
    assert 0 < aci.update(1.0) < 1  # clipped so a threshold exists
    with pytest.raises(ValueError):
        AdaptiveAlpha(1.5)


def test_adaptive_alpha_tracks_coverage_under_drift(model):
    """Classes drift further every session; fixed alpha under-covers, ACI doesn't."""
    rng = np.random.default_rng(14)
    Xc, yc = blobs(300, rng)
    pool = ConformalGestures.calibrate(model, Xc, yc, 0.1, score="lac")
    true = nonconformity(pool.predict_proba(Xc), "lac")[
        np.arange(len(yc)), np.searchsorted(pool.classes, yc)
    ]
    aci = AdaptiveAlpha(0.1, 0.05)
    fixed_cov, aci_cov = [], []
    for session in range(60):
        Xs, ys = blobs(100, rng)
        Xs = Xs + rng.normal(0, 1.0, Xs.shape[1]) * (0.02 * session)  # growing shift
        y = np.searchsorted(pool.classes, ys)
        s = nonconformity(pool.predict_proba(Xs), "lac")
        hit_fixed = s[np.arange(len(y)), y] <= conformal_quantile(true, 0.1)
        hit_aci = s[np.arange(len(y)), y] <= conformal_quantile(true, aci.working_alpha)
        aci.update(1 - hit_aci.mean())
        fixed_cov.append(hit_fixed.mean())
        aci_cov.append(hit_aci.mean())
    assert np.mean(aci_cov[20:]) > np.mean(fixed_cov[20:])
    assert abs(np.mean(aci_cov[20:]) - 0.9) < 0.04
