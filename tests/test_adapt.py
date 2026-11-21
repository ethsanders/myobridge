import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from myobridge.classifier import LDAClassifier, adapt


def blobs(rng, means, n=200, scale=1.0):
    X = np.vstack([rng.normal(m, scale, (n, len(m))) for m in means])
    y = np.repeat(np.arange(len(means)), n)
    return X, y


MEANS = np.array([[0, 0, 0], [4, 0, 1], [0, 4, -1]], dtype=float)


def test_adapt_with_no_prior_moves_means_to_the_new_data():
    rng = np.random.default_rng(0)
    base = LDAClassifier().fit(*blobs(rng, MEANS))
    Xn, yn = blobs(rng, MEANS + 3.0, n=50)
    m = adapt(base, Xn, yn, mean_prior=0.0, covariance_weight=0.0)
    for c in range(3):
        assert np.allclose(m.means_[c], Xn[yn == c].mean(axis=0))
    assert np.allclose(m.covariance_, base.covariance_)  # covariance kept


def test_large_prior_keeps_the_old_model():
    rng = np.random.default_rng(1)
    base = LDAClassifier().fit(*blobs(rng, MEANS))
    Xn, yn = blobs(rng, MEANS + 3.0, n=5)
    m = adapt(base, Xn, yn, mean_prior=1e9)
    assert np.allclose(m.means_, base.means_, atol=1e-6)


def test_prior_is_a_weighted_average():
    rng = np.random.default_rng(2)
    base = LDAClassifier().fit(*blobs(rng, MEANS))
    Xn, yn = blobs(rng, MEANS + 3.0, n=10)
    m = adapt(base, Xn, yn, mean_prior=10.0)
    new = Xn[yn == 1].mean(axis=0)
    assert np.allclose(m.means_[1], (10 * new + 10 * base.means_[1]) / 20)


def test_classes_without_new_data_keep_their_mean():
    rng = np.random.default_rng(3)
    base = LDAClassifier().fit(*blobs(rng, MEANS))
    Xn, yn = blobs(rng, MEANS[:2] + 1.0, n=20)
    m = adapt(base, Xn, yn)
    assert np.allclose(m.means_[2], base.means_[2])
    assert not np.allclose(m.means_[0], base.means_[0])


def test_covariance_weight_one_uses_the_new_covariance():
    rng = np.random.default_rng(4)
    base = LDAClassifier().fit(*blobs(rng, MEANS))
    Xn, yn = blobs(rng, MEANS, n=300, scale=3.0)
    m = adapt(base, Xn, yn, covariance_weight=1.0)
    assert np.allclose(m.covariance_, LDAClassifier().fit(Xn, yn).covariance_)
    half = adapt(base, Xn, yn, covariance_weight=0.5)
    assert np.allclose(half.covariance_, (base.covariance_ + m.covariance_) / 2)


def test_adapt_fixes_a_shift_the_old_model_gets_wrong():
    rng = np.random.default_rng(5)
    base = LDAClassifier().fit(*blobs(rng, MEANS))
    shifted = MEANS + np.array([3.0, -2.0, 0.0])
    Xtest, ytest = blobs(rng, shifted)
    Xcal, ycal = blobs(rng, shifted, n=5)
    assert base.score(Xtest, ytest) < 0.8
    assert adapt(base, Xcal, ycal, mean_prior=0.0, covariance_weight=0.0).score(Xtest, ytest) > 0.95


def test_adapt_validates_inputs():
    rng = np.random.default_rng(6)
    base = LDAClassifier().fit(*blobs(rng, MEANS))
    X, y = blobs(rng, MEANS, n=3)
    with pytest.raises(ValueError, match="classes"):
        adapt(base, X, y + 10)
    with pytest.raises(ValueError):
        adapt(base, X, y, mean_prior=-1)
    with pytest.raises(ValueError):
        adapt(base, X, y, covariance_weight=1.5)
    with pytest.raises(RuntimeError):
        adapt(LDAClassifier(), X, y)


def test_adapted_model_survives_save_and_load(tmp_path):
    rng = np.random.default_rng(7)
    base = LDAClassifier().fit(*blobs(rng, MEANS))
    m = adapt(base, *blobs(rng, MEANS + 1, n=10), mean_prior=5)
    m.save(tmp_path / "m.npz")
    again = LDAClassifier.load(tmp_path / "m.npz")
    X, _ = blobs(rng, MEANS, n=20)
    assert np.array_equal(again.predict(X), m.predict(X))
    assert np.allclose(adapt(again, X[:5], np.zeros(5, int)).means_[1], m.means_[1])


def test_defaults_are_the_benchmarked_setting():
    # plain adapt() should run the setting MultiDay's validation days chose,
    # the one sessions.recalibrate and the conformal recipe use
    from myobridge import sessions

    rng = np.random.default_rng(8)
    base = LDAClassifier().fit(*blobs(rng, MEANS))
    Xn, yn = blobs(rng, MEANS + 1, n=10)
    chosen = adapt(base, Xn, yn, sessions.MEAN_PRIOR, sessions.COVARIANCE_WEIGHT)
    plain = adapt(base, Xn, yn)
    assert np.allclose(plain.means_, chosen.means_)
    assert np.allclose(plain.covariance_, chosen.covariance_)


def test_matches_scikit_learn():
    sk = pytest.importorskip("sklearn.discriminant_analysis")
    rng = np.random.default_rng(8)
    X, y = blobs(rng, np.random.default_rng(9).normal(0, 2, (6, 8)), n=150, scale=1.5)
    ours = LDAClassifier(ridge=0.0).fit(X, y)
    # scikit-learn divides the pooled scatter by n, we divide by n - n_classes
    # (unbiased); after matching that, the models must agree to rounding error.
    n, k = len(y), 6
    ours.set_params(ours.classes_, ours.means_, ours.covariance_ * (n - k) / n, ours.priors_)
    theirs = sk.LinearDiscriminantAnalysis(solver="lsqr").fit(X, y)
    Xt, _ = blobs(rng, np.random.default_rng(9).normal(0, 2, (6, 8)), n=100, scale=1.5)
    assert np.mean(ours.predict(Xt) == theirs.predict(Xt)) > 0.999
    assert np.allclose(ours.predict_proba(Xt), theirs.predict_proba(Xt), atol=1e-6)


@settings(max_examples=30, deadline=None)
@given(st.floats(0, 100), st.floats(0, 1), st.integers(1, 30))
def test_adapted_model_is_always_a_valid_classifier(prior, beta, n):
    rng = np.random.default_rng(n)
    base = LDAClassifier().fit(*blobs(rng, MEANS, n=30))
    m = adapt(base, *blobs(rng, MEANS + 1, n=n), mean_prior=prior, covariance_weight=beta)
    p = m.predict_proba(blobs(rng, MEANS, n=5)[0])
    assert np.all(np.isfinite(p)) and np.allclose(p.sum(axis=1), 1)
