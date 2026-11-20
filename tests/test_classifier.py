import numpy as np
import pytest

from myobridge.classifier import (
    LDAClassifier,
    MajorityVote,
    TrainingSettings,
    calibrate,
    recording_to_features,
)
from myobridge.simulator import GESTURES, EmgSimulator


def test_lda_on_two_blobs():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (100, 2)), rng.normal(5, 1, (100, 2))])
    y = np.array(["a"] * 100 + ["b"] * 100)
    clf = LDAClassifier().fit(X, y)
    assert list(clf.predict([[0, 0], [5, 5]])) == ["a", "b"]
    proba = clf.predict_proba([[0, 0]])
    assert proba.shape == (1, 2)
    assert proba.sum() == pytest.approx(1.0)
    assert proba[0, 0] > 0.99


def test_lda_survives_dead_feature():
    rng = np.random.default_rng(0)
    X = np.hstack([rng.normal(size=(50, 2)), np.zeros((50, 1))])  # constant column
    y = np.array(["a", "b"] * 25)
    LDAClassifier().fit(X, y).predict(X)  # singular without the ridge


def test_unfitted_raises():
    with pytest.raises(RuntimeError):
        LDAClassifier().predict(np.zeros((1, 3)))


def test_save_load_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(60, 3)), np.array(["x", "y", "z"] * 20)
    clf = LDAClassifier().fit(X, y)
    path = tmp_path / "model.npz"
    clf.save(str(path))
    loaded = LDAClassifier.load(str(path))
    np.testing.assert_array_equal(loaded.predict(X), clf.predict(X))
    np.testing.assert_allclose(loaded.predict_proba(X), clf.predict_proba(X))


def test_majority_vote():
    v = MajorityVote(k=3)
    assert v.update("a") == "a"
    assert v.update("b") == "b"  # 1-1 tie -> most recent wins
    assert v.update("b") == "b"
    assert v.update("a") == "b"  # history b, b, a: one-window flicker suppressed
    assert v.update("a") == "a"  # history b, a, a: the new gesture takes over


def test_lda_separates_simulated_gestures_on_held_out_data():
    """Train on one simulated 'session', test on a different seed."""
    clf, _, _ = calibrate(EmgSimulator(seed=1).record, GESTURES, seconds=6.0, fs=1000)
    test = EmgSimulator(seed=99)
    correct = total = 0
    for g in GESTURES:
        X = recording_to_features(test.record(g, 4.0), fs=1000)
        correct += np.sum(clf.predict(X) == g)
        total += len(X)
    assert correct / total > 0.90


def test_lda_ignores_feature_units():
    """Ridge is relative to each feature's variance, so rescaling a column
    (counts vs microvolts, say) never changes a prediction."""
    rng = np.random.default_rng(3)
    X = np.hstack([rng.normal(size=(300, 3)), rng.normal(size=(300, 3)) * 0.01])
    y = np.repeat([0, 1, 2], 100)
    X[y == 1, 3] += 0.01  # a class difference that lives in a small-scale column
    scale = np.array([1.0, 1e3, 1e-3, 50.0, 1e4, 1.0])
    a = LDAClassifier(ridge=0.1).fit(X, y)
    b = LDAClassifier(ridge=0.1).fit(X * scale, y)
    np.testing.assert_allclose(a.predict_proba(X), b.predict_proba(X * scale), atol=1e-9)


def test_feature_set_is_saved_and_old_files_mean_td(tmp_path):
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(40, 4)), np.repeat(["a", "b"], 20)
    for fs in ("td-ar-psd", None):
        LDAClassifier(feature_set=fs).fit(X, y).save(tmp_path / "m.npz")
        assert LDAClassifier.load(tmp_path / "m.npz").feature_set == fs
    clf = LDAClassifier().fit(X, y)
    np.savez(tmp_path / "v01.npz", classes=clf.classes_, weights=clf.weights_, bias=clf.bias_)
    old = LDAClassifier.load(tmp_path / "v01.npz")
    assert old.feature_set == "td"
    np.testing.assert_array_equal(old.predict(X), clf.predict(X))


@pytest.mark.parametrize("feature_set", ["td", "td-ar-psd"])
def test_calibrate_records_the_feature_set(feature_set):
    clf, X, _ = calibrate(
        EmgSimulator(seed=2).record, GESTURES[:3], seconds=2.0, fs=1000, feature_set=feature_set
    )
    assert clf.feature_set == feature_set
    assert X.shape[1] == {"td": 5, "td-ar-psd": 15}[feature_set] * 4


def test_recording_too_short_for_a_window():
    X = recording_to_features(np.zeros((100, 4)), fs=1000)
    assert X.shape == (0, 60)


def test_subset_equals_training_on_those_columns():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, 8)) * rng.uniform(0.01, 100, 8)
    y = np.repeat([0, 1, 2], 100)
    X[y == 1, :3] += 2.0
    cols = [0, 2, 5, 7]
    sub = LDAClassifier().fit(X, y).subset(cols)
    refit = LDAClassifier().fit(X[:, cols], y)
    np.testing.assert_allclose(sub.predict_proba(X[:, cols]), refit.predict_proba(X[:, cols]))


def test_fit_rejects_nan_and_keeps_object_labels_saveable(tmp_path):
    X = np.random.default_rng(6).normal(size=(40, 3))
    y = np.array(["a", "b"] * 20, dtype=object)
    bad = X.copy()
    bad[3, 1] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        LDAClassifier().fit(bad, y)
    clf = LDAClassifier().fit(X, y)
    saved = clf.save(tmp_path / "model")  # suffix added
    assert saved.name == "model.npz"
    loaded = LDAClassifier.load(tmp_path / "model")  # and found again without it
    assert list(loaded.predict(X[:4])) == list(clf.predict(X[:4]))


def test_calibrate_records_how_it_trained_and_save_keeps_it(tmp_path):
    clf, X, _ = calibrate(EmgSimulator(seed=3).record, GESTURES[:3], 2.0, fs=1000, mains_hz=50)
    assert clf.settings == TrainingSettings(fs=1000.0, mains_hz=50.0)
    back = LDAClassifier.load(clf.save(tmp_path / "model"))
    assert back.settings == clf.settings
    assert back.subset([0, 1, 2]).settings == clf.settings
    np.testing.assert_array_equal(back.predict(X), clf.predict(X))
    plain = LDAClassifier().fit(X, np.arange(len(X)) % 2)
    assert LDAClassifier.load(plain.save(tmp_path / "plain")).settings is None
