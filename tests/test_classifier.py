import numpy as np
import pytest

from myobridge.classifier import LDAClassifier, MajorityVote, calibrate, recording_to_features
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
