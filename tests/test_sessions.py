import json

import numpy as np
import pytest

from myobridge.conformal import set_metrics
from myobridge.sessions import (
    CALIBRATION_WINDOW,
    SCORE_FROM,
    Session,
    _rows,
    fit_sets,
    load_sets,
    recalibrate,
    sets_path_for,
)
from myobridge.simulator import GESTURES, EmgSimulator


def session(seed, seconds=5.0, gestures=GESTURES, **sim):
    s = EmgSimulator(seed=seed, **sim)
    return Session(1000.0, tuple(gestures), tuple(s.record(g, seconds) for g in gestures))


@pytest.fixture(scope="module")
def days():
    return [session(seed) for seed in range(3)]


def test_session_round_trips(tmp_path, days):
    path = days[0].save(tmp_path / "day1")
    assert path.name == "day1.npz"
    back = Session.load(path)
    assert back.labels == days[0].labels and back.fs == days[0].fs
    assert back.recorded_at == days[0].recorded_at and back.source == str(path)
    for a, b in zip(back.holds, days[0].holds, strict=True):
        np.testing.assert_array_equal(a, b)


def test_a_model_file_is_not_a_session(tmp_path):
    path = tmp_path / "x.npz"
    np.savez(path, classes=np.array(["a"]))
    with pytest.raises(ValueError, match="not a myobridge session"):
        Session.load(path)


def test_three_sessions_have_three_roles(days):
    fitted = fit_sets(days)
    assert [r["role"] for r in fitted.roles] == ["train", "temperature", "threshold"]
    assert fitted.sets.coverage == pytest.approx(0.9)
    assert fitted.sets.score == "aps"
    assert list(fitted.model.classes) == sorted(GESTURES)


def test_the_promise_holds_on_a_later_day(days):
    """On a new simulated day, recalibrated with 1 s per gesture, coverage is near 90%."""
    fitted = fit_sets(days)
    covs = []
    for seed in (10, 11, 12):
        today = session(seed, seconds=8.0)
        sets = recalibrate(fitted.sets, today)
        s = fitted.model.settings
        X, y = _rows(today, s, fitted.model.feature_set, SCORE_FROM)
        idx = np.searchsorted(sets.classes, y)
        covs.append(set_metrics(sets.predict_sets(X), idx)["coverage"])
    assert 0.8 < np.mean(covs) < 0.99


def test_recalibrating_keeps_temperature_and_threshold(days):
    fitted = fit_sets(days)
    today = session(20, seconds=CALIBRATION_WINDOW[1])  # 2 s per gesture is enough
    after = recalibrate(fitted.sets, today)
    assert after.temperature == fitted.sets.temperature
    np.testing.assert_array_equal(after.thresholds, fitted.sets.thresholds)
    assert after.model is not fitted.model
    assert not np.allclose(after.model.means_, fitted.model.means_)


def test_recalibrating_needs_every_gesture(days):
    fitted = fit_sets(days)
    with pytest.raises(ValueError, match="no rest"):
        recalibrate(fitted.sets, session(21, seconds=2.0, gestures=GESTURES[1:]))


def test_two_sessions_split_the_second(days):
    fitted = fit_sets(days[:2])
    assert [r["role"] for r in fitted.roles] == [
        "train",
        "temperature and threshold (split in time)",
    ]


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda d: d[:1], "at least two days"),
        (lambda d: [d[0], session(5, gestures=GESTURES[:3])], "holds"),
        (lambda d: [d[0], session(5, n_channels=3)], "channels"),
        (lambda d: [d[0], session(5, seconds=2.0)], "needs 3 s"),
    ],
)
def test_sessions_that_dont_fit_together(days, make, message):
    with pytest.raises(ValueError, match=message):
        fit_sets(make(days))


def test_save_and_load_sets(tmp_path, days):
    fitted = fit_sets(days)
    model_path, sets_path = fitted.save(tmp_path / "model")
    assert sets_path == sets_path_for(model_path) == tmp_path / "model.sets.json"
    sets, recipe = load_sets(model_path)
    assert sets.temperature == fitted.sets.temperature
    np.testing.assert_array_equal(sets.thresholds, fitted.sets.thresholds)
    assert [s["role"] for s in recipe["sessions"]] == ["train", "temperature", "threshold"]
    assert json.loads(sets_path.read_text())["recipe"]["score_from_s"] == SCORE_FROM


def test_loading_sets_without_a_sets_file(tmp_path, days):
    fitted = fit_sets(days)
    fitted.model.save(tmp_path / "bare.npz")
    with pytest.raises(FileNotFoundError, match="myobridge fit"):
        load_sets(tmp_path / "bare.npz")
