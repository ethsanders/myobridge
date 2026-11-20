import numpy as np
import pytest

from myobridge.datasets import threedc
from myobridge.evaluation import (
    FeatureConfig,
    accuracy,
    balanced_accuracy,
    confusion,
    featurize,
    filter_recording,
    fit_thresholds,
    prefilter,
    summarize,
)
from myobridge.features import DEFAULT_FEATURE_SET, FEATURE_SETS


@pytest.fixture
def ds(fake_threedc):
    return threedc.load(fake_threedc, subjects=[1, 2])


def test_featurize_shapes_and_window_times(ds):
    cfg = FeatureConfig()
    recs = ds.select(subjects=[1], sessions=[0])
    t = featurize(recs, ds.mains_hz, cfg, fit_thresholds(recs, ds.mains_hz, cfg))
    # 3 s recordings: windows start at 0.25 s, every 50 ms, while they fit
    per_rec = (3000 - 250 - 200) // 50 + 1  # in samples at 1 kHz
    assert len(t) == len(recs) * per_rec
    assert t.X.shape[1] == len(FEATURE_SETS[DEFAULT_FEATURE_SET]) * 10
    assert t.columns([3]).tolist() == list(range(3, t.X.shape[1], 10))
    assert t.t_start.min() == pytest.approx(0.25)
    assert np.allclose(t.t_end - t.t_start, 0.2)
    assert t.t_end.max() <= 3.0 + 1e-9
    assert set(t.subject.tolist()) == {1} and set(t.session.tolist()) == {0}


def test_prefiltered_equals_on_the_fly(ds):
    cfg = FeatureConfig()
    recs = ds.select(subjects=[2], reps=[0])
    clean = prefilter(recs, ds.mains_hz, cfg)
    th = fit_thresholds(recs, ds.mains_hz, cfg)
    th2 = fit_thresholds(recs, ds.mains_hz, cfg, clean=clean)
    assert np.allclose(th.zc, th2.zc)
    a = featurize(recs, ds.mains_hz, cfg, th)
    b = featurize(recs, ds.mains_hz, cfg, th, clean=clean)
    assert np.allclose(a.X, b.X)
    with pytest.raises(ValueError):
        featurize(recs, ds.mains_hz, cfg, th, clean=clean[:-1])


def test_time_splits_never_share_samples(ds):
    cfg = FeatureConfig()
    recs = ds.select(subjects=[1], sessions=[0])
    t = featurize(recs, ds.mains_hz, cfg, fit_thresholds(recs, ds.mains_hz, cfg))
    cal = t.where(t_min=1.0, t_max=1.5)
    test = t.where(t_min=2.0)
    assert cal.t_end.max() <= 1.5 + 1e-9 and test.t_start.min() >= 2.0 - 1e-9
    last = t.where(from_end_s=0.5)
    assert last.t_start.min() >= 2.5 - 1e-9
    assert len(t.where(subjects=[1], sessions=[0], reps=[0, 1])) == len(t) // 2


def test_thresholds_need_rest_data(ds):
    no_rest = [r for r in ds.recordings if r.label != 0]
    with pytest.raises(ValueError, match="no rest"):
        fit_thresholds(no_rest, ds.mains_hz, FeatureConfig())


def test_short_recordings_are_skipped(ds):
    from dataclasses import replace

    cfg = FeatureConfig()
    recs = ds.select(subjects=[1], sessions=[0])
    th = fit_thresholds(recs, ds.mains_hz, cfg)
    short = [replace(recs[0], emg=recs[0].emg[:300]), *recs[1:3]]
    assert set(featurize(short, ds.mains_hz, cfg, th).recording.tolist()) == {1, 2}
    with pytest.raises(ValueError, match="long enough"):
        featurize(short[:1], ds.mains_hz, cfg, th)


def test_filter_caps_the_band_below_nyquist(ds):
    from dataclasses import replace

    rec = replace(ds.recordings[0], fs=800.0)  # 450 Hz would be above Nyquist
    assert filter_recording(rec, 60.0, FeatureConfig()).shape == rec.emg.shape


def test_metrics():
    y = np.array([0, 0, 0, 1])
    p = np.array([0, 0, 0, 0])
    assert accuracy(y, p) == 0.75
    assert balanced_accuracy(y, p) == 0.5  # class 1 is never right
    assert confusion(y, p, 2).tolist() == [[3, 0], [1, 0]]
    with pytest.raises(ValueError):
        accuracy(np.array([]), np.array([]))


def test_summarize_is_reproducible():
    v = [0.8, 0.9, 0.85, 0.7, 0.95]
    a, b = summarize(v), summarize(v)
    assert a == b
    assert a["mean"] == pytest.approx(0.84) and a["n"] == 5
    assert a["ci95_low"] <= a["mean"] <= a["ci95_high"]
    assert summarize([0.5])["std"] == 0.0
    with pytest.raises(ValueError):
        summarize([])
