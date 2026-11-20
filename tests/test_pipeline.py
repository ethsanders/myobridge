import numpy as np

from myobridge.classifier import LDAClassifier
from myobridge.pipeline import GesturePipeline


def test_pipeline_works_with_integer_labels():
    # Models trained by the benchmarks use integer labels; the live pipeline
    # used to assume strings and crashed when looking up the voted label.
    rng = np.random.default_rng(0)
    model = LDAClassifier().fit(rng.normal(size=(60, 20)), np.repeat([0, 1, 2], 20))
    pipe = GesturePipeline(model, fs=1000, n_channels=4)
    preds = [p for _ in range(20) for p in pipe.process_block(rng.normal(0, 50, (20, 4)))]
    assert preds and all(p.gesture in {"0", "1", "2"} for p in preds)
    assert all(0.0 <= p.confidence <= 1.0 for p in preds)
    assert np.all(np.diff([p.sample_index for p in preds]) == 50)


def test_pipeline_uses_the_models_feature_set():
    rng = np.random.default_rng(1)
    y = np.repeat([0, 1], 30)
    rich = LDAClassifier(feature_set="td-ar-psd").fit(rng.normal(size=(60, 60)), y)
    assert GesturePipeline(rich, fs=1000, n_channels=4).feature_set == "td-ar-psd"
    # a model fitted on bare arrays: the feature count says which set it is
    plain = LDAClassifier().fit(rng.normal(size=(60, 20)), y)
    assert GesturePipeline(plain, fs=1000, n_channels=4).feature_set == "td"


def test_pipeline_rejects_a_model_for_other_features():
    import pytest

    model = LDAClassifier(feature_set="td").fit(
        np.random.default_rng(2).normal(size=(60, 60)), np.repeat([0, 1], 30)
    )
    with pytest.raises(ValueError, match="expects 60 features"):
        GesturePipeline(model, fs=1000, n_channels=4)


def _trained_pipeline(monitor=None):
    from myobridge.classifier import calibrate
    from myobridge.simulator import GESTURES, EmgSimulator

    sim = EmgSimulator(seed=3)
    clf, _, _ = calibrate(sim.record, GESTURES, seconds=4.0, fs=1000)
    return sim, GesturePipeline(clf, fs=1000, n_channels=4, monitor=monitor)


def test_monitor_drops_a_dead_channel_and_keeps_decoding():
    from myobridge.quality import QualityConfig, QualityMonitor, rest_floor
    from myobridge.simulator import GESTURES

    sim, plain = _trained_pipeline()
    cfg = QualityConfig(fs=1000.0)
    floor = rest_floor(sim.record("rest", 3.0), cfg)
    _, watched = _trained_pipeline(QualityMonitor(cfg, 4, rest_floor=floor))
    gesture = GESTURES.index("open")
    raw = sim.record(GESTURES[gesture], 6.0)
    raw[2000:, 1] = raw[2000:, 1].mean()  # electrode 1 comes off after 2 s

    def late_predictions(pipe):
        preds = [p for block in np.split(raw, 300) for p in pipe.process_block(block)]
        return preds, [p for p in preds if p.sample_index > 4000]

    preds, late = late_predictions(watched)
    assert late and all(p.dropped_channels == (1,) for p in late)
    assert all(p.dropped_channels == () for p in preds if p.sample_index < 1500)
    assert np.mean([p.gesture == GESTURES[gesture] for p in late]) > 0.75
    _, unwatched = late_predictions(plain)  # without the monitor the dead channel wins
    assert np.mean([p.gesture == GESTURES[gesture] for p in unwatched]) < 0.5


def test_pipeline_checks_its_inputs():
    import pytest

    from myobridge.quality import QualityConfig, QualityMonitor

    with pytest.raises(ValueError, match="not fitted"):
        GesturePipeline(LDAClassifier(), fs=1000, n_channels=4)
    model = LDAClassifier().fit(np.random.default_rng(0).normal(size=(40, 20)), [0, 1] * 20)
    with pytest.raises(ValueError, match="different number"):
        GesturePipeline(model, 1000, 4, monitor=QualityMonitor(QualityConfig(fs=1000.0), 3))


def test_from_model_matches_training_and_mismatches_are_refused():
    import pytest

    from myobridge.classifier import TrainingSettings

    rng = np.random.default_rng(4)
    settings = TrainingSettings(fs=2000.0, window_s=0.25, step_s=0.1, zc_threshold=2.0)
    model = LDAClassifier(settings=settings).fit(rng.normal(size=(60, 20)), np.repeat([0, 1], 30))
    pipe = GesturePipeline.from_model(model, n_channels=4, vote_k=1)
    assert pipe.windower.size == 500 and pipe.windower.step == 200
    assert pipe.thresholds == (2.0, 25.0)
    assert len(pipe.process_block(rng.normal(0, 50, (2000, 4)))) == 8
    with pytest.raises(ValueError, match="sample rate=2000"):
        GesturePipeline(model, fs=1000, n_channels=4)
    with pytest.raises(ValueError, match="window_s=0.25"):
        GesturePipeline(model, fs=2000, n_channels=4)
    with pytest.raises(ValueError, match="doesn't record"):
        GesturePipeline.from_model(LDAClassifier().fit(rng.normal(size=(6, 20)), [0, 1] * 3), 4)


def test_empty_and_misshapen_blocks():
    import pytest

    rng = np.random.default_rng(5)
    model = LDAClassifier().fit(rng.normal(size=(60, 20)), np.repeat([0, 1], 30))
    pipe = GesturePipeline(model, fs=1000, n_channels=4)
    assert pipe.process_block(np.zeros((0, 4))) == [] and pipe.samples_seen == 0
    for bad in (np.zeros((20, 3)), np.zeros(20)):
        with pytest.raises(ValueError, match=r"\(n, 4\)"):
            pipe.process_block(bad)


def test_pipeline_reports_conformal_sets():
    from myobridge.classifier import recording_to_features
    from myobridge.conformal import ConformalGestures
    from myobridge.simulator import GESTURES

    sim, plain = _trained_pipeline()
    model = plain.model
    # calibration data the model never saw: a fresh recording of every gesture
    X, y = [], []
    for g in GESTURES:
        feats = recording_to_features(sim.record(g, 2.0), 1000, feature_set=model.feature_set)
        X.append(feats)
        y += [g] * len(feats)
    cg = ConformalGestures.calibrate(model, np.vstack(X), y, alpha=0.1)
    pipe = GesturePipeline(model, fs=1000, n_channels=4, conformal=cg)
    raw = sim.record("fist", 3.0)
    preds = [p for block in np.split(raw, 60) for p in pipe.process_block(block)]
    assert preds and all(p.gesture_set is not None for p in preds)
    assert all(set(p.gesture_set) <= set(GESTURES) for p in preds)
    late = [p for p in preds if p.sample_index > 1000]
    assert np.mean(["fist" in p.gesture_set for p in late]) > 0.8
    assert any(p.sure for p in late)
    for p in late:
        if p.sure:
            assert p.gesture_set == (p.raw_gesture,)
    assert all(p.gesture_set is None and not p.sure for p in plain.process_block(raw[:500]))
