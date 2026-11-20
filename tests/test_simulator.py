import numpy as np
import pytest

from myobridge.protocol import SequenceTracker, decode_packet
from myobridge.simulator import EmgSimulator, gesture_session, packetize


def test_deterministic_with_seed():
    a = EmgSimulator(seed=3).record("fist", 1.0)
    b = EmgSimulator(seed=3).record("fist", 1.0)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.int16 and a.shape == (1000, 4)


def test_block_size_does_not_matter_for_shape():
    sim = EmgSimulator(seed=0)
    assert sim.generate(20).shape == (20, 4)
    assert sim.t == 20


def test_fist_is_louder_than_rest():
    sim = EmgSimulator(seed=0, hum_amplitude=0)
    rest = sim.record("rest", 2.0).astype(float)
    fist = sim.record("fist", 2.0).astype(float)
    assert np.std(np.diff(fist, axis=0)) > 5 * np.std(np.diff(rest, axis=0))


def test_unknown_gesture():
    with pytest.raises(ValueError):
        EmgSimulator().set_gesture("jazz_hands")


def test_session_labels_align():
    raw, labels = gesture_session(EmgSimulator(seed=0), ("rest", "fist"), 0.5, 2)
    assert raw.shape[0] == labels.shape[0] == 2000
    assert list(labels[[0, 600, 1100, 1600]]) == ["rest", "fist", "rest", "fist"]


def test_packet_loss_injection_is_detected():
    stream = packetize(EmgSimulator(seed=0), frames_per_packet=20, drop_prob=0.1)
    tracker = SequenceTracker()
    for _ in range(500):
        tracker.accept(decode_packet(next(stream)).seq)
    assert tracker.stats.received == 500
    assert 20 < tracker.stats.dropped < 90  # ~10% of ~555 slots


@pytest.mark.parametrize("fs", [250, 500, 2000])
def test_runs_at_slow_and_fast_rates(fs):
    sim = EmgSimulator(fs=fs, seed=1)
    sim.set_gesture("fist")
    x = sim.record("fist", 0.5)
    assert x.shape == (fs // 2, 4)
    assert np.isfinite(x).all()
