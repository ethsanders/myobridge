"""Replaying recordings through the live pipeline, with faults injected on the fly."""

import numpy as np
import pytest

from myobridge.datasets import threedc
from myobridge.faults import Fault
from myobridge.replay import Replay


@pytest.fixture
def replay(fake_threedc):
    ds = threedc.load(fake_threedc, subjects=[1])
    train = ds.select(sessions=[0], reps=[0, 1])
    test = [r for r in ds.select(sessions=[1], reps=[0]) if r.label in (0, 2, 5)]
    return Replay(train, test, ds.gestures, mains_hz=ds.mains_hz, rail=32700.0, seed=1)


def test_replay_decodes_the_test_stream(replay):
    frames = []
    while not replay.done:
        frames.append(replay.step(0.1))
    assert replay.position == len(replay.stream)
    assert frames[-1].t == pytest.approx(replay.duration_s)
    assert frames[0].prediction is None or isinstance(frames[0].prediction, str)
    assert frames[-1].accuracy is not None and frames[-1].accuracy > 0.8
    assert {f.truth for f in frames} == {"neutral", "wrist_flexion", "supination"}
    assert frames[5].raw.shape == (25, 10)  # 100 samples per step, every 4th one
    assert all(not c.issues and not c.dropped for c in frames[-1].channels)


def test_lead_off_is_caught_and_dropped_then_repaired(replay):
    replay.step(1.5)
    replay.inject(3, Fault.LEAD_OFF)
    for _ in range(15):
        frame = replay.step(0.1)
    ch = frame.channels[3]
    assert ch.fault == "lead_off" and "flatline" in ch.issues and ch.dropped
    assert all(not c.dropped for c in frame.channels if c.channel != 3)
    replay.inject(3, None)
    for _ in range(15):
        frame = replay.step(0.1)
    assert frame.channels[3].fault is None and not frame.channels[3].dropped
    rest = slice(replay.position, None)
    np.testing.assert_array_equal(replay.stream[rest], replay.original[rest])


@pytest.mark.parametrize("fault", list(Fault))
def test_every_fault_changes_only_the_future_of_one_channel(replay, fault):
    replay.step(1.0)
    before = replay.stream.copy()
    replay.inject(2, fault)
    pos = replay.position
    np.testing.assert_array_equal(replay.stream[:pos], before[:pos])
    others = [c for c in range(10) if c != 2]
    np.testing.assert_array_equal(replay.stream[:, others], before[:, others])
    assert not np.array_equal(replay.stream[pos:, 2], before[pos:, 2])


def test_channel_subset_and_bad_arguments(fake_threedc):
    ds = threedc.load(fake_threedc, subjects=[1])
    train, test = ds.select(sessions=[0], reps=[0]), ds.select(sessions=[1], reps=[0])[:2]
    r = Replay(train, test, ds.gestures, channels=[0, 2, 5, 7])
    assert r.n_channels == 4 and r.step(0.2).raw.shape[1] == 4
    with pytest.raises(ValueError, match="rail"):
        r.inject(0, "clipping")
    with pytest.raises(ValueError):
        r.inject(4, "lead_off")
    with pytest.raises(ValueError):
        Replay([], test, ds.gestures)


def test_trace_scale_frames_a_gesture_and_shows_lead_off_as_flat(replay):
    assert replay.trace_scale.shape == replay.trace_centre.shape == (replay.n_channels,)
    assert np.all(replay.trace_scale > 0)
    replay.step(2.0)
    replay.inject(0, Fault.LEAD_OFF)
    raw = replay.step(1.0).raw
    size = np.abs(raw - replay.trace_centre).max(axis=0) / replay.trace_scale
    assert size[0] < 0.2  # drawn nearly flat, not rescaled back up
