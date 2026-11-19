import numpy as np
import pytest
from scipy import signal

from myobridge.dsp import EmgFilter, MovingRMS, StreamingSOS, design_bandpass, design_notch

FS = 1000


def random_signal(n=5000, channels=3, seed=0):
    return np.random.default_rng(seed).normal(0, 100, (n, channels))


@pytest.mark.parametrize("block", [1, 7, 20, 333])
def test_streaming_equals_one_shot(block):
    x = random_signal()
    whole = EmgFilter(FS, 3).process(x)
    f = EmgFilter(FS, 3)
    parts = [f.process(x[i : i + block]) for i in range(0, len(x), block)]
    np.testing.assert_allclose(np.vstack([p.clean for p in parts]), whole.clean, atol=1e-9)
    np.testing.assert_allclose(np.vstack([p.envelope for p in parts]), whole.envelope, atol=1e-9)


def test_streaming_sos_matches_scipy():
    x = random_signal()
    sos = design_bandpass(FS)
    f = StreamingSOS(sos, 3)
    chunked = np.vstack([f.process(x[i : i + 50]) for i in range(0, len(x), 50)])
    np.testing.assert_allclose(chunked, signal.sosfilt(sos, x, axis=0), atol=1e-9)


def tone_gain_db(sos, freq, fs=FS, seconds=3.0):
    """Steady-state gain of a filter at one frequency, measured with a sine."""
    t = np.arange(int(seconds * fs)) / fs
    x = np.sin(2 * np.pi * freq * t)[:, None]
    y = StreamingSOS(sos, 1).process(x)
    tail = slice(len(t) // 2, None)  # skip the start-up transient
    return 20 * np.log10(np.std(y[tail]) / np.std(x[tail]))


def test_notch_kills_60hz():
    assert tone_gain_db(design_notch(FS, 60.0), 60.0) < -20


def test_notch_50hz_option():
    assert tone_gain_db(design_notch(FS, 50.0), 50.0) < -20


@pytest.mark.parametrize("freq", [30, 100, 150, 250, 400])
def test_notch_leaves_passband_alone(freq):
    assert abs(tone_gain_db(design_notch(FS, 60.0), freq)) < 0.5


@pytest.mark.parametrize("freq", [40, 100, 200, 350])
def test_bandpass_passes_emg_band(freq):
    assert abs(tone_gain_db(design_bandpass(FS), freq)) < 1.0


@pytest.mark.parametrize("freq", [1, 3])
def test_bandpass_removes_drift(freq):
    assert tone_gain_db(design_bandpass(FS), freq, seconds=10) < -40


def test_bandpass_rejects_bad_corners():
    with pytest.raises(ValueError):
        design_bandpass(FS, 20, 600)


def test_moving_rms_of_constant():
    r = MovingRMS(window=4, n_channels=1)
    out = r.process(np.full((10, 1), 3.0))
    # Before the window fills, the missing history counts as zeros.
    np.testing.assert_allclose(out[:4, 0], np.sqrt([9 / 4, 18 / 4, 27 / 4, 9.0]))
    np.testing.assert_allclose(out[4:, 0], 3.0)


def test_envelope_tracks_amplitude():
    rng = np.random.default_rng(1)
    quiet = rng.normal(0, 10, (2000, 1))
    loud = rng.normal(0, 200, (2000, 1))
    env = EmgFilter(FS, 1).process(np.vstack([quiet, loud])).envelope[:, 0]
    assert env[3500:].mean() > 10 * env[1000:2000].mean()
