import numpy as np
import pytest
from scipy import signal

from myobridge.decimator import (
    HALFBAND_Q15,
    N_TAPS,
    HalfbandDecimator,
    decimate2,
    design_halfband,
)


def test_coefficients_are_the_designed_halfband():
    assert np.array_equal(design_halfband(), HALFBAND_Q15)
    h = HALFBAND_Q15
    assert len(h) == N_TAPS and np.array_equal(h, h[::-1])  # linear phase
    centre = N_TAPS // 2
    assert h[centre] == 2**14  # exactly 0.5
    assert np.all(h[centre + 2 :: 2] == 0)  # every 2nd tap away from the centre is zero
    assert abs(int(h.sum()) - 2**15) <= 8  # DC gain 1.0 to within rounding
    with pytest.raises(ValueError):
        design_halfband(48)


def test_frequency_response():
    w, H = signal.freqz(HALFBAND_Q15 / 2**15, worN=8192, fs=2000)
    gain_db = 20 * np.log10(np.abs(H) + 1e-12)
    assert np.all(np.abs(gain_db[(w >= 20) & (w <= 400)]) < 0.1)  # flat over the EMG band
    assert gain_db[np.argmin(np.abs(w - 450))] > -0.7
    assert np.all(gain_db[w >= 600] < -40)  # what would fold back to < 400 Hz is gone


def test_streaming_matches_batch():
    rng = np.random.default_rng(0)
    x = rng.integers(-32768, 32768, (400, 4)).astype(np.int16)
    d = HalfbandDecimator(4)
    stream = np.array([d.push(x[2 * i], x[2 * i + 1]) for i in range(200)])
    assert np.array_equal(stream, decimate2(x))


def test_a_tone_above_500hz_is_removed_and_one_below_is_kept():
    t = np.arange(4000) / 2000.0
    keep = (8000 * np.sin(2 * np.pi * 150 * t)).astype(np.int16)
    drop = (8000 * np.sin(2 * np.pi * 800 * t)).astype(np.int16)
    y = decimate2(np.column_stack([keep, drop]))[100:].astype(float)
    assert np.std(y[:, 0]) == pytest.approx(8000 / np.sqrt(2), rel=0.01)
    assert np.std(y[:, 1]) < 80  # > 40 dB down


def test_worst_case_input_saturates_instead_of_wrapping():
    # input whose signs line up with the taps: the true output is ~1.6x full scale
    signs = np.sign(HALFBAND_Q15[::-1]).astype(np.int64)
    worst = np.where(signs >= 0, 32767, -32768)
    x = np.concatenate([[0], worst]).astype(np.int16)[:, None]  # output 23 sees `worst`
    y = decimate2(x)
    assert y.dtype == np.int16 and y[-1, 0] == 32767
    assert decimate2(-x - 1)[-1, 0] == -32768


def test_input_checks():
    with pytest.raises(ValueError):
        decimate2(np.zeros((10, 2)))  # not int16
    with pytest.raises(ValueError):
        decimate2(np.zeros((9, 2), dtype=np.int16))
