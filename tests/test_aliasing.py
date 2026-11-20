import numpy as np
import pytest

from myobridge.aliasing import (
    band_power_fraction,
    decimate_chain,
    first_order_lowpass,
    folded_noise_ratio,
    lowpass_power,
    ratio_to_db,
)


def test_lowpass_power():
    f = np.array([0.0, 100.0, 1e6])
    p = lowpass_power(f, 100.0, 1)
    assert p[0] == 1.0 and p[1] == pytest.approx(0.5) and p[2] < 1e-7
    assert lowpass_power(np.array([200.0]), 100.0, 4)[0] == pytest.approx(1 / (1 + 2**8))
    with pytest.raises(ValueError):
        lowpass_power(f, 0, 1)


def test_folded_noise_numbers_quoted_in_the_docs():
    assert folded_noise_ratio(1000, 498, 1) == pytest.approx(0.955, abs=0.005)
    assert folded_noise_ratio(2000, 498, 1) == pytest.approx(0.251, abs=0.005)
    assert ratio_to_db(folded_noise_ratio(1000, 498, 1)) == pytest.approx(2.9, abs=0.05)


def test_folding_shrinks_with_rate_and_order():
    r = {(o, fs): folded_noise_ratio(fs, 498, o) for o in (1, 2, 4) for fs in (1000, 2000, 4000)}
    for o in (1, 2, 4):
        assert r[(o, 1000)] > r[(o, 2000)] > r[(o, 4000)] >= 0
    for fs in (1000, 2000, 4000):
        assert r[(1, fs)] > r[(2, fs)] >= r[(4, fs)]


def test_folded_ratio_is_converged():
    a = folded_noise_ratio(1000, 498, 1)
    b = folded_noise_ratio(1000, 498, 1, n_images=20000, n_grid=2001)
    assert a == pytest.approx(b, rel=0.01)


def test_band_must_fit():
    with pytest.raises(ValueError):
        folded_noise_ratio(800, 498, 1)  # 450 Hz band edge above Nyquist


def test_ratio_to_db():
    assert ratio_to_db(0.0) == 0.0
    assert ratio_to_db(1.0) == pytest.approx(3.0103, abs=1e-4)


def test_first_order_lowpass_has_3db_at_cutoff():
    fs, fc = 20000.0, 500.0
    t = np.arange(int(fs)) / fs
    for f, want in ((50.0, 1.0), (fc, 1 / np.sqrt(2))):
        y = first_order_lowpass(np.sin(2 * np.pi * f * t), fs, fc)
        assert np.std(y[2000:]) / np.std(np.sin(2 * np.pi * f * t)) == pytest.approx(want, rel=0.03)


def test_chains():
    fs = 2048.0
    t = np.arange(4096) / fs
    x = np.column_stack([np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 800 * t)])
    ideal = decimate_chain("ideal", fs)(x)
    none = decimate_chain("none", fs)(x)
    assert ideal.shape == none.shape == (2048, 2)
    # 800 Hz is above the new Nyquist (512 Hz): ideal removes it, "none" folds it to 224 Hz
    assert np.std(ideal[200:-200, 1]) < 0.01 and np.std(none[:, 1]) > 0.5
    assert np.std(ideal[200:-200, 0]) == pytest.approx(1 / np.sqrt(2), rel=0.02)
    assert decimate_chain("rc", fs)(x).shape == (2048, 2)
    with pytest.raises(KeyError):
        decimate_chain("magic", fs)
    with pytest.raises(ValueError):
        decimate_chain("ideal", fs, factor=1)


def test_band_power_fraction():
    fs = 2048.0
    t = np.arange(8192) / fs
    x = np.column_stack([np.sin(2 * np.pi * 100 * t), np.sin(2 * np.pi * 700 * t)])
    frac = band_power_fraction(x, fs, 512.0)
    assert frac[0] < 1e-3 and frac[1] > 0.99


def test_firmware_halfband_is_as_good_as_an_ideal_decimator():
    from myobridge.aliasing import folded_noise_ratio_decimated
    from myobridge.decimator import HALFBAND_Q15

    for order in (1, 2):
        real = ratio_to_db(folded_noise_ratio_decimated(1000.0, 498.0, order, HALFBAND_Q15))
        ideal = ratio_to_db(folded_noise_ratio(2000.0, 498.0, order))
        assert real == pytest.approx(ideal, abs=0.01)
    # the number quoted in the firmware and docs: first order, +2.9 dB -> +1.0 dB
    one = ratio_to_db(folded_noise_ratio_decimated(1000.0, 498.0, 1, HALFBAND_Q15))
    assert one == pytest.approx(0.97, abs=0.01)


def test_a_poor_decimator_lets_noise_through():
    from myobridge.aliasing import folded_noise_ratio_decimated

    # no filter at all (one tap): the 500-1000 Hz band folds straight down
    none = folded_noise_ratio_decimated(1000.0, 498.0, 1, [1.0])
    assert none == pytest.approx(folded_noise_ratio(1000.0, 498.0, 1), rel=0.01)
    with pytest.raises(ValueError):
        folded_noise_ratio_decimated(1000.0, 498.0, 1, [1.0], band=(20.0, 600.0))
