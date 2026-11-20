"""The AR and TD-PSD features: known answers, a reference implementation, edge cases."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy.linalg import solve_toeplitz
from scipy.signal import lfilter

from myobridge.features import (
    AR_NAMES,
    DEFAULT_FEATURE_SET,
    FEATURE_SETS,
    TDPSD_NAMES,
    ar_coefficients,
    extract_features,
    feature_names,
    feature_set_names,
    noise_thresholds,
    sliding_windows,
    tdpsd,
    window_stack,
)

EPS = np.spacing(1)


def test_ar_recovers_a_known_process():
    """x[n] = 1.2 x[n-1] - 0.5 x[n-2] + noise: AR(4) should find (1.2, -0.5, 0, 0)."""
    rng = np.random.default_rng(0)
    x = lfilter([1.0], [1.0, -1.2, 0.5], rng.standard_normal((20000, 1)), axis=0)
    a = np.array(ar_coefficients(x))[:, 0]
    assert np.allclose(a, [1.2, -0.5, 0.0, 0.0], atol=0.03)


def test_ar_matches_scipy_yule_walker():
    rng = np.random.default_rng(1)
    w = rng.normal(size=(200, 3))
    ours = np.array(ar_coefficients(w))  # (order, ch)
    for ch in range(3):
        x = w[:, ch] - w[:, ch].mean()
        r = np.array([x[: len(x) - k] @ x[k:] for k in range(5)]) / len(x)
        ref = solve_toeplitz(r[:4], r[1:])
        assert np.allclose(ours[:, ch], ref, atol=1e-6)


def test_ar_is_scale_free_and_flat_channels_are_zero():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(200, 2))
    assert np.allclose(np.array(ar_coefficients(w)), np.array(ar_coefficients(w * 1000.0)))
    flat = np.column_stack([np.zeros(200), np.full(200, 7.0)])
    assert np.array_equal(np.array(ar_coefficients(flat)), np.zeros((4, 2)))
    with pytest.raises(ValueError):
        ar_coefficients(w, order=0)
    with pytest.raises(ValueError):
        ar_coefficients(w[:4])


def _libemg_tdpsd(windows):
    """LibEMG 2.0.3's getM0feat ... getWLFfeat (MIT licence), transcribed for comparison.

    LibEMG lays windows out as (n_windows, n_channels, n_samples).
    """

    def moments(w):
        n = w.shape[2]
        m0 = np.sqrt(np.sum(w**2, axis=2)) / (n - 1)
        m0 = m0**0.1 / 0.1
        d1 = np.diff(w, n=1, axis=2)
        d2 = np.diff(d1, n=1, axis=2)
        m2 = (np.sqrt(np.sum(d1**2, axis=2) / (n - 1))) ** 0.1 / 0.1
        m4 = (np.sqrt(np.sum(d2**2, axis=2) / (n - 1))) ** 0.1 / 0.1
        return m0, m2, m4, d1, d2

    def closures(w):
        m0, m2, m4, d1, d2 = moments(w)
        return [
            np.log(np.abs(m0)),
            np.log(np.abs(m0 - m2)),
            np.log(np.abs(m0 - m4)),
            np.log(np.abs(np.sqrt(np.abs((m0 - m2) * (m0 - m4))) / m0)),
            np.log(np.abs(m2 / np.sqrt(m0 * m4))),
            np.log(np.abs(np.sqrt(np.sum(np.abs(d1), axis=2) / np.sum(np.abs(d2), axis=2)))),
        ]

    ebp = closures(windows)
    efp = closures(np.log(windows**2 + EPS))
    return [-2 * f * e / (f * f + e * e) for e, f in zip(ebp, efp, strict=True)]


def test_tdpsd_matches_the_libemg_reference():
    rng = np.random.default_rng(3)
    w = rng.normal(0, 40, (25, 200, 4))  # (n_windows, n_samples, n_channels)
    ours = np.array(tdpsd(w))
    ref = np.array(_libemg_tdpsd(np.swapaxes(w, 1, 2)))
    assert ours.shape == ref.shape == (6, 25, 4)
    assert np.allclose(ours, ref, rtol=1e-6, atol=1e-9)


def test_tdpsd_values_are_bounded():
    """-2ab / (a^2 + b^2) always lies in [-1, 1]."""
    rng = np.random.default_rng(4)
    f = np.array(tdpsd(rng.normal(0, 100, (50, 200, 3))))
    assert np.all(np.abs(f) <= 1.0 + 1e-12)


def test_feature_set_layout_and_names():
    assert DEFAULT_FEATURE_SET in FEATURE_SETS
    rich = feature_set_names("td-ar-psd")
    assert rich == FEATURE_SETS["td"] + AR_NAMES + TDPSD_NAMES and len(rich) == 15
    names = feature_names(3, "td-ar-psd")
    assert names[:3] == ["mav_ch0", "mav_ch1", "mav_ch2"] and names[-1] == "wlf_ch2"
    w = np.random.default_rng(5).normal(size=(200, 3))
    f = extract_features(w, feature_set="td-ar-psd")
    assert f.shape == (45,)
    # the first 15 columns are exactly the "td" set
    assert np.array_equal(f[:15], extract_features(w, feature_set="td"))
    assert np.allclose(f[15:27], np.concatenate(ar_coefficients(w)))
    assert np.allclose(f[27:], np.concatenate(tdpsd(w)))
    with pytest.raises(ValueError, match="unknown feature set"):
        extract_features(w, feature_set="fft")
    with pytest.raises(ValueError, match="at least 5 samples"):
        extract_features(w[:4])


def test_rich_stack_equals_loop():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 30, (1000, 4))
    zc, ssc = noise_thresholds(x)
    stack = extract_features(window_stack(x, 200, 50), zc, ssc, "td-ar-psd")
    loop = np.array(
        [extract_features(w, zc, ssc, "td-ar-psd") for _, w in sliding_windows(x, 200, 50)]
    )
    assert np.allclose(stack, loop, rtol=1e-10, atol=1e-12)


def test_dead_channel_stays_finite():
    w = np.random.default_rng(7).normal(size=(200, 3))
    w[:, 1] = 0.0
    assert np.all(np.isfinite(extract_features(w, 0.0, 0.0, "td-ar-psd")))


@settings(max_examples=50, deadline=None)
@given(
    arrays(
        np.float64,
        st.tuples(st.integers(5, 60), st.integers(1, 4)),
        elements=st.floats(-1e4, 1e4, allow_nan=False, width=32),
    )
)
def test_rich_features_are_always_finite(w):
    f = extract_features(w, 0.0, 0.0, "td-ar-psd")
    assert f.shape == (15 * w.shape[1],) and np.all(np.isfinite(f))
