"""The vectorised feature code must equal the one-window-at-a-time maths."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from myobridge.features import (
    channel_columns,
    extract_features,
    feature_names,
    mav,
    noise_thresholds,
    rms,
    sliding_windows,
    window_stack,
    zero_crossings,
)

finite = st.floats(-1e4, 1e4, allow_nan=False, width=32)


def test_stack_equals_loop():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 30, (1000, 4))
    zc, ssc = noise_thresholds(x)
    stack = window_stack(x, 200, 50)
    loop = np.array([extract_features(w, zc, ssc) for _, w in sliding_windows(x, 200, 50)])
    assert stack.shape == (17, 200, 4)
    assert np.array_equal(extract_features(stack, zc, ssc), loop)


def test_window_stack_is_a_view_and_handles_short_input():
    x = np.arange(20.0).reshape(10, 2)
    s = window_stack(x, 4, 3)
    assert s.shape == (3, 4, 2) and np.shares_memory(s, x)
    assert np.array_equal(s[1], x[3:7])
    assert window_stack(x[:3], 4, 1).shape == (0, 4, 2)
    with pytest.raises(ValueError):
        window_stack(x, 0, 1)


def test_per_channel_thresholds():
    w = np.column_stack([np.tile([1.0, -1.0], 50), np.tile([10.0, -10.0], 50)])
    # a threshold of 5 per step ignores channel 0's crossings but not channel 1's
    assert zero_crossings(w, np.array([5.0, 5.0])).tolist() == [0, 99]
    assert zero_crossings(w, np.array([0.0, 50.0])).tolist() == [99, 0]


def test_channel_columns_select_the_right_features():
    names = feature_names(4)
    cols = channel_columns([1, 3], 4)
    assert [names[c] for c in cols] == [
        f"{f}_ch{c}" for f in ("mav", "rms", "wl", "zc", "ssc") for c in (1, 3)
    ]
    with pytest.raises(ValueError):
        channel_columns([4], 4)


def test_noise_thresholds():
    rest = np.random.default_rng(1).normal(0, [2.0, 8.0], (5000, 2))
    zc, ssc = noise_thresholds(rest, k=0.5)
    assert np.allclose(zc, [1.0, 4.0], rtol=0.05)
    assert np.allclose(ssc, zc**2)
    with pytest.raises(ValueError):
        noise_thresholds(rest[:1])


def test_bad_shapes():
    with pytest.raises(ValueError):
        extract_features(np.zeros(10))
    with pytest.raises(ValueError):
        extract_features(np.zeros((2, 3)))


@settings(max_examples=50, deadline=None)
@given(arrays(np.float64, st.tuples(st.integers(3, 60), st.integers(1, 4)), elements=finite))
def test_feature_invariants(w):
    f = extract_features(w, 0.0, 0.0)
    n, ch = w.shape
    assert f.shape == (5 * ch,) and np.all(np.isfinite(f))
    assert np.all(mav(w) <= rms(w) + 1e-9)  # mean |x| never exceeds RMS
    zc, ssc = f[3 * ch : 4 * ch], f[4 * ch :]
    assert np.all((zc >= 0) & (zc <= n - 1)) and np.all((ssc >= 0) & (ssc <= n - 2))


@settings(max_examples=30, deadline=None)
@given(
    arrays(np.float64, st.tuples(st.integers(3, 40), st.just(2)), elements=finite),
    st.sampled_from([0.125, 0.5, 2.0, 16.0]),  # powers of two scale exactly in floating point
)
def test_counts_are_scale_free_when_thresholds_scale_too(w, k):
    """Scaling the signal and the thresholds together leaves ZC/SSC unchanged,
    which is why thresholds come from each recording's own rest noise."""
    a = extract_features(w, 1.0, 1.0)
    b = extract_features(w * k, 1.0 * k, 1.0 * k**2)
    assert np.array_equal(a[6:], b[6:])
