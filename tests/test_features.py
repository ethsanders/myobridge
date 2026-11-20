import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from myobridge.features import (
    StreamingWindower,
    extract_features,
    mav,
    rms,
    sliding_windows,
    slope_sign_changes,
    waveform_length,
    zero_crossings,
)

# One channel, worked out by hand:  x = 1, -2, 3, 3, -1
X = np.array([[1.0], [-2.0], [3.0], [3.0], [-1.0]])


def test_mav():
    assert mav(X)[0] == pytest.approx((1 + 2 + 3 + 3 + 1) / 5)


def test_rms():
    assert rms(X)[0] == pytest.approx(np.sqrt((1 + 4 + 9 + 9 + 1) / 5))


def test_waveform_length():
    # |-2-1| + |3+2| + |3-3| + |-1-3| = 3 + 5 + 0 + 4
    assert waveform_length(X)[0] == 12


def test_zero_crossings():
    # sign changes: 1->-2, -2->3, 3->-1
    assert zero_crossings(X)[0] == 3
    # with threshold 4 only jumps of size >= 4 count: |-2-3|=5, |3+1|=4 -> 2
    assert zero_crossings(X, threshold=4)[0] == 2


def test_slope_sign_changes():
    # i=1 (-2): (-2-1)*(-2-3) = 15 > 0 valley -> counts
    # i=2 (3):  (3+2)*(3-3)  = 0      flat    -> no
    # i=3 (3):  (3-3)*(3+1)  = 0      flat    -> no
    assert slope_sign_changes(X)[0] == 1
    assert slope_sign_changes(X, threshold=20)[0] == 0


def test_multichannel_shapes():
    w = np.random.default_rng(0).normal(size=(200, 4))
    for f in (mav, rms, waveform_length, zero_crossings, slope_sign_changes):
        assert f(w).shape == (4,)
    assert extract_features(w, feature_set="td").shape == (20,)
    assert extract_features(w).shape == (60,)  # default: 15 features per channel


def test_sliding_windows():
    x = np.arange(10)[:, None]
    ends = [end for end, _ in sliding_windows(x, size=4, step=3)]
    assert ends == [4, 7, 10]


@pytest.mark.parametrize("block", [1, 20, 37])
def test_streaming_windower_matches_offline(block):
    x = np.random.default_rng(0).normal(size=(1000, 2))
    offline = [w for _, w in sliding_windows(x, 200, 50)]
    sw = StreamingWindower(200, 50, 2)
    online = []
    for i in range(0, len(x), block):
        online += sw.push(x[i : i + block])
    assert len(online) == len(offline)
    for a, b in zip(online, offline, strict=True):
        np.testing.assert_array_equal(a, b)


@given(
    size=st.integers(1, 40),
    step=st.integers(1, 15),
    cuts=st.lists(st.integers(0, 120), max_size=8),
)
@settings(max_examples=200, deadline=None)
def test_streaming_windows_match_offline_windows_however_the_stream_is_cut(size, step, cuts):
    x = np.arange(240, dtype=np.float64).reshape(120, 2)
    edges = [0, *sorted(cuts), 120]
    sw = StreamingWindower(size, step, 2)
    got = [w for a, b in itertools.pairwise(edges) for w in sw.push(x[a:b])]
    # offline: a window ends on every step-th sample, once `size` samples exist
    want = [x[end - size : end] for end in range(step, 121, step) if end >= size]
    assert len(got) == len(want)
    for g, w in zip(got, want, strict=True):
        np.testing.assert_array_equal(g, w)
