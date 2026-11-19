import numpy as np
import pytest

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
    assert extract_features(w).shape == (20,)


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
