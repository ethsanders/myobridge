"""Classic time-domain EMG features (Hudgins et al. 1993 style).

Every function takes a window shaped (n_samples, n_channels) and returns one
value per channel. They also accept a stack of windows shaped
(n_windows, n_samples, n_channels) and then return (n_windows, n_channels):
the maths runs along the samples axis (axis=-2) either way. They're cheap
enough to run on a microcontroller later.

Thresholds for zero crossings and slope sign changes are in signal units. They
can be a single number or one number per channel; `noise_thresholds()` derives
them from a rest recording so the same code works on 12-bit ESP32 counts, a
research amplifier's microvolts, or anything else.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Threshold = float | FloatArray

FEATURE_NAMES = ("mav", "rms", "wl", "zc", "ssc")
_AXIS = -2  # samples axis, for both a single window and a stack of windows


def mav(w: FloatArray) -> FloatArray:
    """Mean absolute value: average size of the signal."""
    out: FloatArray = np.mean(np.abs(w), axis=_AXIS)
    return out


def rms(w: FloatArray) -> FloatArray:
    """Root mean square: like MAV but weights big spikes more (relates to power)."""
    out: FloatArray = np.sqrt(np.mean(w**2, axis=_AXIS))
    return out


def waveform_length(w: FloatArray) -> FloatArray:
    """Total distance the signal travels: sum of |x[i+1] - x[i]|.

    Grows with both amplitude and frequency content.
    """
    out: FloatArray = np.sum(np.abs(np.diff(w, axis=_AXIS)), axis=_AXIS)
    return out


def zero_crossings(w: FloatArray, threshold: Threshold = 0.0) -> FloatArray:
    """How often the signal changes sign, ignoring tiny wiggles below `threshold`.

    The threshold stops sensor noise around zero from counting as crossings.
    """
    a, b = w[..., :-1, :], w[..., 1:, :]
    crossed = (a * b) < 0
    big_enough = np.abs(a - b) >= threshold
    out: FloatArray = np.sum(crossed & big_enough, axis=_AXIS).astype(np.float64)
    return out


def slope_sign_changes(w: FloatArray, threshold: Threshold = 0.0) -> FloatArray:
    """How often the slope flips (local peaks and valleys), a frequency proxy.

    Sample i is a peak/valley if it is above both neighbours or below both,
    i.e. (x[i] - x[i-1]) * (x[i] - x[i+1]) > 0. The threshold ignores tiny ones.
    Note the threshold compares against a *product* of two differences, so it
    is in signal units squared.
    """
    prev, mid, nxt = w[..., :-2, :], w[..., 1:-1, :], w[..., 2:, :]
    product = (mid - prev) * (mid - nxt)
    out: FloatArray = np.sum(product > threshold, axis=_AXIS).astype(np.float64)
    return out


def extract_features(
    window: ArrayLike, zc_threshold: Threshold = 5.0, ssc_threshold: Threshold = 25.0
) -> FloatArray:
    """Feature vector(s) for one window or a stack of windows.

    Layout is feature-major: [mav_ch0..mav_chN, rms_ch0.., wl.., zc.., ssc..],
    so column `f * n_channels + ch` is feature f of channel ch.
    Amplitude features (mav, rms, wl) are log-scaled: EMG amplitude spans
    orders of magnitude between rest and a hard contraction, and the log makes
    the class clusters roughly the same size, which is what LDA assumes.

    The default thresholds suit the ESP32's 12-bit counts; for other
    recordings pass the output of `noise_thresholds()`.
    """
    w = np.asarray(window, dtype=np.float64)
    if w.ndim not in (2, 3):
        raise ValueError("window must be (n_samples, n_channels) or (n_windows, n, n_channels)")
    if w.shape[-2] < 3:
        raise ValueError("a window needs at least 3 samples")
    amplitude = [mav(w), rms(w), waveform_length(w)]
    counts = [zero_crossings(w, zc_threshold), slope_sign_changes(w, ssc_threshold)]
    return np.concatenate([np.log1p(f) for f in amplitude] + counts, axis=-1)


def feature_names(n_channels: int) -> list[str]:
    """Column names matching extract_features' layout, e.g. 'wl_ch2'."""
    return [f"{f}_ch{c}" for f in FEATURE_NAMES for c in range(n_channels)]


def channel_columns(channels: Sequence[int], n_channels: int) -> NDArray[np.intp]:
    """Feature columns that belong to a subset of channels.

    Every feature is computed per channel, so dropping channels is the same
    as dropping their columns. That makes "what if I only had 4 electrodes?"
    a cheap experiment: compute features once, then select columns.
    """
    if any(not 0 <= c < n_channels for c in channels):
        raise ValueError(f"channels must be in [0, {n_channels})")
    return np.array(
        [f * n_channels + c for f in range(len(FEATURE_NAMES)) for c in channels], dtype=np.intp
    )


def noise_thresholds(rest: ArrayLike, k: float = 0.5) -> tuple[FloatArray, FloatArray]:
    """Per-channel ZC and SSC thresholds from a (filtered) rest recording.

    ZC ignores steps smaller than k * (rest std); SSC ignores slope products
    smaller than (k * rest std)^2. Derive these from *training* data only, or
    the threshold leaks information from the test set.
    """
    r = np.asarray(rest, dtype=np.float64)
    if r.ndim != 2 or len(r) < 2:
        raise ValueError("rest must be (n_samples, n_channels) with n_samples >= 2")
    step = k * r.std(axis=0)
    return step, step**2


def window_stack(x: ArrayLike, size: int, step: int) -> FloatArray:
    """All windows of a recording as one (n_windows, size, n_channels) array.

    Windows end at samples size, size+step, ... (same as sliding_windows), and
    the result is a strided *view*: no data is copied.
    """
    xa = np.asarray(x, dtype=np.float64)
    if size < 1 or step < 1:
        raise ValueError("size and step must be >= 1")
    if xa.shape[0] < size:
        return np.zeros((0, size, xa.shape[1]))
    view = np.lib.stride_tricks.sliding_window_view(xa, size, axis=0)  # (n-size+1, ch, size)
    stacked: FloatArray = np.swapaxes(view[::step], 1, 2)
    return stacked


def sliding_windows(x: FloatArray, size: int, step: int) -> Iterator[tuple[int, FloatArray]]:
    """Yield (end_index, window) pairs over a recording, oldest first."""
    for end in range(size, x.shape[0] + 1, step):
        yield end, x[end - size : end]


class StreamingWindower:
    """Collects incoming blocks and emits a window every `step` new samples."""

    def __init__(self, size: int, step: int, n_channels: int):
        if size < 1 or step < 1:
            raise ValueError("size and step must be >= 1")
        self.size, self.step = size, step
        self.buffer: FloatArray = np.zeros((0, n_channels))
        self.since_last = 0  # samples received since the last emitted window

    def push(self, block: FloatArray) -> list[FloatArray]:
        windows = []
        for row in range(block.shape[0]):  # walk sample by sample: easy to reason about
            self.since_last += 1
            if self.since_last == self.step:
                self.since_last = 0
                full = np.vstack([self.buffer, block[: row + 1]])
                if full.shape[0] >= self.size:
                    windows.append(full[-self.size :])
        self.buffer = np.vstack([self.buffer, block])[-self.size :]
        return windows
