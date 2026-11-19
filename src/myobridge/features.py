"""Classic time-domain EMG features (Hudgins et al. 1993 style).

Every function takes a window shaped (n_samples, n_channels) and returns one
value per channel. They're cheap enough to run on a microcontroller later.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = ("mav", "rms", "wl", "zc", "ssc")


def mav(w: np.ndarray) -> np.ndarray:
    """Mean absolute value: average size of the signal."""
    return np.mean(np.abs(w), axis=0)


def rms(w: np.ndarray) -> np.ndarray:
    """Root mean square: like MAV but weights big spikes more (relates to power)."""
    return np.sqrt(np.mean(w**2, axis=0))


def waveform_length(w: np.ndarray) -> np.ndarray:
    """Total distance the signal travels: sum of |x[i+1] - x[i]|.

    Grows with both amplitude and frequency content.
    """
    return np.sum(np.abs(np.diff(w, axis=0)), axis=0)


def zero_crossings(w: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """How often the signal changes sign, ignoring tiny wiggles below `threshold`.

    The threshold stops sensor noise around zero from counting as crossings.
    """
    a, b = w[:-1], w[1:]
    crossed = (a * b) < 0
    big_enough = np.abs(a - b) >= threshold
    return np.sum(crossed & big_enough, axis=0).astype(float)


def slope_sign_changes(w: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """How often the slope flips (local peaks and valleys), a frequency proxy.

    Sample i is a peak/valley if it is above both neighbours or below both,
    i.e. (x[i] - x[i-1]) * (x[i] - x[i+1]) > 0. The threshold ignores tiny ones.
    """
    prev, mid, nxt = w[:-2], w[1:-1], w[2:]
    product = (mid - prev) * (mid - nxt)
    return np.sum(product > threshold, axis=0).astype(float)


def extract_features(
    window: np.ndarray, zc_threshold: float = 5.0, ssc_threshold: float = 25.0
) -> np.ndarray:
    """One feature vector for one window.

    Layout is feature-major: [mav_ch0..mav_chN, rms_ch0.., wl.., zc.., ssc..].
    Amplitude features (mav, rms, wl) are log-scaled: EMG amplitude spans
    orders of magnitude between rest and a hard contraction, and the log makes
    the class clusters roughly the same size, which is what LDA assumes.
    """
    amplitude = [mav(window), rms(window), waveform_length(window)]
    counts = [zero_crossings(window, zc_threshold), slope_sign_changes(window, ssc_threshold)]
    return np.concatenate([np.log1p(f) for f in amplitude] + counts)


def sliding_windows(x: np.ndarray, size: int, step: int):
    """Yield (end_index, window) pairs over a recording, oldest first."""
    for end in range(size, x.shape[0] + 1, step):
        yield end, x[end - size : end]


class StreamingWindower:
    """Collects incoming blocks and emits a window every `step` new samples."""

    def __init__(self, size: int, step: int, n_channels: int):
        self.size, self.step = size, step
        self.buffer = np.zeros((0, n_channels))
        self.since_last = 0  # samples received since the last emitted window

    def push(self, block: np.ndarray) -> list[np.ndarray]:
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
