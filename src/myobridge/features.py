"""Time-domain EMG features, computed per channel on short windows.

Two feature sets are available (`FEATURE_SETS`):

``"td"``
    The classic Hudgins et al. (1993) set: mean absolute value, RMS, waveform
    length, zero crossings and slope sign changes. 5 numbers per channel.
``"td-ar-psd"`` (the default)
    The same five, plus 4 autoregressive coefficients (the spectral shape of
    the window) and the 6 TD-PSD features of Al-Timemy et al. (2016), which
    are built from spectral moments and were designed to be robust to force
    changes. 15 numbers per channel. On the validation people and days it
    beat "td" on both datasets; docs/results.md has the test-split numbers.

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

TD_NAMES = ("mav", "rms", "wl", "zc", "ssc")
AR_ORDER = 4
AR_NAMES = tuple(f"ar{k}" for k in range(1, AR_ORDER + 1))
TDPSD_NAMES = ("m0", "m2", "m4", "sparsity", "irf", "wlf")

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "td": TD_NAMES,
    "td-ar-psd": TD_NAMES + AR_NAMES + TDPSD_NAMES,
}
DEFAULT_FEATURE_SET = "td-ar-psd"
FEATURE_NAMES = TD_NAMES  # kept for code written against v0.1

_AXIS = -2  # samples axis, for both a single window and a stack of windows
_EPS = float(np.spacing(1))


def feature_set_names(feature_set: str) -> tuple[str, ...]:
    """Per-channel feature names of a feature set, in column-block order."""
    try:
        return FEATURE_SETS[feature_set]
    except KeyError:
        raise ValueError(
            f"unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}"
        ) from None


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


def ar_coefficients(w: FloatArray, order: int = AR_ORDER) -> list[FloatArray]:
    """Autoregressive coefficients a_1..a_order, one array per coefficient.

    An AR model predicts each sample from the previous ones,
    x[n] ~ a_1 x[n-1] + ... + a_p x[n-p], so the coefficients describe the
    window's spectral shape independently of its amplitude. They're fitted
    with the Yule-Walker equations (autocorrelation method), which always
    give a stable model and cost one small linear solve per channel.
    A flat channel gets all-zero coefficients.
    """
    if order < 1:
        raise ValueError("order must be >= 1")
    x = np.moveaxis(w - w.mean(axis=_AXIS, keepdims=True), _AXIS, -1)  # (..., ch, n)
    n = x.shape[-1]
    if n <= order:
        raise ValueError("the window must be longer than the AR order")
    r = np.stack([np.sum(x[..., : n - k] * x[..., k:], axis=-1) for k in range(order + 1)], -1)
    # Normalise by the lag-0 term so the system is unit-free; a flat channel
    # has r0 = 0, and the tiny ridge then makes its solution exactly zero.
    rho = r / np.maximum(r[..., :1], np.finfo(np.float64).tiny)
    lags = np.abs(np.arange(order)[:, None] - np.arange(order)[None, :])
    toeplitz = rho[..., lags] + 1e-9 * np.eye(order)
    a = np.linalg.solve(toeplitz, rho[..., 1:, None])[..., 0]  # (..., ch, order)
    return [np.ascontiguousarray(a[..., k]) for k in range(order)]


def _tdpsd_raw(w: FloatArray) -> list[FloatArray]:
    """The six TD-PSD quantities of one signal, before the orientation step."""
    n = w.shape[_AXIS]
    d1 = np.diff(w, axis=_AXIS)
    d2 = np.diff(d1, axis=_AXIS)
    # Spectral moments via Parseval: m0 from the signal, m2 from its first
    # difference, m4 from its second. The 0.1 power compresses their range.
    m0 = (np.sqrt(np.sum(w**2, axis=_AXIS)) / (n - 1)) ** 0.1 / 0.1
    m2 = np.sqrt(np.sum(d1**2, axis=_AXIS) / (n - 1)) ** 0.1 / 0.1
    m4 = np.sqrt(np.sum(d2**2, axis=_AXIS) / (n - 1)) ** 0.1 / 0.1
    sparsity = np.sqrt(np.abs((m0 - m2) * (m0 - m4))) / (m0 + _EPS)
    irf = m2 / (np.sqrt(m0 * m4) + _EPS)  # irregularity factor
    wlf = np.sqrt(np.sum(np.abs(d1), axis=_AXIS) / (np.sum(np.abs(d2), axis=_AXIS) + _EPS))
    raw = [m0, m0 - m2, m0 - m4, sparsity, irf, wlf]
    return [np.log(np.abs(v) + _EPS) for v in raw]


def tdpsd(w: FloatArray) -> list[FloatArray]:
    """TD-PSD features (Al-Timemy et al., IEEE TNSRE 2016), one array per feature.

    Each of the six quantities is computed on the window and on its log-power
    version log(x^2), and the pair (a, b) is combined as -2ab / (a^2 + b^2),
    the "orientation" step that makes the result insensitive to overall force.
    This follows the reference implementation in LibEMG (MIT licence,
    Eddy et al. 2023), with small guards so a flat channel stays finite.
    """
    a = _tdpsd_raw(w)
    b = _tdpsd_raw(np.log(w**2 + _EPS))
    return [(-2.0 * fa * fb) / (fa * fa + fb * fb + _EPS) for fa, fb in zip(a, b, strict=True)]


def extract_features(
    window: ArrayLike,
    zc_threshold: Threshold = 5.0,
    ssc_threshold: Threshold = 25.0,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> FloatArray:
    """Feature vector(s) for one window or a stack of windows.

    Layout is feature-major: [mav_ch0..mav_chN, rms_ch0.., wl.., zc.., ssc.., ...],
    so column `f * n_channels + ch` is feature f of channel ch, in the order
    of `FEATURE_SETS[feature_set]`.
    Amplitude features (mav, rms, wl) are log-scaled: EMG amplitude spans
    orders of magnitude between rest and a hard contraction, and the log makes
    the class clusters roughly the same size, which is what LDA assumes.

    The default thresholds suit the ESP32's 12-bit counts; for other
    recordings pass the output of `noise_thresholds()`.
    """
    names = feature_set_names(feature_set)
    w = np.asarray(window, dtype=np.float64)
    if w.ndim not in (2, 3):
        raise ValueError("window must be (n_samples, n_channels) or (n_windows, n, n_channels)")
    rich = len(names) > len(TD_NAMES)
    min_samples = AR_ORDER + 1 if rich else 3
    if w.shape[-2] < min_samples:
        raise ValueError(f"a window needs at least {min_samples} samples for {feature_set!r}")
    amplitude = [mav(w), rms(w), waveform_length(w)]
    counts = [zero_crossings(w, zc_threshold), slope_sign_changes(w, ssc_threshold)]
    parts = [np.log1p(f) for f in amplitude] + counts
    if rich:
        parts += ar_coefficients(w) + tdpsd(w)
    return np.concatenate(parts, axis=-1)


def feature_names(n_channels: int, feature_set: str = DEFAULT_FEATURE_SET) -> list[str]:
    """Column names matching extract_features' layout, e.g. 'wl_ch2'."""
    return [f"{f}_ch{c}" for f in feature_set_names(feature_set) for c in range(n_channels)]


def channel_columns(
    channels: Sequence[int], n_channels: int, feature_set: str = DEFAULT_FEATURE_SET
) -> NDArray[np.intp]:
    """Feature columns that belong to a subset of channels.

    Every feature is computed per channel, so dropping channels is the same
    as dropping their columns. That makes "what if I only had 4 electrodes?"
    a cheap experiment: compute features once, then select columns.
    """
    n_per = len(feature_set_names(feature_set))
    if any(not 0 <= c < n_channels for c in channels):
        raise ValueError(f"channels must be in [0, {n_channels})")
    return np.array([f * n_channels + c for f in range(n_per) for c in channels], dtype=np.intp)


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
        """Add samples, shape (n, n_channels); returns the windows completed by them, oldest first.

        A window ends on every `step`-th sample since the stream began, as soon
        as `size` samples have arrived, however the stream is cut into blocks.
        """
        full = np.vstack([self.buffer, block])
        old = len(self.buffer)
        # rows of `block` (1-based) on which the step counter reaches `step`
        first = self.step - self.since_last
        ends = old + np.arange(first, len(block) + 1, self.step)
        windows = [full[end - self.size : end] for end in ends if end >= self.size]
        self.since_last = (self.since_last + len(block)) % self.step
        self.buffer = full[-self.size :]
        return windows
