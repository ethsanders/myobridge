"""Electrode faults, simulated on real recordings.

These are the faults the quality benchmark injects (docs/signal-quality.md),
and what the browser demo's "break a channel" buttons do. Each takes one
channel's raw samples and returns a broken copy; `band_rms` is the channel's
20-450 Hz RMS, which sets how strong hum and motion are relative to its EMG.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


class Fault(str, Enum):
    """A hardware fault the demo and the quality benchmark can simulate."""

    LEAD_OFF = "lead_off"
    CLIPPING = "clipping"
    LINE_NOISE = "line_noise"
    MOTION = "motion"


def lead_off(x: ArrayLike, scale: float = 0.03) -> FloatArray:
    """A lifted electrode: the signal shrinks to `scale` of its size around its mean."""
    xa = np.asarray(x, dtype=np.float64)
    out: FloatArray = xa.mean() + (xa - xa.mean()) * scale
    return out


def mains_hum(
    x: ArrayLike,
    band_rms: float,
    fs: float,
    mains_hz: float = 60.0,
    power_ratio: float = 1.0,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """Add a mains sine with `power_ratio` times the channel's EMG power."""
    xa = np.asarray(x, dtype=np.float64)
    rng = rng or np.random.default_rng()
    t = np.arange(len(xa)) / fs
    amp = np.sqrt(2.0 * power_ratio) * band_rms
    out: FloatArray = xa + amp * np.sin(2 * np.pi * mains_hz * t + rng.uniform(0, 2 * np.pi))
    return out


def motion(
    x: ArrayLike,
    band_rms: float,
    fs: float,
    rms_ratio: float = 3.0,
    rng: np.random.Generator | None = None,
) -> FloatArray:
    """A slowly wandering baseline, like a tugged cable: noise low-passed at ~5 Hz,
    `rms_ratio` times the channel's EMG RMS."""
    xa = np.asarray(x, dtype=np.float64)
    rng = rng or np.random.default_rng()
    n = len(xa)
    knots = rng.standard_normal(int(n / fs * 10) + 2)  # 10 random points per second
    wander = np.interp(np.arange(n) / fs * 10, np.arange(len(knots)), knots)
    wander *= rms_ratio * band_rms / max(float(np.std(wander)), 1e-12)
    out: FloatArray = xa + wander
    return out


def clipping(x: ArrayLike, rail: float, overdrive: float = 2.0) -> FloatArray:
    """Too much gain: scale so the 99th-percentile peak is `overdrive` x the ADC
    rail, then clip just outside +-rail, as a saturating amplifier would."""
    xa = np.asarray(x, dtype=np.float64)
    centred = xa - xa.mean()
    gain = overdrive * rail / max(float(np.percentile(np.abs(centred), 99)), 1e-12)
    out: FloatArray = np.clip(centred * gain, -rail - 1, rail + 1)
    return out
