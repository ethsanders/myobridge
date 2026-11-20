"""Streaming EMG conditioning: bandpass -> notch -> rectify -> RMS envelope.

"Streaming" means every filter remembers its internal state between calls, so
feeding the signal in 20-sample packets gives exactly the same output as
filtering the whole recording at once. Without that state, every packet would
start from zero and you'd get a click (filter transient) every 20 ms.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


class StreamingSOS:
    """A multi-channel IIR filter (second-order sections) that keeps its state."""

    def __init__(self, sos: np.ndarray, n_channels: int):
        self.sos = sos
        # One (2 delay values) state per biquad section per channel.
        self.zi = np.zeros((sos.shape[0], 2, n_channels))

    def process(self, x: np.ndarray) -> np.ndarray:
        """Filter a (n_samples, n_channels) block; the state carries over."""
        y, self.zi = signal.sosfilt(self.sos, x, axis=0, zi=self.zi)
        out: np.ndarray = y
        return out


class MovingRMS:
    """Root-mean-square over the last `window` samples, computed block by block.

    We keep the last window-1 squared samples from the previous block so the
    window can slide across packet boundaries.
    """

    def __init__(self, window: int, n_channels: int):
        self.window = window
        self.history = np.zeros((window - 1, n_channels))

    def process(self, x: np.ndarray) -> np.ndarray:
        squared = np.vstack([self.history, x**2])
        # Running sum trick: mean over each length-`window` slice in O(n).
        csum = np.cumsum(squared, axis=0)
        csum = np.vstack([np.zeros((1, x.shape[1])), csum])
        mean_sq = (csum[self.window :] - csum[: -self.window]) / self.window
        self.history = squared[-(self.window - 1) :] if self.window > 1 else self.history
        out: np.ndarray = np.sqrt(np.maximum(mean_sq, 0.0))  # max() guards tiny negative round-off
        return out


def design_bandpass(
    fs: float, low: float = 20.0, high: float = 450.0, order: int = 4
) -> np.ndarray:
    """Butterworth bandpass as second-order sections (numerically stable)."""
    if not 0 < low < high < fs / 2:
        raise ValueError(f"need 0 < low < high < Nyquist ({fs / 2} Hz)")
    sos: np.ndarray = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sos


def design_notch(fs: float, freq: float = 60.0, quality: float = 30.0) -> np.ndarray:
    """Narrow IIR notch at the mains frequency. Q=30 -> ~2 Hz wide at 60 Hz."""
    b, a = signal.iirnotch(freq, quality, fs=fs)
    sos: np.ndarray = signal.tf2sos(b, a)
    return sos


@dataclass
class FilterOutput:
    clean: np.ndarray  # bandpassed + notched EMG, same units as input
    envelope: np.ndarray  # RMS envelope of the rectified signal (>= 0)


class EmgFilter:
    """The full conditioning chain for one multi-channel stream."""

    def __init__(
        self,
        fs: float,
        n_channels: int,
        band: tuple[float, float] = (20.0, 450.0),
        mains_hz: float = 60.0,
        rms_window_s: float = 0.1,
    ):
        self.bandpass = StreamingSOS(design_bandpass(fs, *band), n_channels)
        self.notch = StreamingSOS(design_notch(fs, mains_hz), n_channels)
        self.rms = MovingRMS(max(1, int(round(rms_window_s * fs))), n_channels)

    def process(self, raw: np.ndarray) -> FilterOutput:
        x = np.asarray(raw, dtype=np.float64)
        clean = self.notch.process(self.bandpass.process(x))
        # Full-wave rectification. (RMS squares anyway, so the sign would vanish
        # regardless, but rectified EMG is the textbook intermediate signal.)
        rectified = np.abs(clean)
        return FilterOutput(clean=clean, envelope=self.rms.process(rectified))
