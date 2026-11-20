"""How much does sampling at 1 kHz fold out-of-band energy into the EMG band?

Two separate questions, because they have different answers:

1. **The EMG itself.** Surface EMG has very little power above ~500 Hz (the
   skin and fat between muscle and electrode act as a low-pass filter). The
   MultiDay recordings, sampled at 2048 Hz, let us *measure* this and re-run
   the whole pipeline at an emulated 1024 Hz with and without anti-alias
   filtering (`decimate_chain`; results in docs/ALIASING.md).

2. **Broadband noise.** Amplifier noise, ADC noise and digital interference
   are roughly white up to tens of kHz. Whatever is left of them above
   fs/2 after the analog filter folds straight into the band. With only a
   first-order roll-off that is a lot. `folded_noise_ratio` computes it for
   white noise through an n-th order Butterworth-shaped low-pass.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import integrate, signal

FloatArray = NDArray[np.float64]


def lowpass_power(f: FloatArray, cutoff_hz: float, order: int) -> FloatArray:
    """|H(f)|^2 of an order-n Butterworth-shaped low-pass (order 1 = a simple RC)."""
    if cutoff_hz <= 0 or order < 1:
        raise ValueError("cutoff must be > 0 and order >= 1")
    out: FloatArray = 1.0 / (1.0 + (np.abs(f) / cutoff_hz) ** (2 * order))
    return out


def folded_noise_ratio(
    fs: float,
    cutoff_hz: float,
    order: int,
    band: tuple[float, float] = (20.0, 450.0),
    n_images: int = 5000,
    n_grid: int = 501,
) -> float:
    """Folded noise power / in-band noise power, for white noise before the ADC.

    After sampling at `fs`, the component at frequency f inside `band` also
    contains everything that was at k*fs - f and k*fs + f (k = 1, 2, ...).
    With white input noise, each contributes |H|^2 at that frequency. The
    ratio says how much the in-band noise floor grows because of aliasing:
    0.0 = no growth, 1.0 = doubled (+3 dB). Assumes everything after the ADC
    (digital filtering, decimation) is ideal.

    The sum over images is truncated at `n_images`; the remaining tail falls
    off like 1/k^(2*order) and is negligible at the default.
    """
    lo, hi = band
    if not 0 <= lo < hi <= fs / 2:
        raise ValueError("band must lie in [0, fs/2]")
    f = np.linspace(lo, hi, n_grid)
    in_band = integrate.trapezoid(lowpass_power(f, cutoff_hz, order), f)
    k = np.arange(1, n_images + 1)[:, None] * fs
    images = lowpass_power(k - f, cutoff_hz, order) + lowpass_power(k + f, cutoff_hz, order)
    folded = integrate.trapezoid(images.sum(axis=0), f)
    return float(folded / in_band)


def folded_noise_ratio_decimated(
    fs_out: float,
    cutoff_hz: float,
    order: int,
    taps: ArrayLike,
    band: tuple[float, float] = (20.0, 450.0),
    n_images: int = 5000,
    n_grid: int = 501,
) -> float:
    """Like folded_noise_ratio, for an ADC at 2 * fs_out followed by a real
    decimate-by-2 FIR (`taps`, at the ADC rate) instead of an ideal one.

    Output frequency f (in `band`) receives, through the FIR's gain G:
      G(f)^2        x everything the ADC folded onto f       (images of k * 2fs_out)
      G(fs_out-f)^2 x everything the ADC put at fs_out - f   (folds at decimation)
    and the "wanted" noise is G(f)^2 |H(f)|^2. So this is the noise-floor
    growth the firmware's OVERSAMPLE 2 path actually gives, filter flaws included.
    """
    lo, hi = band
    if not 0 <= lo < hi <= fs_out / 2:
        raise ValueError("band must lie in [0, fs_out/2]")
    fs_adc = 2.0 * fs_out
    h = np.asarray(taps, dtype=np.float64)
    h = h / h.sum()  # unity gain at DC, whatever the fixed-point scale

    def gain2(freq: FloatArray) -> FloatArray:
        _, resp = signal.freqz(h, worN=freq, fs=fs_adc)
        out: FloatArray = np.abs(resp) ** 2
        return out

    def adc_psd(freq: FloatArray) -> tuple[FloatArray, FloatArray]:
        """(the frequency's own noise, what folds onto it) right after the ADC."""
        own = lowpass_power(freq, cutoff_hz, order)
        k = np.arange(1, n_images + 1)[:, None] * fs_adc
        imgs = lowpass_power(k - freq, cutoff_hz, order) + lowpass_power(k + freq, cutoff_hz, order)
        return own, imgs.sum(axis=0)

    f = np.linspace(lo, hi, n_grid)
    own, folded_adc = adc_psd(f)
    own_m, folded_m = adc_psd(fs_out - f)
    g, g_m = gain2(f), gain2(fs_out - f)
    wanted = integrate.trapezoid(g * own, f)
    folded = integrate.trapezoid(g * folded_adc + g_m * (own_m + folded_m), f)
    return float(folded / wanted)


def ratio_to_db(ratio: float) -> float:
    """Noise-floor increase in dB for a folded/in-band power ratio."""
    return float(10.0 * np.log10(1.0 + ratio))


# --------------------------------------------------------- emulated sampling


def first_order_lowpass(x: FloatArray, fs: float, cutoff_hz: float) -> FloatArray:
    """Digital stand-in for an analog RC low-pass (impulse-invariant one-pole)."""
    a = np.exp(-2.0 * np.pi * cutoff_hz / fs)
    out: FloatArray = signal.lfilter([1.0 - a], [1.0, -a], x, axis=0)
    return out


Chain = Callable[[FloatArray], FloatArray]


def decimate_chain(name: str, fs: float, factor: int = 2, rc_cutoff_hz: float = 498.0) -> Chain:
    """A way to turn a recording at `fs` into one at fs/factor.

    - "ideal":    proper anti-alias FIR, then keep every factor-th sample.
    - "rc":       first-order low-pass at `rc_cutoff_hz` (MyoWare-like), then
                  keep every factor-th sample. Aliasing AND the RC's in-band droop.
    - "rc_ideal": the same RC, then an ideal decimator. Comparing "rc" with
                  "rc_ideal" isolates the aliasing part.
    - "none":     keep every factor-th sample, no filter at all.
    """
    if factor < 2:
        raise ValueError("factor must be >= 2")

    def ideal(x: FloatArray) -> FloatArray:
        out: FloatArray = signal.decimate(x, factor, ftype="fir", n=64, axis=0, zero_phase=True)
        return out

    chains: dict[str, Chain] = {
        "ideal": ideal,
        "rc": lambda x: first_order_lowpass(x, fs, rc_cutoff_hz)[::factor],
        "rc_ideal": lambda x: ideal(first_order_lowpass(x, fs, rc_cutoff_hz)),
        "none": lambda x: np.asarray(x)[::factor],
    }
    if name not in chains:
        raise KeyError(f"unknown chain {name!r}; choose from {sorted(chains)}")
    return chains[name]


def band_power_fraction(
    x: FloatArray, fs: float, above_hz: float, floor_hz: float = 20.0
) -> FloatArray:
    """Per channel: power above `above_hz` / power above `floor_hz` (Welch PSD)."""
    nperseg = min(len(x), int(fs))
    f, p = signal.welch(x, fs=fs, nperseg=nperseg, axis=0)
    total = p[f >= floor_hz].sum(axis=0)
    out: FloatArray = p[f > above_hz].sum(axis=0) / np.where(total > 0, total, 1.0)
    return out
