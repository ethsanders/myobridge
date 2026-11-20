"""Tiny fake copies of the public datasets' folder layouts.

The real datasets are hundreds of MB and are not downloaded in CI; these
fixtures write a few small files with the same names and formats, so the
loaders, splits and benchmarks can be tested end to end in seconds.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from myobridge.datasets import multiday, threedc


def _signal(rng: np.random.Generator, n: int, n_ch: int, label: int, fs: float) -> np.ndarray:
    """Noise whose per-channel loudness depends on the gesture: easy to classify."""
    gain = 1.0 + 4.0 * np.abs(np.sin(np.arange(n_ch) * (label + 1)))
    if label == 0:
        gain = np.full(n_ch, 0.3)
    t = np.arange(n) / fs
    return rng.normal(0, 1, (n, n_ch)) * gain * 50 + 5 * np.sin(2 * np.pi * 60 * t)[:, None]


def write_threedc(root: Path, subjects=(1, 2), seconds: float = 3.0, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = int(seconds * threedc.FS)
    for s in subjects:
        for folder in ("train", "test"):
            d = root / f"Participant{s}" / folder / "EMG"
            d.mkdir(parents=True)
            for rep in range(threedc.N_REPS):
                for label in range(len(threedc.GESTURES)):
                    x = _signal(rng, n, threedc.N_CHANNELS, label, threedc.FS)
                    np.savetxt(
                        d / f"3dc_EMG_gesture_{rep}_{label}.txt", x, delimiter=",", fmt="%.1f"
                    )
    return root


def write_multiday(root: Path, days=(1, 2, 3), seconds: float = 3.0, seed: int = 0) -> Path:
    rng = np.random.default_rng(seed)
    n = int(seconds * multiday.FS)
    root.mkdir(parents=True, exist_ok=True)
    for day in days:
        for label in range(len(multiday.GESTURES)):
            x = _signal(rng, n, multiday.N_CHANNELS, label, multiday.FS)
            np.savetxt(root / f"S0_D{day}_C{label}.csv", x, fmt="%.2f")
    return root
