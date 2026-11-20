"""The MultiDay dataset (Kaufmann, Englehart & Platzner 2010): one person, many days.

What the mirror's README says (github.com/LibEMG/MultiDay): a single-subject
collection over about 120 days. The mirror holds 121 day indices x 11 classes,
4 channels at 2048 Hz, roughly 6 s per class per day, one file per class per
day. Electrodes were re-applied each day, so day-to-day changes in placement,
skin and posture are all in the data. That is exactly the drift a wearable
has to survive.

Files are S0_D<day>_C<label>.csv, whitespace-separated. Class names follow
LibEMG's loader. The 4th channel's amplitude is about 1000x smaller than the
others on every day (the mirror doesn't say why); everything downstream is
per-channel and scale-free, so it is used as-is.
"""

from __future__ import annotations

import re
from pathlib import Path

from .core import Dataset, DatasetError, Source, build_recordings, check_version, scan

SOURCE = Source(
    url="https://github.com/LibEMG/MultiDay",
    commit="2b11c859ca2dc3c7b52188c54bc0f5edeb12245a",
    citation=(
        "P. Kaufmann, K. Englehart and M. Platzner, 'Fluctuating EMG signals: Investigating "
        "long-term effects of pattern matching algorithms', IEEE EMBC 2010, pp. 6357-6360."
    ),
)
FS = 2048.0
N_CHANNELS = 4
MAINS_HZ = 50.0  # recorded in Germany (Paderborn); the spectra show a 50 Hz line
GESTURES = (
    "no_motion",
    "wrist_extension",
    "wrist_flexion",
    "wrist_adduction",
    "wrist_abduction",
    "supination",
    "pronation",
    "hand_open",
    "hand_closed",
    "key_grip",
    "index_point",
)
N_DAYS = 121

_PATTERN = re.compile(r"S0_D(\d+)_C(\d+)\.csv")


def load(folder: str | Path, days: list[int] | None = None, cache: bool = True) -> Dataset:
    """Load MultiDay from a local clone. `days` are the 1-based day indices."""
    root = Path(folder)
    check_version(root, SOURCE)
    files = scan(root, _PATTERN, lambda m: (0, int(m[1]), 0, int(m[2])))
    if days is not None:
        files = {k: v for k, v in files.items() if k[1] in set(days)}
    wanted = len(days) if days is not None else N_DAYS
    if len(files) != wanted * len(GESTURES):
        raise DatasetError(
            f"MultiDay: found {len(files)} files in {root}, expected {wanted * len(GESTURES)}"
        )
    recordings = tuple(build_recordings(files, FS, GESTURES, None, N_CHANNELS, cache))
    return Dataset("multiday", FS, MAINS_HZ, GESTURES, recordings, SOURCE)
