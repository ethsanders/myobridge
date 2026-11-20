"""The 3DC dataset (Côté-Allard et al.): 22 people, 10 channels, 1 kHz, 11 gestures.

What the mirror's README says (github.com/libemg/3DCDataset):
- 10 bipolar EMG channels around the forearm from the 3DC armband, sampled at
  1000 Hz after an analog anti-aliasing low-pass at 460 Hz.
- 11 classes, 5 s per gesture. One sitting: 4 cycles through all gestures
  (the "train" folder), a 5 minute break, 4 more cycles ("test" folder).
  The armband stays on, so train vs test is a *within-session* split.

Files are Participant<subject>/<train|test>/EMG/3dc_EMG_gesture_<rep>_<label>.txt,
comma-separated, one row per sample. Class names follow LibEMG's loader.
"""

from __future__ import annotations

import re
from pathlib import Path

from .core import Dataset, DatasetError, Source, build_recordings, check_version, scan

SOURCE = Source(
    url="https://github.com/libemg/3DCDataset",
    commit="1af9058237a5d79a883feb2ce61f4d59090288ac",
    citation=(
        "Côté-Allard et al., 'Interpreting deep learning features for myoelectric control: "
        "a comparison with handcrafted features', Frontiers in Bioengineering and "
        "Biotechnology 8:158 (2020); data first collected for Côté-Allard et al., IEEE TNSRE "
        "27(4):760-771 (2019)."
    ),
)
FS = 1000.0
N_CHANNELS = 10
MAINS_HZ = 60.0  # recorded in Québec
GESTURES = (
    "neutral",
    "radial_deviation",
    "wrist_flexion",
    "ulnar_deviation",
    "wrist_extension",
    "supination",
    "pronation",
    "power_grip",
    "open_hand",
    "chuck_grip",
    "pinch_grip",
)
SESSIONS = {"train": 0, "test": 1}
N_SUBJECTS, N_REPS = 22, 4

_PATTERN = re.compile(r"Participant(\d+)/(train|test)/EMG/3dc_EMG_gesture_(\d+)_(\d+)\.txt")


def load(folder: str | Path, subjects: list[int] | None = None, cache: bool = True) -> Dataset:
    """Load 3DC from a local clone. `subjects` are 1-based, as in the folder names."""
    root = Path(folder)
    check_version(root, SOURCE)
    files = scan(
        root,
        _PATTERN,
        lambda m: (int(m[1]), SESSIONS[m[2]], int(m[3]), int(m[4])),
    )
    if subjects is not None:
        files = {k: v for k, v in files.items() if k[0] in set(subjects)}
    wanted = len(subjects) if subjects is not None else N_SUBJECTS
    expected = wanted * len(SESSIONS) * N_REPS * len(GESTURES)
    if len(files) != expected:
        raise DatasetError(f"3DC: found {len(files)} EMG files in {root}, expected {expected}")
    recordings = tuple(build_recordings(files, FS, GESTURES, ",", N_CHANNELS, cache))
    return Dataset("3dc", FS, MAINS_HZ, GESTURES, recordings, SOURCE)
