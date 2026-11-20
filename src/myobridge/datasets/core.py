"""Shared plumbing for public EMG datasets: the Recording type, download, caching.

Nothing here is specific to one dataset. Each dataset module describes where
its files live and how a filename maps to (subject, session, rep, gesture);
this module turns that into a list of `Recording`s.

The data itself is NOT part of this repository. `fetch()` clones the public
mirror the LibEMG project keeps on GitHub, at a pinned commit, so every number
in docs/RESULTS.md comes from exactly the same bytes.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ENV_VAR = "MYOBRIDGE_DATA"


class DatasetError(RuntimeError):
    """The dataset is missing, incomplete, or not the pinned version."""


@dataclass(frozen=True)
class Source:
    """Where a dataset comes from and which exact version we expect."""

    url: str
    commit: str  # full git SHA of the mirror we benchmarked
    citation: str


@dataclass(frozen=True)
class Recording:
    """One continuous recording of one gesture.

    `session` is whatever separates recordings in time for that dataset
    (a block within a sitting for 3DC, a day for MultiDay). Two recordings
    with the same (subject, session) were made without taking the electrodes
    off; different sessions may differ in electrode placement.
    """

    emg: FloatArray = field(repr=False)  # (n_samples, n_channels)
    fs: float
    gesture: str
    label: int
    subject: int
    session: int
    rep: int

    @property
    def duration_s(self) -> float:
        return float(self.emg.shape[0] / self.fs)

    @property
    def n_channels(self) -> int:
        return int(self.emg.shape[1])


@dataclass(frozen=True)
class Dataset:
    name: str
    fs: float
    mains_hz: float  # powerline frequency where it was recorded
    gestures: tuple[str, ...]  # index = label
    recordings: tuple[Recording, ...] = field(repr=False)
    source: Source

    @property
    def n_channels(self) -> int:
        return self.recordings[0].n_channels

    @property
    def subjects(self) -> list[int]:
        return sorted({r.subject for r in self.recordings})

    @property
    def sessions(self) -> list[int]:
        return sorted({r.session for r in self.recordings})

    def select(
        self,
        subjects: Iterable[int] | None = None,
        sessions: Iterable[int] | None = None,
        reps: Iterable[int] | None = None,
    ) -> list[Recording]:
        """Recordings matching every given filter (None = no filter on that field)."""
        subj = None if subjects is None else set(subjects)
        sess = None if sessions is None else set(sessions)
        rep_set = None if reps is None else set(reps)
        return [
            r
            for r in self.recordings
            if (subj is None or r.subject in subj)
            and (sess is None or r.session in sess)
            and (rep_set is None or r.rep in rep_set)
        ]


# ------------------------------------------------------------------ locations


def data_root(root: str | Path | None = None) -> Path:
    """Where datasets live: `root`, else $MYOBRIDGE_DATA, else ~/.cache/myobridge."""
    if root is not None:
        return Path(root).expanduser()
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser()
    return Path.home() / ".cache" / "myobridge"


def _git(*args: str, cwd: Path | None = None) -> str:
    # A list of arguments, never a shell string: paths with spaces or odd
    # characters can't be misread as extra commands.
    done = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=1800
    )
    return done.stdout.strip()


def fetch(name: str, source: Source, root: str | Path | None = None) -> Path:
    """Make sure the pinned version of a dataset is on disk; return its folder.

    Clones a shallow copy the first time. If the mirror has moved on since we
    pinned it, fetches the pinned commit explicitly; if that's impossible we
    stop rather than silently benchmark different data.
    """
    dest = data_root(root) / name
    if not (dest / ".git").exists():
        if dest.exists() and any(dest.iterdir()):
            raise DatasetError(f"{dest} exists but is not a git clone; move it away first")
        dest.parent.mkdir(parents=True, exist_ok=True)
        _git("clone", "--depth", "1", source.url, str(dest))
    head = _git("rev-parse", "HEAD", cwd=dest)
    if head != source.commit:
        try:
            _git("fetch", "--depth", "1", "origin", source.commit, cwd=dest)
            _git("checkout", "--detach", source.commit, cwd=dest)
        except subprocess.CalledProcessError as err:
            raise DatasetError(
                f"{name}: checkout is at {head}, expected pinned commit {source.commit} "
                f"and it could not be fetched ({err.stderr.strip()})"
            ) from err
    return dest


def check_version(folder: Path, source: Source) -> str | None:
    """The folder's git commit, or None if it's not a git checkout.

    Raises DatasetError if it IS a checkout of a different commit.
    """
    if not (folder / ".git").exists():
        return None
    head = _git("rev-parse", "HEAD", cwd=folder)
    if head != source.commit:
        raise DatasetError(f"{folder} is at commit {head}, expected {source.commit}")
    return head


# -------------------------------------------------------------------- parsing


FileKey = tuple[int, int, int, int]  # subject, session, rep, label


def scan(
    folder: Path, pattern: re.Pattern[str], key: Callable[[re.Match[str]], FileKey]
) -> dict[FileKey, Path]:
    """Map every file under `folder` whose relative path matches `pattern` to its key."""
    found: dict[FileKey, Path] = {}
    for path in sorted(folder.rglob("*")):
        rel = path.relative_to(folder).as_posix()
        m = pattern.fullmatch(rel)
        if m is None:
            continue
        k = key(m)
        if k in found:
            raise DatasetError(f"two files map to {k}: {found[k]} and {path}")
        found[k] = path
    return found


def load_matrix(path: Path, delimiter: str | None, n_channels: int, cache: bool) -> FloatArray:
    """Read one text matrix (rows = samples), using a .npy cache next to the data.

    The cache lives in <dataset>/.myobridge-cache/ and is ignored by git. It is
    rebuilt whenever the text file is newer than the cached copy.
    """
    cached = path.parent / ".myobridge-cache" / (path.name + ".npy")
    if cache and cached.exists() and cached.stat().st_mtime >= path.stat().st_mtime:
        out: FloatArray = np.load(cached)
        return out
    data = np.loadtxt(path, delimiter=delimiter, dtype=np.float64, ndmin=2)
    if data.shape[1] != n_channels or len(data) == 0:
        raise DatasetError(f"{path}: expected n x {n_channels} numbers, got {data.shape}")
    if not np.all(np.isfinite(data)):
        raise DatasetError(f"{path}: contains NaN or inf")
    if cache:
        cached.parent.mkdir(exist_ok=True)
        np.save(cached, data)
    return data


def build_recordings(
    files: dict[FileKey, Path],
    fs: float,
    gestures: tuple[str, ...],
    delimiter: str | None,
    n_channels: int,
    cache: bool,
) -> Iterator[Recording]:
    for (subject, session, rep, label), path in sorted(files.items()):
        if not 0 <= label < len(gestures):
            raise DatasetError(f"{path}: label {label} out of range")
        yield Recording(
            emg=load_matrix(path, delimiter, n_channels, cache),
            fs=fs,
            gesture=gestures[label],
            label=label,
            subject=subject,
            session=session,
            rep=rep,
        )
