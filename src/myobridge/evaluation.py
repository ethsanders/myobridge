"""Turn dataset recordings into feature tables, and score models on them honestly.

The rule every experiment follows: a window used for testing never shares a
single sample with a window used for training or for choosing a setting.
That is guaranteed structurally, not by hope:

- Splits are made on whole recordings (subject, session, rep), or on time
  ranges inside a recording with a gap of at least one window between them.
- Feature thresholds (zero crossings, slope sign changes) are estimated from
  the training part only (`fit_thresholds`).
- Hyperparameters such as the recalibration prior are chosen on validation
  subjects/days that are not reported (see experiments.py).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
from numpy.typing import NDArray

from .datasets.core import Recording
from .dsp import EmgFilter
from .features import extract_features, noise_thresholds, window_stack

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class FeatureConfig:
    """How recordings become feature windows. Defaults match the live pipeline."""

    window_s: float = 0.2
    step_s: float = 0.05
    settle_s: float = 0.25  # skip filter start-up and the move into the pose
    band: tuple[float, float] = (20.0, 450.0)
    threshold_k: float = 0.5  # ZC/SSC thresholds = k * rest std (see noise_thresholds)

    def window_samples(self, fs: float) -> int:
        return int(round(self.window_s * fs))

    def step_samples(self, fs: float) -> int:
        return int(round(self.step_s * fs))


@dataclass(frozen=True)
class Thresholds:
    zc: FloatArray
    ssc: FloatArray


@dataclass
class FeatureTable:
    """All windows of a set of recordings, one row per window.

    `t_start`/`t_end` are the window's position (seconds) in its recording,
    which is what time-based splits use. `recording` indexes into the list of
    recordings the table was built from.
    """

    X: FloatArray
    label: IntArray
    subject: IntArray
    session: IntArray
    rep: IntArray
    recording: IntArray
    t_start: FloatArray
    t_end: FloatArray
    n_channels: int
    meta: dict[str, object] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def take(self, mask: NDArray[np.bool_] | NDArray[np.intp]) -> FeatureTable:
        return replace(
            self,
            X=self.X[mask],
            label=self.label[mask],
            subject=self.subject[mask],
            session=self.session[mask],
            rep=self.rep[mask],
            recording=self.recording[mask],
            t_start=self.t_start[mask],
            t_end=self.t_end[mask],
        )

    def where(
        self,
        subjects: Sequence[int] | None = None,
        sessions: Sequence[int] | None = None,
        reps: Sequence[int] | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
        from_end_s: float | None = None,
    ) -> FeatureTable:
        """Rows matching every filter.

        t_min / t_max keep windows lying entirely inside [t_min, t_max] seconds
        of their recording. from_end_s keeps windows inside the last
        `from_end_s` seconds of their recording.
        """
        mask = np.ones(len(self), dtype=bool)
        if subjects is not None:
            mask &= np.isin(self.subject, subjects)
        if sessions is not None:
            mask &= np.isin(self.session, sessions)
        if reps is not None:
            mask &= np.isin(self.rep, reps)
        if t_min is not None:
            mask &= self.t_start >= t_min - 1e-9
        if t_max is not None:
            mask &= self.t_end <= t_max + 1e-9
        if from_end_s is not None:
            duration = np.asarray(self.meta["durations"], dtype=np.float64)[self.recording]
            mask &= self.t_start >= duration - from_end_s - 1e-9
        return self.take(mask)


def filter_recording(rec: Recording, mains_hz: float, cfg: FeatureConfig) -> FloatArray:
    """Bandpass + notch one recording (fresh filter state, as if streamed from t=0)."""
    high = min(cfg.band[1], 0.45 * rec.fs)
    f = EmgFilter(rec.fs, rec.n_channels, band=(cfg.band[0], high), mains_hz=mains_hz)
    clean: FloatArray = f.process(rec.emg).clean
    return clean


def prefilter(
    recordings: Sequence[Recording], mains_hz: float, cfg: FeatureConfig
) -> list[FloatArray]:
    """filter_recording() for every recording. Filtering is the slow step, so
    experiments that featurize the same recordings many times do it once."""
    return [filter_recording(r, mains_hz, cfg) for r in recordings]


def fit_thresholds(
    recordings: Sequence[Recording],
    mains_hz: float,
    cfg: FeatureConfig,
    rest_label: int = 0,
    clean: Sequence[FloatArray] | None = None,
) -> Thresholds:
    """ZC/SSC thresholds from the rest recordings among `recordings` (training data).

    `clean`, if given, holds the already-filtered recordings (same order).
    """
    if clean is not None and len(clean) != len(recordings):
        raise ValueError("clean must match recordings one to one")
    rest_idx = [i for i, r in enumerate(recordings) if r.label == rest_label]
    if not rest_idx:
        raise ValueError("no rest recordings to estimate noise thresholds from")
    parts = []
    for i in rest_idx:
        r = recordings[i]
        x = clean[i] if clean is not None else filter_recording(r, mains_hz, cfg)
        parts.append(x[int(round(cfg.settle_s * r.fs)) :])
    zc, ssc = noise_thresholds(np.vstack(parts), cfg.threshold_k)
    return Thresholds(zc=zc, ssc=ssc)


def featurize(
    recordings: Sequence[Recording],
    mains_hz: float,
    cfg: FeatureConfig,
    thresholds: Thresholds,
    clean: Sequence[FloatArray] | None = None,
) -> FeatureTable:
    """Filter every recording, cut it into windows and compute features.

    `clean`, if given, holds the already-filtered recordings (same order).
    """
    if not recordings:
        raise ValueError("no recordings")
    if clean is not None and len(clean) != len(recordings):
        raise ValueError("clean must match recordings one to one")
    fs = recordings[0].fs
    n_ch = recordings[0].n_channels
    if any(r.fs != fs or r.n_channels != n_ch for r in recordings):
        raise ValueError("all recordings must share a sample rate and channel count")
    size, step = cfg.window_samples(fs), cfg.step_samples(fs)
    skip = int(round(cfg.settle_s * fs))

    blocks: list[FloatArray] = []
    cols: dict[str, list[NDArray[np.generic]]] = {k: [] for k in ("lab", "sub", "ses", "rep")}
    rec_idx: list[IntArray] = []
    starts: list[FloatArray] = []
    for i, rec in enumerate(recordings):
        x = clean[i] if clean is not None else filter_recording(rec, mains_hz, cfg)
        windows = window_stack(x[skip:], size, step)
        n = len(windows)
        if n == 0:
            continue  # recording shorter than settle + one window
        blocks.append(extract_features(windows, thresholds.zc, thresholds.ssc))
        cols["lab"].append(np.full(n, rec.label))
        cols["sub"].append(np.full(n, rec.subject))
        cols["ses"].append(np.full(n, rec.session))
        cols["rep"].append(np.full(n, rec.rep))
        rec_idx.append(np.full(n, i, dtype=np.int64))
        starts.append((skip + np.arange(n) * step) / fs)

    n_feat = 5 * n_ch
    if not blocks:
        raise ValueError("no recording was long enough for a single window")
    t_start = np.concatenate(starts)
    return FeatureTable(
        X=np.vstack(blocks).reshape(-1, n_feat),
        label=np.concatenate(cols["lab"]).astype(np.int64),
        subject=np.concatenate(cols["sub"]).astype(np.int64),
        session=np.concatenate(cols["ses"]).astype(np.int64),
        rep=np.concatenate(cols["rep"]).astype(np.int64),
        recording=np.concatenate(rec_idx),
        t_start=t_start,
        t_end=t_start + size / fs,
        n_channels=n_ch,
        meta={"durations": [r.duration_s for r in recordings], "fs": fs},
    )


# ------------------------------------------------------------------ metrics


def accuracy(y_true: NDArray[np.generic], y_pred: NDArray[np.generic]) -> float:
    if len(y_true) == 0:
        raise ValueError("empty test set")
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def balanced_accuracy(y_true: NDArray[np.generic], y_pred: NDArray[np.generic]) -> float:
    """Mean per-class recall: not fooled by one class having more windows."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    classes = np.unique(yt)
    return float(np.mean([np.mean(yp[yt == c] == c) for c in classes]))


def confusion(
    y_true: NDArray[np.generic], y_pred: NDArray[np.generic], n_classes: int
) -> NDArray[np.int64]:
    """Counts, rows = true class, columns = predicted class."""
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(m, (np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)), 1)
    return m


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean, standard deviation (n-1), min, max, and a 95% bootstrap CI of the mean.

    The bootstrap uses a fixed seed so results files are reproducible.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(0)
    boot = rng.choice(v, size=(2000, v.size), replace=True).mean(axis=1)
    return {
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "ci95_low": float(np.percentile(boot, 2.5)),
        "ci95_high": float(np.percentile(boot, 97.5)),
        "n": int(v.size),
    }
