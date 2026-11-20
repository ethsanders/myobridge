"""Your own armband, your own promise: the across-day recipe for one person.

`bench conformal` shows the recipe on public data: gesture sets that keep a
90% promise on 91 later days. This module runs the same recipe on sessions
you record yourself, so the promise is about *your* arm.

1. On each of a few days, record a `Session` (`myobridge calibrate --session
   day1.npz` does it): every gesture held for a few seconds.
2. `fit_sets(sessions)` trains the decoder on the early sessions, fits the
   temperature on the second-to-last one and sets the threshold on the last
   one. Both later sessions are first recalibrated the way you will
   recalibrate every day after (`adapt` with one second of each gesture),
   so the threshold is set on the kind of model you will actually use.
3. On any later day, `recalibrate(sets, today)` adapts the decoder to
   today's electrode placement with one second per gesture and keeps the
   temperature and threshold: that is what keeps the promise across days.

Within each hold, windows lying in `CALIBRATION_WINDOW` are the ones a
recalibration uses, and windows starting at `SCORE_FROM` or later are the
ones scored. These are the bench's own choices (see `bench.multiday_bench`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .classifier import LDAClassifier, TrainingSettings, adapt, recording_to_features
from .conformal import ConformalGestures, fit_temperature
from .features import DEFAULT_FEATURE_SET

CALIBRATION_WINDOW = (1.0, 2.0)  # seconds into each hold used to recalibrate
SCORE_FROM = 2.5  # windows starting this far into a hold are scored
SETTLE_S = 0.25  # training skips the first moment of each hold, as `calibrate` does
# adapt()'s setting, chosen on MultiDay's validation days by `bench multiday`
MEAN_PRIOR = 20.0
COVARIANCE_WEIGHT = 0.5
MIN_HOLD_S = SCORE_FROM + 0.5  # so every hold has some scored windows


@dataclass
class Session:
    """One sitting with the armband: each gesture held once (or more), raw counts kept.

    `holds[i]` is an (n_samples, n_channels) recording of `labels[i]`, from
    the moment the person was asked to hold it.
    """

    fs: float
    labels: tuple[str, ...]
    holds: tuple[NDArray[Any], ...]
    mains_hz: float = 60.0
    recorded_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    source: str = ""  # the file it was loaded from, if any

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.holds) or not self.holds:
            raise ValueError("need one recording per label, and at least one")
        widths = {h.shape[1] for h in self.holds if h.ndim == 2}
        if len(widths) != 1 or any(h.ndim != 2 for h in self.holds):
            raise ValueError("every hold must be (n_samples, n_channels) with the same channels")

    @property
    def n_channels(self) -> int:
        return int(self.holds[0].shape[1])

    @property
    def gestures(self) -> list[str]:
        """The gestures held, sorted (the order a model's classes come in)."""
        return sorted(set(self.labels))

    def save(self, path: str | Path) -> Path:
        """Write the session to an .npz file (the suffix is added if missing)."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        meta = {
            "fs": self.fs,
            "labels": list(self.labels),
            "mains_hz": self.mains_hz,
            "recorded_at": self.recorded_at,
        }
        arrays = {f"hold_{i}": h for i, h in enumerate(self.holds)}
        np.savez(path, meta=np.array(json.dumps(meta)), **arrays)  # type: ignore[arg-type]
        return path

    @classmethod
    def load(cls, path: str | Path) -> Session:
        """Read a session written by `save`."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "meta" not in data:
                raise ValueError(f"{path} is not a myobridge session (no metadata)")
            meta = json.loads(str(data["meta"]))
            holds = tuple(data[f"hold_{i}"] for i in range(len(meta["labels"])))
        return cls(
            float(meta["fs"]),
            tuple(meta["labels"]),
            holds,
            float(meta["mains_hz"]),
            meta["recorded_at"],
            str(path),
        )


def _windows(
    raw: NDArray[Any], fs: float, settings: TrainingSettings, feature_set: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Features of every window of one hold, and each window's start time (s)."""
    X = recording_to_features(
        raw,
        int(round(fs)),
        settings.window_s,
        settings.step_s,
        settle_s=0.0,
        mains_hz=settings.mains_hz,
        feature_set=feature_set,
    )
    starts = np.arange(len(X), dtype=np.float64) * settings.step_s
    return X, starts


def _rows(
    session: Session,
    settings: TrainingSettings,
    feature_set: str,
    t_min: float,
    t_max: float = np.inf,
) -> tuple[NDArray[np.float64], NDArray[Any]]:
    """Feature rows of windows lying wholly inside [t_min, t_max) of every hold."""
    Xs, ys = [], []
    for label, raw in zip(session.labels, session.holds, strict=True):
        X, start = _windows(raw, session.fs, settings, feature_set)
        keep = (start >= t_min - 1e-9) & (start + settings.window_s <= t_max + 1e-9)
        Xs.append(X[keep])
        ys += [label] * int(keep.sum())
    width = Xs[0].shape[1]
    return (np.vstack(Xs) if Xs else np.zeros((0, width))), np.array(ys)


def _check(sessions: Sequence[Session]) -> None:
    first = sessions[0]
    for s in sessions[1:]:
        name = s.source or s.recorded_at
        if not np.isclose(s.fs, first.fs):
            raise ValueError(f"{name} was recorded at {s.fs:g} Hz, the first at {first.fs:g} Hz")
        if s.n_channels != first.n_channels:
            raise ValueError(f"{name} has {s.n_channels} channels, the first {first.n_channels}")
        if s.gestures != first.gestures:
            raise ValueError(
                f"{name} holds {', '.join(s.gestures)}, the first {', '.join(first.gestures)}"
            )
    for s in sessions:
        short = min(len(h) for h in s.holds) / s.fs
        if short < MIN_HOLD_S:
            raise ValueError(
                f"{s.source or s.recorded_at} has a {short:.1f} s hold; the recipe needs "
                f"{MIN_HOLD_S:g} s or more (calibrate --seconds)"
            )


def recalibrate(sets: ConformalGestures, today: Session) -> ConformalGestures:
    """Adapt the decoder to today's placement; the temperature and threshold stay.

    Uses the windows in `CALIBRATION_WINDOW` of each of today's holds (one
    second per gesture), so `today` can be short: 2 s per gesture is enough.
    """
    base = sets.model
    if not isinstance(base, LDAClassifier) or base.settings is None:
        raise ValueError("recalibration needs a model saved by `fit_sets` or `calibrate`")
    X, y = _rows(today, base.settings, base.feature_set or DEFAULT_FEATURE_SET, *CALIBRATION_WINDOW)
    missing = set(np.asarray(base.classes).tolist()) - set(y.tolist())
    if missing:
        raise ValueError(f"today's recording has no {', '.join(sorted(map(str, missing)))}")
    return sets.with_model(adapt(base, X, y, MEAN_PRIOR, COVARIANCE_WEIGHT))


@dataclass
class FittedSets:
    """What `fit_sets` made: the sets (on the base model) and which session did what."""

    sets: ConformalGestures
    roles: list[dict[str, str]]  # one per session: source, recorded_at, role

    @property
    def model(self) -> LDAClassifier:
        assert isinstance(self.sets.model, LDAClassifier)
        return self.sets.model

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Write the model (`path`, .npz) and its sets beside it (`<name>.sets.json`)."""
        model_path = self.model.save(path)
        info = {
            **self.sets.to_dict(),
            "recipe": {
                "sessions": self.roles,
                "calibration_window_s": list(CALIBRATION_WINDOW),
                "score_from_s": SCORE_FROM,
                "adapt": {"mean_prior": MEAN_PRIOR, "covariance_weight": COVARIANCE_WEIGHT},
            },
        }
        sets_path = sets_path_for(model_path)
        sets_path.write_text(json.dumps(info, indent=2) + "\n")
        return model_path, sets_path


def sets_path_for(model_path: str | Path) -> Path:
    """Where `FittedSets.save` puts the sets for a model file."""
    p = Path(model_path)
    if p.suffix != ".npz":
        p = p.with_name(p.name + ".npz")
    return p.with_name(p.stem + ".sets.json")


def load_sets(model_path: str | Path) -> tuple[ConformalGestures, dict[str, Any]]:
    """The model and its sets, as `FittedSets.save` wrote them, and the recipe notes."""
    model = LDAClassifier.load(model_path)
    path = sets_path_for(model_path)
    if not path.exists():
        raise FileNotFoundError(f"no sets for this model at {path}; make them with `myobridge fit`")
    return ConformalGestures.load(path, model), json.loads(path.read_text()).get("recipe", {})


def fit_sets(
    sessions: Sequence[Session],
    alpha: float = 0.1,
    feature_set: str = DEFAULT_FEATURE_SET,
    seed: int | None = 0,
) -> FittedSets:
    """The across-day recipe on your own sessions, oldest first.

    With three or more sessions: train on all but the last two, fit the
    temperature on the second-to-last and set the threshold on the last.
    With two: train on the first, and split the second's scored windows by
    time within each hold, the earlier half for the temperature and the later
    half for the threshold. That is a weaker test of a new day (both halves
    share one placement), so record a third session when you can.
    """
    if len(sessions) < 2:
        raise ValueError(
            "the promise is about later days, so it needs sessions from at least two days "
            "(three is better)"
        )
    _check(sessions)
    first = sessions[0]
    settings = TrainingSettings(fs=float(first.fs), mains_hz=first.mains_hz)
    train = sessions[:-2] if len(sessions) >= 3 else sessions[:1]
    Xs, ys = zip(
        *(_rows(s, settings, feature_set, SETTLE_S) for s in train),
        strict=True,
    )
    base = LDAClassifier(feature_set=feature_set, settings=settings).fit(
        np.vstack(Xs), np.concatenate(ys)
    )

    def scored(s: Session, t_min: float, t_max: float = np.inf) -> tuple[Any, Any]:
        X_cal, y_cal = _rows(s, settings, feature_set, *CALIBRATION_WINDOW)
        model = adapt(base, X_cal, y_cal, MEAN_PRIOR, COVARIANCE_WEIGHT)
        X, y = _rows(s, settings, feature_set, t_min, t_max)
        index = np.searchsorted(np.asarray(base.classes), y)
        return model.decision_function(X), index

    if len(sessions) >= 3:
        dec_t, y_t = scored(sessions[-2], SCORE_FROM)
        dec_q, y_q = scored(sessions[-1], SCORE_FROM)
        roles = ["train"] * len(train) + ["temperature", "threshold"]
    else:
        shortest = min(len(h) for h in sessions[1].holds) / sessions[1].fs
        middle = SCORE_FROM + (shortest - SCORE_FROM) / 2
        dec_t, y_t = scored(sessions[1], SCORE_FROM, middle)
        dec_q, y_q = scored(sessions[1], middle)
        roles = ["train", "temperature and threshold (split in time)"]
    temperature = fit_temperature(dec_t, y_t)
    sets = ConformalGestures.from_scores(base, dec_q, y_q, alpha, temperature, "aps", seed=seed)
    return FittedSets(
        sets,
        [
            {"source": s.source, "recorded_at": s.recorded_at, "role": r}
            for s, r in zip(sessions, roles, strict=True)
        ],
    )
