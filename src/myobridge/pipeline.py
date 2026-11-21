"""Raw blocks in, smoothed gesture predictions out. Used live, in replays and in the demo.

Optionally the pipeline also watches signal quality: give it a
`QualityMonitor` and it stops using any channel the monitor flags for one of
`drop_on` (an electrode that came off, or one that clips), switching to the
same model minus that channel's features until the channel recovers.

Give it a `ConformalGestures` calibration too and each prediction also
carries the window's gesture *set*: one gesture when the model is sure, more
(or none) when it isn't. See `myobridge.conformal`.

Give it a `DecisionPolicy` (see `myobridge.gating`) and each prediction also
carries the policy's `action`: the gesture to act on, or None to hold.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from .classifier import LDAClassifier, MajorityVote
from .dsp import EmgFilter
from .features import FEATURE_SETS, StreamingWindower, Threshold, channel_columns, extract_features
from .quality import DROP_CHANNEL_ON, Issue, QualityMonitor, QualityReport

if TYPE_CHECKING:
    from .conformal import ConformalGestures
    from .gating import DecisionPolicy


@dataclass
class Prediction:
    """One decision of the live pipeline, made at the end of a window."""

    gesture: str  # after majority-vote smoothing
    raw_gesture: str  # what LDA said for this single window
    confidence: float  # LDA probability of `gesture` for this window
    sample_index: int  # stream position (samples) where the window ended
    dropped_channels: tuple[int, ...] = ()  # channels left out because the monitor flagged them
    gesture_set: tuple[str, ...] | None = None  # this window's conformal set, if calibrated
    action: str | None = None  # the policy's decision (None: hold); `gesture` without a policy

    @property
    def sure(self) -> bool:
        """True when a conformal set was computed and holds exactly one gesture."""
        return self.gesture_set is not None and len(self.gesture_set) == 1


class GesturePipeline:
    """Filter, window, extract features, classify and smooth a stream of raw EMG.

    `model` must have been trained on features of the same kind: its
    `feature_set` (or, for a model fitted on bare arrays, its feature count)
    picks them. `zc_threshold` and `ssc_threshold` are the zero-crossing and
    slope-sign-change thresholds used in training; the defaults suit the
    ESP32's 12-bit counts (see features.noise_thresholds for other signals).

    A model from `calibrate` (or a file it saved) knows the sample rate and
    window it was trained with; `GesturePipeline.from_model` uses them, and
    passing a different sample rate or window here raises an error.

    `policy` (see `myobridge.gating`) decides, window by window, whether to
    act and on which gesture; its answer is `Prediction.action`. It sees the
    conformal calibration's temperature-scaled probabilities when there is
    one (else the model's own) and the conformal sets. Without a policy,
    `action` is the voted `gesture`, as before.
    """

    def __init__(
        self,
        model: LDAClassifier,
        fs: float,
        n_channels: int,
        window_s: float = 0.2,
        step_s: float = 0.05,
        vote_k: int = 5,
        mains_hz: float = 60.0,
        zc_threshold: Threshold = 5.0,
        ssc_threshold: Threshold = 25.0,
        monitor: QualityMonitor | None = None,
        drop_on: Collection[Issue] = DROP_CHANNEL_ON,
        conformal: ConformalGestures | None = None,
        policy: DecisionPolicy | None = None,
    ):
        if not model.is_fitted:
            raise ValueError("the model is not fitted")
        if monitor is not None and monitor.n_channels != n_channels:
            raise ValueError("the monitor watches a different number of channels")
        if policy is not None and policy.needs_sets and conformal is None:
            raise ValueError(f"{type(policy).__name__} needs conformal sets: pass conformal=...")
        if conformal is not None and conformal.model is not model:
            conformal = conformal.with_model(model)  # same classes required
        trained = model.settings
        if trained is not None:
            for name, ours, theirs in (
                ("sample rate", fs, trained.fs),
                ("window_s", window_s, trained.window_s),
                ("step_s", step_s, trained.step_s),
            ):
                if not np.isclose(ours, theirs):
                    raise ValueError(
                        f"the model was trained with {name}={theirs:g}, not {ours:g}; "
                        "use GesturePipeline.from_model to match it"
                    )
        self.model = model
        self.n_channels = n_channels
        self.feature_set = _feature_set_for(model, n_channels)
        self.thresholds = (zc_threshold, ssc_threshold)
        self.filter = EmgFilter(fs, n_channels, mains_hz=mains_hz)
        size, step = int(round(window_s * fs)), int(round(step_s * fs))
        self.windower = StreamingWindower(size, step, n_channels)
        self.vote = MajorityVote(vote_k)
        self.monitor = monitor
        self.drop_on = frozenset(drop_on)
        self.samples_seen = 0
        self.last_envelope = np.zeros(n_channels)  # handy for live displays
        self.last_quality: QualityReport | None = None
        self.dropped: tuple[int, ...] = ()
        self._labels = [str(c) for c in model.classes]  # labels may be ints or strings
        self._models: dict[tuple[int, ...], LDAClassifier] = {(): model}
        self.conformal = conformal
        self._conformal: dict[tuple[int, ...], ConformalGestures] = {}
        if conformal is not None:
            self._conformal[()] = conformal
        self.policy = policy

    @classmethod
    def from_model(cls, model: LDAClassifier, n_channels: int, **overrides: Any) -> GesturePipeline:
        """A pipeline set up exactly as `model` was trained (see `TrainingSettings`).

        `overrides` are passed on to the constructor, e.g. `vote_k=3` or
        `monitor=...`.
        """
        if model.settings is None:
            raise ValueError("the model doesn't record how it was trained; use the constructor")
        s = model.settings
        kwargs: dict[str, Any] = {
            "window_s": s.window_s,
            "step_s": s.step_s,
            "mains_hz": s.mains_hz,
            "zc_threshold": s.zc_threshold,
            "ssc_threshold": s.ssc_threshold,
        }
        kwargs.update(overrides)
        return cls(model, s.fs, n_channels, **kwargs)

    def process_block(self, raw: ArrayLike) -> list[Prediction]:
        """Feed one packet's worth of samples; returns 0 or more predictions."""
        block = np.asarray(raw, dtype=np.float64)
        if block.ndim != 2 or block.shape[1] != self.n_channels:
            raise ValueError(f"expected samples shaped (n, {self.n_channels}), got {block.shape}")
        if len(block) == 0:  # e.g. a zero-frame packet: nothing to do
            return []
        if self.monitor is not None:
            report = self.monitor.push(block)
            if report is not None:
                self.last_quality = report
                self.dropped = self._channels_to_drop(report)
        out = self.filter.process(block)
        self.last_envelope = out.envelope[-1]
        self.samples_seen += block.shape[0]

        windows = self.windower.push(out.clean)
        if not windows:
            return []
        # Windows are `step` samples apart; the newest ended `since_last` samples ago.
        step = self.windower.step
        last_end = self.samples_seen - self.windower.since_last
        # One batched feature call for every window this block completed.
        X = extract_features(np.stack(windows), *self.thresholds, self.feature_set)
        model = self._model_without(self.dropped)
        if self.dropped:
            keep = [c for c in range(self.n_channels) if c not in self.dropped]
            X = X[:, channel_columns(keep, self.n_channels, self.feature_set)]
        sets = None
        scaled = None
        if self.conformal is not None:
            conf = self._conformal_without(self.dropped)
            sets = conf.predict_sets(X)
            if self.policy is not None:
                scaled = conf.predict_proba(X)
        predictions = []
        for i, proba in enumerate(model.predict_proba(X)):
            label = self._labels[int(np.argmax(proba))]
            voted = self.vote.update(label)
            action: str | None = voted
            if self.policy is not None:
                chosen = self.policy.step(
                    proba if scaled is None else scaled[i], None if sets is None else sets[i]
                )
                action = None if chosen is None else self._labels[chosen]
            predictions.append(
                Prediction(
                    gesture=voted,
                    raw_gesture=label,
                    confidence=float(proba[self._labels.index(voted)]),
                    sample_index=last_end - (len(windows) - 1 - i) * step,
                    dropped_channels=self.dropped,
                    gesture_set=None
                    if sets is None
                    else tuple(self._labels[j] for j in np.flatnonzero(sets[i])),
                    action=action,
                )
            )
        return predictions

    def _channels_to_drop(self, report: QualityReport) -> tuple[int, ...]:
        bad = tuple(c.channel for c in report.channels if self.drop_on & set(c.issues))
        # With every channel flagged there is nothing better to fall back on.
        return () if len(bad) == self.n_channels else bad

    def _conformal_without(self, channels: tuple[int, ...]) -> ConformalGestures:
        # A model without some channels keeps the full model's thresholds.
        # The promise was calibrated for the full model only; bench conformal
        # measures what a dropped channel does to it.
        assert self.conformal is not None
        if channels not in self._conformal:
            self._conformal[channels] = self.conformal.with_model(self._model_without(channels))
        return self._conformal[channels]

    def _model_without(self, channels: tuple[int, ...]) -> LDAClassifier:
        if channels not in self._models:
            keep = [c for c in range(self.n_channels) if c not in channels]
            cols = channel_columns(keep, self.n_channels, self.feature_set)
            self._models[channels] = self.model.subset(cols)
        return self._models[channels]


def _feature_set_for(model: LDAClassifier, n_channels: int) -> str:
    """The feature set to compute for `model`: the one it records, or else the
    one whose size matches its weights (a model fitted directly on arrays)."""
    n_features = model.n_features
    candidates = [model.feature_set] if model.feature_set else list(FEATURE_SETS)
    for name in candidates:
        if name in FEATURE_SETS and len(FEATURE_SETS[name]) * n_channels == n_features:
            return name
    raise ValueError(
        f"the model expects {n_features} features, which doesn't match "
        f"{model.feature_set or 'any feature set'} with {n_channels} channels"
    )
