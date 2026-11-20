"""Replay recorded EMG through the live pipeline, as if it were streaming in.

This is what the browser demo runs: a model is calibrated on one part of a
person's recordings, then another part is fed through `GesturePipeline`
20 samples at a time, exactly the way packets arrive from the armband, with
the signal-quality monitor watching every channel. `inject()` breaks a
channel from the current moment on, so you can watch the monitor catch it
and the decoder carry on without it.

Everything here works on arrays, so it runs the same in Python and in the
browser (Pyodide), whichever way the recordings were loaded.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from . import faults
from .classifier import LDAClassifier
from .conformal import ConformalGestures, blocked_scores
from .datasets.core import Recording
from .evaluation import FeatureConfig, featurize, filter_recording, fit_thresholds
from .faults import Fault
from .features import DEFAULT_FEATURE_SET
from .pipeline import GesturePipeline
from .quality import DROP_CHANNEL_ON, QualityConfig, QualityMonitor, rest_floor

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ChannelState:
    """One channel as the monitor and the decoder see it after a `step()`."""

    channel: int
    issues: tuple[str, ...]  # empty when the channel looks healthy
    dropped: bool  # the decoder is ignoring this channel right now
    fault: str | None  # the fault injected into it, if any


@dataclass(frozen=True)
class Frame:
    """What happened during one `step()`."""

    t: float  # seconds of the replay played so far
    truth: str  # the gesture actually being held at time t
    prediction: str | None  # the pipeline's smoothed output (None before the first window)
    confidence: float
    channels: tuple[ChannelState, ...]
    raw: FloatArray  # the raw samples of this step, every `display_every`-th one
    accuracy: float | None  # share of predictions so far that matched the truth
    gesture_set: tuple[str, ...] | None = None  # the newest window's set, if calibrated
    sets: SetCounts | None = None  # how the sets have done so far


@dataclass(frozen=True)
class SetCounts:
    """Running counts for the gesture sets, window by window."""

    windows: int
    covered: int  # sets that held the gesture being held
    singletons: int  # sets of exactly one gesture
    singletons_right: int  # ... that were the gesture being held

    @property
    def coverage(self) -> float | None:
        return self.covered / self.windows if self.windows else None

    @property
    def singleton_rate(self) -> float | None:
        return self.singletons / self.windows if self.windows else None

    @property
    def accuracy_when_singleton(self) -> float | None:
        return self.singletons_right / self.singletons if self.singletons else None


class Replay:
    """Calibrate on `train`, then stream `test` through the live pipeline.

    `train` must include at least one rest recording (label `rest_label`): it
    sets the zero-crossing thresholds and the monitor's rest noise floor.
    `channels` picks a subset of electrodes, e.g. four to match the
    myobridge armband. `rail` is the ADC's full scale in signal units, if
    known, so clipping can be detected.

    With `alpha`, every window also gets a conformal gesture set that should
    hold the true gesture 1 - alpha of the time. Its threshold comes from
    out-of-repetition scores on `train` (fit on the other repetitions, score
    this one), so `train` needs at least two repetitions; `temperature`
    softens the scores first (see `myobridge.conformal`).
    """

    def __init__(
        self,
        train: Sequence[Recording],
        test: Sequence[Recording],
        gestures: Sequence[str],
        mains_hz: float = 60.0,
        channels: Sequence[int] | None = None,
        rail: float | None = None,
        block: int = 20,
        feature_set: str = DEFAULT_FEATURE_SET,
        rest_label: int = 0,
        display_every: int = 4,
        seed: int = 0,
        alpha: float | None = None,
        temperature: float = 1.0,
    ):
        if not train or not test:
            raise ValueError("need recordings to train on and to replay")
        if block < 1 or display_every < 1:
            raise ValueError("block and display_every must be >= 1")
        if channels is not None:
            cols = list(channels)
            train = [replace(r, emg=r.emg[:, cols]) for r in train]
            test = [replace(r, emg=r.emg[:, cols]) for r in test]
        self.fs = train[0].fs
        self.n_channels = train[0].n_channels
        if any(r.fs != self.fs or r.n_channels != self.n_channels for r in [*train, *test]):
            raise ValueError("all recordings must share a sample rate and channel count")
        self.gestures = tuple(gestures)
        self.block, self.display_every = block, display_every

        cfg = FeatureConfig(feature_set=feature_set)
        thresholds = fit_thresholds(train, mains_hz, cfg, rest_label=rest_label)
        table = featurize(train, mains_hz, cfg, thresholds)
        self.model = LDAClassifier(feature_set=feature_set).fit(table.X, table.label)
        self.conformal: ConformalGestures | None = None
        if alpha is not None:
            decision, y, _ = blocked_scores(
                lambda X, yy: LDAClassifier(feature_set=feature_set).fit(X, yy),
                table.X,
                table.label,
                table.rep,
            )
            self.conformal = ConformalGestures.from_scores(
                self.model, decision, y, alpha, temperature, "aps", seed=seed
            )

        rails = {"clip_low": -rail, "clip_high": rail} if rail is not None else {}
        qcfg = QualityConfig(fs=self.fs, mains_hz=mains_hz, **rails)  # type: ignore[arg-type]
        rest = np.vstack([r.emg for r in train if r.label == rest_label])
        self.monitor = QualityMonitor(qcfg, self.n_channels, rest_floor=rest_floor(rest, qcfg))
        self.pipeline = GesturePipeline(
            self.model,
            self.fs,
            self.n_channels,
            mains_hz=mains_hz,
            zc_threshold=thresholds.zc,
            ssc_threshold=thresholds.ssc,
            monitor=self.monitor,
            conformal=self.conformal,
        )
        # Each channel's typical EMG size while gesturing, which sets how
        # strong an injected hum or motion artefact is.
        active = [filter_recording(r, mains_hz, cfg) for r in train if r.label != rest_label]
        self.band_rms = np.sqrt(np.mean(np.vstack(active or [rest]) ** 2, axis=0))
        self.mains_hz = mains_hz
        self.rail = rail
        # A fixed vertical scale for drawing each channel's raw trace, taken
        # from the training gestures, so a broken channel visibly changes size.
        raw_active = np.vstack([r.emg for r in train if r.label != rest_label] or [rest])
        self.trace_centre = np.median(raw_active, axis=0)
        dev = np.abs(raw_active - self.trace_centre)
        self.trace_scale = np.maximum(np.percentile(dev, 99.5, axis=0), 1e-9)

        self.original = np.vstack([r.emg for r in test])
        self.stream = self.original.copy()
        self.truth = np.concatenate([np.full(len(r.emg), r.label) for r in test])
        self.position = 0
        self.faults: dict[int, Fault] = {}
        self._rng = np.random.default_rng(seed)
        self._last: tuple[str, float] | None = None
        self._last_set: tuple[str, ...] | None = None
        self._hits = self._total = 0
        self._sets = SetCounts(0, 0, 0, 0)

    @property
    def drops_flagged(self) -> bool:
        """Whether the decoder stops using a channel the monitor flags."""
        return bool(self.pipeline.drop_on)

    def drop_flagged(self, on: bool) -> None:
        """Let the monitor take broken channels out of the decoder (the default), or not.

        With it off the monitor still reports problems, but the decoder keeps
        every channel: what a decoder without a quality monitor would do.
        """
        self.pipeline.drop_on = DROP_CHANNEL_ON if on else frozenset()

    @property
    def duration_s(self) -> float:
        """Length of the replayed stream in seconds."""
        return len(self.stream) / self.fs

    @property
    def done(self) -> bool:
        """True once every sample has been played."""
        return self.position >= len(self.stream)

    def inject(self, channel: int, fault: Fault | str | None) -> None:
        """Break `channel` from now on with `fault`, or repair it (None)."""
        if not 0 <= channel < self.n_channels:
            raise ValueError(f"channel must be in [0, {self.n_channels})")
        rest_of = slice(self.position, None)
        x = self.original[rest_of, channel]
        if fault is None:
            self.stream[rest_of, channel] = x
            self.faults.pop(channel, None)
            return
        kind = Fault(fault)
        if kind is Fault.CLIPPING and self.rail is None:
            raise ValueError("simulating clipping needs the ADC rail (pass rail=...)")
        if len(x) == 0:
            return
        size, fs, rail = float(self.band_rms[channel]), self.fs, self.rail or 0.0
        broken = {
            Fault.LEAD_OFF: lambda: faults.lead_off(x),
            Fault.CLIPPING: lambda: faults.clipping(x, rail),
            Fault.LINE_NOISE: lambda: faults.mains_hum(x, size, fs, self.mains_hz, 1.0, self._rng),
            Fault.MOTION: lambda: faults.motion(x, size, fs, 3.0, self._rng),
        }[kind]()
        self.stream[rest_of, channel] = broken
        self.faults[channel] = kind

    def step(self, seconds: float = 0.1) -> Frame:
        """Play the next `seconds` of the stream, one packet-sized block at a time."""
        n = max(self.block, int(round(seconds * self.fs)))
        start = self.position
        end = min(len(self.stream), start + n)
        for b in range(start, end, self.block):
            stop = min(end, b + self.block)
            for p in self.pipeline.process_block(self.stream[b:stop]):
                truth = self.gestures[int(self.truth[p.sample_index - 1])]
                gesture = self.gestures[int(p.gesture)]
                self._hits += gesture == truth
                self._total += 1
                self._last = (gesture, p.confidence)
                if p.gesture_set is not None:
                    named = [self.gestures[int(g)] for g in p.gesture_set]
                    # the decoder's own pick first, then the rest in gesture order
                    raw = self.gestures[int(p.raw_gesture)]
                    named.sort(key=lambda g: g != raw)
                    self._last_set = tuple(named)
                    one = len(named) == 1
                    c = self._sets
                    self._sets = SetCounts(
                        c.windows + 1,
                        c.covered + (truth in named),
                        c.singletons + one,
                        c.singletons_right + (one and named[0] == truth),
                    )
        self.position = end
        report = self.pipeline.last_quality
        dropped = set(self.pipeline.dropped)
        states = tuple(
            ChannelState(
                channel=c,
                issues=tuple(i.value for i in report.channels[c].issues) if report else (),
                dropped=c in dropped,
                fault=self.faults[c].value if c in self.faults else None,
            )
            for c in range(self.n_channels)
        )
        return Frame(
            t=end / self.fs,
            truth=self.gestures[int(self.truth[max(end - 1, 0)])],
            prediction=self._last[0] if self._last else None,
            confidence=self._last[1] if self._last else 0.0,
            channels=states,
            raw=self.stream[start : end : self.display_every].copy(),
            accuracy=self._hits / self._total if self._total else None,
            gesture_set=self._last_set,
            sets=self._sets if self.conformal is not None else None,
        )
