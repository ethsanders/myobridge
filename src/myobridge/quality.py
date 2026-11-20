"""Live signal-quality monitoring: is each electrode actually giving us EMG?

A gesture decoder fails silently when an electrode lifts, a cable picks up
mains hum, or the ADC clips: it just outputs confident wrong answers. For
anything that moves a body (a prosthesis, an exoskeleton) that's the failure
you most need to catch. So the monitor checks every channel, once per hop,
on the *raw* signal (before the notch and bandpass hide the problem):

    check          what it catches                        how
    -------------  -------------------------------------  ----------------------------------
    flatline       lifted electrode, broken lead, dead    band RMS far below this channel's
                   sensor                                 rest noise floor from calibration
    clipping       gain too high, electrode gel drying    fraction of samples at the ADC rails
    line_noise     poor skin contact, unshielded cable    share of 20-450 Hz power in narrow
                                                          peaks at mains and its harmonics
    motion         cable tug, electrode sliding           share of power below 10 Hz

Thresholds are deliberately simple numbers in a dataclass so they are easy
to explain and to tune; docs/signal-quality.md says how they were picked and
how well the checks work on real recordings with injected faults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import signal

FloatArray = NDArray[np.float64]


class Issue(str, Enum):
    """A problem `assess` can find on a channel. The value is the name used in JSON."""

    FLATLINE = "flatline"
    CLIPPING = "clipping"
    LINE_NOISE = "line_noise"
    MOTION = "motion"


@dataclass(frozen=True)
class QualityConfig:
    """Thresholds for the four checks. Only `fs` is required.

    Give `clip_low`/`clip_high` (the ADC rails, in the same units as the
    signal) to check for clipping; without them that check is skipped.
    """

    fs: float
    mains_hz: float = 60.0
    window_s: float = 1.0  # how much history each check looks at
    # The three thresholds below are rounded from `myobridge bench quality`,
    # which sets each one for ~1% false alarms on clean 3DC validation people
    # (results/quality.json; test_docs checks they still agree). Recheck them
    # on new hardware: docs/hardware-bringup.md.
    flatline_ratio: float = 0.65  # flag if band RMS < this x the rest level
    clip_low: float | None = None  # ADC rails in signal units (None = don't check)
    clip_high: float | None = None
    clip_fraction: float = 0.01  # flag if more than 1% of samples sit on a rail
    line_noise_share: float = 0.6  # flag if mains peaks hold > 60% of 20-450 Hz power
    motion_share: float = 0.25  # flag if < 10 Hz holds > 25% of total power
    n_harmonics: int = 3  # mains, 2x mains, 3x mains

    def __post_init__(self) -> None:
        if self.fs <= 0 or self.window_s <= 0:
            raise ValueError("fs and window_s must be > 0")
        if (self.clip_low is None) != (self.clip_high is None):
            raise ValueError("give both ADC rails or neither")


@dataclass(frozen=True)
class ChannelReport:
    """The measurements behind one channel's verdict, and the issues found."""

    channel: int
    band_rms: float  # RMS of the 20-450 Hz component
    clip_fraction: float
    line_noise_share: float
    motion_share: float
    issues: tuple[Issue, ...]

    @property
    def ok(self) -> bool:
        """True when no check fired."""
        return not self.issues


# Faults the decoder should stop trusting a channel for. Hum and slow motion
# are reported too, but the notch and the 20 Hz high-pass already remove most
# of them before feature extraction, so dropping the channel costs more than
# it saves (docs/signal-quality.md has the numbers for both policies).
DROP_CHANNEL_ON = frozenset({Issue.FLATLINE, Issue.CLIPPING})


@dataclass(frozen=True)
class QualityReport:
    """One `ChannelReport` per channel, from a single call to `assess`."""

    channels: tuple[ChannelReport, ...]

    @property
    def ok(self) -> bool:
        """True when every channel is healthy."""
        return all(c.ok for c in self.channels)

    @property
    def bad_channels(self) -> list[int]:
        """Indices of the channels with at least one issue."""
        return [c.channel for c in self.channels if not c.ok]


def _band_power(f: FloatArray, p: FloatArray, lo: float, hi: float) -> FloatArray:
    sel = (f >= lo) & (f <= hi)
    out: FloatArray = p[sel].sum(axis=0)
    return out


def assess(
    window: ArrayLike, cfg: QualityConfig, rest_floor: ArrayLike | None = None
) -> QualityReport:
    """Check one (n_samples, n_channels) window of RAW samples.

    `rest_floor` is each channel's band RMS during calibration rest (from
    `rest_floor()`); without it the flatline check is skipped.
    """
    x = np.asarray(window, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 16:
        raise ValueError("window must be (n_samples >= 16, n_channels)")
    n_ch = x.shape[1]
    # Welch PSD with nperseg = fs/4, so bins are 4 Hz wide: fine enough to
    # separate a mains line from the EMG around it, coarse enough to be stable.
    nperseg = int(min(len(x), max(16, cfg.fs // 4)))
    f, p = signal.welch(x - x.mean(axis=0), fs=cfg.fs, nperseg=nperseg, axis=0)
    df = float(f[1] - f[0])
    high = min(450.0, 0.45 * cfg.fs)
    band = _band_power(f, p, 20.0, high)
    total = p.sum(axis=0)
    # Mains hum is a narrow peak. EMG also has power at 60 Hz, so count only
    # the EXCESS over the local floor, estimated from the bins 8-20 Hz away.
    line = np.zeros(n_ch)
    for h in range(1, cfg.n_harmonics + 1):
        fh = h * cfg.mains_hz
        if fh + 20.0 >= high:
            break
        near = np.abs(f - fh) <= 1.5 * df
        side = (np.abs(f - fh) >= 8.0) & (np.abs(f - fh) <= 20.0)
        floor = p[side].mean(axis=0) * near.sum()
        line += np.maximum(p[near].sum(axis=0) - floor, 0.0)
    low = _band_power(f, p, 0.0, 10.0)

    band_rms = np.sqrt(band * df)
    line_share = np.divide(line, band, out=np.zeros(n_ch), where=band > 0)
    motion_share = np.divide(low, total, out=np.zeros(n_ch), where=total > 0)
    if cfg.clip_low is not None and cfg.clip_high is not None:
        clip = np.mean((x <= cfg.clip_low) | (x >= cfg.clip_high), axis=0)
    else:
        clip = np.zeros(n_ch)
    floor = None if rest_floor is None else np.asarray(rest_floor, dtype=np.float64)

    reports = []
    for ch in range(n_ch):
        issues: list[Issue] = []
        if floor is not None and band_rms[ch] < cfg.flatline_ratio * floor[ch]:
            issues.append(Issue.FLATLINE)
        if clip[ch] > cfg.clip_fraction:
            issues.append(Issue.CLIPPING)
        if line_share[ch] > cfg.line_noise_share:
            issues.append(Issue.LINE_NOISE)
        if motion_share[ch] > cfg.motion_share:
            issues.append(Issue.MOTION)
        reports.append(
            ChannelReport(
                channel=ch,
                band_rms=float(band_rms[ch]),
                clip_fraction=float(clip[ch]),
                line_noise_share=float(line_share[ch]),
                motion_share=float(motion_share[ch]),
                issues=tuple(issues),
            )
        )
    return QualityReport(channels=tuple(reports))


def rest_floor(rest: ArrayLike, cfg: QualityConfig) -> FloatArray:
    """Each channel's typical 20-450 Hz RMS at rest (the flatline reference).

    The rest recording is cut into `cfg.window_s` pieces, the same length the
    monitor checks, and we take each channel's median over the pieces. One
    noisy second during "rest" (a twitch, a bumped cable) would otherwise set
    the reference far too high and make every quiet moment look like a dead
    electrode.
    """
    x = np.asarray(rest, dtype=np.float64)
    size = int(round(cfg.window_s * cfg.fs))
    pieces = [x[i : i + size] for i in range(0, len(x) - size + 1, size)] or [x]
    rms = np.array([[c.band_rms for c in assess(p, cfg).channels] for p in pieces])
    out: FloatArray = np.median(rms, axis=0)
    return out


@dataclass
class QualityMonitor:
    """Streaming wrapper: push raw blocks, get a report every `hop_s` seconds.

    `rest_floor` is each channel's band RMS at rest (see `rest_floor`); the
    flatline check needs it, and is skipped without it.
    """

    cfg: QualityConfig
    n_channels: int
    rest_floor: FloatArray | None = None
    hop_s: float = 0.25
    _buffer: FloatArray = field(init=False)
    _since: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._buffer = np.zeros((0, self.n_channels))

    @property
    def window_samples(self) -> int:
        """How many samples each report looks back over."""
        return int(round(self.cfg.window_s * self.cfg.fs))

    def push(self, block: ArrayLike) -> QualityReport | None:
        """Add raw samples, shape (n, n_channels); returns a report when one is due."""
        b = np.asarray(block, dtype=np.float64)
        self._buffer = np.vstack([self._buffer, b])[-self.window_samples :]
        self._since += len(b)
        hop = max(1, int(round(self.hop_s * self.cfg.fs)))
        if self._since < hop or len(self._buffer) < self.window_samples:
            return None
        self._since %= hop  # keep the remainder so reports don't drift off the hop grid
        return assess(self._buffer, self.cfg, self.rest_floor)


def scan_recording(
    raw: ArrayLike,
    cfg: QualityConfig,
    rest_floor: ArrayLike | None = None,
    hop_s: float = 0.25,
    min_history_s: float = 0.25,
    only: frozenset[Issue] | None = None,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Run the monitor over a whole recording, as it would run live.

    Returns (times, flagged): `times[i]` is when the i-th report is made (s),
    `flagged[i, ch]` whether channel ch had any issue. Each report looks at the
    last `cfg.window_s` seconds, or at everything so far near the start (but
    at least `min_history_s`), so nothing from the future is used. With
    `only`, a channel counts as flagged only for those issues.
    """
    x = np.asarray(raw, dtype=np.float64)
    hop = max(1, int(round(hop_s * cfg.fs)))
    first = max(16, int(round(min_history_s * cfg.fs)))
    size = int(round(cfg.window_s * cfg.fs))
    ends = np.arange(first, len(x) + 1, hop)
    flagged = np.zeros((len(ends), x.shape[1]), dtype=bool)
    for i, end in enumerate(ends):
        report = assess(x[max(0, end - size) : end], cfg, rest_floor)
        flagged[i] = [
            bool(c.issues) if only is None else bool(set(c.issues) & only) for c in report.channels
        ]
    return ends / cfg.fs, flagged
