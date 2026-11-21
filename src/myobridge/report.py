"""Hardware evidence from your own recordings: `myobridge report`.

Every accuracy elsewhere in the docs comes from public datasets. This module
turns sessions recorded on the armband itself (`myobridge calibrate
--session`) and raw recordings (`myobridge record`) into a report a reader
can check: the raw signal, the noise floor at rest, mains hum, clipping,
the quality monitor's verdicts, how well gestures decode within a sitting
and across days, and how the WiFi link behaved.

`write_report` writes `report.md`, `report.json` (every number) and PNG
figures to one folder. The numbers are worked out by `analyse`, which needs
no plotting library.

Two hold names are checks, not gestures (`CHECK_LABELS`): `lifted` is a
hold with one electrode lifted on purpose, and `mains` a relaxed hold with a
mains lead laid near the sensor cables. They are kept out of the decoding
results; docs/first-session.md has the protocol.

The conversion from ADC counts to microvolts at the electrodes is
`SignalChain`. Its ADC side comes from the firmware (12 bits, 11 dB
attenuation, centred on 2048). The sensor's gain doesn't appear anywhere in
this repository, so it has no default: without `gain`, the report gives
counts and millivolts at the ADC pin, and says why there are no microvolts.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import __version__
from .classifier import LDAClassifier, TrainingSettings
from .dsp import EmgFilter
from .evaluation import balanced_accuracy, confusion
from .features import DEFAULT_FEATURE_SET
from .quality import QualityConfig, assess, rest_floor
from .sessions import MIN_HOLD_S, SCORE_FROM, SETTLE_S, Session, _check, _rows, adapt_to

FloatArray = NDArray[np.float64]

CHECK_LABELS = {
    "lifted": "one electrode lifted on purpose",
    "mains": "relaxed, with a mains lead laid near the sensor cables",
}
SNIPPET_S = 0.5  # length of the raw traces in the figure
JSON_NAME, MARKDOWN_NAME = "report.json", "report.md"


# ------------------------------------------------------------- signal chain


@dataclass(frozen=True)
class SignalChain:
    """How an ADC count relates to a voltage, at the ADC pin and at the electrodes.

    The firmware reads each MyoWare's RAW output with `analogRead` at 12 bits
    and 11 dB attenuation, and subtracts 2048 because RAW idles at half the
    3.3 V supply. So one count is `adc_full_scale_v / 2**adc_bits` volts at
    the pin, and dividing by the sensor's gain from the electrodes to RAW
    gives volts at the skin.

    Args:
        gain: The sensor's voltage gain from the electrodes to its RAW output
            (V/V), from the datasheet of your board revision. None (the
            default) leaves microvolts out: this repository doesn't pin it.
        adc_full_scale_v: Volts at the pin for the full 0-4095 range, as the
            firmware assumes ("full ~0-3.3 V input range").
        adc_bits: ADC resolution set by the firmware.
        rail: |count| at which a sample counts as clipped (`cli.ESP32_RAIL`).
    """

    gain: float | None = None
    adc_full_scale_v: float = 3.3
    adc_bits: int = 12
    rail: float = 2040.0

    def __post_init__(self) -> None:
        if self.gain is not None and not self.gain > 0:
            raise ValueError(f"gain must be > 0, got {self.gain}")
        if not self.adc_full_scale_v > 0 or self.adc_bits < 1 or not self.rail > 0:
            raise ValueError("adc_full_scale_v, adc_bits and rail must be > 0")

    @property
    def mv_per_count(self) -> float:
        """Millivolts at the ADC pin per count."""
        return 1000.0 * self.adc_full_scale_v / float(2**self.adc_bits)

    def to_mv(self, counts: float) -> float:
        """Counts to millivolts at the ADC pin."""
        return counts * self.mv_per_count

    def to_uv(self, counts: float) -> float | None:
        """Counts to microvolts at the electrodes, or None without a gain."""
        if self.gain is None:
            return None
        return counts * self.mv_per_count * 1000.0 / self.gain

    def assumptions(self) -> list[str]:
        """The chain in plain sentences, for the report."""
        lines = [
            f"ADC: {self.adc_bits} bits over {self.adc_full_scale_v:g} V (`analogRead`, "
            f"11 dB attenuation, as in firmware/), so 1 count = {self.mv_per_count:.4f} mV "
            "at the pin. `--adc-full-scale` changes the voltage.",
            "The firmware subtracts 2048 from every sample because the MyoWare's RAW "
            "output idles at half its 3.3 V supply; the report also removes each "
            "channel's own offset before every RMS.",
            "`analogRead` is not calibrated per chip, and the ESP32's ADC is least "
            "linear near the ends of its range, so treat volts as within about 10%. "
            "Measuring a known voltage on one pin would pin this down.",
            "RMS values are of the 20-450 Hz band with the mains notch, the same "
            "filter `myobridge record` and the decoder use. The MyoWare's own "
            "band-pass is taken as flat across that band.",
        ]
        if self.gain is None:
            lines.append(
                "No microvolts: the MyoWare's gain from the electrodes to RAW isn't "
                "written down anywhere in this repository. Look it up in the datasheet "
                "of your board revision and run again with `--gain`."
            )
        else:
            lines.append(
                f"Sensor gain {self.gain:g} V/V from the electrodes to RAW (`--gain`), "
                f"so 1 count = {self.to_uv(1.0):.3f} µV at the electrodes."
            )
        lines.append(f"A sample at |count| >= {self.rail:g} counts as clipped (`--rail`).")
        return lines

    def to_dict(self) -> dict[str, Any]:
        return {
            "gain": self.gain,
            "adc_full_scale_v": self.adc_full_scale_v,
            "adc_bits": self.adc_bits,
            "rail_counts": self.rail,
            "mv_per_count_at_adc": self.mv_per_count,
            "uv_per_count_at_electrodes": self.to_uv(1.0),
            "assumptions": self.assumptions(),
        }


# ------------------------------------------------------------------ inputs


@dataclass(frozen=True)
class Recording:
    """A raw recording from `myobridge record`: free use, say."""

    fs: float
    samples: NDArray[Any]
    source: str = ""
    link: dict[str, Any] | None = None  # None in files from before link stats were saved

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            if "samples" not in data or "fs" not in data:
                raise ValueError(f"{path} is not a myobridge recording")
            link: dict[str, Any] | None = None
            if "link" in data:
                link = json.loads(str(data["link"]))
            elif "dropped_packets" in data:
                link = {"dropped": int(data["dropped_packets"]), "partial": True}
            return cls(float(data["fs"]), np.asarray(data["samples"]), str(path), link)


def load_inputs(paths: Sequence[str | Path]) -> tuple[list[Session], list[Recording]]:
    """Sessions and raw recordings, told apart by what the files hold."""
    sessions, recordings = [], []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            is_session = "meta" in data
        if is_session:
            sessions.append(Session.load(path))
        else:
            recordings.append(Recording.load(path))
    return sessions, recordings


# ----------------------------------------------------------------- helpers


def filtered_rms(raw: ArrayLike, fs: float, mains_hz: float, settle_s: float = 0.25) -> FloatArray:
    """Each channel's RMS after the 20-450 Hz band-pass and mains notch.

    The first `settle_s` seconds are skipped while the filters settle. This
    is the number `myobridge record` prints.
    """
    x = np.asarray(raw, dtype=np.float64)
    clean = EmgFilter(fs, x.shape[1], mains_hz=mains_hz).process(x - x.mean(axis=0)).clean
    rest = clean[int(settle_s * fs) :]
    out: FloatArray = np.sqrt(np.mean(rest**2, axis=0)) if len(rest) else np.zeros(x.shape[1])
    return out


def _f(x: Any) -> float | None:
    """A JSON-safe float (None for nan and infinities)."""
    v = float(x)
    return v if np.isfinite(v) else None


def _name(s: Session | Recording) -> str:
    if s.source:
        return Path(s.source).name
    when = getattr(s, "recorded_at", "")
    return f"recorded {when[:19].replace('T', ' ')}" if when else "unnamed"


def _gestures(s: Session) -> list[str]:
    return sorted(set(s.labels) - set(CHECK_LABELS))


def _without_checks(s: Session) -> Session:
    keep = [i for i, label in enumerate(s.labels) if label not in CHECK_LABELS]
    return replace(
        s, labels=tuple(s.labels[i] for i in keep), holds=tuple(s.holds[i] for i in keep)
    )


def _qcfg(s: Session, chain: SignalChain) -> QualityConfig:
    return QualityConfig(fs=s.fs, mains_hz=s.mains_hz, clip_low=-chain.rail, clip_high=chain.rail)


def _rest(s: Session) -> NDArray[Any] | None:
    if "rest" not in s.labels:
        return None
    return s.holds[s.labels.index("rest")]


def _settled(raw: NDArray[Any], fs: float) -> FloatArray:
    out: FloatArray = np.asarray(raw, dtype=np.float64)[int(SETTLE_S * fs) :]
    return out


def scan_issues(
    raw: ArrayLike, cfg: QualityConfig, floor: ArrayLike | None, hop_s: float = 0.25
) -> dict[str, list[float]]:
    """Share of the monitor's reports that flag each issue, per channel.

    Reports come every `hop_s` once a full `cfg.window_s` is in, as
    `QualityMonitor` makes them live.
    """
    x = np.asarray(raw, dtype=np.float64)
    size, hop = int(round(cfg.window_s * cfg.fs)), max(1, int(round(hop_s * cfg.fs)))
    counts: dict[str, FloatArray] = {
        k: np.zeros(x.shape[1]) for k in ("any", "flatline", "clipping", "line_noise", "motion")
    }
    ends = range(size, len(x) + 1, hop)
    for end in ends:
        for c in assess(x[end - size : end], cfg, floor).channels:
            counts["any"][c.channel] += bool(c.issues)
            for issue in c.issues:
                counts[issue.value][c.channel] += 1
    n = max(len(ends), 1)
    return {k: [float(v) for v in arr / n] for k, arr in counts.items()}


def _confusion_rows(
    y_true: NDArray[Any], y_pred: NDArray[Any], classes: list[str]
) -> NDArray[np.int64]:
    index = {c: i for i, c in enumerate(classes)}
    return confusion(
        np.array([index[c] for c in y_true], dtype=np.int64),
        np.array([index[c] for c in y_pred], dtype=np.int64),
        len(classes),
    )


def _score(
    model: LDAClassifier, X: FloatArray, y: NDArray[Any], classes: list[str]
) -> dict[str, Any]:
    pred = model.predict(X)
    return {
        "accuracy": float(np.mean(pred == y)),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "windows": len(y),
        "confusion": _confusion_rows(y, pred, classes).tolist(),
    }


# ---------------------------------------------------------------- analysis


def _facts(s: Session | Recording) -> dict[str, Any]:
    if isinstance(s, Recording):
        return {
            "source": _name(s),
            "kind": "recording",
            "fs": s.fs,
            "n_channels": int(s.samples.shape[1]),
            "seconds": len(s.samples) / s.fs,
            "link_recorded": s.link is not None and not s.link.get("partial", False),
        }
    return {
        "source": _name(s),
        "kind": "session",
        "recorded_at": s.recorded_at,
        "fs": s.fs,
        "n_channels": s.n_channels,
        "mains_hz": s.mains_hz,
        "gestures": _gestures(s),
        "checks": sorted(set(s.labels) & set(CHECK_LABELS)),
        "holds": [
            {"label": label, "seconds": len(h) / s.fs}
            for label, h in zip(s.labels, s.holds, strict=True)
        ],
        "link_recorded": s.link is not None,
    }


def _signal(s: Session, chain: SignalChain) -> dict[str, Any]:
    """Noise floor, hum, clipping and the monitor's verdicts for one session."""
    cfg = _qcfg(s, chain)
    rest = _rest(s)
    out: dict[str, Any] = {"source": _name(s), "has_rest": rest is not None}
    floor = None
    if rest is not None:
        counts = filtered_rms(rest, s.fs, s.mains_hz)
        quiet = _settled(rest, s.fs)
        rest_report = assess(quiet, cfg).channels
        floor = rest_floor(quiet, cfg)
        channels = []
        for ch, c in enumerate(rest_report):
            share = c.line_noise_share
            hum = c.band_rms * np.sqrt(share)
            channels.append(
                {
                    "channel": ch,
                    "rms_counts": float(counts[ch]),
                    "rms_mv_at_adc": chain.to_mv(float(counts[ch])),
                    "rms_uv_at_electrodes": chain.to_uv(float(counts[ch])),
                    "mean_counts": float(np.mean(rest[:, ch])),
                    "hum_share": share,
                    "hum_rms_counts": float(hum),
                    "hum_rms_uv_at_electrodes": chain.to_uv(float(hum)),
                    "hum_to_emg_db": _f(10 * np.log10(share / (1 - share)))
                    if 0 < share < 1
                    else None,
                }
            )
        out["rest"] = channels
    clipped = np.zeros(s.n_channels)
    total = 0
    worst: list[dict[str, Any]] = [{"share": 0.0, "label": None} for _ in range(s.n_channels)]
    holds = []
    for label, raw in zip(s.labels, s.holds, strict=True):
        x = np.asarray(raw, dtype=np.float64)
        at_rail = np.abs(x) >= chain.rail
        clipped += at_rail.sum(axis=0)
        total += len(x)
        for ch in range(s.n_channels):
            share = float(at_rail[:, ch].mean()) if len(x) else 0.0
            if share > worst[ch]["share"]:
                worst[ch] = {"share": share, "label": label}
        quiet = _settled(raw, s.fs)
        hum = [c.line_noise_share for c in assess(quiet, cfg).channels] if len(quiet) > 16 else []
        holds.append(
            {
                "label": label,
                "check": label in CHECK_LABELS,
                "seconds": len(x) / s.fs,
                "rms_counts": filtered_rms(raw, s.fs, s.mains_hz).tolist(),
                "hum_share": hum,
                "flags": scan_issues(raw, cfg, floor),
            }
        )
    out["clipping"] = [
        {"channel": ch, "share": float(clipped[ch] / max(total, 1)), "worst_hold": worst[ch]}
        for ch in range(s.n_channels)
    ]
    out["holds"] = holds
    gestures = [h for h in holds if not h["check"] and h["label"] != "rest"]
    if gestures:
        out["flatline_in_gestures"] = [
            float(np.mean([h["flags"]["flatline"][ch] for h in gestures]))
            for ch in range(s.n_channels)
        ]
    return out


def _free_use(
    r: Recording, floor_from: Session | None, chain: SignalChain, mains: float
) -> dict[str, Any]:
    cfg = QualityConfig(fs=r.fs, mains_hz=mains, clip_low=-chain.rail, clip_high=chain.rail)
    floor, note = None, "no session with a rest hold at this rate, so no flatline check"
    if floor_from is not None:
        rest = _rest(floor_from)
        same = np.isclose(floor_from.fs, r.fs) and floor_from.n_channels == r.samples.shape[1]
        if rest is not None and same:
            floor = rest_floor(_settled(rest, floor_from.fs), cfg)
            note = f"flatline compared with the rest hold of {_name(floor_from)}"
    counts = filtered_rms(r.samples, r.fs, mains)
    return {
        "source": _name(r),
        "seconds": len(r.samples) / r.fs,
        "mains_hz": mains,
        "rms_counts": counts.tolist(),
        "flags": scan_issues(r.samples, cfg, floor),
        "note": note,
    }


def _decoding(gesture_sessions: list[Session]) -> dict[str, Any]:
    """Within-session, leave-one-session-out and across-day accuracy."""
    out: dict[str, Any] = {"split": {"settle_s": SETTLE_S, "score_from_s": SCORE_FROM}}
    long_enough = [min(len(h) for h in s.holds) / s.fs >= MIN_HOLD_S for s in gesture_sessions]
    usable = [s for s, ok in zip(gesture_sessions, long_enough, strict=True) if ok]
    skipped = [_name(s) for s, ok in zip(gesture_sessions, long_enough, strict=True) if not ok]
    if skipped:
        out["skipped"] = {"sessions": skipped, "why": f"holds shorter than {MIN_HOLD_S:g} s"}
    if not usable:
        return out
    first = usable[0]
    classes = _gestures(first)
    out["gestures"] = classes
    out["chance"] = 1.0 / len(classes)
    within = []
    pooled = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for s in usable:
        settings = TrainingSettings(fs=float(s.fs), mains_hz=s.mains_hz)
        X_tr, y_tr = _rows(s, settings, DEFAULT_FEATURE_SET, SETTLE_S, SCORE_FROM)
        X_te, y_te = _rows(s, settings, DEFAULT_FEATURE_SET, SCORE_FROM)
        model = LDAClassifier(feature_set=DEFAULT_FEATURE_SET, settings=settings).fit(X_tr, y_tr)
        cls = _gestures(s)
        r = _score(model, X_te, y_te, cls) | {"source": _name(s), "gestures": cls}
        within.append(r)
        if cls == classes:
            pooled += np.asarray(r["confusion"], dtype=np.int64)
    out["within_session"] = {
        "sessions": within,
        "accuracy": float(np.trace(pooled) / max(pooled.sum(), 1)),
        "windows": int(pooled.sum()),
        "confusion": pooled.tolist(),
    }
    if len(usable) < 2:
        return out
    try:
        _check(usable)
    except ValueError as err:
        out["across_sessions_skipped"] = str(err)
        return out
    settings = TrainingSettings(fs=float(first.fs), mains_hz=first.mains_hz)
    rows = [_rows(s, settings, DEFAULT_FEATURE_SET, SETTLE_S) for s in usable]
    tests = [_rows(s, settings, DEFAULT_FEATURE_SET, SCORE_FROM) for s in usable]

    def train(idx: Sequence[int]) -> LDAClassifier:
        X = np.vstack([rows[i][0] for i in idx])
        y = np.concatenate([rows[i][1] for i in idx])
        return LDAClassifier(feature_set=DEFAULT_FEATURE_SET, settings=settings).fit(X, y)

    loso, total = [], np.zeros_like(pooled)
    for i, s in enumerate(usable):
        model = train([j for j in range(len(usable)) if j != i])
        r = _score(model, *tests[i], classes) | {"source": _name(s)}
        loso.append(r)
        total += np.asarray(r["confusion"], dtype=np.int64)
    out["leave_one_session_out"] = {
        "sessions": loso,
        "accuracy": float(np.trace(total) / max(total.sum(), 1)),
        "windows": int(total.sum()),
        "confusion": total.tolist(),
    }
    base = train(range(len(usable) - 1))
    last = usable[-1]
    out["across_days"] = {
        "train": [_name(s) for s in usable[:-1]],
        "test": _name(last),
        "without_adapt": _score(base, *tests[-1], classes),
        "with_adapt": _score(adapt_to(base, last), *tests[-1], classes),
    }
    return out


def _model_scores(model_path: str | Path, gesture_sessions: list[Session]) -> dict[str, Any]:
    """A saved model (and its sets, if `fit` made it) scored on every session."""
    from .conformal import set_metrics
    from .sessions import load_sets, recalibrate

    model = LDAClassifier.load(model_path)
    if model.settings is None:
        raise ValueError(f"{model_path} has no training settings; save it with this version")
    try:
        sets, _ = load_sets(model_path)
    except FileNotFoundError:
        sets = None
    classes = [str(c) for c in model.classes]
    scores = []
    for s in gesture_sessions:
        if not np.isclose(s.fs, model.settings.fs) or not set(classes) <= set(_gestures(s)):
            scores.append({"source": _name(s), "skipped": "different rate or gestures"})
            continue
        X, y = _rows(s, model.settings, model.feature_set or DEFAULT_FEATURE_SET, SCORE_FROM)
        keep = np.isin(y, classes)
        X, y = X[keep], y[keep]
        r: dict[str, Any] = {
            "source": _name(s),
            "without_adapt": _score(model, X, y, classes),
            "with_adapt": _score(adapt_to(model, s), X, y, classes),
        }
        if sets is not None:
            today = recalibrate(sets, s)
            idx = np.searchsorted(np.asarray(today.classes), y)
            r["sets_after_recalibration"] = {
                k: _f(v) for k, v in set_metrics(today.predict_sets(X), idx).items()
            } | {"promised": sets.coverage}
        scores.append(r)
    return {"model": Path(model_path).name, "gestures": classes, "sessions": scores}


def analyse(
    sessions: Sequence[Session],
    recordings: Sequence[Recording] = (),
    chain: SignalChain | None = None,
    model: str | Path | None = None,
) -> dict[str, Any]:
    """Every number of the report, as JSON-ready values.

    `sessions` are from `calibrate --session`, oldest first; the across-day
    result trains on all but the last and tests on the last. `recordings`
    are raw `myobridge record` files, reported as free use. `model` is a
    saved model (from `calibrate` or `fit`) to score on every session too.
    """
    if not sessions:
        raise ValueError("the report needs at least one session from `calibrate --session`")
    chain = chain or SignalChain()
    gesture_sessions = [_without_checks(s) for s in sessions if len(_gestures(s)) >= 2]
    rest_session = next((s for s in sessions if _rest(s) is not None), None)
    files: list[Session | Recording] = [*sessions, *recordings]
    out: dict[str, Any] = {
        "myobridge_version": __version__,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "signal_chain": chain.to_dict(),
        "files": [_facts(s) for s in files],
        "signal": [_signal(s, chain) for s in sessions],
        "free_use": [_free_use(r, rest_session, chain, sessions[0].mains_hz) for r in recordings],
        "link": [{"source": _name(s), "stats": s.link} for s in files if s.link is not None],
        "decoding": _decoding(gesture_sessions),
    }
    if gesture_sessions:
        g = gesture_sessions[0]
        names = _gestures(g)
        per = []
        for s in gesture_sessions:
            if _gestures(s) != names or s.n_channels != g.n_channels:
                continue
            per.append(
                [
                    np.mean(
                        [
                            filtered_rms(h, s.fs, s.mains_hz)
                            for label, h in zip(s.labels, s.holds, strict=True)
                            if label == name
                        ],
                        axis=0,
                    )
                    for name in names
                ]
            )
        out["gesture_rms"] = {
            "gestures": names,
            "sessions": len(per),
            "rms_counts": np.mean(per, axis=0).tolist(),
        }
    if model is not None:
        out["model"] = _model_scores(model, gesture_sessions)
    return out


# ----------------------------------------------------------------- figures


def _figure_session(sessions: Sequence[Session]) -> Session | None:
    return next((s for s in sessions if _rest(s) is not None and len(_gestures(s)) >= 2), None)


def draw_figures(result: dict[str, Any], sessions: Sequence[Session], out: Path) -> dict[str, str]:
    """Draw the PNGs; returns {figure name: file name}. Needs matplotlib."""
    from .bench.figures import BLUE, INK, INK2, ORANGE, _plt

    plt = _plt()
    made: dict[str, str] = {}
    s = _figure_session(sessions)

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        fig.savefig(out / f"{name}.png", dpi=150)
        plt.close(fig)
        made[name] = f"{name}.png"

    if s is not None:
        rest = np.asarray(_rest(s), dtype=np.float64)
        others = [
            (label, np.asarray(h, dtype=np.float64))
            for label, h in zip(s.labels, s.holds, strict=True)
            if label != "rest" and label not in CHECK_LABELS
        ]
        label, strong = max(
            others, key=lambda lh: float(filtered_rms(lh[1], s.fs, s.mains_hz).sum())
        )
        n = int(SNIPPET_S * s.fs)

        def snippet(x: FloatArray) -> FloatArray:
            start = max(0, min(len(x) - n, int(2.0 * s.fs)))
            seg: FloatArray = x[start : start + n]
            return seg

        t = np.arange(n) / s.fs * 1000
        fig, axes = plt.subplots(
            s.n_channels, 2, figsize=(9, 1.5 + 1.3 * s.n_channels), sharex=True, squeeze=False
        )
        for ch in range(s.n_channels):
            a, b = snippet(rest[:, ch]), snippet(strong[:, ch])
            lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
            pad = 0.05 * (hi - lo) + 1
            for col, (seg, color) in enumerate(((a, INK2), (b, BLUE))):
                ax = axes[ch, col]
                ax.plot(t[: len(seg)], seg, color=color, linewidth=0.8)
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_ylabel(f"ch{ch}\ncounts")
        axes[0, 0].set_title("rest")
        axes[0, 1].set_title(label)
        for col in range(2):
            axes[-1, col].set_xlabel("ms")
        fig.suptitle(f"Raw samples, {SNIPPET_S:g} s from 2 s into each hold ({_name(s)})")
        save(fig, "traces")

        from scipy import signal as sps

        quiet = [_settled(rest, s.fs), *(_settled(h, s.fs) for _, h in others)]
        nper = int(min(s.fs, *(len(x) for x in quiet)))  # one frequency grid for all

        def psd(x: FloatArray) -> tuple[FloatArray, FloatArray]:
            f, p = sps.welch(x - x.mean(axis=0), fs=s.fs, nperseg=nper, axis=0)
            return f, p

        f, p_rest = psd(quiet[0])
        p_move = np.mean([psd(x)[1] for x in quiet[1:]], axis=0)
        cols = min(s.n_channels, 2)
        rows = int(np.ceil(s.n_channels / cols))
        fig, axes = plt.subplots(
            rows, cols, figsize=(9, 2.6 * rows), sharex=True, sharey=True, squeeze=False
        )
        for ch in range(s.n_channels):
            ax = axes[ch // cols, ch % cols]
            ax.semilogy(f, p_rest[:, ch] + 1e-12, color=INK2, linewidth=1, label="rest")
            ax.semilogy(f, p_move[:, ch] + 1e-12, color=BLUE, linewidth=1, label="gestures")
            for h in range(1, 4):
                if h * s.mains_hz < s.fs / 2:
                    ax.axvline(h * s.mains_hz, color=ORANGE, linewidth=0.8, linestyle=":")
            ax.set_title(f"ch{ch}")
            ax.set_xlim(0, s.fs / 2)
        for ax in axes[-1]:
            ax.set_xlabel("Hz")
        for ax in axes[:, 0]:
            ax.set_ylabel("counts²/Hz")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"Spectrum per channel; dotted: {s.mains_hz:g} Hz mains and harmonics")
        save(fig, "spectra")

    rms = result.get("gesture_rms")
    if rms:
        m = np.asarray(rms["rms_counts"], dtype=np.float64)
        names = rms["gestures"]
        fig, ax = plt.subplots(figsize=(1.6 + 1.1 * m.shape[1], 1.2 + 0.45 * len(names)))
        ax.grid(False)
        im = ax.imshow(m, cmap="Blues", vmin=0, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{m[i, j]:.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if m[i, j] > 0.6 * m.max() else INK,
                )
        ax.set_xticks(range(m.shape[1]), [f"ch{j}" for j in range(m.shape[1])])
        ax.set_yticks(range(len(names)), names)
        ax.set_title("20-450 Hz RMS per gesture (counts)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        save(fig, "gesture_rms")

    dec = result["decoding"]
    panels = [
        (key, title)
        for key, title in (
            ("within_session", "Within a sitting"),
            ("leave_one_session_out", "Another sitting's model"),
        )
        if key in dec and dec[key]["windows"]
    ]
    if panels:
        names = dec["gestures"]
        fig, axes = plt.subplots(
            1, len(panels), figsize=(4.6 * len(panels), 1.6 + 0.42 * len(names)), squeeze=False
        )
        for ax, (key, title) in zip(axes[0], panels, strict=True):
            c = np.asarray(dec[key]["confusion"], dtype=np.float64)
            c = c / np.maximum(c.sum(axis=1, keepdims=True), 1)
            ax.grid(False)
            ax.imshow(c, cmap="Blues", vmin=0, vmax=1)
            for i in range(len(names)):
                for j in range(len(names)):
                    if c[i, j] >= 0.02:
                        ax.text(
                            j,
                            i,
                            f"{c[i, j]:.0%}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if c[i, j] > 0.5 else INK,
                        )
            ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(names)), names, fontsize=8)
            ax.set_xlabel("predicted")
            ax.set_ylabel("performed")
            ax.set_title(f"{title}: {dec[key]['accuracy']:.1%}")
        save(fig, "confusion")
    return made


# ---------------------------------------------------------------- markdown


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{100 * x:.{digits}f}%"


def _num(x: float | None, fmt: str = ".1f") -> str:
    return "n/a" if x is None else format(x, fmt)


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
        "",
    ]


def _link_rows(link: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for item in link:
        st = item["stats"]
        timing = st.get("timing", {})
        interval = timing.get("interval_error_ms", {})
        extra = timing.get("extra_delay_ms", {})
        invalid = st.get("invalid_by_kind", {})
        rows.append(
            [
                item["source"],
                str(st.get("received", "n/a")),
                f"{st.get('dropped', 'n/a')} ({_num(st.get('loss_percent'), '.2f')}%)",
                str(st.get("gaps", "n/a")),
                str(st.get("longest_gap", "n/a")),
                str(invalid.get("crc", 0)) + f" of {st.get('corrupt', 'n/a')}",
                f"{st.get('reordered', 'n/a')} / {st.get('duplicates', 'n/a')} / "
                f"{st.get('restarts', 'n/a')}",
                _num(interval.get("std"), ".2f"),
                f"{_num(extra.get('p95'), '.1f')} / {_num(extra.get('p99'), '.1f')}",
            ]
        )
    return rows


def render_markdown(result: dict[str, Any], figures: dict[str, str]) -> str:
    """The report as Markdown, from `analyse`'s result and the figure files."""
    chain = result["signal_chain"]
    uv = chain["gain"] is not None
    L: list[str] = [
        "# Hardware report",
        "",
        f"Made by `myobridge report` ({result['myobridge_version']}) on "
        f"{result['generated_at'][:10]}. Every number here is also in `{JSON_NAME}`.",
        "",
        "## Recordings",
        "",
    ]
    rows = []
    for f in result["files"]:
        if f["kind"] == "session":
            holds = ", ".join(f"{h['label']} {h['seconds']:.1f} s" for h in f["holds"])
            when = f["recorded_at"][:16].replace("T", " ") + " UTC"
            mains = f"{f['mains_hz']:g} Hz"
        else:
            holds, when, mains = f"free use, {f['seconds']:.0f} s", "", ""
        rows.append(
            [
                f"`{f['source']}`",
                when,
                f"{f['fs']:g} Hz",
                str(f["n_channels"]),
                mains,
                holds,
                "yes" if f["link_recorded"] else "not recorded",
            ]
        )
    L += _table(["File", "Recorded", "Rate", "Channels", "Mains", "Holds", "Link stats"], rows)

    L += ["## Signal chain", ""]
    L += [f"- {line}" for line in chain["assumptions"]]
    L.append("")

    if "traces" in figures:
        L += ["## Raw signal", "", f"![Raw traces]({figures['traces']})", ""]

    L += ["## Noise floor at rest", ""]
    with_rest = [s for s in result["signal"] if s["has_rest"]]
    if not with_rest:
        L += ["No session has a `rest` hold, so there is no noise floor.", ""]
    else:
        L += [
            "20-450 Hz RMS of each `rest` hold after the first "
            f"{SETTLE_S:g} s, with the mains notch. This is rest EMG, sensor noise and "
            "ADC noise together. Each cell is counts, then "
            + ("microvolts at the electrodes." if uv else "millivolts at the ADC pin."),
            "",
        ]
        n_ch = max(len(s["rest"]) for s in with_rest)
        rows = []
        for s in with_rest:
            row = [s["source"]]
            for c in s["rest"]:
                other = (
                    f"{c['rms_uv_at_electrodes']:.2f} µV" if uv else f"{c['rms_mv_at_adc']:.2f} mV"
                )
                row.append(f"{c['rms_counts']:.2f} ({other})")
            rows.append(row + [""] * (n_ch - len(s["rest"])))
        L += _table(["File", *(f"ch{ch}" for ch in range(n_ch))], rows)
        L += [
            "## Mains hum, clipping and flatline",
            "",
            "Hum is the power in narrow peaks at the mains frequency and its "
            "harmonics, above the spectrum around them, as a share of the 20-450 Hz "
            "power at rest (the monitor flags over 60%). Clipped is the share of all "
            "samples at the rail. Flatline is the share of the monitor's reports that "
            "called a channel flat during the gestures, against its own rest level; it "
            "should be near 0.",
            "",
        ]
        rows = []
        for s in result["signal"]:
            for ch, clip in enumerate(s["clipping"]):
                rest = s["rest"][ch] if s.get("rest") else None
                worst = clip["worst_hold"]
                flat = s.get("flatline_in_gestures")
                rows.append(
                    [
                        s["source"],
                        f"ch{ch}",
                        _pct(rest and rest["hum_share"]),
                        _num(rest and rest["hum_to_emg_db"], "+.1f"),
                        _pct(clip["share"], 2)
                        + (f" (most in {worst['label']})" if worst["label"] else ""),
                        _pct(flat[ch]) if flat else "n/a",
                    ]
                )
        L += _table(
            ["File", "Channel", "Hum at rest", "Hum vs EMG (dB)", "Clipped", "Flatline"],
            rows,
        )

    checks = [(s["source"], h) for s in result["signal"] for h in s["holds"] if h["check"]]
    L += ["## Deliberate faults", ""]
    if not checks:
        L += [
            "Not recorded. Add a `lifted` and a `mains` hold "
            "(docs/first-session.md) to see the monitor catch both.",
            "",
        ]
    else:
        L += [
            "Share of the monitor's reports (every 250 ms) that flag each channel "
            "during each check hold.",
            "",
        ]
        rows = []
        for source, h in checks:
            fl = h["flags"]
            for ch in range(len(fl["any"])):
                rows.append(
                    [
                        source,
                        h["label"],
                        f"ch{ch}",
                        _pct(fl["flatline"][ch], 0),
                        _pct(fl["line_noise"][ch], 0),
                        _pct(h["hum_share"][ch]) if h["hum_share"] else "n/a",
                    ]
                )
        L += _table(["File", "Check", "Channel", "Flatline", "Line noise", "Hum share"], rows)

    if result["free_use"]:
        L += ["## Free use", ""]
        rows = []
        for r in result["free_use"]:
            fl = r["flags"]
            for ch in range(len(fl["any"])):
                rows.append(
                    [
                        r["source"],
                        f"ch{ch}",
                        _num(r["rms_counts"][ch], ".1f"),
                        _pct(fl["any"][ch]),
                        _pct(fl["flatline"][ch]),
                        _pct(fl["clipping"][ch]),
                        _pct(fl["line_noise"][ch]),
                        _pct(fl["motion"][ch]),
                    ]
                )
        L += [
            "How often the monitor raised each flag while the arm moved freely. "
            + "; ".join(f"{r['source']}: {r['note']}" for r in result["free_use"])
            + ".",
            "",
        ]
        L += _table(
            ["File", "Channel", "RMS (counts)", "Any flag", "Flat", "Clip", "Hum", "Motion"],
            rows,
        )

    L += ["## Gestures", ""]
    for key, alt in (("spectra", "Spectra"), ("gesture_rms", "Gesture RMS per channel")):
        if key in figures:
            L += [f"![{alt}]({figures[key]})", ""]
    dec = result["decoding"]
    if "within_session" not in dec:
        L += ["No session has two or more gestures with holds long enough to decode.", ""]
    else:
        ws = dec["within_session"]
        L += [
            f"{len(dec['gestures'])} gestures, chance {_pct(dec['chance'], 0)}. LDA on "
            "the library's own features. Within a sitting, the first "
            f"{SCORE_FROM:g} s of every hold (after {SETTLE_S:g} s to settle) trains "
            "and the rest tests. Windows overlap and follow each other, so these are "
            "fewer independent tests than the window count suggests.",
            "",
        ]
        rows = [
            [s["source"], _pct(s["accuracy"]), _pct(s["balanced_accuracy"]), str(s["windows"])]
            for s in ws["sessions"]
        ]
        if "leave_one_session_out" in dec:
            rows += [
                [
                    f"{s['source']}, model from the other sessions",
                    _pct(s["accuracy"]),
                    _pct(s["balanced_accuracy"]),
                    str(s["windows"]),
                ]
                for s in dec["leave_one_session_out"]["sessions"]
            ]
        if "across_days" in dec:
            ad = dec["across_days"]
            for k, what in (("without_adapt", "no recalibration"), ("with_adapt", "`adapt()`")):
                rows.append(
                    [
                        f"{ad['test']}, model from {', '.join(ad['train'])}, {what}",
                        _pct(ad[k]["accuracy"]),
                        _pct(ad[k]["balanced_accuracy"]),
                        str(ad[k]["windows"]),
                    ]
                )
        L += _table(["Test", "Accuracy", "Balanced", "Windows"], rows)
        if "across_days" in dec:
            L += [
                "The `adapt()` row recalibrates with the windows from 1-2 s into each of "
                "the last session's holds (`sessions.recalibrate`), and tests on the "
                f"windows from {SCORE_FROM:g} s on, as `myobridge sets` does each day.",
                "",
            ]
        if "across_sessions_skipped" in dec:
            L += [f"No across-session result: {dec['across_sessions_skipped']}.", ""]
        if "confusion" in figures:
            L += [f"![Confusion matrices]({figures['confusion']})", ""]
    if "model" in result:
        m = result["model"]
        L += [f"Saved model `{m['model']}` on every session (windows from {SCORE_FROM:g} s):", ""]
        rows = []
        for s in m["sessions"]:
            if "skipped" in s:
                rows.append([s["source"], s["skipped"], "", ""])
                continue
            sets = s.get("sets_after_recalibration")
            rows.append(
                [
                    s["source"],
                    _pct(s["without_adapt"]["accuracy"]),
                    _pct(s["with_adapt"]["accuracy"]),
                    f"{_pct(sets['coverage'])} (promised {_pct(sets['promised'], 0)})"
                    if sets
                    else "no sets",
                ]
            )
        L += _table(["Session", "As saved", "After `adapt()`", "Sets hold the gesture"], rows)

    L += ["## Link", ""]
    if not result["link"]:
        L += [
            "Not recorded. Link stats are saved by `calibrate --session` and `record` "
            "on the armband; the simulator has no link.",
            "",
        ]
    else:
        L += [
            "Loss counts every packet whose sequence number never arrived, over the "
            "whole command, waits included. Jitter compares arrival times with the "
            "device's own timestamps, for packets read as they came in; the two "
            "clocks aren't shared, so the delay itself isn't measured.",
            "",
        ]
        L += _table(
            [
                "File",
                "Received",
                "Lost",
                "Gaps",
                "Longest gap",
                "CRC errors",
                "Late / dup. / restarts",
                "Interval jitter (ms, sd)",
                "Extra delay p95 / p99 (ms)",
            ],
            _link_rows(result["link"]),
        )
        missing = [f["source"] for f in result["files"] if not f["link_recorded"]]
        if missing:
            L += [f"Not recorded for {', '.join(missing)}.", ""]

    L += ["## For the README", ""]
    L += [f"- {line}" for line in readme_lines(result)]
    L.append("")
    return "\n".join(L)


def readme_lines(result: dict[str, Any]) -> list[str]:
    """The headline numbers, one sentence each, dated."""
    day = result["generated_at"][:10]
    lines = []
    floors = [c for s in result["signal"] for c in s.get("rest", [])]
    if floors:
        lo = min(c["rms_counts"] for c in floors)
        hi = max(c["rms_counts"] for c in floors)
        line = f"Rest noise floor {lo:.1f}-{hi:.1f} counts RMS (20-450 Hz)"
        if result["signal_chain"]["gain"] is not None:
            ulo = min(c["rms_uv_at_electrodes"] for c in floors)
            uhi = max(c["rms_uv_at_electrodes"] for c in floors)
            line += f", {ulo:.1f}-{uhi:.1f} µV at the electrodes"
        hum = max(c["hum_share"] for c in floors)
        lines.append(f"{line}; mains hum at most {_pct(hum)} of the band at rest ({day}).")
    dec = result["decoding"]
    if "within_session" in dec:
        n = len(dec["gestures"])
        lines.append(
            f"{n} gestures within one sitting: {_pct(dec['within_session']['accuracy'])} "
            f"(time-blocked, {len(dec['within_session']['sessions'])} session(s), {day})."
        )
    if "across_days" in dec:
        ad = dec["across_days"]
        lines.append(
            f"A model from {len(ad['train'])} earlier session(s) on the next: "
            f"{_pct(ad['without_adapt']['accuracy'])}, and "
            f"{_pct(ad['with_adapt']['accuracy'])} after `adapt()` with 1 s per gesture."
        )
    timed = [item for item in result["link"] if "loss_percent" in item["stats"]]
    if timed:
        received = sum(int(i["stats"].get("received", 0)) for i in timed)
        dropped = sum(int(i["stats"].get("dropped", 0)) for i in timed)
        p99 = [i["stats"].get("timing", {}).get("extra_delay_ms", {}).get("p99") for i in timed]
        worst = max((v for v in p99 if v is not None), default=None)
        lines.append(
            f"WiFi link: {_pct(dropped / max(received + dropped, 1), 2)} of "
            f"{received + dropped} packets lost"
            + (f", 99% within {worst:.0f} ms of the quickest" if worst is not None else "")
            + "."
        )
    return lines


def write_report(
    sessions: Sequence[Session],
    out: str | Path,
    recordings: Sequence[Recording] = (),
    chain: SignalChain | None = None,
    model: str | Path | None = None,
    figures: bool = True,
) -> dict[str, Any]:
    """Write report.md, report.json and the figures to `out`; returns the result."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    result = analyse(sessions, recordings, chain, model)
    made = draw_figures(result, sessions, out) if figures else {}
    result["figures"] = made
    (out / JSON_NAME).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    (out / MARKDOWN_NAME).write_text(render_markdown(result, made), encoding="utf-8")
    return result
