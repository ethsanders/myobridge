"""Does the signal-quality monitor catch real faults, and does acting on it help?

Uses the raw (unfiltered) 3DC recordings, where every channel's true state
is "fine" unless we break it on purpose. Three parts:

1. Thresholds. For each check, the threshold is set so that on clean data
   from persons 1-6 only 1% of reports raise that check. Nothing about the
   faults below is used to choose them.
2. Detection. On persons 7-22, faults are injected into one channel of a
   recording from t = 1.5 s on, at three severities each, and we measure how
   often (and how fast) the right check fires, plus how often any check
   fires on the untouched recordings.
3. Fault tolerance. The same persons' gesture decoding with one channel
   broken for the whole test session, for each of four faults (electrode
   off, clipping, mains hum, motion). Decoders compared: no monitor (the bad
   channel stays in); the monitor as shipped (drop a channel only when it is
   flagged for one of ``DROP_CHANNEL_ON``); the monitor dropping a channel on
   any flag; and an oracle that always drops the broken channel. Both
   monitor policies are reported because the shipped one was chosen after
   seeing an earlier run. A model per left-out channel set is cheap:
   features are per channel, so it's the same training data minus some
   columns.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import datasets
from ..classifier import LDAClassifier
from ..datasets.core import Dataset, Recording
from ..evaluation import FeatureConfig, FeatureTable, accuracy, featurize, fit_thresholds, summarize
from ..features import channel_columns
from ..quality import DROP_CHANNEL_ON, Issue, QualityConfig, assess, rest_floor, scan_recording
from .common import RunOptions, provenance, timed, write_result

FloatArray = NDArray[np.float64]

VAL_SUBJECTS = list(range(1, 7))
REPORT_SUBJECTS = list(range(7, 23))
RAIL = 32700.0  # the 3DC armband's ADC is signed 16-bit; clean data reaches -32763
FAULT_ONSET_S = 1.5
FALSE_ALARM_TARGET = 0.01
HOP_S = 0.25


def _val(ds: Dataset) -> list[int]:
    return [s for s in ds.subjects if s in VAL_SUBJECTS]


def _report(ds: Dataset) -> list[int]:
    return [s for s in ds.subjects if s in REPORT_SUBJECTS]


Fault = Callable[[FloatArray, float, float, np.random.Generator], FloatArray]


def _lead_off(scale: float) -> Fault:
    def f(x: FloatArray, band_rms: float, fs: float, rng: np.random.Generator) -> FloatArray:
        return x.mean() + (x - x.mean()) * scale

    return f


def _hum(power_ratio: float, mains: float) -> Fault:
    def f(x: FloatArray, band_rms: float, fs: float, rng: np.random.Generator) -> FloatArray:
        t = np.arange(len(x)) / fs
        amp = np.sqrt(2.0 * power_ratio) * band_rms
        out: FloatArray = x + amp * np.sin(2 * np.pi * mains * t + rng.uniform(0, 2 * np.pi))
        return out

    return f


def _motion(rms_ratio: float) -> Fault:
    """Slow wandering baseline, like a cable being tugged: noise low-passed at ~5 Hz."""

    def f(x: FloatArray, band_rms: float, fs: float, rng: np.random.Generator) -> FloatArray:
        n = len(x)
        knots = rng.standard_normal(int(n / fs * 10) + 2)  # 10 random points per second
        wander = np.interp(np.arange(n) / fs * 10, np.arange(len(knots)), knots)
        wander *= rms_ratio * band_rms / max(float(np.std(wander)), 1e-12)
        return x + wander

    return f


def _clip(overdrive: float) -> Fault:
    """Too much gain: scale so the 99th-percentile peak is `overdrive` x the ADC rail."""

    def f(x: FloatArray, band_rms: float, fs: float, rng: np.random.Generator) -> FloatArray:
        centred = x - x.mean()
        gain = overdrive * RAIL / max(float(np.percentile(np.abs(centred), 99)), 1e-12)
        return np.clip(centred * gain, -RAIL - 1, RAIL + 1)

    return f


def fault_menu(mains: float) -> dict[str, tuple[Issue, dict[str, Fault]]]:
    return {
        "lead_off": (
            Issue.FLATLINE,
            {"x0.1": _lead_off(0.1), "x0.03": _lead_off(0.03), "x0.01": _lead_off(0.01)},
        ),
        "line_noise": (
            Issue.LINE_NOISE,
            {"0.25x": _hum(0.25, mains), "1x": _hum(1.0, mains), "4x": _hum(4.0, mains)},
        ),
        "motion": (Issue.MOTION, {"1x": _motion(1.0), "3x": _motion(3.0), "10x": _motion(10.0)}),
        "clipping": (
            Issue.CLIPPING,
            {"peak1x": _clip(1.0), "peak2x": _clip(2.0), "peak4x": _clip(4.0)},
        ),
    }


def _floors(ds: Dataset, cfg: QualityConfig, subjects: Sequence[int]) -> dict[int, FloatArray]:
    out = {}
    for s in subjects:
        rest = [r.emg for r in ds.select(subjects=[s], sessions=[0]) if r.label == 0]
        out[s] = rest_floor(np.vstack(rest), cfg)
    return out


def choose_thresholds(ds: Dataset, base: QualityConfig) -> tuple[QualityConfig, dict[str, Any]]:
    floors = _floors(ds, base, _val(ds))
    line, motion, ratio = [], [], []
    size = int(base.window_s * base.fs)
    for r in ds.select(subjects=_val(ds)):
        for end in range(size, len(r.emg) + 1, int(HOP_S * base.fs)):
            rep = assess(r.emg[end - size : end], base)
            line += [c.line_noise_share for c in rep.channels]
            motion += [c.motion_share for c in rep.channels]
            ratio += [c.band_rms / f for c, f in zip(rep.channels, floors[r.subject], strict=True)]
    q = 100 * (1 - FALSE_ALARM_TARGET)
    cfg = replace(
        base,
        line_noise_share=float(np.percentile(line, q)),
        motion_share=float(np.percentile(motion, q)),
        flatline_ratio=float(np.percentile(ratio, 100 - q)),
    )
    info = {
        "rule": f"each threshold = the {q:g}th (flatline: {100 - q:g}th) percentile of that "
        "statistic on clean reports from validation persons",
        "n_reports": len(line) // ds.n_channels,
        "chosen": {
            "line_noise_share": cfg.line_noise_share,
            "motion_share": cfg.motion_share,
            "flatline_ratio": cfg.flatline_ratio,
        },
    }
    return cfg, info


def false_alarms(ds: Dataset, cfg: QualityConfig, floors: dict[int, FloatArray]) -> dict[str, Any]:
    """Share of clean reports (after the first full second) raising each check."""
    per_issue = {i.value: 0 for i in Issue}
    any_flag = total = 0
    worst: dict[str, float] = {}
    size = int(cfg.window_s * cfg.fs)
    for r in ds.select(subjects=_report(ds), sessions=[1]):
        for end in range(size, len(r.emg) + 1, int(HOP_S * cfg.fs)):
            rep = assess(r.emg[end - size : end], cfg, floors[r.subject])
            for c in rep.channels:
                total += 1
                any_flag += not c.ok
                for i in c.issues:
                    per_issue[i.value] += 1
                key = f"subject{r.subject}_ch{c.channel}"
                worst[key] = worst.get(key, 0.0) + (not c.ok)
    counts = {k: v for k, v in worst.items() if v}
    top = sorted(counts.items(), key=lambda kv: -kv[1])[:5]
    return {
        "reports": total,
        "any_check": any_flag / total,
        "per_check": {k: v / total for k, v in per_issue.items()},
        "most_flagged_channels": [{"channel": k, "reports_flagged": int(v)} for k, v in top],
    }


def detection(
    ds: Dataset, cfg: QualityConfig, floors: dict[int, FloatArray], quick: bool
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    recs = [r for r in ds.select(subjects=_report(ds), sessions=[1]) if r.duration_s >= 3.0]
    if quick:
        recs = recs[::10]
    onset = int(FAULT_ONSET_S * cfg.fs)
    hop = int(HOP_S * cfg.fs)
    size = int(cfg.window_s * cfg.fs)
    out: dict[str, Any] = {}
    for name, (issue, levels) in fault_menu(cfg.mains_hz).items():
        out[name] = {}
        for level, fault in levels.items():
            hits, delays, n = 0, [], 0
            for k, r in enumerate(recs):
                ch = k % r.n_channels
                x = r.emg.copy()
                band = assess(x[:onset], cfg).channels[ch].band_rms
                x[onset:, ch] = fault(x[onset:, ch], band, cfg.fs, rng)
                # does a report after the onset raise *this* check on the broken channel?
                ends = np.arange(onset + hop, len(x) + 1, hop)
                fired = [
                    issue
                    in assess(x[max(0, e - size) : e], cfg, floors[r.subject]).channels[ch].issues
                    for e in ends
                ]
                n += 1
                if any(fired):
                    hits += 1
                    delays.append((ends[int(np.argmax(fired))] - onset) / cfg.fs)
            out[name][level] = {
                "detected": hits / n,
                "n": n,
                "median_delay_s": float(np.median(delays)) if delays else None,
            }
    return out


class _DropModels:
    """LDA models trained without some channels, made on first use."""

    def __init__(self, train: FeatureTable, n_channels: int):
        self.train, self.n_channels = train, n_channels
        self.cache: dict[tuple[int, ...], LDAClassifier] = {}

    def get(self, bad: tuple[int, ...]) -> LDAClassifier:
        if bad not in self.cache:
            keep = [c for c in range(self.n_channels) if c not in bad]
            cols = channel_columns(keep, self.n_channels)
            self.cache[bad] = LDAClassifier().fit(self.train.X[:, cols], self.train.label)
        return self.cache[bad]


def _decode_with_monitor(
    t: FeatureTable,
    timelines: Sequence[tuple[FloatArray, NDArray[np.bool_]]],
    models: _DropModels,
    broken_channel: int,
) -> tuple[NDArray[np.int64], float]:
    """Predict every window with the model that leaves out whatever the monitor
    had flagged by the end of that window. Also returns the share of windows
    where a healthy channel was dropped too."""
    n_ch = models.n_channels
    pred = np.empty(len(t), dtype=np.int64)
    extra = np.zeros(len(t), dtype=bool)
    for i in range(len(t)):
        times, flagged = timelines[int(t.recording[i])]
        ready = np.nonzero(times <= t.t_end[i] + 1e-9)[0]
        bad = tuple(int(c) for c in np.nonzero(flagged[ready[-1]])[0]) if len(ready) else ()
        if len(bad) == n_ch:  # everything flagged: nothing left to fall back on
            bad = ()
        extra[i] = bool(set(bad) - {broken_channel})
        keep = [c for c in range(n_ch) if c not in bad]
        X = t.X[i : i + 1] if not bad else t.X[i : i + 1, channel_columns(keep, n_ch)]
        pred[i] = int(models.get(bad).predict(X)[0])
    return pred, float(extra.mean())


POLICIES: dict[str, frozenset[Issue] | None] = {
    "monitor": DROP_CHANNEL_ON,  # what myobridge ships
    "monitor_drop_on_any_flag": None,  # the naive first version
}


def fault_tolerance(
    ds: Dataset, qcfg: QualityConfig, floors: dict[int, FloatArray], quick: bool
) -> dict[str, Any]:
    fcfg = FeatureConfig()
    rng = np.random.default_rng(1)
    menu = fault_menu(qcfg.mains_hz)
    scenarios = {
        "lead_off": menu["lead_off"][1]["x0.01"],
        "clipping": menu["clipping"][1]["peak2x"],
        "line_noise": menu["line_noise"][1]["4x"],
        "motion": menu["motion"][1]["3x"],
    }
    subjects = _report(ds)[:2] if quick else _report(ds)
    channels = [0, 5] if quick else list(range(ds.n_channels))
    n_ch = ds.n_channels
    keys = ["clean", "no_monitor", *POLICIES, "oracle_drop"]
    results: dict[str, dict[str, list[float]]] = {sc: {k: [] for k in keys} for sc in scenarios}
    healthy_dropped: dict[str, dict[str, list[float]]] = {
        sc: {p: [] for p in POLICIES} for sc in scenarios
    }
    for s in subjects:
        train_recs = ds.select(subjects=[s], sessions=[0])
        test_recs = ds.select(subjects=[s], sessions=[1])
        th = fit_thresholds(train_recs, ds.mains_hz, fcfg)
        train = featurize(train_recs, ds.mains_hz, fcfg, th)
        clean_test = featurize(test_recs, ds.mains_hz, fcfg, th)
        models = _DropModels(train, n_ch)
        full = models.get(())
        clean_acc = accuracy(clean_test.label, full.predict(clean_test.X))
        for scen, fault in scenarios.items():
            for ch in channels:
                broken: list[Recording] = []
                for r in test_recs:
                    x = r.emg.copy()
                    band = assess(x, qcfg).channels[ch].band_rms
                    x[:, ch] = fault(x[:, ch], band, qcfg.fs, rng)
                    broken.append(replace(r, emg=x))
                t = featurize(broken, ds.mains_hz, fcfg, th)
                res = results[scen]
                res["clean"].append(clean_acc)
                res["no_monitor"].append(accuracy(t.label, full.predict(t.X)))
                for name, only in POLICIES.items():
                    timelines = [
                        scan_recording(b.emg, qcfg, floors[s], HOP_S, only=only) for b in broken
                    ]
                    pred, extra = _decode_with_monitor(t, timelines, models, ch)
                    res[name].append(accuracy(t.label, pred))
                    healthy_dropped[scen][name].append(extra)
                keep = [c for c in range(n_ch) if c != ch]
                oracle = models.get((ch,)).predict(t.X[:, channel_columns(keep, n_ch)])
                res["oracle_drop"].append(accuracy(t.label, oracle))
    return {
        "scenarios": {
            "lead_off": "signal scaled by 0.01",
            "clipping": "gain raised until the 99th-percentile peak is 2x the ADC rail",
            "line_noise": "mains hum with 4x the EMG power",
            "motion": "slow baseline wander with 3x the EMG RMS",
        },
        "drop_channel_on": sorted(i.value for i in DROP_CHANNEL_ON),
        "averaged_over": f"{len(subjects)} persons x {len(channels)} failed channels",
        "windows_also_dropping_a_healthy_channel": {
            sc: {p: float(np.mean(v)) for p, v in d.items()} for sc, d in healthy_dropped.items()
        },
        "results": {scen: {k: summarize(v) for k, v in r.items()} for scen, r in results.items()},
    }


def run(opts: RunOptions) -> dict[str, Any]:
    with timed("load 3DC"):
        subjects = [1, 2, 7, 8] if opts.quick else None
        ds = datasets.load("3dc", root=opts.data_root, download=opts.download, subjects=subjects)
    base = QualityConfig(fs=ds.fs, mains_hz=ds.mains_hz, clip_low=-RAIL, clip_high=RAIL)
    with timed("choose thresholds"):
        cfg, chosen = choose_thresholds(ds, base)
    floors = _floors(ds, cfg, _report(ds))
    with timed("false alarms"):
        fa = false_alarms(ds, cfg, floors)
    with timed("fault detection"):
        det = detection(ds, cfg, floors, opts.quick)
    with timed("fault tolerance"):
        tol = fault_tolerance(ds, cfg, floors, opts.quick)
    payload = {
        "provenance": provenance(
            ds,
            {
                "quality": asdict(cfg),
                "validation_subjects": _val(ds),
                "report_subjects": _report(ds),
                "fault_onset_s": FAULT_ONSET_S,
                "hop_s": HOP_S,
            },
        ),
        "thresholds": chosen,
        "false_alarms_clean": fa,
        "detection": det,
        "fault_tolerance": tol,
    }
    print(f"  wrote {write_result(opts, 'quality', payload)}")
    return payload
