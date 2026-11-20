"""Partial recalibration on MultiDay: can a day be fixed from rest alone?

`adapt()` needs about a second of every gesture each day. Asking only for a
few seconds of rest (or rest plus one or two gestures) would be much easier
on a user. This benchmark tests that, with the MultiDay splits used
everywhere else: base model from days 1-10, every setting chosen on days
11-30, results on days 31-121, calibration windows from 1.0 s into each hold
and test windows from 2.5 s.

Methods (each with its prior weight chosen on days 11-30):

    none            the base model, unchanged
    adapt_all       adapt() with every gesture (the usual recalibration)
    adapt_rest      adapt() with the rest windows only
    shift_rest      every class mean moved by one shared offset, from rest
    gain_rest       one offset per channel on its amplitude features, from rest
    shift_all       one shared offset, estimated from all 11 gestures
    gain_all        per-channel amplitude offsets, from all 11 gestures
    hybrid_rest+1   rest plus one more gesture: those classes get their own
                    new means, the others the shared shift; the extra gesture
                    is chosen on days 11-30
    hybrid_rest+2   the same with two extra gestures (greedy, on days 11-30)

`shared_translation_share` measures how much of each test day's class-mean
movement (whitened by the base covariance) one shared offset can explain.
`end_of_hold` describes the last 0.5 s of each non-rest hold: its band RMS
relative to mid-hold (2.5-4.0 s), and how often the adapted model calls
those windows rest.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import datasets
from ..classifier import LDAClassifier, adapt
from ..evaluation import (
    FeatureConfig,
    FeatureTable,
    featurize,
    fit_thresholds,
    prefilter,
    summarize,
)
from ..features import feature_set_names
from . import multiday_bench as mdb
from .common import (
    ADAPT_GRID,
    AdaptSetting,
    RunOptions,
    best_setting,
    provenance,
    timed,
    write_result,
)

FloatArray = NDArray[np.float64]

CAL_SECONDS = (0.5, 1.0, 1.5)
PRIORS = (0.0, 2.0, 5.0, 10.0, 20.0, 50.0)
REST = 0
AMPLITUDE_FEATURES = ("mav", "rms", "wl")
TAIL_S = 0.5
MID_HOLD_S = (2.5, 4.0)
TAIL_ADAPT_SECONDS = 1.0

Method = Callable[[FeatureTable, float], LDAClassifier]


def _with_means(base: LDAClassifier, means: FloatArray) -> LDAClassifier:
    assert base.classes_ is not None and base.covariance_ is not None
    assert base.priors_ is not None
    return LDAClassifier(base.ridge, base.feature_set, base.settings).set_params(
        base.classes_, means, base.covariance_, base.priors_
    )


def _class_shift(
    base: LDAClassifier, cal: FeatureTable, classes: Sequence[int]
) -> tuple[FloatArray, int]:
    """The count-weighted mean offset of the given classes from the base means."""
    assert base.means_ is not None
    known = list(np.asarray(base.classes))
    deltas, counts = [], []
    for c in classes:
        rows = cal.X[cal.label == c]
        if len(rows):
            deltas.append(rows.mean(axis=0) - base.means_[known.index(c)])
            counts.append(len(rows))
    if not counts:
        return np.zeros(base.means_.shape[1]), 0
    return np.average(deltas, axis=0, weights=counts), int(sum(counts))


def _amplitude_columns(t: FeatureTable, feature_set: str) -> list[NDArray[np.intp]]:
    names = feature_set_names(feature_set)
    picks = [names.index(k) for k in AMPLITUDE_FEATURES]
    return [np.asarray(t.columns([ch]))[picks] for ch in range(t.n_channels)]


def shared_shift(
    base: LDAClassifier,
    cal: FeatureTable,
    classes: Sequence[int],
    prior: float,
    gain_columns: list[NDArray[np.intp]] | None = None,
) -> LDAClassifier:
    """Move every class mean by one offset estimated from `classes`.

    The offset shrinks toward zero with `prior` (in windows). With
    `gain_columns`, only each channel's amplitude features move, all by the
    same amount: a per-channel gain change on a log scale.
    """
    assert base.means_ is not None
    delta, n = _class_shift(base, cal, classes)
    delta = delta * (n / (n + prior)) if n + prior > 0 else delta
    if gain_columns is not None:
        gain = np.zeros_like(delta)
        for cols in gain_columns:
            gain[cols] = delta[cols].mean()
        delta = gain
    return _with_means(base, base.means_ + delta)


def hybrid(
    base: LDAClassifier, cal: FeatureTable, classes: Sequence[int], prior: float
) -> LDAClassifier:
    """Calibrated classes get their own (shrunk) new means; the rest the shared shift."""
    shifted = shared_shift(base, cal, classes, prior)
    assert shifted.means_ is not None and base.means_ is not None
    means = shifted.means_.copy()
    known = list(np.asarray(base.classes))
    for c in classes:
        rows = cal.X[cal.label == c]
        i = known.index(c)
        means[i] = (len(rows) * rows.mean(axis=0) + prior * base.means_[i]) / (len(rows) + prior)
    return _with_means(base, means)


def _accuracy(model: LDAClassifier, t: FeatureTable) -> float:
    return float(np.mean(model.predict(t.X) == t.label))


def _per_day(
    t: FeatureTable,
    days: Sequence[int],
    seconds: float,
    make: Callable[[FeatureTable], LDAClassifier],
) -> list[float]:
    return [_accuracy(make(mdb._cal(t, d, seconds)), mdb._test(t, d)) for d in days]


def _at(
    build: Callable[[FeatureTable, Any], LDAClassifier], setting: Any
) -> Callable[[FeatureTable], LDAClassifier]:
    """`build` with its setting fixed, as a function of the calibration table."""

    def make(cal: FeatureTable) -> LDAClassifier:
        return build(cal, setting)

    return make


def _adapter(base: LDAClassifier) -> Callable[[FeatureTable, AdaptSetting], LDAClassifier]:
    return lambda cal, s: _adapt(base, cal, s)


def _hybrid(base: LDAClassifier, cls: list[int]) -> Callable[[FeatureTable, float], LDAClassifier]:
    return lambda cal, p: hybrid(base, cal, cls, p)


def _choose_prior(
    t: FeatureTable, val: Sequence[int], seconds: float, build: Callable[[FeatureTable, float], Any]
) -> tuple[float, float]:
    """The prior with the best mean accuracy on the validation days (ties: smaller)."""
    scores = {p: float(np.mean(_per_day(t, val, seconds, _at(build, p)))) for p in PRIORS}
    best = max(PRIORS, key=lambda p: (scores[p], -PRIORS.index(p)))
    return best, scores[best]


def partial(
    t: FeatureTable, base: LDAClassifier, val: list[int], test: list[int]
) -> dict[str, Any]:
    """Every method at every calibration length, with settings chosen on `val`."""
    gains = _amplitude_columns(t, base.feature_set or "td-ar-psd")
    others = [int(c) for c in np.asarray(base.classes) if c != REST]
    builders: dict[str, Callable[[FeatureTable, float], LDAClassifier]] = {
        "shift_rest": lambda c, p: shared_shift(base, c, [REST], p),
        "gain_rest": lambda c, p: shared_shift(base, c, [REST], p, gains),
        "shift_all": lambda c, p: shared_shift(base, c, list(np.asarray(base.classes)), p),
        "gain_all": lambda c, p: shared_shift(base, c, list(np.asarray(base.classes)), p, gains),
        "adapt_rest": lambda c, p: adapt(
            base, c.X[c.label == REST], c.label[c.label == REST], p, 0.0
        ),
    }
    out: dict[str, Any] = {}
    for seconds in CAL_SECONDS:
        res: dict[str, Any] = {}
        none = [_accuracy(base, mdb._test(t, d)) for d in test]
        res["none"] = {"test": summarize(none), "per_day": none}

        grid: dict[AdaptSetting, list[float]] = {
            s: _per_day(t, val, seconds, _at(_adapter(base), s)) for s in ADAPT_GRID
        }
        s = best_setting(grid)
        per = _per_day(t, test, seconds, _at(_adapter(base), s))
        res["adapt_all"] = {"setting": asdict(s), "val": float(np.mean(grid[s])), "per_day": per}

        for name, build in builders.items():
            prior, v = _choose_prior(t, val, seconds, build)
            per = _per_day(t, test, seconds, _at(build, prior))
            res[name] = {"prior": prior, "val": v, "per_day": per}

        chosen: list[int] = []
        for k in (1, 2):  # greedy: add the extra gesture that helps most on validation days
            best: tuple[float, int, float] | None = None
            for g in others:
                if g in chosen:
                    continue
                cls = [REST, *chosen, g]
                prior, v = _choose_prior(t, val, seconds, _hybrid(base, cls))
                if best is None or v > best[0]:
                    best = (v, g, prior)
            assert best is not None
            v, g, prior = best
            chosen.append(g)
            cls = [REST, *chosen]
            per = _per_day(t, test, seconds, _at(_hybrid(base, cls), prior))
            res[f"hybrid_rest+{k}"] = {
                "extra_gestures": list(chosen),
                "prior": prior,
                "val": v,
                "per_day": per,
            }

        for name, r in res.items():
            if name == "none":
                continue
            r["test"] = summarize(r["per_day"])
            r["minus_none"] = summarize(np.subtract(r["per_day"], none).tolist())
        out[str(seconds)] = res
    return out


def _adapt(base: LDAClassifier, cal: FeatureTable, s: AdaptSetting) -> LDAClassifier:
    return adapt(base, cal.X, cal.label, s.mean_prior, s.covariance_weight)


def shared_translation_share(
    t: FeatureTable, base: LDAClassifier, days: list[int]
) -> dict[str, float]:
    """Share of each day's whitened class-mean movement that one shared offset explains."""
    assert base.means_ is not None and base.covariance_ is not None
    cov = base.covariance_ + base.ridge * np.diag(np.diag(base.covariance_))
    whiten = np.linalg.cholesky(np.linalg.inv(cov))
    shares = []
    for d in days:
        day = t.where(sessions=[d])
        moved = np.array(
            [
                day.X[day.label == c].mean(axis=0) - base.means_[i]
                for i, c in enumerate(base.classes)
            ]
        )
        total = np.sum((moved @ whiten) ** 2)
        left = np.sum(((moved - moved.mean(axis=0)) @ whiten) ** 2)
        shares.append(1.0 - left / total)
    return summarize(shares)


def end_of_hold(
    ds: datasets.Dataset,
    clean: Sequence[FloatArray],
    t: FeatureTable,
    base: LDAClassifier,
    setting: AdaptSetting,
    days: list[int],
) -> dict[str, Any]:
    """How the last 0.5 s of a held gesture differs from mid-hold, on `days`."""
    ratios = []
    for i, r in enumerate(ds.recordings):
        if r.session not in days or r.label == REST:
            continue
        x = clean[i]
        a, b = (int(s * r.fs) for s in MID_HOLD_S)
        mid = np.sqrt(np.mean(x[a:b] ** 2, axis=0))
        end = np.sqrt(np.mean(x[-int(TAIL_S * r.fs) :] ** 2, axis=0))
        ratios.append(float(np.median(end / mid)))
    as_rest, right, n = 0, 0, 0
    for d in days:
        model = _adapt(base, mdb._cal(t, d, TAIL_ADAPT_SECONDS), setting)
        te = mdb._test(t, d)
        duration = np.asarray(t.meta["durations"], dtype=np.float64)[te.recording]
        tail = te.take((te.t_end > duration - TAIL_S) & (te.label != REST))
        pred = model.predict(tail.X)
        as_rest += int(np.sum(pred == REST))
        right += int(np.sum(pred == tail.label))
        n += len(tail)
    q25, q50, q75 = np.percentile(ratios, [25, 50, 75])
    return {
        "tail_s": TAIL_S,
        "mid_hold_s": MID_HOLD_S,
        "non_rest_holds": len(ratios),
        "rms_ratio_end_to_mid": {"median": float(q50), "q25": float(q25), "q75": float(q75)},
        "adapted_tail_windows": n,
        "share_predicted_rest": as_rest / n if n else float("nan"),
        "accuracy_in_tail": right / n if n else float("nan"),
        "adapt_setting": asdict(setting),
        "adapt_seconds": TAIL_ADAPT_SECONDS,
    }


def run(opts: RunOptions) -> dict[str, Any]:
    cfg = FeatureConfig()
    days = None
    if opts.quick:
        days = mdb.TRAIN_DAYS[:3] + mdb.VAL_DAYS[:3] + mdb.TEST_DAYS[:6]
    with timed("load MultiDay"):
        ds = datasets.load("multiday", root=opts.data_root, download=opts.download, days=days)
    with timed("filter + features"):
        clean = prefilter(ds.recordings, ds.mains_hz, cfg)
        train_idx = [i for i, r in enumerate(ds.recordings) if r.session in mdb.TRAIN_DAYS]
        th = fit_thresholds(
            [ds.recordings[i] for i in train_idx],
            ds.mains_hz,
            cfg,
            clean=[clean[i] for i in train_idx],
        )
        t = featurize(ds.recordings, ds.mains_hz, cfg, th, clean=clean)
    present = set(t.session.tolist())
    train = [d for d in mdb.TRAIN_DAYS if d in present]
    val = [d for d in mdb.VAL_DAYS if d in present]
    test = [d for d in mdb.TEST_DAYS if d in present]
    tr = t.where(sessions=train)
    base = LDAClassifier(feature_set=cfg.feature_set).fit(tr.X, tr.label)
    with timed("partial recalibration"):
        results = partial(t, base, val, test)
    with timed("shared translation + end of hold"):
        share = shared_translation_share(t, base, test)
        setting = AdaptSetting(**results[str(TAIL_ADAPT_SECONDS)]["adapt_all"]["setting"])
        tail = end_of_hold(ds, clean, t, base, setting, test)
    payload = {
        "provenance": provenance(
            ds,
            {
                "features": asdict(cfg),
                "train_days": mdb.TRAIN_DAYS,
                "validation_days": mdb.VAL_DAYS,
                "test_days": mdb.TEST_DAYS,
                "calibration_start_s": mdb.CAL_START_S,
                "test_start_s": mdb.TEST_START_S,
                "calibration_seconds": CAL_SECONDS,
                "priors": PRIORS,
                "adapt_grid": [asdict(s) for s in ADAPT_GRID],
            },
        ),
        "gestures": list(ds.gestures),
        "test_days": test,
        "by_calibration_seconds": results,
        "shared_translation_share": share,
        "end_of_hold": tail,
    }
    print(f"  wrote {write_result(opts, 'partial_calibration', payload)}")
    return payload
