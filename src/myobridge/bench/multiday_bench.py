"""MultiDay benchmarks: how fast does a gesture model go stale, and what fixes it?

One person, 121 day indices, electrodes re-applied every day. Splits:

    days 1-10    training days for the base model and the ZC/SSC thresholds
    days 11-30   validation: the only days used to choose recalibration settings
    days 31-121  reported (never looked at while choosing anything)

Inside a day each gesture is one continuous ~6 s hold. Windows starting at or
after 2.5 s are the day's test windows. Calibration data, when a method uses
it, comes from [1.0, 1.0 + C] s of the same recordings, so there is always a
gap of at least one window between calibration and test samples.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import numpy as np

from .. import datasets
from ..classifier import LDAClassifier
from ..evaluation import (
    FeatureConfig,
    FeatureTable,
    accuracy,
    balanced_accuracy,
    featurize,
    fit_thresholds,
    prefilter,
    summarize,
)
from .common import (
    ADAPT_GRID,
    METHODS,
    AdaptSetting,
    RunOptions,
    best_setting,
    provenance,
    recalibrate,
    timed,
    write_result,
)

TRAIN_DAYS = list(range(1, 11))
VAL_DAYS = list(range(11, 31))
TEST_DAYS = list(range(31, 122))
CAL_START_S = 1.0
TEST_START_S = 2.5
CAL_SECONDS = (0.25, 0.5, 1.0)
GAPS = (1, 2, 3, 5, 10, 20, 30, 50, 90)
N_TRAIN_DAYS = (1, 2, 5, 10, 20, 30)


def _test(t: FeatureTable, day: int) -> FeatureTable:
    return t.where(sessions=[day], t_min=TEST_START_S)


def _cal(t: FeatureTable, day: int, seconds: float) -> FeatureTable:
    return t.where(sessions=[day], t_min=CAL_START_S, t_max=CAL_START_S + seconds)


def _scores(model: LDAClassifier, t: FeatureTable) -> tuple[float, float]:
    pred = model.predict(t.X)
    return accuracy(t.label, pred), balanced_accuracy(t.label, pred)


def build_table(opts: RunOptions, cfg: FeatureConfig) -> tuple[datasets.Dataset, FeatureTable]:
    days = None
    if opts.quick:
        days = TRAIN_DAYS[:3] + VAL_DAYS[:3] + TEST_DAYS[:6]
    with timed("load MultiDay"):
        ds = datasets.load("multiday", root=opts.data_root, download=opts.download, days=days)
    with timed("filter + features"):
        clean = prefilter(ds.recordings, ds.mains_hz, cfg)
        train_idx = [i for i, r in enumerate(ds.recordings) if r.session in TRAIN_DAYS]
        th = fit_thresholds(
            [ds.recordings[i] for i in train_idx],
            ds.mains_hz,
            cfg,
            clean=[clean[i] for i in train_idx],
        )
        table = featurize(ds.recordings, ds.mains_hz, cfg, th, clean=clean)
    return ds, table


def recalibration(table: FeatureTable) -> dict[str, Any]:
    days = sorted(set(table.session.tolist()))
    train = [d for d in days if d in TRAIN_DAYS]
    val = [d for d in days if d in VAL_DAYS]
    test = [d for d in days if d in TEST_DAYS]
    base_t = table.where(sessions=train)
    base = LDAClassifier().fit(base_t.X, base_t.label)

    out: dict[str, Any] = {"by_calibration_seconds": {}}
    # Reference point: a model trained only on that day's [1.0, 2.5) s, with no
    # older days at all. Not a ceiling: adapt can beat it by also using the prior.
    same_day = []
    for d in test:
        tr = table.where(sessions=[d], t_min=CAL_START_S, t_max=TEST_START_S)
        same_day.append(_scores(LDAClassifier().fit(tr.X, tr.label), _test(table, d))[0])
    out["same_day_only"] = {
        "calibration_seconds": TEST_START_S - CAL_START_S,
        "accuracy": summarize(same_day),
        "per_day": same_day,
    }

    for seconds in CAL_SECONDS:
        # 1) choose the adapt setting on validation days only
        val_scores: dict[AdaptSetting, list[float]] = {s: [] for s in ADAPT_GRID}
        for d in val:
            cal = _cal(table, d, seconds)
            te = _test(table, d)
            for s in ADAPT_GRID:
                m = recalibrate("adapt", base, base_t.X, base_t.label, cal.X, cal.label, s)
                val_scores[s].append(_scores(m, te)[0])
        chosen = best_setting(val_scores)

        # 2) report every method on the test days with that setting frozen
        per_method: dict[str, Any] = {}
        for method in METHODS:
            acc, bal = [], []
            for d in test:
                cal = _cal(table, d, seconds)
                m = recalibrate(method, base, base_t.X, base_t.label, cal.X, cal.label, chosen)
                a, b = _scores(m, _test(table, d))
                acc.append(a)
                bal.append(b)
            per_method[method] = {
                "accuracy": summarize(acc),
                "balanced_accuracy": summarize(bal),
                "per_day": acc,
            }
        paired = {
            f"adapt_minus_{other}": summarize(
                np.subtract(per_method["adapt"]["per_day"], per_method[other]["per_day"]).tolist()
            )
            for other in ("none", "new_only", "pooled")
        }
        n_cal = int(np.mean([len(_cal(table, d, seconds)) for d in test]))
        out["by_calibration_seconds"][str(seconds)] = {
            "chosen_setting": asdict(chosen),
            "validation_accuracy_of_chosen": float(np.mean(val_scores[chosen])),
            "calibration_windows_per_day": n_cal,
            "methods": per_method,
            "paired_differences": paired,
        }
    out["test_days"] = test
    return out


def decay(table: FeatureTable) -> dict[str, Any]:
    """Train on one whole day, test on the day `gap` indices later."""
    days = sorted(set(table.session.tolist()))
    models = {
        d: LDAClassifier().fit(t.X, t.label) for d in days for t in [table.where(sessions=[d])]
    }
    out: dict[str, Any] = {}
    for gap in GAPS:
        accs = [_scores(models[d], _test(table, d + gap))[0] for d in days if d + gap in models]
        if accs:
            out[str(gap)] = summarize(accs)
    return out


def training_days(table: FeatureTable) -> dict[str, Any]:
    """Does training on more days make a model that survives new days better?"""
    days = sorted(set(table.session.tolist()))
    test = [d for d in days if d in TEST_DAYS]
    out: dict[str, Any] = {}
    for n in N_TRAIN_DAYS:
        train = [d for d in days if d <= n]
        if len(train) < n:
            continue
        t = table.where(sessions=train)
        model = LDAClassifier().fit(t.X, t.label)
        out[str(n)] = summarize([_scores(model, _test(table, d))[0] for d in test])
    return out


def run(opts: RunOptions) -> dict[str, Any]:
    cfg = FeatureConfig()
    ds, table = build_table(opts, cfg)
    with timed("recalibration"):
        recal = recalibration(table)
    with timed("decay + training days"):
        dec = decay(table)
        tdays = training_days(table)
    payload = {
        "provenance": provenance(
            ds,
            {
                "features": asdict(cfg),
                "train_days": TRAIN_DAYS,
                "validation_days": VAL_DAYS,
                "test_days": TEST_DAYS,
                "calibration_start_s": CAL_START_S,
                "test_start_s": TEST_START_S,
                "adapt_grid": [asdict(s) for s in ADAPT_GRID],
            },
        ),
        "recalibration": recal,
        "decay_by_gap_days": dec,
        "accuracy_by_training_days": tdays,
    }
    print(f"  wrote {write_result(opts, 'multiday', payload)}")
    return payload
