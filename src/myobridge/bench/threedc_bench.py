"""3DC benchmarks: 22 people, 10 forearm channels at 1 kHz (like myobridge's ESP32).

Each person did 4 cycles of 11 gestures ("train" = session 0), rested 5
minutes, then 4 more cycles ("test" = session 1) without taking the armband
off. Experiments:

within_subject   train on session 0, test on session 1, per person: the
                 split the dataset itself is organised around.
channels         the same, using only k of the 10 channels, for every subset.
                 myobridge has 4 sensors; this says what that costs.
windows          the same, for analysis windows of 100-300 ms (latency vs accuracy).
cross_subject    leave one person out: train on the other 21, test on the new
                 person, with and without a few seconds of their calibration.
                 Recalibration settings are chosen on persons 1-6 only;
                 persons 7-22 are reported.
feature_sets     within-subject accuracy for each feature set, on all 10 and
                 the even 4 channels. The default set was picked on persons
                 1-6; persons 7-22 are the unbiased comparison.
classifiers      LDA against SVM, random forest and an MLP (scikit-learn, if
                 installed) on persons 7-22 with the default features.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import asdict, replace
from typing import Any

import numpy as np

from .. import datasets
from ..classifier import LDAClassifier
from ..datasets.core import Dataset, Recording
from ..evaluation import (
    FeatureConfig,
    FeatureTable,
    accuracy,
    balanced_accuracy,
    confusion,
    featurize,
    fit_thresholds,
    prefilter,
    summarize,
)
from ..features import FEATURE_SETS, channel_columns
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

TRAIN, TEST = 0, 1
VAL_SUBJECTS = list(range(1, 7))
REPORT_SUBJECTS = list(range(7, 23))
EVEN_4 = (0, 2, 5, 7)  # four channels spread evenly around the forearm, fixed in advance
WINDOWS_S = (0.1, 0.15, 0.2, 0.25, 0.3)
CAL_START_S = 1.0
CAL_SECONDS = (0.5, 1.0, 2.0)


class Prepared:
    """The dataset plus its filtered recordings, so filtering happens once."""

    def __init__(self, ds: Dataset, cfg: FeatureConfig):
        self.ds = ds
        self.cfg = cfg
        self.clean = prefilter(ds.recordings, ds.mains_hz, cfg)

    def indices(self, subjects: Sequence[int], sessions: Sequence[int] | None = None) -> list[int]:
        return [
            i
            for i, r in enumerate(self.ds.recordings)
            if r.subject in subjects and (sessions is None or r.session in sessions)
        ]

    def table(
        self, idx: Sequence[int], threshold_idx: Sequence[int], cfg: FeatureConfig | None = None
    ) -> FeatureTable:
        c = cfg or self.cfg
        recs: list[Recording] = [self.ds.recordings[i] for i in idx]
        th_recs = [self.ds.recordings[i] for i in threshold_idx]
        th = fit_thresholds(
            th_recs, self.ds.mains_hz, c, clean=[self.clean[i] for i in threshold_idx]
        )
        return featurize(recs, self.ds.mains_hz, c, th, clean=[self.clean[i] for i in idx])


def _fit_score(train: FeatureTable, test: FeatureTable, cols: Any = None) -> float:
    Xtr = train.X if cols is None else train.X[:, cols]
    Xte = test.X if cols is None else test.X[:, cols]
    return accuracy(test.label, LDAClassifier().fit(Xtr, train.label).predict(Xte))


def within_subject(p: Prepared, subjects: Sequence[int]) -> dict[str, Any]:
    accs, bals = [], []
    conf = np.zeros((len(p.ds.gestures),) * 2, dtype=np.int64)
    for s in subjects:
        tr_idx = p.indices([s], [TRAIN])
        t = p.table(p.indices([s]), tr_idx)
        train, test = t.where(sessions=[TRAIN]), t.where(sessions=[TEST])
        pred = LDAClassifier().fit(train.X, train.label).predict(test.X)
        accs.append(accuracy(test.label, pred))
        bals.append(balanced_accuracy(test.label, pred))
        conf += confusion(test.label, pred, len(p.ds.gestures))
    return {
        "accuracy": summarize(accs),
        "balanced_accuracy": summarize(bals),
        "per_subject": dict(zip(map(str, subjects), accs, strict=True)),
        "confusion_counts": conf,
        "gestures": list(p.ds.gestures),
    }


def channels(p: Prepared, subjects: Sequence[int], quick: bool) -> dict[str, Any]:
    n_ch = p.ds.n_channels
    tables = {}
    for s in subjects:
        t = p.table(p.indices([s]), p.indices([s], [TRAIN]))
        tables[s] = (t.where(sessions=[TRAIN]), t.where(sessions=[TEST]))
    by_k: dict[str, Any] = {}
    for k in range(1, n_ch + 1):
        subsets = list(itertools.combinations(range(n_ch), k))
        if quick:
            subsets = subsets[:5]
        # mean over people, for every subset of size k
        means = []
        for sub in subsets:
            cols = channel_columns(sub, n_ch, p.cfg.feature_set)
            means.append(float(np.mean([_fit_score(tr, te, cols) for tr, te in tables.values()])))
        order = np.argsort(means)
        by_k[str(k)] = {
            "n_subsets": len(subsets),
            "mean_over_subsets": float(np.mean(means)),
            "worst": {"channels": subsets[order[0]], "accuracy": means[order[0]]},
            "best": {"channels": subsets[order[-1]], "accuracy": means[order[-1]]},
        }
    cols = channel_columns(EVEN_4, n_ch, p.cfg.feature_set)
    even = [_fit_score(tr, te, cols) for tr, te in tables.values()]
    return {
        "by_k": by_k,
        "even_4": {"channels": EVEN_4, "accuracy": summarize(even)},
        "note": "best/worst subsets are picked after seeing test accuracy; only mean_over_subsets "
        "and even_4 (fixed in advance) are unbiased estimates",
    }


def windows(p: Prepared, subjects: Sequence[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for w in WINDOWS_S:
        cfg = replace(p.cfg, window_s=w)
        accs = []
        for s in subjects:
            t = p.table(p.indices([s]), p.indices([s], [TRAIN]), cfg)
            accs.append(_fit_score(t.where(sessions=[TRAIN]), t.where(sessions=[TEST])))
        out[str(w)] = summarize(accs)
    return out


def cross_subject(p: Prepared, val: Sequence[int], report: Sequence[int]) -> dict[str, Any]:
    everyone = p.ds.subjects
    val_scores: dict[float, dict[AdaptSetting, list[float]]] = {
        c: {s: [] for s in ADAPT_GRID} for c in CAL_SECONDS
    }
    fold_data = {}
    for held in [*val, *report]:
        others = [s for s in everyone if s != held]
        # thresholds from the training people only; the new person's data is unseen
        t = p.table(p.indices(everyone), p.indices(others, [TRAIN]))
        train = t.where(subjects=others)
        base = LDAClassifier().fit(train.X, train.label)
        test = t.where(subjects=[held], sessions=[TEST])
        cals = {
            c: t.where(
                subjects=[held],
                sessions=[TRAIN],
                reps=[0],
                t_min=CAL_START_S,
                t_max=CAL_START_S + c,
            )
            for c in CAL_SECONDS
        }
        fold_data[held] = (train, base, test, cals)
        if held in val:
            for c in CAL_SECONDS:
                for s in ADAPT_GRID:
                    m = recalibrate(
                        "adapt", base, train.X, train.label, cals[c].X, cals[c].label, s
                    )
                    val_scores[c][s].append(accuracy(test.label, m.predict(test.X)))

    out: dict[str, Any] = {"by_calibration_seconds": {}}
    for c in CAL_SECONDS:
        chosen = best_setting(val_scores[c])
        results: dict[str, dict[str, list[float]]] = {m: {"acc": []} for m in METHODS}
        for held in report:
            train, base, test, cals = fold_data[held]
            cal = cals[c]
            for method in METHODS:
                m = recalibrate(method, base, train.X, train.label, cal.X, cal.label, chosen)
                results[method]["acc"].append(accuracy(test.label, m.predict(test.X)))
        out["by_calibration_seconds"][str(c)] = {
            "chosen_setting": asdict(chosen),
            "calibration_windows_mean": float(np.mean([len(fold_data[h][3][c]) for h in report])),
            "methods": {m: {"accuracy": summarize(v["acc"])} for m, v in results.items()},
            "paired_differences": {
                f"adapt_minus_{o}": summarize(
                    np.subtract(results["adapt"]["acc"], results[o]["acc"]).tolist()
                )
                for o in ("none", "new_only", "pooled")
            },
        }
    return out


def feature_sets(p: Prepared, val: Sequence[int], report: Sequence[int]) -> dict[str, Any]:
    acc: dict[tuple[str, str, int], float] = {}  # (feature set, channels, person) -> accuracy
    for name in FEATURE_SETS:
        cfg = replace(p.cfg, feature_set=name)
        for s in [*val, *report]:
            t = p.table(p.indices([s]), p.indices([s], [TRAIN]), cfg)
            tr, te = t.where(sessions=[TRAIN]), t.where(sessions=[TEST])
            acc[name, "all_channels", s] = _fit_score(tr, te)
            acc[name, "even_4", s] = _fit_score(tr, te, t.columns(EVEN_4))
    out: dict[str, Any] = {"default": p.cfg.feature_set, "chosen_on": "validation_subjects"}
    keys = ("all_channels", "even_4")
    for name in FEATURE_SETS:
        out[name] = {
            group: {k: summarize([acc[name, k, s] for s in people]) for k in keys}
            for group, people in (("validation", val), ("report", report))
        }
    if p.cfg.feature_set != "td":  # paired: same person, same split
        out["default_minus_td_report"] = {
            k: summarize([acc[p.cfg.feature_set, k, s] - acc["td", k, s] for s in report])
            for k in keys
        }
    return out


def classifiers(p: Prepared, report: Sequence[int]) -> dict[str, Any]:
    """LDA vs common nonlinear classifiers. Needs scikit-learn (the dev extra)."""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError:
        return {"skipped": "scikit-learn is not installed"}

    def models() -> dict[str, Any]:
        return {
            "lda": LDAClassifier(),
            "svm_rbf": make_pipeline(StandardScaler(), SVC(C=10.0, gamma="scale")),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1),
            "mlp": make_pipeline(
                StandardScaler(),
                MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=0),
            ),
        }

    scores: dict[str, list[float]] = {k: [] for k in models()}
    for s in report:
        t = p.table(p.indices([s]), p.indices([s], [TRAIN]))
        tr, te = t.where(sessions=[TRAIN]), t.where(sessions=[TEST])
        for name, model in models().items():
            scores[name].append(accuracy(te.label, model.fit(tr.X, tr.label).predict(te.X)))
    return {
        "feature_set": p.cfg.feature_set,
        "settings_note": "fixed in advance, not tuned",
        "accuracy": {k: summarize(v) for k, v in scores.items()},
        "lda_minus": {
            k: summarize(np.subtract(scores["lda"], v).tolist())
            for k, v in scores.items()
            if k != "lda"
        },
    }


def run(opts: RunOptions) -> dict[str, Any]:
    cfg = FeatureConfig()
    subjects = list(range(1, 23))
    val, report = VAL_SUBJECTS, REPORT_SUBJECTS
    if opts.quick:
        subjects, val, report = [1, 2, 7, 8], [1, 2], [7, 8]
    with timed("load 3DC"):
        ds = datasets.load("3dc", root=opts.data_root, download=opts.download, subjects=subjects)
    with timed("filter"):
        p = Prepared(ds, cfg)
    with timed("within-subject"):
        within = within_subject(p, subjects)
    with timed("channel subsets"):
        chans = channels(p, subjects, opts.quick)
    with timed("window lengths"):
        wins = windows(p, subjects)
    with timed("cross-subject"):
        cross = cross_subject(p, val, report)
    with timed("feature sets"):
        fsets = feature_sets(p, val, report)
    with timed("classifiers"):
        clfs = classifiers(p, report)
    payload = {
        "provenance": provenance(
            ds,
            {
                "features": asdict(cfg),
                "subjects": subjects,
                "validation_subjects": val,
                "report_subjects": report,
                "calibration": f"held-out person's session 0, rep 0, windows in "
                f"[{CAL_START_S}, {CAL_START_S} + C] s of each gesture",
                "adapt_grid": [asdict(s) for s in ADAPT_GRID],
            },
        ),
        "within_subject": within,
        "channels": chans,
        "windows": wins,
        "cross_subject": cross,
        "feature_sets": fsets,
        "classifiers": clfs,
    }
    print(f"  wrote {write_result(opts, 'threedc', payload)}")
    return payload
