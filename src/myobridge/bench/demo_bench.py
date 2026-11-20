"""What the browser demo plays back: real decoder output on four MultiDay test days.

The demo page has to load in a second on a phone, so it can't train an
LDA on ten days of 2048 Hz recordings. This benchmark does that part
offline with the library itself, exactly as `bench conformal` does, and
writes every test window's gesture set to results/demo.json (the page
reads it as docs/demo/days.json, a link to that file):

    recalibrate   adapt() with 1 s per gesture that day, or the day 1-10 model
    method        "myobridge": temperature + APS, threshold from adapted days
                  21-30 (the recipe); "textbook": the usual score (one minus
                  the probability of the true gesture), no temperature,
                  threshold from days 11-30
    electrode     working; electrode 1 lifted for the second half of the day
                  with no quality monitor; the same with the monitor, which
                  drops the flagged electrode and keeps the same threshold,
                  as GesturePipeline does

The sets come from `ConformalGestures`, the class an application uses.
The page only replays them and counts, so every number it shows is
worked out in the browser from these sets. The headline numbers it shows
beside them are copied from results/conformal.json, which this benchmark
reads, so run `bench conformal` first.

The file holds no EMG: only gesture sets, decoded gestures and labels.
"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import datasets, faults
from ..classifier import LDAClassifier
from ..conformal import ConformalGestures, fit_temperature
from ..datasets import multiday as multiday_data
from ..datasets.core import Dataset
from ..evaluation import FeatureConfig, FeatureTable, featurize, fit_thresholds, prefilter
from ..quality import DROP_CHANNEL_ON, QualityConfig, rest_floor, scan_recording
from . import conformal_bench as cb
from . import multiday_bench as mdb
from . import quality_bench as qb
from .common import ADAPT_GRID, RunOptions, _jsonable, best_setting, provenance, timed

IntArray = NDArray[np.intp]

DEMO_DAYS = (31, 61, 91, 121)  # evenly spread over the 91 test days, not picked
ALPHA = 0.1
FAULT_CHANNEL = 0  # "electrode 1"
FAULT_FROM = 0.5  # the electrode lifts for the second half of the day's gestures
SEED = 0


def _encode(sets: NDArray[np.bool_]) -> list[int]:
    """Each row's set as a bitmask: bit c is gesture c."""
    weights = 1 << np.arange(sets.shape[1], dtype=np.int64)
    return [int(v) for v in sets.astype(np.int64) @ weights]


# the page's four "Sets from" x "Recalibrate" settings and their benchmark rows
ROWS = {
    "adapt-myobridge": "adapt_temp_aps",
    "adapt-textbook": "adapt_validation_days",
    "static-myobridge": "static_temp_aps",
    "static-textbook": "static_validation_days",
}


def _spread(per_group: dict[str, Any]) -> dict[str, float]:
    """How one setting's coverage varies from day to day over the test days."""
    c = np.array([g["coverage"] for g in per_group.values()])
    return {
        "within_5_points": float(np.mean(np.abs(c - (1 - ALPHA)) <= 0.05)),
        "worst_day": float(c.min()),
        "sd": float(c.std(ddof=1)),
    }


def _headline(conformal: dict[str, Any], demo_days: list[int]) -> dict[str, Any]:
    """The benchmark numbers the page shows beside its own counts."""
    md = conformal["multiday"]
    rows = md["rows"]
    a = str(ALPHA)
    flt = conformal["threedc"]["faults"]["results"]["lead_off"]

    def cov(row: dict[str, Any]) -> dict[str, float]:
        c = row[a]["coverage"]
        return {"mean": c["mean"], "low": c["ci95_low"], "high": c["ci95_high"]}

    per = {k: rows[r][a]["per_group"] for k, r in ROWS.items()}
    recipe, rival = per["adapt-myobridge"], per["adapt-textbook"]
    return {
        "promised": 1 - ALPHA,
        "test_days": len(md["test_days"]),
        "multiday": {k: cov(rows[r]) for k, r in ROWS.items()},
        "spread": {k: _spread(v) for k, v in per.items()},
        # the textbook recipe recalibrated each day covers at least as much as
        # myobridge on this many test days
        "rival_at_least_recipe_days": sum(
            rival[d]["coverage"] >= recipe[d]["coverage"] for d in recipe
        ),
        # the benchmark's own coverage on each demo day. The textbook rows have
        # no random part and match the page exactly; myobridge's APS tie-breaks
        # are drawn from the bench's random stream there and from the library's
        # here, so those days can differ by about a point.
        "bench_per_day": {
            str(d): {k: v[str(d)]["coverage"] for k, v in per.items()} for d in demo_days
        },
        "singleton_rate": rows["adapt_temp_aps"][a]["singleton_rate"],
        "accuracy_when_singleton": rows["adapt_temp_aps"][a]["accuracy_when_singleton"],
        "threedc_lead_off": {
            "no_monitor": flt["no_monitor"][a]["coverage"]["mean"],
            "monitor": flt["monitor"][a]["coverage"]["mean"],
            "monitor_shared_threshold": flt["monitor_shared_threshold"][a]["coverage"]["mean"],
            # "16 persons x 4 failed channels"
            "people": int(conformal["threedc"]["faults"]["averaged_over"].split()[0]),
        },
        # the live 3DC panel uses this temperature with its own per-person threshold
        "threedc_temperature": conformal["threedc"]["within_sitting"]["temperature"],
    }


class _Day:
    """One demo day: its test windows, the broken copy, and the monitor's timeline."""

    def __init__(self, ds: Dataset, t: FeatureTable, day: int, th: Any, cfg: FeatureConfig):
        self.clean = mdb._test(t, day)
        recs = [r for r in ds.recordings if r.session == day]
        recs.sort(key=lambda r: r.label)
        self.labels = [r.label for r in recs]
        n_broken = int(round(len(recs) * (1 - FAULT_FROM)))
        self.broken_labels = set(self.labels[len(recs) - n_broken :])
        broken = []
        for r in recs:
            if r.label in self.broken_labels:
                x = r.emg.copy()
                x[:, FAULT_CHANNEL] = faults.lead_off(x[:, FAULT_CHANNEL])
                broken.append(replace(r, emg=x))
            else:
                broken.append(r)
        bt = featurize(broken, ds.mains_hz, cfg, th)
        self.broken = bt.where(t_min=mdb.TEST_START_S)
        qcfg = QualityConfig(fs=ds.fs, mains_hz=ds.mains_hz)
        # the monitor's rest reference, from that day's rest recording before the test part
        rest = next(r for r in recs if r.label == 0)
        floor = rest_floor(rest.emg[: int(mdb.TEST_START_S * ds.fs)], qcfg)
        self.timelines = [
            scan_recording(b.emg, qcfg, floor, qb.HOP_S, only=DROP_CHANNEL_ON) for b in broken
        ]
        self.recording_labels = [r.label for r in broken]
        self.durations = [r.duration_s for r in recs]

    def order(self, table: FeatureTable) -> IntArray:
        """Windows in play order: gesture by gesture, then by time."""
        return np.lexsort((table.t_start, table.label))

    def flagged(self) -> list[tuple[int, ...]]:
        """For each window of `broken`, the channels the monitor had flagged by its end."""
        out = []
        b = self.broken
        for i in range(len(b)):
            times, flagged = self.timelines[int(b.recording[i])]
            ready = np.nonzero(times <= b.t_end[i] + 1e-9)[0]
            bad = tuple(int(c) for c in np.nonzero(flagged[ready[-1]])[0]) if len(ready) else ()
            out.append(() if len(bad) == flagged.shape[1] else bad)
        return out


def _sets_for(
    cg: ConformalGestures, model: LDAClassifier, table: FeatureTable, bad: list[tuple[int, ...]]
) -> tuple[NDArray[np.bool_], IntArray]:
    """Sets and top gestures, each window decoded without its flagged channels."""
    k = len(model.classes)
    sets = np.zeros((len(table), k), dtype=bool)
    top = np.empty(len(table), dtype=np.intp)
    for b in sorted(set(bad)):
        rows = np.array([x == b for x in bad])
        if b:
            keep = [c for c in range(table.n_channels) if c not in b]
            cols = table.columns(keep)
            m, X = model.subset(cols), table.X[rows][:, cols]
        else:
            m, X = model, table.X[rows]
        sets[rows] = cg.with_model(m).predict_sets(X)
        top[rows] = np.argmax(m.decision_function(X), axis=1)
    return sets, top


def run(opts: RunOptions) -> dict[str, Any]:
    src = opts.out_dir / ("conformal.quick.json" if opts.quick else "conformal.json")
    if not src.exists():
        raise FileNotFoundError(f"{src} is missing: run `myobridge bench conformal` first")
    conformal = json.loads(src.read_text())
    cfg = FeatureConfig()
    days_wanted = sorted(set(mdb.TRAIN_DAYS + mdb.VAL_DAYS + list(DEMO_DAYS)))
    if opts.quick:
        days_wanted = mdb.TRAIN_DAYS[:3] + mdb.VAL_DAYS[:3] + mdb.TEST_DAYS[:2]
    with timed("load MultiDay"):
        ds = datasets.load(
            "multiday", root=opts.data_root, download=opts.download, days=days_wanted
        )
    with timed("features"):
        clean = prefilter(ds.recordings, ds.mains_hz, cfg)
        train_idx = [i for i, r in enumerate(ds.recordings) if r.session in mdb.TRAIN_DAYS]
        th = fit_thresholds(
            [ds.recordings[i] for i in train_idx],
            ds.mains_hz,
            cfg,
            clean=[clean[i] for i in train_idx],
        )
        t = featurize(ds.recordings, ds.mains_hz, cfg, th, clean=clean)
    days = sorted(set(t.session.tolist()))
    train = [d for d in days if d in mdb.TRAIN_DAYS]
    val = [d for d in days if d in mdb.VAL_DAYS]
    demo = [d for d in days if d not in mdb.TRAIN_DAYS and d not in mdb.VAL_DAYS]
    half = len(val) // 2
    temp_days, thresh_days = val[:half], val[half:]

    base_t = t.where(sessions=train)
    base = LDAClassifier().fit(base_t.X, base_t.label)

    def cal(d: int) -> FeatureTable:
        return cb._window(t, d, mdb.CAL_START_S, mdb.CAL_START_S + cb.ADAPT_SECONDS)

    def val_accuracy(s: Any) -> list[float]:
        out = []
        for d in val:
            te = mdb._test(t, d)
            out.append(float(np.mean(cb._adapted(base, cal(d), s).predict(te.X) == te.label)))
        return out

    with timed("adapt setting"):
        setting = best_setting({s: val_accuracy(s) for s in ADAPT_GRID})
    models: dict[str, dict[int, LDAClassifier]] = {
        "static": {d: base for d in val + demo},
        "adapt": {d: cb._adapted(base, cal(d), setting) for d in val + demo},
    }

    def pooled(kind: str, ds_: list[int]) -> tuple[NDArray[np.float64], IntArray]:
        sc = [cb.Scored.of(models[kind][d], mdb._test(t, d)) for d in ds_]
        return np.vstack([s.decision for s in sc]), np.concatenate([s.y for s in sc])

    calibrations: dict[str, ConformalGestures] = {}
    for kind in ("static", "adapt"):
        dec, y = pooled(kind, val)
        calibrations[f"{kind}-textbook"] = ConformalGestures.from_scores(
            base, dec, y, ALPHA, 1.0, "lac", seed=SEED
        )
        tdec, ty = pooled(kind, temp_days)
        temp = float(fit_temperature(tdec, ty))
        hdec, hy = pooled(kind, thresh_days)
        calibrations[f"{kind}-myobridge"] = ConformalGestures.from_scores(
            base, hdec, hy, ALPHA, temp, "aps", seed=SEED
        )

    out_days = []
    with timed("demo days"):
        for d in demo:
            day = _Day(ds, t, d, th, cfg)
            o_clean = day.order(day.clean)
            o_broken = day.order(day.broken)
            clean_t, broken_t = day.clean.take(o_clean), day.broken.take(o_broken)
            if not np.array_equal(clean_t.label, broken_t.label) or not np.allclose(
                clean_t.t_start, broken_t.t_start
            ):
                raise RuntimeError("the broken copy's windows don't line up with the clean ones")
            every = day.flagged()
            flagged = [every[i] for i in o_broken]
            # replay time: gesture by gesture, each from 2.5 s to the end of its recording
            starts, segments, at = {}, [], 0.0
            for label, dur in zip(day.labels, day.durations, strict=True):
                starts[label] = at
                length = dur - mdb.TEST_START_S
                segments.append(
                    {"gesture": label, "start": round(at, 3), "length": round(length, 3)}
                )
                at += length
            t_end = np.array(
                [
                    starts[int(lb)] + te - mdb.TEST_START_S
                    for lb, te in zip(clean_t.label, clean_t.t_end, strict=True)
                ]
            )
            y = np.searchsorted(base.classes, clean_t.label)
            variants: dict[str, Any] = {}
            for name, cg in calibrations.items():
                kind = name.split("-")[0]
                model = models[kind][d]
                none: list[tuple[int, ...]] = [()] * len(clean_t)
                for electrode, table, bad in (
                    ("ok", clean_t, none),
                    ("off", broken_t, none),
                    ("off_monitor", broken_t, flagged),
                ):
                    sets, top = _sets_for(cg, model, table, bad)
                    single = sets.sum(axis=1) == 1
                    hit = sets[np.arange(len(y)), y]
                    variants[f"{name}-{electrode}"] = {
                        "sets": _encode(sets),
                        "top": top.tolist(),
                        "dropped": [1 if b else 0 for b in bad],
                        # the page counts these itself; demo.test.mjs checks it agrees
                        "check": {
                            "covered": int(hit.sum()),
                            "singletons": int(single.sum()),
                            "singletons_right": int(np.sum(single & hit)),
                        },
                    }
            out_days.append(
                {
                    "day": d,
                    "segments": segments,
                    "fault_from": round(starts[min(day.broken_labels, key=day.labels.index)], 3),
                    "t": np.round(t_end, 3).tolist(),
                    "truth": y.tolist(),
                    "variants": variants,
                }
            )

    headline = _headline(conformal, [d["day"] for d in out_days])
    payload: dict[str, Any] = {
        "about": "Gesture sets for the browser demo, made by `myobridge bench demo`. No EMG.",
        "provenance": provenance(
            ds,
            {
                "features": asdict(cfg),
                "alpha": ALPHA,
                "demo_days": demo,
                "adapt_setting": asdict(setting),
                "temperature_days": temp_days,
                "threshold_days": thresh_days,
                "fault": {"channel": FAULT_CHANNEL, "kind": "lead_off", "from_share": FAULT_FROM},
                "seed": SEED,
            },
        ),
        "gestures": list(multiday_data.GESTURES),
        "temperatures": {k: v.temperature for k, v in calibrations.items()},
        "thresholds": {k: float(v.thresholds[0]) for k, v in calibrations.items()},
        "headline": headline,
        # the page opens on, and the README GIF shows, the most ordinary day
        "typical_day": _typical(out_days, headline),
        "days": out_days,
    }
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    path = opts.out_dir / ("demo.quick.json" if opts.quick else "demo.json")
    # compact, because the page downloads it
    path.write_text(json.dumps(_jsonable({"quick": opts.quick, **payload}), separators=(",", ":")))
    print(f"  wrote {path} ({path.stat().st_size / 1024:.0f} KB)")
    return payload


def _typical(days: list[dict[str, Any]], headline: dict[str, Any]) -> int:
    """The demo day whose recipe and textbook coverage are, together, closest to
    their 91-day benchmark means, so neither the best nor the worst day is shown."""

    def gap(day: dict[str, Any]) -> float:
        n = len(day["truth"])
        total = 0.0
        for variant, key in (
            ("adapt-myobridge-ok", "adapt-myobridge"),
            ("static-textbook-ok", "static-textbook"),
        ):
            cov = day["variants"][variant]["check"]["covered"] / n
            total += abs(cov - headline["multiday"][key]["mean"])
        return total

    return int(min(days, key=gap)["day"])
