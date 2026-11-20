"""Does "one of these gestures, 90% of the time" stay true across days and people?

Split conformal prediction promises that a gesture set contains the true
gesture with probability 1 - alpha, *if* the data it is used on is
exchangeable with the data it was calibrated on. EMG breaks that in three
ways, and this benchmark measures each, plus the recipe that held up:

MultiDay (one person, electrodes re-applied daily; splits as in
multiday_bench: base model on days 1-10, everything chosen on days 11-30,
days 31-121 reported). Each row is one way to get the threshold:

    static_training_days    base model; threshold from leave-one-day-out
                            scores on the training days themselves.
    static_validation_days  base model; threshold from days 11-30.
    adapt_validation_days   adapt() each day with 1 s per gesture; threshold
                            from days 11-30 handled the same way.
    adapt_same_day          adapt on [1.0, 2.0) s; threshold from the same
                            day's [2.0, 2.5) s (the textbook "recalibrate").
    adapt_blocked_cross     adapt on [1.0, 2.5) s (1.5 s); out-of-block
                            scores from three 0.5 s blocks; the nominal alpha
                            is chosen on days 11-30 so validation coverage
                            reaches the target. An empirical recipe: tuning
                            alpha gives up the finite-sample guarantee.
    adapt_aci               adaptive conformal inference across days, fed
                            each day's out-of-block miss rate on [1.0, 2.0) s.
    static_temp_aps         base model + temperature + APS (the recipe below
                            without recalibration).
    adapt_temp_aps          the recipe: adapt each day; temperature fitted on
                            adapted days 11-20; APS threshold from adapted
                            days 21-30; nominal alpha used as is.
    adapt_temp_aps_mondrian the same with one threshold per gesture.
    adapt_temp_aps_swapped  the recipe with days 11-20 and 21-30 swapped.

3DC (22 people, persons 1-6 choose, 7-22 reported):

    within_sitting  per person, out-of-block scores from the 4 cycles of
                    session 0 (fit on 3, score 1), model on all 4, tested on
                    session 1 after the 5-minute break.
    cross_person    leave one person out; threshold from the other people's
                    leave-one-person-out scores; with and without 2 s of
                    the new person's data through adapt().
    faults          within_sitting with one of the even-4 channels broken in
                    session 1 (electrode off, clipping, mains hum, motion),
                    with no monitor, with the shipped quality monitor
                    dropping flagged channels, and dropping the broken one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import datasets
from ..classifier import LDAClassifier, adapt
from ..conformal import (
    AdaptiveAlpha,
    blocked_scores,
    conformal_quantile,
    fit_temperature,
    nonconformity,
    set_metrics,
    softmax,
)
from ..datasets import multiday as multiday_data
from ..datasets import threedc
from ..datasets.core import Dataset, Recording
from ..evaluation import FeatureConfig, FeatureTable, featurize, fit_thresholds, summarize
from ..quality import DROP_CHANNEL_ON, QualityConfig, assess, scan_recording
from . import multiday_bench as mdb
from . import quality_bench as qb
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
IntArray = NDArray[np.intp]

ALPHAS = (0.05, 0.1, 0.2)
WITHIN_5 = 0.05  # "days within 5 points of the target"
ADAPT_SECONDS = 1.0
TEMP_DAYS = mdb.VAL_DAYS[:10]  # 11-20
THRESH_DAYS = mdb.VAL_DAYS[10:]  # 21-30
BLOCK_EDGES = (1.0, 1.5, 2.0, 2.5)  # three 0.5 s blocks before the test windows
ALPHA_GRID = tuple(float(a) for a in np.geomspace(0.001, 0.5, 60))
ACI_GAMMAS = (0.005, 0.01, 0.02, 0.05, 0.1)
EVEN_4 = (0, 2, 5, 7)
CROSS_PERSON_CAL_S = 2.0
SEED = 0
SEEDS = (0, 1, 2, 3, 4)  # the APS random share: spread over seeds for the headline row
TAIL_S = 1.0  # sensitivity: drop test windows ending in the last second of a hold
HOLD_BIN_S = 0.5


# ------------------------------------------------------------------- scoring


class Scored:
    """Decision scores and true class indices for one group (a day, a person).

    `t_start` (seconds into each window's recording) is kept when known, so
    set sizes can be reported against time into the hold.
    """

    def __init__(self, decision: FloatArray, y: IntArray, t_start: FloatArray | None = None):
        self.decision, self.y, self.t_start = decision, y, t_start

    @classmethod
    def of(cls, model: LDAClassifier, t: FeatureTable) -> Scored:
        return cls(model.decision_function(t.X), np.searchsorted(model.classes, t.label), t.t_start)


def _true_scores(groups: Sequence[Scored], temp: float, score: str, seed: int) -> FloatArray:
    rng = np.random.default_rng(seed)
    out = []
    for g in groups:
        s = nonconformity(softmax(g.decision, temp), score, rng)
        out.append(s[np.arange(len(g.y)), g.y])
    return np.concatenate(out)


def _thresholds(
    groups: Sequence[Scored], alpha: float, temp: float, score: str, mondrian: bool, seed: int
) -> FloatArray:
    rng = np.random.default_rng(seed)
    true, ys = [], []
    for g in groups:
        s = nonconformity(softmax(g.decision, temp), score, rng)
        true.append(s[np.arange(len(g.y)), g.y])
        ys.append(g.y)
    t, y = np.concatenate(true), np.concatenate(ys)
    k = groups[0].decision.shape[1]
    if mondrian:
        return np.array([conformal_quantile(t[y == c], alpha) for c in range(k)])
    return np.full(k, conformal_quantile(t, alpha))


def _sets(
    g: Scored, thresholds: FloatArray, temp: float, score: str, rng: np.random.Generator
) -> tuple[NDArray[np.bool_], IntArray]:
    p = softmax(g.decision, temp)
    return nonconformity(p, score, rng) <= thresholds, p.argmax(axis=1)


def _evaluate(
    groups: dict[Any, Scored],
    thresholds: dict[Any, FloatArray] | FloatArray,
    alpha: float,
    temp: float,
    score: str,
    seed: int,
) -> dict[str, Any]:
    """Per-group and pooled metrics; `thresholds` may differ per group."""
    rng = np.random.default_rng(seed + 1)
    per: dict[str, dict[str, float]] = {}
    k = next(iter(groups.values())).decision.shape[1]
    hit_by_class, n_by_class = np.zeros(k), np.zeros(k)
    pooled_single, pooled_correct = 0, 0
    by_time: dict[float, list[FloatArray]] = {}
    for key, g in groups.items():
        th = thresholds[key] if isinstance(thresholds, dict) else thresholds
        sets, top = _sets(g, th, temp, score, rng)
        m = set_metrics(sets, g.y, top)
        per[str(key)] = m
        hit = sets[np.arange(len(g.y)), g.y]
        np.add.at(hit_by_class, g.y, hit)
        np.add.at(n_by_class, g.y, 1)
        single = sets.sum(axis=1) == 1
        pooled_single += int(single.sum())
        pooled_correct += int(np.sum(top[single] == g.y[single]))
        if g.t_start is not None:
            bins = np.floor(g.t_start / HOLD_BIN_S) * HOLD_BIN_S
            for b in np.unique(bins):
                rows = bins == b
                by_time.setdefault(float(b), []).append(
                    np.column_stack([sets[rows].sum(axis=1), hit[rows], single[rows]])
                )
    cov = [m["coverage"] for m in per.values()]
    target = 1 - alpha
    by_class = hit_by_class / np.maximum(n_by_class, 1)
    return {
        "target": target,
        "coverage": summarize(cov),
        "share_within_5_points": float(np.mean(np.array(cov) >= target - WITHIN_5)),
        "share_at_or_above_target": float(np.mean(np.array(cov) >= target)),
        "mean_set_size": float(np.mean([m["mean_set_size"] for m in per.values()])),
        "singleton_rate": float(np.mean([m["singleton_rate"] for m in per.values()])),
        "empty_rate": float(np.mean([m["empty_rate"] for m in per.values()])),
        "set_size_over_groups": summarize([m["mean_set_size"] for m in per.values()]),
        "singleton_rate_over_groups": summarize([m["singleton_rate"] for m in per.values()]),
        "accuracy_when_singleton": pooled_correct / pooled_single if pooled_single else None,
        "coverage_by_gesture": by_class.tolist(),
        "worst_gesture_coverage": float(by_class.min()),
        "by_time_into_hold": {
            f"{b:.1f}": {
                "mean_set_size": float(np.mean(np.vstack(v)[:, 0])),
                "coverage": float(np.mean(np.vstack(v)[:, 1])),
                "singleton_rate": float(np.mean(np.vstack(v)[:, 2])),
                "n_windows": int(len(np.vstack(v))),
            }
            for b, v in sorted(by_time.items())
        },
        "per_group": per,
    }


def _rows(
    groups: dict[Any, Scored],
    cal: Sequence[Scored] | Mapping[Any, Sequence[Scored]],
    temp: float = 1.0,
    score: str = "lac",
    mondrian: bool = False,
    seed: int = SEED,
) -> dict[str, Any]:
    """Evaluate one threshold recipe at every alpha.

    `cal` is either one calibration pool for all groups, or a pool per group.
    """
    out: dict[str, Any] = {"temperature": temp, "score": score, "mondrian": mondrian}
    for a in ALPHAS:
        if isinstance(cal, Mapping):
            th: dict[Any, FloatArray] | FloatArray = {
                k: _thresholds(c, a, temp, score, mondrian, seed) for k, c in cal.items()
            }
        else:
            th = _thresholds(cal, a, temp, score, mondrian, seed)
        res = _evaluate(groups, th, a, temp, score, seed)
        res["threshold"] = _describe_thresholds(th)
        out[str(a)] = res
    return out


def _describe_thresholds(th: Mapping[Any, FloatArray] | FloatArray) -> Any:
    """Thresholds for the record: the shared value, the per-gesture list, or (one
    per group) a summary of the shared values. Infinity is written as null."""

    def clean(v: float) -> float | None:
        return float(v) if np.isfinite(v) else None

    if isinstance(th, Mapping):
        firsts = np.array([v[0] for v in th.values()])
        finite = firsts[np.isfinite(firsts)]
        return {
            "per_group": True,
            "median": clean(float(np.median(finite))) if len(finite) else None,
            "share_infinite": float(np.mean(~np.isfinite(firsts))),
        }
    arr = np.asarray(th)
    return clean(float(arr[0])) if np.all(arr == arr[0]) else [clean(float(v)) for v in arr]


def _accuracy(groups: Mapping[Any, Scored]) -> float:
    return float(np.mean([np.mean(g.decision.argmax(axis=1) == g.y) for g in groups.values()]))


def _fit_temp(groups: Sequence[Scored]) -> float:
    return fit_temperature(
        np.vstack([g.decision for g in groups]), np.concatenate([g.y for g in groups])
    )


# ------------------------------------------------------------------- MultiDay


def _window(t: FeatureTable, day: int, start: float, end: float) -> FeatureTable:
    return t.where(sessions=[day], t_min=start, t_max=end)


def _adapted(base: LDAClassifier, cal: FeatureTable, s: AdaptSetting) -> LDAClassifier:
    return adapt(base, cal.X, cal.label, s.mean_prior, s.covariance_weight)


def _blocked_day(
    base: LDAClassifier, t: FeatureTable, day: int, s: AdaptSetting, edges: Sequence[float]
) -> tuple[Scored, LDAClassifier]:
    """Out-of-block scores on [edges[0], edges[-1]) and the model adapted on all of it."""
    span = _window(t, day, edges[0], edges[-1])
    block = np.full(len(span), -1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        block[(span.t_start >= lo - 1e-9) & (span.t_end <= hi + 1e-9)] = i
    keep = block >= 0  # windows straddling two blocks belong to neither
    span = span.take(keep)
    block = block[keep]
    decision, y, _ = blocked_scores(
        lambda X, yy: adapt(base, X, yy, s.mean_prior, s.covariance_weight),
        span.X,
        span.label,
        block,
    )
    return Scored(decision, y), _adapted(base, span, s)


def multiday(t: FeatureTable) -> dict[str, Any]:
    days = sorted(set(t.session.tolist()))
    train = [d for d in days if d in mdb.TRAIN_DAYS]
    val = [d for d in days if d in mdb.VAL_DAYS]
    test = [d for d in days if d in mdb.TEST_DAYS]
    # days 11-20 and 21-30; a --quick run has fewer, so it halves what it has
    half = len(val) // 2 if not set(TEMP_DAYS) <= set(val) else len(TEMP_DAYS)
    temp_days, thresh_days = val[:half], val[half:]
    base_t = t.where(sessions=train)
    base = LDAClassifier().fit(base_t.X, base_t.label)

    def cal(d: int, sec: float = ADAPT_SECONDS) -> FeatureTable:
        return _window(t, d, mdb.CAL_START_S, mdb.CAL_START_S + sec)

    # the adapt setting, chosen exactly as multiday_bench does (accuracy, days 11-30)
    def val_accuracy(s: AdaptSetting) -> list[float]:
        out = []
        for d in val:
            te = mdb._test(t, d)
            out.append(float(np.mean(_adapted(base, cal(d), s).predict(te.X) == te.label)))
        return out

    setting = best_setting({s: val_accuracy(s) for s in ADAPT_GRID})
    adapted = {d: _adapted(base, cal(d), setting) for d in val + test}
    static = {d: Scored.of(base, mdb._test(t, d)) for d in val + test}
    adapt_sc = {d: Scored.of(adapted[d], mdb._test(t, d)) for d in val + test}
    test_static = {d: static[d] for d in test}
    test_adapt = {d: adapt_sc[d] for d in test}

    rows: dict[str, Any] = {}

    # 1. leave-one-day-out scores on the training days
    lodo = []
    for d in train:
        rest = t.where(sessions=[x for x in train if x != d])
        lodo.append(Scored.of(LDAClassifier().fit(rest.X, rest.label), mdb._test(t, d)))
    rows["static_training_days"] = _rows(test_static, lodo)
    rows["static_validation_days"] = _rows(test_static, [static[d] for d in val])
    rows["adapt_validation_days"] = _rows(test_adapt, [adapt_sc[d] for d in val])

    # 4. same day: adapt on [1.0, 2.0), threshold from [2.0, 2.5)
    same_day = {d: [Scored.of(adapted[d], _window(t, d, 2.0, mdb.TEST_START_S))] for d in test}
    rows["adapt_same_day"] = _rows(test_adapt, same_day)

    # 5. blocked cross-conformal on [1.0, 2.5) with alpha tuned on validation days
    rows["adapt_blocked_cross"] = _blocked_cross(base, t, val, test, setting)

    # 6. adaptive conformal inference across days
    rows["adapt_aci"] = _aci(base, t, val, test, setting, adapt_sc)

    # 7-10. temperature + APS
    t_static = _fit_temp([static[d] for d in temp_days])
    rows["static_temp_aps"] = _rows(test_static, [static[d] for d in thresh_days], t_static, "aps")
    t_adapt = _fit_temp([adapt_sc[d] for d in temp_days])
    rows["adapt_temp_aps"] = _rows(test_adapt, [adapt_sc[d] for d in thresh_days], t_adapt, "aps")
    rows["adapt_temp_aps_mondrian"] = _rows(
        test_adapt, [adapt_sc[d] for d in thresh_days], t_adapt, "aps", mondrian=True
    )
    t_swap = _fit_temp([adapt_sc[d] for d in thresh_days])
    rows["adapt_temp_aps_swapped"] = _rows(
        test_adapt, [adapt_sc[d] for d in temp_days], t_swap, "aps"
    )
    rows["adapt_temp_lac"] = _rows(test_adapt, [adapt_sc[d] for d in thresh_days], t_adapt, "lac")

    trend = _trend(
        rows["adapt_temp_aps"]["0.1"]["per_group"], rows["static_temp_aps"]["0.1"]["per_group"]
    )

    # the APS random share: how much the headline row moves with the seed
    seeds = {
        str(sd): _rows(test_adapt, [adapt_sc[d] for d in thresh_days], t_adapt, "aps", seed=sd)
        for sd in SEEDS
    }
    seed_spread = {
        str(a): {
            k: summarize([seeds[str(sd)][str(a)][k] for sd in SEEDS])
            for k in ("mean_set_size", "singleton_rate")
        }
        | {"coverage": summarize([seeds[str(sd)][str(a)]["coverage"]["mean"] for sd in SEEDS])}
        for a in ALPHAS
    }

    # sensitivity: the last second of each hold behaves differently (the
    # gesture weakens before the recording stops), so repeat without it
    def trim(g: Scored, d: int) -> Scored:
        te = mdb._test(t, d)
        end = np.asarray(t.meta["durations"], dtype=np.float64)[te.recording]
        keep = te.t_end <= end - TAIL_S + 1e-9
        return Scored(g.decision[keep], g.y[keep], None if g.t_start is None else g.t_start[keep])

    trimmed_static = {d: trim(static[d], d) for d in val + test}
    trimmed_adapt = {d: trim(adapt_sc[d], d) for d in val + test}
    tail = {
        "dropped_seconds": TAIL_S,
        "share_of_test_windows_dropped": 1
        - sum(len(trimmed_adapt[d].y) for d in test) / sum(len(adapt_sc[d].y) for d in test),
        "accuracy": {
            "static": _accuracy({d: trimmed_static[d] for d in test}),
            "adapted": _accuracy({d: trimmed_adapt[d] for d in test}),
        },
        "rows": {
            # thresholds as deployed (from untrimmed days), tested on trimmed days
            "adapt_temp_aps": _rows(
                {d: trimmed_adapt[d] for d in test},
                [adapt_sc[d] for d in thresh_days],
                t_adapt,
                "aps",
            ),
            # every step trimmed, including the temperature and threshold days
            "adapt_temp_aps_all_trimmed": _rows(
                {d: trimmed_adapt[d] for d in test},
                [trimmed_adapt[d] for d in thresh_days],
                _fit_temp([trimmed_adapt[d] for d in temp_days]),
                "aps",
            ),
            "adapt_validation_days": _rows(
                {d: trimmed_adapt[d] for d in test}, [trimmed_adapt[d] for d in val]
            ),
            "static_validation_days": _rows(
                {d: trimmed_static[d] for d in test}, [trimmed_static[d] for d in val]
            ),
            "adapt_same_day": _rows({d: trimmed_adapt[d] for d in test}, same_day),
        },
    }
    return {
        "adapt_setting": asdict(setting),
        "accuracy": {"static": _accuracy(test_static), "adapted": _accuracy(test_adapt)},
        "temperature_days": temp_days,
        "threshold_days": thresh_days,
        "test_days": test,
        "rows": rows,
        "coverage_trend_at_0.1": trend,
        "seed_spread_adapt_temp_aps": {"seeds": SEEDS, "by_alpha": seed_spread},
        "without_last_second_of_hold": tail,
    }


def _trend(
    adapted: dict[str, dict[str, float]], static: dict[str, dict[str, float]]
) -> dict[str, Any]:
    """Coverage by thirds of the test days and its slope per 100 days."""
    out: dict[str, Any] = {}
    for name, per in (("adapt_temp_aps", adapted), ("static_temp_aps", static)):
        days = np.array([int(k) for k in per])
        cov = np.array([per[k]["coverage"] for k in per])
        thirds = np.array_split(np.argsort(days), 3)
        out[name] = {
            "by_thirds": [float(cov[i].mean()) for i in thirds],
            "slope_per_100_days": float(np.polyfit(days, cov, 1)[0] * 100)
            if len(days) > 1
            else 0.0,
        }
    return out


def _blocked_cross(
    base: LDAClassifier, t: FeatureTable, val: list[int], test: list[int], s: AdaptSetting
) -> dict[str, Any]:
    per_day = {d: _blocked_day(base, t, d, s, BLOCK_EDGES) for d in val + test}
    groups = {d: Scored.of(per_day[d][1], mdb._test(t, d)) for d in val + test}
    out: dict[str, Any] = {"temperature": 1.0, "score": "lac", "calibration_seconds": 1.5}
    for a in ALPHAS:
        target = 1 - a
        chosen = None
        for nominal in sorted(ALPHA_GRID, reverse=True):  # largest alpha that reaches target
            th = {d: _thresholds([per_day[d][0]], nominal, 1.0, "lac", False, SEED) for d in val}
            cov = _evaluate({d: groups[d] for d in val}, th, nominal, 1.0, "lac", SEED)
            if cov["coverage"]["mean"] >= target:
                chosen = nominal
                break
        if chosen is None:
            out[str(a)] = {
                "target": target,
                "chosen_nominal_alpha": None,
                "note": "no nominal alpha reached the target on the validation days",
            }
            continue
        th = {d: _thresholds([per_day[d][0]], chosen, 1.0, "lac", False, SEED) for d in test}
        res = _evaluate({d: groups[d] for d in test}, th, chosen, 1.0, "lac", SEED)
        res["target"] = target
        res["share_within_5_points"] = float(
            np.mean([m["coverage"] >= target - WITHIN_5 for m in res["per_group"].values()])
        )
        res["share_at_or_above_target"] = float(
            np.mean([m["coverage"] >= target for m in res["per_group"].values()])
        )
        res["chosen_nominal_alpha"] = chosen
        out[str(a)] = res
    return out


def _aci(
    base: LDAClassifier,
    t: FeatureTable,
    val: list[int],
    test: list[int],
    s: AdaptSetting,
    adapt_sc: dict[int, Scored],
) -> dict[str, Any]:
    """ACI over days: a pooled threshold from days 11-30 at the working alpha,
    updated after each day from that day's out-of-block miss rate on [1, 2) s."""
    edges = (1.0, 1.5, 2.0)
    feedback = {d: _blocked_day(base, t, d, s, edges)[0] for d in val + test}
    pool = [adapt_sc[d] for d in val]
    pool_true = _true_scores(pool, 1.0, "lac", SEED)
    out: dict[str, Any] = {
        "temperature": 1.0,
        "score": "lac",
        "feedback": "out-of-block, [1.0, 2.0) s",
    }
    for a in ALPHAS:

        def run(days: list[int], gamma: float, state: AdaptiveAlpha) -> dict[int, FloatArray]:
            th: dict[int, FloatArray] = {}
            k = pool[0].decision.shape[1]
            for d in days:
                th[d] = np.full(k, conformal_quantile(pool_true, state.working_alpha))
                fb = feedback[d]
                sets, _ = _sets(fb, th[d], 1.0, "lac", np.random.default_rng(SEED))
                state.update(1 - float(np.mean(sets[np.arange(len(fb.y)), fb.y])))
            return th

        # gamma chosen on validation days (closest mean coverage to target)
        best, best_err = ACI_GAMMAS[0], np.inf
        for g in ACI_GAMMAS:
            th = run(val, g, AdaptiveAlpha(a, g))
            cov = _evaluate({d: adapt_sc[d] for d in val}, th, a, 1.0, "lac", SEED)["coverage"][
                "mean"
            ]
            if abs(cov - (1 - a)) < best_err:
                best, best_err = g, abs(cov - (1 - a))
        state = AdaptiveAlpha(a, best)
        run(val, best, state)  # warm up through the validation days, then carry on
        th = run(test, best, state)
        res = _evaluate({d: adapt_sc[d] for d in test}, th, a, 1.0, "lac", SEED)
        res["gamma"] = best
        out[str(a)] = res
    return out


# ------------------------------------------------------------------------ 3DC


def _person_blocks(
    tr: FeatureTable, cols: NDArray[np.intp] | None = None
) -> tuple[Scored, LDAClassifier]:
    X = tr.X if cols is None else tr.X[:, cols]
    decision, y, _ = blocked_scores(lambda A, b: LDAClassifier().fit(A, b), X, tr.label, tr.rep)
    return Scored(decision, y), LDAClassifier().fit(X, tr.label)


def within_sitting(
    tables: dict[int, FeatureTable], val: list[int], report: list[int]
) -> dict[str, Any]:
    oof: dict[int, Scored] = {}
    test: dict[int, Scored] = {}
    for s in val + report:
        t = tables[s]
        sc, model = _person_blocks(t.where(sessions=[0]))
        oof[s] = sc
        test[s] = Scored.of(model, t.where(sessions=[1]))
    temp = _fit_temp([oof[s] for s in val])
    rep_test = {s: test[s] for s in report}
    per_person_cal = {s: [oof[s]] for s in report}
    return {
        "temperature": temp,
        "accuracy": float(
            np.mean([np.mean(g.decision.argmax(1) == g.y) for g in rep_test.values()])
        ),
        "rows": {
            "lac": _rows(rep_test, per_person_cal),
            "aps": _rows(rep_test, per_person_cal, 1.0, "aps"),
            "temp_aps": _rows(rep_test, per_person_cal, temp, "aps"),
            "temp_aps_mondrian": _rows(rep_test, per_person_cal, temp, "aps", mondrian=True),
        },
    }


def cross_person(
    ds: Dataset, clean: list[FloatArray], cfg: FeatureConfig, val: list[int], report: list[int]
) -> dict[str, Any]:
    """Leave one person out, with and without 2 s per gesture through adapt().

    Person p's threshold comes from the other people's leave-one-person-out
    scores, so nothing about p's own labels is used.
    """
    everyone = list(ds.subjects)
    folds: dict[int, tuple[LDAClassifier, FeatureTable, FeatureTable]] = {}
    for held in everyone:
        others = [s for s in everyone if s != held]
        th_idx = [i for i, r in enumerate(ds.recordings) if r.subject in others and r.session == 0]
        th = fit_thresholds(
            [ds.recordings[i] for i in th_idx],
            ds.mains_hz,
            cfg,
            clean=[clean[i] for i in th_idx],
        )
        t = featurize(ds.recordings, ds.mains_hz, cfg, th, clean=clean)
        tr = t.where(subjects=others)
        base = LDAClassifier().fit(tr.X, tr.label)
        cal = t.where(
            subjects=[held], sessions=[0], reps=[0], t_min=1.0, t_max=1.0 + CROSS_PERSON_CAL_S
        )
        folds[held] = (base, cal, t.where(subjects=[held], sessions=[1]))
    setting = best_setting(
        {
            s: [
                float(np.mean(_adapted(b, c, s).predict(te.X) == te.label))
                for b, c, te in (folds[h] for h in val)
            ]
            for s in ADAPT_GRID
        }
    )
    none = {h: Scored.of(b, te) for h, (b, _, te) in folds.items()}
    adapted = {h: Scored.of(_adapted(b, c, setting), te) for h, (b, c, te) in folds.items()}
    out: dict[str, Any] = {
        "calibration_seconds": CROSS_PERSON_CAL_S,
        "adapt_setting": asdict(setting),
    }
    for name, scores in (("no_calibration", none), ("adapt", adapted)):
        temp = _fit_temp([scores[h] for h in val])
        pools = {h: [scores[o] for o in everyone if o != h] for h in report}
        rep = {h: scores[h] for h in report}
        out[name] = {
            "accuracy": float(
                np.mean([np.mean(g.decision.argmax(1) == g.y) for g in rep.values()])
            ),
            "temperature": temp,
            "rows": {"lac": _rows(rep, pools), "temp_aps": _rows(rep, pools, temp, "aps")},
        }
    return out


class _DropCache:
    """Per person: models without some channels, their out-of-block scores and thresholds."""

    def __init__(self, train: FeatureTable, n_channels: int, temp: float, score: str = "aps"):
        self.train, self.n_channels, self.temp, self.score = train, n_channels, temp, score
        self.models: dict[tuple[int, ...], tuple[Scored, LDAClassifier]] = {}
        self.th: dict[tuple[tuple[int, ...], float], FloatArray] = {}

    def keep(self, bad: tuple[int, ...]) -> list[int]:
        return [c for c in range(self.n_channels) if c not in bad]

    def get(self, bad: tuple[int, ...]) -> tuple[Scored, LDAClassifier]:
        if bad not in self.models:
            cols = self.train.columns(self.keep(bad)) if bad else None
            self.models[bad] = _person_blocks(self.train, cols)
        return self.models[bad]

    def threshold(self, bad: tuple[int, ...], alpha: float) -> FloatArray:
        if (bad, alpha) not in self.th:
            oof = self.get(bad)[0]
            self.th[bad, alpha] = _thresholds([oof], alpha, self.temp, self.score, False, SEED)
        return self.th[bad, alpha]

    def scored(self, bad: tuple[int, ...], t: FeatureTable) -> Scored:
        model = self.get(bad)[1]
        X = t.X[:, t.columns(self.keep(bad))] if bad else t.X
        return Scored(model.decision_function(X), np.searchsorted(model.classes, t.label))


def _measure(
    drops: _DropCache, bad: tuple[int, ...], t: FeatureTable, rng: np.random.Generator
) -> dict[str, dict[str, float]]:
    g = drops.scored(bad, t)
    out = {}
    for a in ALPHAS:
        sets, top = _sets(g, drops.threshold(bad, a), drops.temp, drops.score, rng)
        out[str(a)] = set_metrics(sets, g.y, top)
    return out


def _monitor_sets(
    t: FeatureTable,
    timelines: Sequence[tuple[FloatArray, NDArray[np.bool_]]],
    drops: _DropCache,
    alpha: float,
    rng: np.random.Generator,
    shared: bool = False,
) -> tuple[NDArray[np.bool_], IntArray, IntArray]:
    """Each window decoded by the model without whatever the monitor had flagged
    by the end of that window, against that model's own threshold, or with
    `shared` against the full model's threshold (what GesturePipeline does)."""
    n_ch = drops.n_channels
    bads: list[tuple[int, ...]] = []
    for i in range(len(t)):
        times, flagged = timelines[int(t.recording[i])]
        ready = np.nonzero(times <= t.t_end[i] + 1e-9)[0]
        bad = tuple(int(c) for c in np.nonzero(flagged[ready[-1]])[0]) if len(ready) else ()
        bads.append(() if len(bad) == n_ch else bad)
    k = len(drops.get(())[1].classes)
    sets = np.zeros((len(t), k), dtype=bool)
    top = np.empty(len(t), dtype=np.intp)
    y = np.empty(len(t), dtype=np.intp)
    for bad in set(bads):
        rows = np.array([b == bad for b in bads])
        g = drops.scored(bad, t.take(rows))
        th = drops.threshold(() if shared else bad, alpha)
        s, tp = _sets(g, th, drops.temp, drops.score, rng)
        sets[rows], top[rows], y[rows] = s, tp, g.y
    return sets, top, y


def faults(ds: Dataset, temp: float, report: list[int], quick: bool) -> dict[str, Any]:
    """Coverage with one electrode broken for the whole test session."""
    fcfg = FeatureConfig()
    rng = np.random.default_rng(1)
    base_q = QualityConfig(
        fs=ds.fs, mains_hz=ds.mains_hz, clip_low=-threedc.ADC_RAIL, clip_high=threedc.ADC_RAIL
    )
    qcfg, _ = qb.choose_thresholds(ds, base_q)
    floors = qb._floors(ds, qcfg, report)
    menu = qb.fault_menu(qcfg.mains_hz)
    scenarios = {
        "lead_off": menu["lead_off"][1]["x0.01"],
        "clipping": menu["clipping"][1]["peak2x"],
        "line_noise": menu["line_noise"][1]["4x"],
        "motion": menu["motion"][1]["3x"],
    }
    people = report[:2] if quick else report
    channels = EVEN_4[:2] if quick else EVEN_4
    n_ch = ds.n_channels
    decoders = ("clean", "no_monitor", "monitor", "monitor_shared_threshold", "oracle_drop")
    shared_rng = np.random.default_rng(2)  # its own stream, so the other rows don't move
    acc: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {
        sc: {d: {str(a): [] for a in ALPHAS} for d in decoders} for sc in scenarios
    }
    for s in people:
        train_recs = ds.select(subjects=[s], sessions=[0])
        test_recs = ds.select(subjects=[s], sessions=[1])
        th = fit_thresholds(train_recs, ds.mains_hz, fcfg)
        train = featurize(train_recs, ds.mains_hz, fcfg, th)
        clean_test = featurize(test_recs, ds.mains_hz, fcfg, th)
        drops = _DropCache(train, n_ch, temp)

        clean_m = _measure(drops, (), clean_test, rng)
        for scen, fault in scenarios.items():
            for ch in channels:
                broken: list[Recording] = []
                for r in test_recs:
                    x = r.emg.copy()
                    band = assess(x, qcfg).channels[ch].band_rms
                    x[:, ch] = fault(x[:, ch], band, qcfg.fs, rng)
                    broken.append(replace(r, emg=x))
                t = featurize(broken, ds.mains_hz, fcfg, th)
                res = acc[scen]
                found = {
                    "clean": clean_m,
                    "no_monitor": _measure(drops, (), t, rng),
                    "oracle_drop": _measure(drops, (ch,), t, rng),
                }
                timelines = [
                    scan_recording(b.emg, qcfg, floors[s], qb.HOP_S, only=DROP_CHANNEL_ON)
                    for b in broken
                ]
                found["monitor"] = {}
                found["monitor_shared_threshold"] = {}
                for a in ALPHAS:
                    sets, top, y = _monitor_sets(t, timelines, drops, a, rng)
                    found["monitor"][str(a)] = set_metrics(sets, y, top)
                    sets, top, y = _monitor_sets(t, timelines, drops, a, shared_rng, shared=True)
                    found["monitor_shared_threshold"][str(a)] = set_metrics(sets, y, top)
                for d in decoders:
                    for a in ALPHAS:
                        res[d][str(a)].append(found[d][str(a)])
    summary: dict[str, Any] = {}
    for scen, by_dec in acc.items():
        summary[scen] = {}
        for d, by_a in by_dec.items():
            summary[scen][d] = {}
            for key, ms in by_a.items():
                summary[scen][d][key] = {
                    "coverage": summarize([m["coverage"] for m in ms]),
                    "mean_set_size": float(np.mean([m["mean_set_size"] for m in ms])),
                    "singleton_rate": float(np.mean([m["singleton_rate"] for m in ms])),
                    "accuracy_when_singleton": float(
                        np.nanmean([m["accuracy_when_singleton"] for m in ms])
                    ),
                }
    return {
        "recipe": "temperature (from within_sitting) + APS, per-person out-of-block thresholds",
        "monitor": "thresholds recalibrated for the model without the flagged channels",
        "monitor_shared_threshold": "the full model's threshold kept, as GesturePipeline does",
        "failed_channels": list(channels),
        "averaged_over": f"{len(people)} persons x {len(channels)} failed channels",
        "results": summary,
    }


# ------------------------------------------------------------------------ run


def run(opts: RunOptions) -> dict[str, Any]:
    cfg = FeatureConfig()
    _, tables = mdb.build_tables(opts, cfg)
    with timed("MultiDay conformal"):
        md = multiday(tables[cfg.feature_set])
    del tables
    with timed("load 3DC"):
        subjects = [1, 2, 7, 8] if opts.quick else None
        ds = datasets.load("3dc", root=opts.data_root, download=opts.download, subjects=subjects)
    val = [s for s in ds.subjects if s in qb.VAL_SUBJECTS]
    report = [s for s in ds.subjects if s in qb.REPORT_SUBJECTS]
    from .threedc_bench import Prepared

    with timed("3DC within sitting"):
        prep = Prepared(ds, cfg)
        per_person = {
            s: prep.table(prep.indices([s]), prep.indices([s], [0])) for s in val + report
        }
        within = within_sitting(per_person, val, report)
    del per_person
    with timed("3DC cross person"):
        cross = cross_person(ds, prep.clean, cfg, val, report)
    with timed("3DC faults"):
        flt = faults(ds, within["temperature"], report, opts.quick)
    payload = {
        "provenance": provenance(
            None,
            {
                "features": asdict(cfg),
                "alphas": ALPHAS,
                "seed": SEED,
                "seeds_for_spread": SEEDS,
                "tail_sensitivity_s": TAIL_S,
                "multiday": {
                    "train_days": mdb.TRAIN_DAYS,
                    "temperature_days": TEMP_DAYS,
                    "threshold_days": THRESH_DAYS,
                    "test_days": mdb.TEST_DAYS,
                    "adapt_seconds": ADAPT_SECONDS,
                    "calibration_start_s": mdb.CAL_START_S,
                    "test_start_s": mdb.TEST_START_S,
                    "blocked_cross_edges_s": BLOCK_EDGES,
                },
                "threedc": {
                    "validation_subjects": val,
                    "report_subjects": report,
                    "cross_person_calibration_s": CROSS_PERSON_CAL_S,
                },
            },
        ),
        "datasets": {
            src_name: {"url": src.url, "commit": src.commit, "citation": src.citation}
            for src_name, src in (("multiday", multiday_data.SOURCE), ("3dc", threedc.SOURCE))
        },
        "multiday": md,
        "threedc": {"within_sitting": within, "cross_person": cross, "faults": flt},
    }
    print(f"  wrote {write_result(opts, 'conformal', payload)}")
    return payload
