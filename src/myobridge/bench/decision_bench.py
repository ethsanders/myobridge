"""Decisions, not windows: does knowing when it doesn't know mean fewer wrong actions?

Per-window accuracy and set coverage are not what a user feels. A
prosthesis or a game acts on *decisions*: how often it does the wrong thing,
how long the right thing takes to appear, and how much of the time it holds.
This benchmark replays real recordings window by window through the
decision policies in `myobridge.gating` and counts exactly that.

Every recording is one held gesture. A group's recordings are played back
to back, in a random order where no gesture follows itself, window by
window in time order, through one policy. So each hold starts with the
previous gesture still in the policy's memory, as when a user changes
gesture. (Played one at a time from a fresh policy, every vote size looks
equally fast: the first window of a MultiDay hold is nearly always right.)
Per group (a MultiDay day, a 3DC person):

    wrong_per_min      wrong actions per minute: runs of the same wrong
                       gesture that overlap the scored span, per minute of it
    wrong_share        share of the scored time spent on a wrong action
    acting_share       share of the scored time acting (not holding)
    correct_share      share of the scored time acting on the right gesture
    right_when_acting  correct_share / acting_share
    ever_wrong         share of holds with any wrong action in the span
    time_to_correct_s  from the hold's first window to its first correct
                       action (a hold that never gets one counts its full
                       length, and is also counted in never_correct)

The span is either from 2.5 s into each hold (as in the other benchmarks)
or every window of it ("from_onset").

Groups and recipes (the conformal sets are the recipe from bench conformal):

    multiday   base model on days 1-10; each day adapted with 1 s per
               gesture from [1.0, 2.0) s; temperature from adapted days
               11-20, APS thresholds from adapted days 21-30. Policies are
               chosen on days 11-30, reported on days 31-121.
    within     3DC, per person: model and out-of-block thresholds from the
               session before the break, tested on the session after it;
               temperature from persons 1-6. Chosen on persons 1-6,
               reported on 7-22.
    cross      3DC, leave one person out, 2 s per gesture through adapt();
               thresholds from the other people. Chosen on 1-6, reported
               on 7-22.

How a policy is chosen, on validation groups only: among its settings whose
mean time to the first correct action is no longer than the baseline's
(the pipeline's 5-window vote), and whose share of time acting on the right
gesture from 2.5 s is no smaller ("matched"), the one with the fewest
wrong actions per minute from 2.5 s. "ttfc_only" drops the second
condition, so a gate may hold in the middle of a gesture. The same rule at
other vote sizes' validation numbers gives the curve.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .. import datasets
from ..classifier import LDAClassifier
from ..conformal import nonconformity, softmax
from ..datasets import multiday as multiday_data
from ..datasets import threedc
from ..datasets.core import Dataset
from ..evaluation import FeatureConfig, FeatureTable, featurize, fit_thresholds, summarize
from ..gating import HOLD, ConfidenceGate, DecisionPolicy, EvidenceGate, SetGate, VotePolicy
from . import conformal_bench as cb
from . import multiday_bench as mdb
from . import quality_bench as qb
from .common import ADAPT_GRID, AdaptSetting, RunOptions, best_setting, provenance, timed
from .common import write_result as _write

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.intp]

ALPHAS = (0.05, 0.1, 0.2)  # set gating may use the sets of any of these
SCORE_FROM_S = mdb.TEST_START_S  # 2.5 s
LATE_S = mdb.CAL_START_S  # report how often the first correct action comes after 1.0 s
BASELINE = "vote_k5"  # GesturePipeline's default vote
SEED = cb.SEED

VOTE_KS = (1, 3, 5, 7, 9, 11, 15, 21)
CURVE_KS = (3, 5, 7, 9, 11, 15)  # the baseline vote sizes the curve matches
CONF_KS = (1, 3, 5)
CONF_CUTOFFS = (0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.98)
SET_NM = (
    (1, 1),
    (2, 2),
    (3, 2),
    (3, 3),
    (4, 3),
    (4, 4),
    (5, 3),
    (5, 4),
    (5, 5),
    (6, 4),
    (6, 5),
    (6, 6),
    (8, 5),
    (8, 6),
    (8, 8),
)
EV_MEMORY = (0.5, 0.7, 0.8, 0.9, 0.95)
EV_THRESHOLD = (0.5, 0.7, 0.9, 0.95, 0.99, 0.999)
FAMILIES = ("confidence", "set", "evidence")
HOLD_MODES = ("release", "latch")


# ----------------------------------------------------------------- streams


@dataclass
class Stream:
    """One held gesture, window by window in time order."""

    proba: FloatArray  # (n, k) temperature-scaled probabilities
    sets: dict[float, BoolArray]  # alpha -> (n, k) conformal sets
    y: int  # the true class index
    t_start: FloatArray  # window starts, seconds into the recording
    t_end: FloatArray  # window ends: when each decision is made


def play_order(t: FeatureTable, rng: np.random.Generator, tries: int = 10_000) -> IntArray:
    """The recordings of `t` shuffled so that no gesture comes twice in a row
    (when that is possible; otherwise the last shuffle tried)."""
    recordings = np.unique(t.recording)
    label = np.array([t.label[np.argmax(t.recording == r)] for r in recordings])
    order = rng.permutation(len(recordings))
    for _ in range(tries):
        if not np.any(label[order][1:] == label[order][:-1]):
            break
        order = rng.permutation(len(recordings))
    out: IntArray = recordings[order]
    return out


def streams_of(
    model: LDAClassifier,
    t: FeatureTable,
    temp: float,
    thresholds: Mapping[float, FloatArray],
    rng: np.random.Generator,
) -> list[Stream]:
    """Every recording in `t` as a hold, with APS sets at each alpha, in a
    random order drawn from `rng` (the order they are played in) where no
    gesture follows itself."""
    out = []
    for r in play_order(t, rng):
        w = t.take(np.flatnonzero(t.recording == r))
        order = np.argsort(w.t_start, kind="stable")
        w = w.take(order)
        p = softmax(model.decision_function(w.X), temp)
        scores = nonconformity(p, "aps", rng)  # one random share for every alpha
        y = int(np.searchsorted(model.classes, w.label[0]))
        out.append(
            Stream(p, {a: scores <= th for a, th in thresholds.items()}, y, w.t_start, w.t_end)
        )
    return out


# ----------------------------------------------------------------- policies


@dataclass(frozen=True)
class Config:
    """One policy setting: a family, its parameters and its hold mode."""

    family: str  # "vote", "confidence", "set", "evidence"
    params: tuple[tuple[str, float], ...]
    latch: bool = False

    @property
    def name(self) -> str:
        p = "_".join(f"{k}{v:g}" for k, v in self.params)
        return f"{self.family}_{p}" + ("_latch" if self.latch else "")

    def make(self) -> DecisionPolicy:
        p = dict(self.params)
        if self.family == "vote":
            return VotePolicy(int(p["k"]))
        if self.family == "confidence":
            return ConfidenceGate(p["cutoff"], int(p["k"]), latch=self.latch)
        if self.family == "set":
            return SetGate(int(p["n"]), int(p["m"]), latch=self.latch)
        if self.family == "evidence":
            return EvidenceGate(p["threshold"], p["memory"], latch=self.latch)
        raise KeyError(self.family)

    @property
    def alpha(self) -> float | None:
        return dict(self.params).get("alpha")

    def describe(self) -> dict[str, Any]:
        return {"family": self.family, **dict(self.params), "latch": self.latch}


def grid() -> list[Config]:
    out = [Config("vote", (("k", k),)) for k in VOTE_KS]
    for latch in (False, True):
        out += [
            Config("confidence", (("k", k), ("cutoff", c)), latch)
            for k in CONF_KS
            for c in CONF_CUTOFFS
        ]
        out += [
            Config("set", (("alpha", a), ("n", n), ("m", m)), latch)
            for a in ALPHAS
            for n, m in SET_NM
        ]
        out += [
            Config("evidence", (("memory", mem), ("threshold", th)), latch)
            for mem in EV_MEMORY
            for th in EV_THRESHOLD
        ]
    return out


def quick_grid() -> list[Config]:
    """A few settings per family, for --quick smoke runs."""
    out = [Config("vote", (("k", k),)) for k in VOTE_KS]
    for latch in (False, True):
        out += [Config("confidence", (("k", 1), ("cutoff", c)), latch) for c in (0.5, 0.8)]
        out += [Config("set", (("alpha", 0.1), ("n", n), ("m", n)), latch) for n in (1, 3)]
        out += [
            Config("evidence", (("memory", 0.8), ("threshold", th)), latch) for th in (0.9, 0.99)
        ]
    return out


def decide(cfg: Config, holds: Sequence[Stream]) -> list[IntArray]:
    """One policy run through a group's holds played back to back, so each
    hold starts with whatever the previous one left behind (a gesture
    change); returns each hold's decisions."""
    a = cfg.alpha
    proba = np.vstack([h.proba for h in holds])
    sets = None if a is None else np.vstack([h.sets[a] for h in holds])
    d = cfg.make().run(proba, sets)
    return np.split(d, np.cumsum([len(h.t_start) for h in holds])[:-1])


# ----------------------------------------------------------------- metrics


def _runs(d: IntArray) -> tuple[IntArray, IntArray]:
    """Start and end (exclusive) of each run of equal values."""
    edges = np.flatnonzero(np.diff(d)) + 1
    starts = np.concatenate([[0], edges]).astype(np.intp)
    ends = np.concatenate([edges, [len(d)]]).astype(np.intp)
    return starts, ends


def _step(s: Stream) -> float:
    return float(np.median(np.diff(s.t_start))) if len(s.t_start) > 1 else 0.05


def group_metrics(streams: Sequence[Stream], decisions: Sequence[IntArray]) -> dict[str, Any]:
    """Decision metrics for one group's holds, from 2.5 s and from onset."""
    out: dict[str, Any] = {}
    for span, start in (("from_2.5s", SCORE_FROM_S), ("from_onset", None)):
        n = acting = correct = wrong = episodes = ever = 0
        minutes = 0.0
        for s, d in zip(streams, decisions, strict=True):
            keep = np.ones(len(d), bool) if start is None else s.t_start >= start - 1e-9
            if not keep.any():
                continue
            dk = d[keep]
            n += len(dk)
            minutes += len(dk) * _step(s) / 60
            acting += int(np.sum(dk != HOLD))
            correct += int(np.sum(dk == s.y))
            bad = (d != HOLD) & (d != s.y)
            wrong += int(np.sum(bad[keep]))
            ever += int(bad[keep].any())
            first = int(np.argmax(keep))
            for a, b in zip(*_runs(d), strict=True):
                if bad[a] and b > first:
                    episodes += 1
        out[span] = {
            "wrong_per_min": episodes / minutes if minutes else float("nan"),
            "wrong_share": wrong / n if n else float("nan"),
            "acting_share": acting / n if n else float("nan"),
            "correct_share": correct / n if n else float("nan"),
            "right_when_acting": correct / acting if acting else float("nan"),
            "ever_wrong": ever / len(streams),
        }
    ttc, never, late = [], 0, 0
    for s, d in zip(streams, decisions, strict=True):
        hit = np.flatnonzero(d == s.y)
        if len(hit):
            ttc.append(float(s.t_end[hit[0]] - s.t_end[0]))
            late += int(s.t_end[hit[0]] > LATE_S + 1e-9)
        else:  # never: count the whole hold (a lower bound)
            ttc.append(float(s.t_end[-1] - s.t_end[0] + _step(s)))
            never += 1
    out["time_to_correct_s"] = float(np.mean(ttc))
    out["median_time_to_correct_s"] = float(np.median(ttc))
    out["never_correct"] = never / len(streams)
    out["first_correct_after_1s"] = late / len(streams)
    return out


METRICS = (
    "wrong_per_min",
    "wrong_share",
    "acting_share",
    "correct_share",
    "right_when_acting",
    "ever_wrong",
)
TOP = ("time_to_correct_s", "median_time_to_correct_s", "never_correct", "first_correct_after_1s")


def evaluate(cfg: Config, groups: Mapping[Any, Sequence[Stream]]) -> dict[str, dict[str, Any]]:
    """Per-group metrics of one setting."""
    return {str(g): group_metrics(ss, decide(cfg, ss)) for g, ss in groups.items()}


def _flat(m: Mapping[str, Any]) -> dict[str, float]:
    out = {
        f"{span}.{k}": float(m[span][k]) for span in ("from_2.5s", "from_onset") for k in METRICS
    }
    out.update({k: float(m[k]) for k in TOP})
    return out


def mean_over_groups(per: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    flats = [_flat(m) for m in per.values()]
    return {k: float(np.nanmean([f[k] for f in flats])) for k in flats[0]}


def summary(per: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Mean and bootstrap CI over groups of every metric."""
    flats = [_flat(m) for m in per.values()]
    return {
        k: summarize([f[k] for f in flats if np.isfinite(f[k])])
        for k in flats[0]
        if any(np.isfinite(f[k]) for f in flats)
    }


def paired(
    per: Mapping[str, Mapping[str, Any]], base: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """This setting minus the baseline, group by group, with bootstrap CIs."""
    a = {g: _flat(m) for g, m in per.items()}
    b = {g: _flat(m) for g, m in base.items()}
    keys = ("from_2.5s.wrong_per_min", "from_2.5s.wrong_share", "from_2.5s.acting_share",
            "from_2.5s.correct_share", "from_2.5s.ever_wrong", "from_onset.wrong_per_min",
            "from_onset.wrong_share", "from_onset.acting_share", "from_onset.ever_wrong",
            "time_to_correct_s")  # fmt: skip
    out = {}
    for k in keys:
        diffs = [a[g][k] - b[g][k] for g in a if np.isfinite(a[g][k]) and np.isfinite(b[g][k])]
        if diffs:
            out[k] = summarize(diffs)
    return out


# ----------------------------------------------------------------- choosing


OBJECTIVE = "from_2.5s.wrong_per_min"
SPEED = "time_to_correct_s"
COVER = "from_2.5s.correct_share"


def choose(
    val: Mapping[Config, Mapping[str, float]],
    family: str,
    latch: bool,
    max_ttc: float,
    min_correct: float | None,
) -> Config | None:
    """Fewest validation wrong actions per minute among the settings at least as
    responsive as the target; ties go to the faster one, then grid order."""
    ok = [
        c
        for c, m in val.items()
        if c.family == family
        and c.latch == latch
        and m[SPEED] <= max_ttc + 1e-9
        and (min_correct is None or m[COVER] >= min_correct - 1e-9)
    ]
    if not ok:
        return None
    return min(ok, key=lambda c: (val[c][OBJECTIVE], val[c][SPEED]))


def verdict(diff: Mapping[str, Mapping[str, float]]) -> dict[str, bool]:
    """Fewer wrong actions (the 95% interval of the paired difference from
    2.5 s is below 0), and neither slower nor less often right (neither
    interval is entirely on the bad side)."""
    fewer = diff[OBJECTIVE]["ci95_high"] < 0
    not_slower = diff[SPEED]["ci95_low"] <= 0
    not_less_right = diff[COVER]["ci95_high"] >= 0
    return {
        "fewer_wrong_actions": bool(fewer),
        "not_slower": bool(not_slower),
        "not_less_often_right": bool(not_less_right),
        "wins": bool(fewer and not_slower and not_less_right),
    }


def compare(
    val_groups: Mapping[Any, Sequence[Stream]],
    report_groups: Mapping[Any, Sequence[Stream]],
    configs: Sequence[Config],
    fixed: Mapping[str, Config] | None = None,
) -> dict[str, Any]:
    """Score every setting on the validation groups, choose, report the choices.

    `fixed` adds settings chosen elsewhere (e.g. on MultiDay), reported as is.
    """
    val = {c: mean_over_groups(evaluate(c, val_groups)) for c in configs}
    votes = {int(dict(c.params)["k"]): c for c in configs if c.family == "vote"}

    chosen: dict[str, dict[str, Config | None]] = {}
    for k in CURVE_KS:
        target = val[votes[k]]
        for rule, min_corr in (("matched", target[COVER]), ("ttfc_only", None)):
            for fam in FAMILIES:
                for mode in HOLD_MODES:
                    key = f"{fam}_{mode}"
                    chosen.setdefault(f"{rule}@vote_k{k}", {})[key] = choose(
                        val, fam, mode == "latch", target[SPEED], min_corr
                    )

    to_report: dict[str, Config] = {f"vote_k{k}": c for k, c in votes.items()}
    for picks in chosen.values():
        for c in picks.values():
            if c is not None:
                to_report[c.name] = c
    for c in (fixed or {}).values():
        to_report[c.name] = c
    rep_per = {name: evaluate(c, report_groups) for name, c in to_report.items()}
    base_rep = rep_per[BASELINE]
    at_baseline = {c.name for c in chosen[f"matched@{BASELINE}"].values() if c is not None}
    at_baseline |= {c.name for c in chosen[f"ttfc_only@{BASELINE}"].values() if c is not None}
    at_baseline |= {c.name for c in (fixed or {}).values()} | {BASELINE}

    def row(name: str) -> dict[str, Any]:
        c = to_report[name]
        r: dict[str, Any] = {
            "setting": c.describe(),
            "validation": val[c],
            "report": summary(rep_per[name]),
        }
        if name != BASELINE:
            r["minus_baseline"] = diff = paired(rep_per[name], base_rep)
            r["verdict"] = verdict(diff)
        if name in at_baseline:  # per-group numbers for the headline rows only
            r["per_group"] = rep_per[name]
        return r

    return {
        "baseline": BASELINE,
        "n_settings_tried": len(configs),
        "validation_groups": [str(g) for g in val_groups],
        "report_groups": [str(g) for g in report_groups],
        "chosen": {
            point: {k: (None if c is None else c.name) for k, c in picks.items()}
            for point, picks in chosen.items()
        },
        "settings": {name: row(name) for name in to_report},
        "validation_all": {c.name: val[c] for c in configs},
        "fixed_from_elsewhere": {k: c.name for k, c in (fixed or {}).items()},
    }


def headline_choices(res: Mapping[str, Any], cfgs: Sequence[Config]) -> dict[str, Config]:
    """The settings chosen at the baseline's own point, by name -> Config."""
    by_name = {c.name: c for c in cfgs}
    out = {}
    for rule in ("matched", "ttfc_only"):
        for key, name in res["chosen"][f"{rule}@{BASELINE}"].items():
            if name is not None:
                out[f"{rule}:{key}"] = by_name[name]
    return out


# ----------------------------------------------------------------- MultiDay


def multiday_groups(
    t: FeatureTable,
) -> tuple[
    dict[int, list[Stream]],
    dict[int, list[Stream]],
    dict[str, Any],
    Callable[[int], dict[int, list[Stream]]],
]:
    """Validation and test days as streams, with the bench conformal recipe,
    and a way to stream the test days again in another playback order."""
    days = sorted(set(t.session.tolist()))
    train = [d for d in days if d in mdb.TRAIN_DAYS]
    val = [d for d in days if d in mdb.VAL_DAYS]
    test = [d for d in days if d in mdb.TEST_DAYS]
    half = len(val) // 2 if not set(cb.TEMP_DAYS) <= set(val) else len(cb.TEMP_DAYS)
    temp_days, thresh_days = val[:half], val[half:]
    base_t = t.where(sessions=train)
    base = LDAClassifier().fit(base_t.X, base_t.label)

    def cal(d: int) -> FeatureTable:
        return cb._window(t, d, mdb.CAL_START_S, mdb.CAL_START_S + cb.ADAPT_SECONDS)

    def val_accuracy(s: AdaptSetting) -> list[float]:
        out = []
        for d in val:
            te = mdb._test(t, d)
            out.append(float(np.mean(cb._adapted(base, cal(d), s).predict(te.X) == te.label)))
        return out

    setting = best_setting({s: val_accuracy(s) for s in ADAPT_GRID})
    adapted = {d: cb._adapted(base, cal(d), setting) for d in val + test}
    scored = {d: cb.Scored.of(adapted[d], mdb._test(t, d)) for d in val}
    temp = cb._fit_temp([scored[d] for d in temp_days])
    th = {
        a: cb._thresholds([scored[d] for d in thresh_days], a, temp, "aps", False, SEED)
        for a in ALPHAS
    }

    def restream(days: Sequence[int], *seed: int) -> dict[int, list[Stream]]:
        return {
            d: streams_of(
                adapted[d], t.where(sessions=[d]), temp, th, np.random.default_rng([SEED, d, *seed])
            )
            for d in days
        }

    groups = restream(val + test)
    info = {
        "adapt_setting": asdict(setting),
        "temperature": temp,
        "temperature_days": temp_days,
        "threshold_days": thresh_days,
        "thresholds": {str(a): float(v[0]) for a, v in th.items()},
        "validation_days": val,
        "test_days": test,
    }
    return (
        {d: groups[d] for d in val},
        {d: groups[d] for d in test},
        info,
        lambda seed: restream(test, seed),
    )


ORDER_SEEDS = tuple(range(1, 11))  # other playback orders for the headline rows


def order_spread(
    restream: Callable[[int], Mapping[Any, Sequence[Stream]]],
    picks: Mapping[str, Config],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """The headline rows again on other playback orders of the test groups.

    Each seed redraws the order the holds are played in on every test group
    (and the APS sets' random share), so this shows how much the headline
    depends on the one order behind it. Settings are not chosen again.
    """
    base_cfg = Config("vote", (("k", 5),))
    cfgs = {BASELINE: base_cfg, **{c.name: c for c in picks.values()}}
    rows: dict[str, dict[str, list[float]]] = {
        n: {"wrong_per_min": [], "time_to_correct_s": [], "correct_share": []} for n in cfgs
    }
    diffs: dict[str, dict[str, list[float]]] = {
        n: {"mean": [], "ci95_low": [], "ci95_high": [], "wins": []} for n in cfgs if n != BASELINE
    }
    for seed in seeds:
        groups = restream(seed)
        per = {n: evaluate(c, groups) for n, c in cfgs.items()}
        for n in cfgs:
            m = mean_over_groups(per[n])
            rows[n]["wrong_per_min"].append(m[OBJECTIVE])
            rows[n]["time_to_correct_s"].append(m[SPEED])
            rows[n]["correct_share"].append(m[COVER])
            if n == BASELINE:
                continue
            d = paired(per[n], per[BASELINE])
            for k in ("mean", "ci95_low", "ci95_high"):
                diffs[n][k].append(float(d[OBJECTIVE][k]))
            diffs[n]["wins"].append(float(verdict(d)["wins"]))
    return {
        "seeds": list(seeds),
        "settings": {
            n: {
                "by_order": rows[n],
                **(
                    {}
                    if n == BASELINE
                    else {
                        "minus_baseline_by_order": diffs[n],
                        "wins_on_every_order": bool(all(diffs[n]["wins"])),
                        "fewer_wrong_on_every_order": bool(max(diffs[n]["ci95_high"]) < 0),
                    }
                ),
            }
            for n in cfgs
        },
    }


# ----------------------------------------------------------------- 3DC


def within_groups(
    tables: Mapping[int, FeatureTable], val: list[int], report: list[int]
) -> tuple[dict[int, list[Stream]], dict[int, list[Stream]], dict[str, Any]]:
    """Per person: fit before the break, stream the session after it."""
    oof, models = {}, {}
    for s in val + report:
        oof[s], models[s] = cb._person_blocks(tables[s].where(sessions=[0]))
    temp = cb._fit_temp([oof[s] for s in val])
    groups = {}
    for s in val + report:
        th = {a: cb._thresholds([oof[s]], a, temp, "aps", False, SEED) for a in ALPHAS}
        groups[s] = streams_of(
            models[s], tables[s].where(sessions=[1]), temp, th, np.random.default_rng([SEED, s])
        )
    info = {"temperature": temp, "validation_persons": val, "report_persons": report}
    return {s: groups[s] for s in val}, {s: groups[s] for s in report}, info


def cross_groups(
    ds: Dataset, clean: list[FloatArray], cfg: FeatureConfig, val: list[int], report: list[int]
) -> tuple[dict[int, list[Stream]], dict[int, list[Stream]], dict[str, Any]]:
    """Leave one person out with 2 s per gesture through adapt(), as in bench
    conformal's cross_person: the threshold comes from the other people."""
    everyone = list(ds.subjects)
    folds: dict[int, tuple[LDAClassifier, FeatureTable, FeatureTable]] = {}
    for held in everyone:
        others = [s for s in everyone if s != held]
        th_idx = [i for i, r in enumerate(ds.recordings) if r.subject in others and r.session == 0]
        th = fit_thresholds(
            [ds.recordings[i] for i in th_idx], ds.mains_hz, cfg, clean=[clean[i] for i in th_idx]
        )
        t = featurize(ds.recordings, ds.mains_hz, cfg, th, clean=clean)
        tr = t.where(subjects=others)
        cal = t.where(
            subjects=[held], sessions=[0], reps=[0], t_min=1.0, t_max=1.0 + cb.CROSS_PERSON_CAL_S
        )
        folds[held] = (
            LDAClassifier().fit(tr.X, tr.label),
            cal,
            t.where(subjects=[held], sessions=[1]),
        )
    setting = best_setting(
        {
            s: [
                float(np.mean(cb._adapted(b, c, s).predict(te.X) == te.label))
                for b, c, te in (folds[h] for h in val)
            ]
            for s in ADAPT_GRID
        }
    )
    models = {h: cb._adapted(b, c, setting) for h, (b, c, _) in folds.items()}
    scores = {h: cb.Scored.of(models[h], folds[h][2]) for h in everyone}
    temp = cb._fit_temp([scores[h] for h in val])
    groups = {}
    for h in val + report:
        pool = [scores[o] for o in everyone if o != h]
        sets_th = {a: cb._thresholds(pool, a, temp, "aps", False, SEED) for a in ALPHAS}
        rng = np.random.default_rng([SEED, h])
        groups[h] = streams_of(models[h], folds[h][2], temp, sets_th, rng)
    info = {
        "temperature": temp,
        "adapt_setting": asdict(setting),
        "calibration_seconds": cb.CROSS_PERSON_CAL_S,
        "validation_persons": val,
        "report_persons": report,
    }
    return {h: groups[h] for h in val}, {h: groups[h] for h in report}, info


# ----------------------------------------------------------------- run


def run(opts: RunOptions) -> dict[str, Any]:
    cfg = FeatureConfig()
    configs = quick_grid() if opts.quick else grid()
    _, tables = mdb.build_tables(opts, cfg)
    with timed("MultiDay streams"):
        md_val, md_test, md_info, md_restream = multiday_groups(tables[cfg.feature_set])
    del tables
    with timed("MultiDay decisions"):
        md = compare(md_val, md_test, configs)
    md_pick = headline_choices(md, configs)
    md["streams"] = md_info
    with timed("MultiDay playback orders"):
        seeds = ORDER_SEEDS[:2] if opts.quick else ORDER_SEEDS
        md["playback_orders"] = order_spread(md_restream, md_pick, seeds)
    del md_restream

    with timed("load 3DC"):
        subjects = [1, 7] if opts.quick else None  # quick: one to choose, one to report
        ds = datasets.load("3dc", root=opts.data_root, download=opts.download, subjects=subjects)
    val = [s for s in ds.subjects if s in qb.VAL_SUBJECTS]
    report = [s for s in ds.subjects if s in qb.REPORT_SUBJECTS]
    from .threedc_bench import Prepared

    prep = Prepared(ds, cfg)
    with timed("3DC within sitting"):
        per_person = {
            s: prep.table(prep.indices([s]), prep.indices([s], [0])) for s in val + report
        }
        w_val, w_rep, w_info = within_groups(per_person, val, report)
        del per_person
        within = compare(w_val, w_rep, configs, fixed=md_pick)
        within["streams"] = w_info
    with timed("3DC cross person"):
        c_val, c_rep, c_info = cross_groups(ds, prep.clean, cfg, val, report)
        cross = compare(c_val, c_rep, configs, fixed=md_pick)
        cross["streams"] = c_info

    payload = {
        "provenance": provenance(
            None,
            {
                "features": asdict(cfg),
                "seed": SEED,
                "alphas": ALPHAS,
                "score_from_s": SCORE_FROM_S,
                "baseline": BASELINE,
                "objective": OBJECTIVE,
                "responsiveness": [SPEED, COVER],
                "grid": {
                    "vote_k": VOTE_KS,
                    "confidence_k": CONF_KS,
                    "confidence_cutoff": CONF_CUTOFFS,
                    "set_n_m": SET_NM,
                    "evidence_memory": EV_MEMORY,
                    "evidence_threshold": EV_THRESHOLD,
                    "hold_modes": HOLD_MODES,
                },
                "curve_vote_k": CURVE_KS,
            },
        ),
        "datasets": {
            name: {"url": src.url, "commit": src.commit, "citation": src.citation}
            for name, src in (("multiday", multiday_data.SOURCE), ("3dc", threedc.SOURCE))
        },
        "multiday": md,
        "threedc": {"within_sitting": within, "cross_person": cross},
    }
    print(f"  wrote {_write(opts, 'decision', payload)}")
    return payload
