"""Does "when it answers, it's right 97% of the time" stay true on later days?

A reject option names the most probable gesture only when a confidence score
clears a cut-off. `myobridge.selective` picks the cut-off with Learn then
Test, which promises the accuracy of the answers (with probability 1 - delta)
when new windows are exchangeable with the held-out ones. Days are not, so
this benchmark measures the promise the way `bench conformal` measures the
sets, and compares it with the naive cut-off.

Rules for the cut-off, all on the same grid of cut-offs (`default_grid`):

    naive           the loosest cut-off whose held-out answers were right
                    `target` of the time; no promise.
    naive_exact     the same over every held-out score value instead of the
                    grid (max_prob only); conformal_bench's _confidence_cutoff.
    fixed_sequence  LTT, binomial p-values, tested strict to loose, stop at
    fallback        the first failure / fallback procedure / Bonferroni.
    bonferroni

Scores: max_prob, margin, aps_singleton (see `myobridge.selective`).

MultiDay (one person; base model days 1-10; temperature days 11-20; cut-off
days 21-30; reported days 31-121), with and without daily adapt() (1 s per
gesture from [1.0, 2.0) s, the setting chosen as conformal_bench does), and
with and without the temperature. The testing procedure is chosen on
validation days alone: temperature on days 11-15, cut-off on 16-20, compared
on 21-30.

3DC (22 people; persons 1-6 choose, 7-22 reported):

    within_sitting  per person, cut-off from the blocked out-of-fold scores
                    of session 0; tested on session 1 after the break.
    cross_person    leave one person out; cut-off from the other people;
                    with no calibration and with 2 s per gesture of adapt().
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from .. import datasets
from ..classifier import LDAClassifier
from ..conformal import softmax
from ..datasets import multiday as multiday_data
from ..datasets import threedc
from ..datasets.core import Dataset
from ..evaluation import FeatureConfig, FeatureTable, featurize, fit_thresholds, summarize
from ..selective import (
    METHODS,
    SCORES,
    accept_matrix,
    binomial_pvalues,
    default_grid,
    grid_from,
    learn_then_test,
)
from . import conformal_bench as cb
from . import multiday_bench as mdb
from . import quality_bench as qb
from .common import ADAPT_GRID, RunOptions, best_setting, provenance, timed, write_result
from .conformal_bench import Scored, _adapted, _fit_temp, _window

TARGETS = (0.95, 0.97)
DELTA = 0.1
SLACK = 0.02  # "within 2 points of the target"
SEED = 0
GRID = default_grid()
HELD_OUT = "_held_out_grid"  # suffix: the grid comes from the temperature data (grid_from)
LTT_RULES = (*METHODS, *(m + HELD_OUT for m in ("fixed_sequence", "fallback")))
RULES = ("naive", "naive_exact", *LTT_RULES)
GRID_SCORES = ("max_prob", "margin")  # the scores grid_from can make a grid for
DEFAULT_METHOD = "fallback"  # used when there are too few validation days to choose


def _rule_for(score: str, rule: str) -> str:
    """aps_singleton has no held-out grid; it uses the fallback procedure on the
    fixed one instead, since fixed-sequence testing there never answers once a
    temperature is used (the strictest cut-offs accept no windows)."""
    if score in GRID_SCORES or not rule.endswith(HELD_OUT):
        return rule
    return "fallback"


# ------------------------------------------------------------- the rules


class _Counts:
    """Per group: windows answered and wrong ones at every grid cut-off.

    A pool's counts are the sums of its groups', so a leave-one-out pool
    costs nothing, and a group's own counts give its result at any grid
    cut-off. APS draws its random share once per window, in group order.
    """

    def __init__(
        self, scored: Mapping[Any, Scored], temp: float, score: str, grid: np.ndarray = GRID
    ):
        self.grid = grid
        rng = np.random.default_rng(SEED)
        self.n: dict[Any, np.ndarray] = {}
        self.errors: dict[Any, np.ndarray] = {}
        self.total: dict[Any, int] = {}
        for key, g in scored.items():
            p = softmax(g.decision, temp)
            acc = accept_matrix(p, grid, score, rng)
            wrong = p.argmax(axis=1) != g.y
            self.n[key] = acc.sum(axis=0)
            self.errors[key] = (acc & wrong[:, None]).sum(axis=0)
            self.total[key] = len(p)

    def pooled(self, keys: Sequence[Any]) -> tuple[np.ndarray, np.ndarray, int]:
        n = np.sum([self.n[k] for k in keys], axis=0)
        e = np.sum([self.errors[k] for k in keys], axis=0)
        return n, e, sum(self.total[k] for k in keys)


def _grid_index(n: np.ndarray, errors: np.ndarray, target: float, rule: str) -> int | None:
    """Which grid cut-off a rule picks from pooled counts (None: never answer)."""
    if rule == "naive":
        ok = np.nonzero((n > 0) & (n - errors >= target * n - 1e-9))[0]
        return int(ok[-1]) if len(ok) else None
    method = rule.removesuffix(HELD_OUT)
    return learn_then_test(binomial_pvalues(n, errors, 1.0 - target), DELTA, method)


def _summary(per: dict[str, tuple[int, int, int]], target: float) -> dict[str, Any]:
    """From (answered, wrong, windows) per group: accuracy when answering and answer rate."""
    acc = [(n - e) / n for n, e, _ in per.values() if n]
    rate = [n / tot for n, _, tot in per.values()]
    n_named = sum(n for n, _, _ in per.values())
    n_wrong = sum(e for _, e, _ in per.values())
    arr = np.array(acc)
    return {
        "accuracy_when_answering": summarize(acc) if acc else None,
        "pooled_accuracy_when_answering": (n_named - n_wrong) / n_named if n_named else None,
        "answer_rate": summarize(rate),
        "pooled_answer_rate": n_named / sum(tot for _, _, tot in per.values()),
        "share_meeting_target": float(np.mean(arr >= target - 1e-12)) if acc else None,
        "share_within_2_points": float(np.mean(arr >= target - SLACK - 1e-12)) if acc else None,
        "worst_group": float(arr.min()) if acc else None,
        "groups_with_no_answer": int(len(rate) - len(acc)),
        "per_group": {
            k: {"accuracy_when_answering": (n - e) / n if n else None, "answer_rate": n / tot}
            for k, (n, e, tot) in per.items()
        },
    }


def _exact(scored: Mapping[Any, Scored], keys: Sequence[Any], temp: float, cut: float) -> Any:
    """(answered, wrong, windows) for a max-prob cut-off off the grid."""
    out = {}
    for k in keys:
        p = softmax(scored[k].decision, temp)
        named = p.max(axis=1) >= cut
        wrong = p.argmax(axis=1) != scored[k].y
        out[str(k)] = (int(named.sum()), int((named & wrong).sum()), len(p))
    return out


def _grid_rows(
    scored: Mapping[Any, Scored],
    groups: Sequence[Any],
    pools: Sequence[Any] | Mapping[Any, Sequence[Any]],
    temp: float,
    held_out: Sequence[Any] | None,
    rules: Sequence[str] = RULES,
) -> dict[str, Any]:
    """Every score x target x rule on `groups`, with the cut-off from one pool
    of groups for all of them, or (a mapping) from a pool per group.

    `held_out` names the groups the "_held_out_grid" rules build their grid
    from; they must not be in any pool (None: skip those rules)."""
    per_pool = isinstance(pools, Mapping)
    pool_of = {g: (pools[g] if isinstance(pools, Mapping) else pools) for g in groups}
    if held_out is not None:
        assert not set(held_out) & {k for g in groups for k in pool_of[g]}
    out: dict[str, Any] = {"temperature": temp}
    for score in SCORES:
        fixed = _Counts(scored, temp, score)
        grown = None
        if held_out is not None and score in GRID_SCORES:
            proba = np.vstack([softmax(scored[k].decision, temp) for k in held_out])
            grown = _Counts(scored, temp, score, grid_from(proba, score))
        out[score] = {}
        for target in TARGETS:
            res: dict[str, Any] = {}
            for rule in rules:
                counts = grown if rule.endswith(HELD_OUT) else fixed
                if counts is None:
                    continue
                if rule == "naive_exact":
                    if score != "max_prob":
                        continue
                    per: dict[str, tuple[int, int, int]] = {}
                    cuts: dict[Any, float | None] = {}
                    for g in groups:
                        c = cb._confidence_cutoff([scored[k] for k in pool_of[g]], temp, target)
                        cuts[g] = float(c) if np.isfinite(c) else None
                        per |= (
                            _exact(scored, [g], temp, c)
                            if np.isfinite(c)
                            else {str(g): (0, 0, counts.total[g])}
                        )
                    r = _summary(per, target)
                    if not per_pool:
                        r["cutoff"] = cuts[groups[0]] if groups else None
                    res[rule] = r
                    continue
                per, picked, cal_rate, cal_acc = {}, {}, [], []
                for g in groups:
                    n, e, tot = counts.pooled(pool_of[g])
                    j = _grid_index(n, e, target, rule)
                    picked[g] = j
                    if j is None:
                        per[str(g)] = (0, 0, counts.total[g])
                        cal_rate.append(0.0)
                        continue
                    per[str(g)] = (int(counts.n[g][j]), int(counts.errors[g][j]), counts.total[g])
                    cal_rate.append(float(n[j] / tot))
                    cal_acc.append(float((n[j] - e[j]) / n[j]))
                r = _summary(per, target)
                r["calibration_answer_rate"] = float(np.mean(cal_rate)) if cal_rate else None
                r["calibration_accuracy"] = float(np.mean(cal_acc)) if cal_acc else None
                if per_pool:
                    r["share_of_groups_never_answering"] = float(
                        np.mean([j is None for j in picked.values()])
                    )
                else:
                    j0 = picked[groups[0]] if groups else None
                    r["cutoff"] = None if j0 is None else float(counts.grid[j0])
                r["grid_size"] = len(counts.grid)
                res[rule] = r
            out[score][str(target)] = res
    return out


def _choose_method(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The LTT rule with the highest mean answer rate on the comparison groups,
    over max_prob and margin (the scores every rule applies to) and both targets.
    All of them keep the promise on exchangeable data, so the most answers wins."""
    candidates = [m for m in LTT_RULES if all(m in r["max_prob"][str(TARGETS[0])] for r in rows)]
    rate: dict[str, list[float]] = {m: [] for m in candidates}
    for r in rows:
        for score in GRID_SCORES:
            for target in TARGETS:
                for m in candidates:
                    rate[m].append(r[score][str(target)][m]["answer_rate"]["mean"])
    means = {m: float(np.mean(v)) for m, v in rate.items()}
    return {"mean_answer_rate": means, "chosen": max(candidates, key=lambda m: means[m])}


def _score_ranking(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    """Per target: each score's answer rate and accuracy, and the highest answer rate."""
    out: dict[str, Any] = {}
    for target in TARGETS:
        by = {
            s: {
                "rule": _rule_for(s, method),
                "answer_rate": row[s][str(target)][_rule_for(s, method)]["answer_rate"]["mean"],
                "accuracy_when_answering": (
                    row[s][str(target)][_rule_for(s, method)]["accuracy_when_answering"] or {}
                ).get("mean"),
            }
            for s in SCORES
        }
        out[str(target)] = by | {
            "highest_answer_rate": max(SCORES, key=lambda s: by[s]["answer_rate"])
        }
    return out


# ------------------------------------------------------------------- MultiDay


def multiday(t: FeatureTable) -> dict[str, Any]:
    days = sorted(set(t.session.tolist()))
    train = [d for d in days if d in mdb.TRAIN_DAYS]
    val = [d for d in days if d in mdb.VAL_DAYS]
    test = [d for d in days if d in mdb.TEST_DAYS]
    half = len(val) // 2 if not set(cb.TEMP_DAYS) <= set(val) else len(cb.TEMP_DAYS)
    temp_days, cut_days = val[:half], val[half:]
    base_t = t.where(sessions=train)
    base = LDAClassifier().fit(base_t.X, base_t.label)

    def cal(d: int) -> FeatureTable:
        return _window(t, d, mdb.CAL_START_S, mdb.CAL_START_S + cb.ADAPT_SECONDS)

    def val_accuracy(s: Any) -> list[float]:
        out = []
        for d in val:
            te = mdb._test(t, d)
            out.append(float(np.mean(_adapted(base, cal(d), s).predict(te.X) == te.label)))
        return out

    setting = best_setting({s: val_accuracy(s) for s in ADAPT_GRID})
    scored = {
        "static": {d: Scored.of(base, mdb._test(t, d)) for d in val + test},
        "adapt": {
            d: Scored.of(_adapted(base, cal(d), setting), mdb._test(t, d)) for d in val + test
        },
    }

    # choose the testing procedure on validation days alone
    q = max(len(val) // 4, 1)
    sel_temp, sel_cut, sel_eval = val[:q], val[q : 2 * q], val[2 * q :]
    selection: dict[str, Any]
    if sel_temp and sel_cut and sel_eval:
        sel_rows = {}
        for name, sc in scored.items():
            for tname, temp in (
                ("no_temperature", 1.0),
                ("temperature", _fit_temp([sc[d] for d in sel_temp])),
            ):
                sel_rows[f"{name}_{tname}"] = _grid_rows(
                    {d: sc[d] for d in sel_temp + sel_cut + sel_eval},
                    sel_eval,
                    sel_cut,
                    temp,
                    sel_temp,
                    LTT_RULES,
                )
        choice = _choose_method([sel_rows["adapt_no_temperature"], sel_rows["adapt_temperature"]])
        selection = {
            "temperature_days": sel_temp,
            "cutoff_days": sel_cut,
            "comparison_days": sel_eval,
            "method": choice,
            "scores": _score_ranking(sel_rows["adapt_temperature"], choice["chosen"]),
            "rows": sel_rows,
        }
        method = choice["chosen"]
    else:
        selection = {"note": "too few validation days; default procedure used"}
        method = DEFAULT_METHOD

    rows: dict[str, Any] = {}
    temps: dict[str, float] = {}
    for name, sc in scored.items():
        temps[name] = _fit_temp([sc[d] for d in temp_days])
        for tname, temp in (("no_temperature", 1.0), ("temperature", temps[name])):
            rows[f"{name}_{tname}"] = _grid_rows(
                {d: sc[d] for d in temp_days + cut_days + test}, test, cut_days, temp, temp_days
            )
    accuracy = {
        name: float(np.mean([np.mean(sc[d].decision.argmax(1) == sc[d].y) for d in test]))
        for name, sc in scored.items()
    }
    return {
        "adapt_setting": asdict(setting),
        "accuracy": accuracy,
        "temperatures": temps,
        "temperature_days": temp_days,
        "cutoff_days": cut_days,
        "test_days": test,
        "method": method,
        "selection_check": selection,
        "scores_on_test_days": _score_ranking(rows["adapt_temperature"], method),
        "rows": rows,
    }


# ------------------------------------------------------------------------ 3DC


def within_sitting(
    tables: dict[int, FeatureTable], val: list[int], report: list[int]
) -> dict[str, Any]:
    oof: dict[int, Scored] = {}
    test: dict[int, Scored] = {}
    for s in val + report:
        t = tables[s]
        sc, model = cb._person_blocks(t.where(sessions=[0]))
        oof[s] = sc
        test[s] = Scored.of(model, t.where(sessions=[1]))
    temp = _fit_temp([oof[s] for s in val])
    # one table of scores: session 1 under the person's number, session 0's
    # out-of-fold scores (the cut-off pool) under "oof-<number>"
    both: dict[Any, Scored] = {s: test[s] for s in val + report}
    both |= {f"oof-{s}": oof[s] for s in val + report}
    # the procedure is chosen on persons 4-6, with the held-out grid from 1-3
    half = max(len(val) // 2, 1)
    choose_on = val[half:] or val
    grid_on = [f"oof-{s}" for s in val[:half]] if val[half:] else None
    choice = _choose_method(
        [
            _grid_rows(
                both, choose_on, {s: [f"oof-{s}"] for s in choose_on}, tp, grid_on, LTT_RULES
            )
            for tp in (1.0, temp)
        ]
    )
    rep = {s: test[s] for s in report}
    pools = {s: [f"oof-{s}"] for s in report}
    grid_src = [f"oof-{s}" for s in val]
    rows = {
        "no_temperature": _grid_rows(both, report, pools, 1.0, grid_src),
        "temperature": _grid_rows(both, report, pools, temp, grid_src),
    }
    return {
        "temperature": temp,
        "procedure_chosen_on": choose_on,
        "held_out_grid_for_choosing_from": grid_on,
        "accuracy": float(np.mean([np.mean(g.decision.argmax(1) == g.y) for g in rep.values()])),
        "method_choice": choice,
        "method": choice["chosen"],
        "scores": _score_ranking(rows["temperature"], choice["chosen"]),
        "rows": rows,
    }


def cross_person(
    ds: Dataset, clean: list[np.ndarray], cfg: FeatureConfig, val: list[int], report: list[int]
) -> dict[str, Any]:
    """Leave one person out, as conformal_bench.cross_person: the cut-off for
    person p comes from the other people's scores, with and without 2 s per
    gesture of p's data through adapt()."""
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
        base = LDAClassifier().fit(tr.X, tr.label)
        cal = t.where(
            subjects=[held], sessions=[0], reps=[0], t_min=1.0, t_max=1.0 + cb.CROSS_PERSON_CAL_S
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
    scored = {
        "no_calibration": {h: Scored.of(b, te) for h, (b, _, te) in folds.items()},
        "adapt": {h: Scored.of(_adapted(b, c, setting), te) for h, (b, c, te) in folds.items()},
    }
    out: dict[str, Any] = {
        "calibration_seconds": cb.CROSS_PERSON_CAL_S,
        "adapt_setting": asdict(setting),
    }
    for name, sc in scored.items():
        temp = _fit_temp([sc[h] for h in val])
        rest = {h: [o for o in everyone if o != h] for h in everyone}
        choice = _choose_method(
            [
                _grid_rows(sc, val, {h: rest[h] for h in val}, tp, None, LTT_RULES)
                for tp in (1.0, temp)
            ]
        )
        rep = {h: sc[h] for h in report}
        pools = {h: rest[h] for h in report}
        rows = {
            # every other person is in some pool, so there is no held-out grid
            "no_temperature": _grid_rows(sc, report, pools, 1.0, None),
            "temperature": _grid_rows(sc, report, pools, temp, None),
        }
        out[name] = {
            "accuracy": float(
                np.mean([np.mean(g.decision.argmax(1) == g.y) for g in rep.values()])
            ),
            "temperature": temp,
            "method_choice": choice,
            "method": choice["chosen"],
            "scores": _score_ranking(rows["temperature"], choice["chosen"]),
            "rows": rows,
        }
    return out


# ------------------------------------------------------------------------ run


def run(opts: RunOptions) -> dict[str, Any]:
    cfg = FeatureConfig()
    _, tables = mdb.build_tables(opts, cfg)
    with timed("MultiDay selective"):
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
    payload = {
        "provenance": provenance(
            None,
            {
                "features": asdict(cfg),
                "targets": TARGETS,
                "delta": DELTA,
                "slack": SLACK,
                "seed": SEED,
                "rules": RULES,
                "scores": SCORES,
                "grid": {"n": len(GRID), "strictest": float(GRID[0]), "loosest": float(GRID[-1])},
                "multiday": {
                    "train_days": mdb.TRAIN_DAYS,
                    "temperature_days": cb.TEMP_DAYS,
                    "cutoff_days": cb.THRESH_DAYS,
                    "test_days": mdb.TEST_DAYS,
                    "adapt_seconds": cb.ADAPT_SECONDS,
                    "calibration_start_s": mdb.CAL_START_S,
                    "test_start_s": mdb.TEST_START_S,
                },
                "threedc": {
                    "validation_subjects": val,
                    "report_subjects": report,
                    "cross_person_calibration_s": cb.CROSS_PERSON_CAL_S,
                },
            },
        ),
        "datasets": {
            src_name: {"url": src.url, "commit": src.commit, "citation": src.citation}
            for src_name, src in (("multiday", multiday_data.SOURCE), ("3dc", threedc.SOURCE))
        },
        "multiday": md,
        "threedc": {"within_sitting": within, "cross_person": cross},
    }
    print(f"  wrote {write_result(opts, 'selective', payload)}")
    return payload
