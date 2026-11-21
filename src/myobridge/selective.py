"""Act only when sure: "when it names a gesture, it's right 97% of the time".

A gesture decoder can hold back on the windows it is least sure of and act
on the rest. That is the classic reject option (Chow 1970; for EMG, Scheme,
Hudgins & Englehart 2013). The question is where to put the cut-off. Picked by
eye, or as "the loosest cut-off whose accepted windows were right 97% of the
time on held-out data", it gives no promise: the held-out accuracy at that
cut-off is an estimate, and about half the time the truth is below it.

`SelectiveGestures` picks the cut-off with Learn then Test (LTT; Angelopoulos,
Bates, Candès, Jordan & Lei 2021, "Learn then Test: Calibrating predictive
algorithms to achieve risk control"; the selective-classification example is
in Angelopoulos & Bates 2021, "A gentle introduction to conformal prediction
and distribution-free uncertainty quantification", section 5). For each
cut-off on a grid fixed in advance, from strict to loose, it tests the null
hypothesis "answers accepted at this cut-off are right less than `target` of
the time". Given that `n` held-out windows are accepted, their errors are
binomial, so the exact p-value is the binomial tail
`P(Binomial(n, 1 - target) <= errors)` (the test behind the Clopper-Pearson
bound). The hypotheses are tested in order with a procedure that controls the
family-wise error rate at `delta`, and the loosest rejected cut-off is used.
If none is rejected, it never answers.

The promise: if the held-out windows and the new ones are independent draws
from the same distribution, then with probability at least 1 - delta over the
held-out draw, the chosen cut-off's answers are right at least `target` of the
time. Nothing about monotonicity is assumed.

Three procedures order the tests:

- "fixed_sequence": test strict to loose at level `delta`, stop at the first
  failure. The most powerful, but it never answers if the strictest cut-off
  accepts too few held-out windows to pass (with 0 errors, a 97% promise
  needs 76 windows at delta = 0.1). That happens on `default_grid()` once a
  temperature has softened the probabilities; a grid from `grid_from` on
  other held-out windows avoids it.
- "fallback" (the default): the fallback procedure (Wiens 2003; a graphical
  procedure in the sense of Bretz et al. 2009). Each cut-off gets
  `delta / m` of its own, plus the level of the one before it if that one
  was rejected, and a failure doesn't stop the sequence.
- "bonferroni": every cut-off at `delta / m`.

The grid must be fixed without looking at the windows it is tested on:
`default_grid()` is fixed in advance, and `grid_from` builds one from other
held-out windows (in the benchmark, the ones the temperature was fitted on).

Scores (all name the most probable gesture when they answer):

- "max_prob": the top probability after temperature scaling.
- "margin": the top probability minus the second.
- "aps_singleton": answer when the randomized APS gesture set
  (`myobridge.conformal`) at threshold `cutoff` holds exactly one gesture.

Like the sets, the promise assumes new windows look like the held-out ones,
and EMG breaks that every time the armband is put back on. Across days it
has to be measured, which is what `myobridge bench selective` does with the
same recipe as the sets: `adapt()` each day, the temperature from held-out
adapted days, and the cut-off from other held-out adapted days, kept as is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import beta, binom

from .conformal import ScoringModel, _label_index, _streams, nonconformity, softmax

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
SCORES = ("max_prob", "margin", "aps_singleton")
METHODS = ("fixed_sequence", "fallback", "bonferroni")


# ------------------------------------------------------------ building blocks


def default_grid(per_decade: int = 20, strictest: float = 1e-12) -> FloatArray:
    """Cut-offs from strict to loose, fixed before seeing any data.

    `1 - cutoff` runs from `strictest` to 1 in `per_decade` steps per factor
    of ten, so the grid is fine where LDA probabilities pile up near 1. The
    grid must not depend on the windows used to test it; build it from other
    data, or use this one.
    """
    if per_decade < 1 or not 0.0 < strictest < 1.0:
        raise ValueError("per_decade must be >= 1 and strictest in (0, 1)")
    n = int(round(-np.log10(strictest) * per_decade)) + 1
    out: FloatArray = 1.0 - np.geomspace(strictest, 1.0, n)
    return out


def grid_from(
    proba: ArrayLike, score: str = "max_prob", rates: ArrayLike | None = None
) -> FloatArray:
    """Cut-offs that answer `rates` of these windows, strict to loose.

    Use windows that are *not* the ones you will pick the cut-off on, or the
    promise is lost. `rates` defaults to 1%, 2%, ..., 100%. Repeated values
    (ties at a probability of 1) are kept once. Not for "aps_singleton",
    whose answer rate doesn't fall steadily as its threshold rises.
    """
    if score not in ("max_prob", "margin"):
        raise ValueError("grid_from needs score 'max_prob' or 'margin'")
    p = np.atleast_2d(np.asarray(proba, dtype=np.float64))
    if len(p) == 0:
        raise ValueError("no windows")
    r = np.arange(1, 101) / 100 if rates is None else np.asarray(rates, dtype=np.float64)
    if np.any((r <= 0) | (r > 1)):
        raise ValueError("rates must be in (0, 1]")
    ranked = np.sort(p, axis=1)
    value = ranked[:, -1] if score == "max_prob" else ranked[:, -1] - ranked[:, -2]
    cuts = np.quantile(value, 1.0 - r, method="higher")
    out: FloatArray = np.unique(cuts)[::-1].copy()
    return out


def accept_matrix(
    proba: ArrayLike,
    cutoffs: ArrayLike,
    score: str = "max_prob",
    rng: np.random.Generator | None = None,
) -> BoolArray:
    """(n, m) booleans: would row i be answered at cut-off j?

    The answer, when there is one, is always the most probable gesture.
    """
    p = np.atleast_2d(np.asarray(proba, dtype=np.float64))
    cut = np.asarray(cutoffs, dtype=np.float64).ravel()
    if score not in SCORES:
        raise ValueError(f"score must be one of {SCORES}")
    if score == "aps_singleton":
        s = np.sort(nonconformity(p, "aps", rng), axis=1)
        lo, hi = s[:, :1], s[:, 1:2] if p.shape[1] > 1 else np.full((len(p), 1), np.inf)
        out: BoolArray = (lo <= cut[None, :]) & (cut[None, :] < hi)
        return out
    ranked = np.sort(p, axis=1)
    value = ranked[:, -1] if score == "max_prob" else ranked[:, -1] - ranked[:, -2]
    accepted: BoolArray = value[:, None] >= cut[None, :]
    return accepted


def binomial_pvalues(n: ArrayLike, errors: ArrayLike, risk: float) -> FloatArray:
    """Exact p-values for "the error rate among accepted windows exceeds `risk`".

    `P(Binomial(n, risk) <= errors)`; 1 where nothing was accepted.
    """
    na = np.asarray(n, dtype=np.int64)
    ea = np.asarray(errors, dtype=np.int64)
    out: FloatArray = np.where(na > 0, binom.cdf(ea, np.maximum(na, 1), risk), 1.0)
    return out


def learn_then_test(pvalues: ArrayLike, delta: float, method: str = "fallback") -> int | None:
    """Index of the loosest cut-off rejected with family-wise error `delta`.

    `pvalues` are ordered from the strictest cut-off to the loosest. Returns
    None when no cut-off can be certified.
    """
    p = np.asarray(pvalues, dtype=np.float64).ravel()
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    m = len(p)
    if m == 0:
        return None
    if method == "bonferroni":
        ok = np.nonzero(p <= delta / m)[0]
        return int(ok[-1]) if len(ok) else None
    if method == "fixed_sequence":
        chosen = None
        for j in range(m):
            if p[j] > delta:
                break
            chosen = j
        return chosen
    w = delta / m
    level, chosen, carried = w, None, 0.0
    for j in range(m):
        level = w + carried
        if p[j] <= level:
            chosen, carried = j, level
        else:
            carried = 0.0
    return chosen


def clopper_pearson_lower(correct: int, n: int, delta: float) -> float:
    """One-sided 1 - delta lower bound on a success rate (Clopper & Pearson 1934)."""
    if n == 0 or correct == 0:
        return 0.0
    return float(beta.ppf(delta, correct, n - correct + 1))


# --------------------------------------------------------------- the wrapper


@dataclass
class SelectiveGestures:
    """Name a gesture only when sure enough, with an accuracy promise.

    Build it with `calibrate` (labelled windows the model did not train on)
    or `from_scores`. The cut-off only depends on probabilities, so after
    `adapt()` you can keep it on the new model with `with_model`.

    Args:
        model: The classifier whose `decision_function` is used.
        cutoff: Answer when the score is at least this (for "aps_singleton",
            the APS threshold); None means never answer.
        target: The promised accuracy when it answers.
        delta: The chance the promise fails because of an unlucky held-out draw.
        temperature: Divides the decision scores before the softmax.
        score: "max_prob", "margin" or "aps_singleton".
        method: The testing procedure used to pick the cut-off.
        answer_rate: The share of held-out windows answered at the cut-off.
        calibration_accuracy: Their accuracy.
        accuracy_bound: A Clopper-Pearson 1 - delta lower bound on it, for
            that one cut-off (not a promise for the selected rule).
        n_calibration: How many held-out windows were used.
        seed: Seeds the APS random share, so results repeat.
    """

    model: ScoringModel
    cutoff: float | None
    target: float
    delta: float = 0.1
    temperature: float = 1.0
    score: str = "max_prob"
    method: str = "fallback"
    answer_rate: float = float("nan")
    calibration_accuracy: float = float("nan")
    accuracy_bound: float = float("nan")
    n_calibration: int = 0
    seed: int | None = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}")
        if not 0.0 < self.target < 1.0 or not 0.0 < self.delta < 1.0:
            raise ValueError("target and delta must be in (0, 1)")
        if self.cutoff is not None and not np.isfinite(self.cutoff):
            self.cutoff = None
        self._rng = _streams(self.seed)[1]

    @classmethod
    def from_scores(
        cls,
        model: ScoringModel,
        decision: ArrayLike,
        y_index: ArrayLike,
        target: float = 0.97,
        delta: float = 0.1,
        temperature: float = 1.0,
        score: str = "max_prob",
        method: str = "fallback",
        grid: ArrayLike | None = None,
        seed: int | None = 0,
    ) -> SelectiveGestures:
        """Pick the cut-off from decision scores you already have.

        `grid` lists candidate cut-offs from strict to loose and must be fixed
        without looking at these windows (default: `default_grid()`).
        """
        if not 0.0 < target < 1.0:
            raise ValueError("target must be in (0, 1)")
        y = np.asarray(y_index, dtype=np.intp)
        p = softmax(decision, temperature)
        if len(p) != len(y):
            raise ValueError("decision and y_index have different lengths")
        cuts = default_grid() if grid is None else np.asarray(grid, dtype=np.float64).ravel()
        acc = accept_matrix(p, cuts, score, _streams(seed)[0])
        wrong = p.argmax(axis=1) != y
        n = acc.sum(axis=0)
        errors = (acc & wrong[:, None]).sum(axis=0)
        j = learn_then_test(binomial_pvalues(n, errors, 1.0 - target), delta, method)
        common: dict[str, Any] = {
            "target": target,
            "delta": delta,
            "temperature": temperature,
            "score": score,
            "method": method,
            "n_calibration": len(y),
            "seed": seed,
        }
        if j is None:
            return cls(model, None, answer_rate=0.0, **common)
        nj, ej = int(n[j]), int(errors[j])
        return cls(
            model,
            float(cuts[j]),
            answer_rate=nj / len(y),
            calibration_accuracy=(nj - ej) / nj,
            accuracy_bound=clopper_pearson_lower(nj - ej, nj, delta),
            **common,
        )

    @classmethod
    def calibrate(
        cls,
        model: ScoringModel,
        X: ArrayLike,
        y: ArrayLike,
        target: float = 0.97,
        delta: float = 0.1,
        temperature: float = 1.0,
        score: str = "max_prob",
        method: str = "fallback",
        grid: ArrayLike | None = None,
        seed: int | None = 0,
    ) -> SelectiveGestures:
        """Pick the cut-off on labelled feature rows the model was *not* trained on."""
        y_index = _label_index(model, y)
        return cls.from_scores(
            model,
            model.decision_function(X),
            y_index,
            target,
            delta,
            temperature,
            score,
            method,
            grid,
            seed,
        )

    def with_model(self, model: ScoringModel) -> SelectiveGestures:
        """The same cut-off on another model with the same classes (e.g. after adapt)."""
        if not np.array_equal(np.asarray(model.classes), np.asarray(self.model.classes)):
            raise ValueError("the new model has different classes")
        d = self.to_dict()
        return SelectiveGestures._from_dict(d, model)

    @property
    def classes(self) -> NDArray[Any]:
        return np.asarray(self.model.classes)

    @property
    def abstains(self) -> bool:
        """True when no cut-off could be certified, so it never answers."""
        return self.cutoff is None

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """The model's probabilities after temperature scaling."""
        return softmax(self.model.decision_function(X), self.temperature)

    def accept(self, X: ArrayLike) -> BoolArray:
        """(n,) booleans: is each row sure enough to answer?"""
        p = self.predict_proba(X)
        if self.cutoff is None:
            return np.zeros(len(p), dtype=bool)
        out: BoolArray = accept_matrix(p, [self.cutoff], self.score, self._rng)[:, 0]
        return out

    def predict(self, X: ArrayLike, unsure: Any = None) -> list[Any]:
        """The most probable gesture when sure enough, else `unsure`."""
        p = self.predict_proba(X)
        if self.cutoff is None:
            return [unsure] * len(p)
        ok = accept_matrix(p, [self.cutoff], self.score, self._rng)[:, 0]
        best = p.argmax(axis=1)
        return [self.classes[b] if a else unsure for b, a in zip(best, ok, strict=True)]

    # ---------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        """Everything but the model, as JSON-friendly values."""

        def num(v: float) -> float | None:
            return float(v) if np.isfinite(v) else None

        return {
            "classes": [str(c) for c in self.classes],
            "cutoff": self.cutoff,
            "target": self.target,
            "delta": self.delta,
            "temperature": self.temperature,
            "score": self.score,
            "method": self.method,
            "answer_rate": num(self.answer_rate),
            "calibration_accuracy": num(self.calibration_accuracy),
            "accuracy_bound": num(self.accuracy_bound),
            "n_calibration": self.n_calibration,
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> Path:
        """Write the cut-off (not the model) to a JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path, model: ScoringModel) -> SelectiveGestures:
        """Read a cut-off written by `save` and attach it to `model`."""
        return cls._from_dict(json.loads(Path(path).read_text()), model)

    @classmethod
    def _from_dict(cls, d: dict[str, Any], model: ScoringModel) -> SelectiveGestures:
        if d["classes"] != [str(c) for c in np.asarray(model.classes)]:
            raise ValueError("the cut-off was made for a model with other classes")

        def num(v: float | None) -> float:
            return float("nan") if v is None else float(v)

        return cls(
            model,
            d["cutoff"],
            d["target"],
            d["delta"],
            d["temperature"],
            d["score"],
            d["method"],
            num(d["answer_rate"]),
            num(d["calibration_accuracy"]),
            num(d["accuracy_bound"]),
            int(d["n_calibration"]),
            d["seed"],
        )
