"""Gesture sets with a coverage promise: "one of these, 90% of the time".

A classifier always names one gesture, even when two look alike or the
electrodes have moved. Conformal prediction turns its scores into a *set* of
gestures that contains the true one with a chosen probability, say 90%. When
the model is sure, the set has one gesture; when it is torn, it has two or
three; when nothing fits, it can be empty. An application can act only on
one-gesture sets and treat the rest as "not sure".

The promise holds on new data that looks like the calibration data. EMG breaks
that in three ways (days, positions within a hold, people), which is what
`myobridge bench conformal` measures. The recipe that kept its promise across
three months of daily re-donning on MultiDay was:

1. recalibrate the LDA each day with `adapt()` and about a second per gesture;
2. soften its scores with one temperature fitted on held-out days
   (`fit_temperature`); LDA probabilities from dozens of features are
   nearly always 0 or 1, which leaves conformal nothing to work with;
3. use the randomized APS score and a threshold from other held-out days
   (`ConformalGestures.calibrate`), then keep that threshold on later days.

References: split conformal prediction (Vovk, Gammerman & Shafer 2005; Lei et
al. 2018), APS (Romano, Sesia & Candès 2020), Mondrian (class-conditional)
conformal (Vovk et al. 2003), temperature scaling (Guo et al. 2017) and
adaptive conformal inference (Gibbs & Candès 2021).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
SCORES = ("aps", "lac")


class ScoringModel(Protocol):
    """Anything with LDAClassifier's `decision_function` and `classes`."""

    @property
    def classes(self) -> NDArray[Any]: ...

    def decision_function(self, X: ArrayLike) -> FloatArray: ...


# ------------------------------------------------------------ building blocks


def softmax(decision: ArrayLike, temperature: float = 1.0) -> FloatArray:
    """Class probabilities from unnormalised log-scores divided by `temperature`."""
    if not temperature > 0:
        raise ValueError("temperature must be > 0")
    z = np.atleast_2d(np.asarray(decision, dtype=np.float64)) / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    out: FloatArray = p / p.sum(axis=1, keepdims=True)
    return out


def fit_temperature(
    decision: ArrayLike, y_index: ArrayLike, bounds: tuple[float, float] = (0.05, 3000.0)
) -> float:
    """The temperature that makes `softmax(decision / T)` best match the labels.

    Minimises the negative log-likelihood of the true classes (`y_index` are
    column indices into `decision`). Fit it on data the model was not trained
    on, and not on the data used for `ConformalGestures.calibrate`.
    """
    z = np.asarray(decision, dtype=np.float64)
    y = np.asarray(y_index, dtype=np.intp)
    if z.ndim != 2 or len(z) != len(y) or len(y) == 0:
        raise ValueError("decision must be (n, k) and y_index (n,), with n > 0")
    rows = np.arange(len(y))

    def nll(log_t: float) -> float:
        zt = z / np.exp(log_t)
        zt = zt - zt.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(zt).sum(axis=1))
        return float(np.mean(log_norm - zt[rows, y]))

    lo, hi = np.log(bounds[0]), np.log(bounds[1])
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-4})
    return float(np.exp(res.x))


def temperature_for(model: ScoringModel, X: ArrayLike, y: ArrayLike) -> float:
    """`fit_temperature` on labelled feature rows, for a model with `classes`.

    Use rows the model was not trained on, and different rows from the ones
    you pass to `ConformalGestures.calibrate`.
    """
    return fit_temperature(model.decision_function(X), _label_index(model, y))


def _label_index(model: ScoringModel, y: ArrayLike) -> NDArray[np.intp]:
    ya = np.asarray(y)
    classes = np.asarray(model.classes)
    unknown = set(np.unique(ya).tolist()) - set(classes.tolist())
    if unknown:
        raise ValueError(f"labels the model doesn't know: {unknown}")
    out: NDArray[np.intp] = np.searchsorted(classes, ya).astype(np.intp)
    return out


def nonconformity(
    proba: ArrayLike, score: str = "aps", rng: np.random.Generator | None = None
) -> FloatArray:
    """How badly each class fits each row; small means "this class is plausible".

    - "lac": 1 - p(class). Gives the smallest sets on average, but can return
      empty sets and gives easy rows the same threshold as hard ones.
    - "aps": the probability of every class ranked above this one, plus a
      uniformly random share of this class's own. Sets grow on hard rows and
      shrink on easy ones. The random share is what makes the promise exact;
      pass a seeded `rng` for repeatable results.
    """
    p = np.atleast_2d(np.asarray(proba, dtype=np.float64))
    if score == "lac":
        lac: FloatArray = 1.0 - p
        return lac
    if score != "aps":
        raise ValueError(f"score must be one of {SCORES}")
    rng = rng if rng is not None else np.random.default_rng()
    order = np.argsort(-p, axis=1, kind="stable")
    ranked = np.take_along_axis(p, order, axis=1)
    above_and_self = np.cumsum(ranked, axis=1)
    u = rng.random((len(p), 1))
    out = np.empty_like(p)
    np.put_along_axis(out, order, above_and_self - u * ranked, axis=1)
    return out


def conformal_quantile(scores: ArrayLike, alpha: float) -> float:
    """The split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest score.

    Returns +inf when there are too few scores to promise 1 - alpha (then
    every set holds every class, which is the honest answer).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = len(s)
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float("inf") if k > n else float(s[k - 1])


def blocked_scores(
    fit: Callable[[FloatArray, NDArray[Any]], ScoringModel],
    X: ArrayLike,
    y: ArrayLike,
    blocks: ArrayLike,
) -> tuple[FloatArray, NDArray[np.intp], NDArray[Any]]:
    """Out-of-block decision scores, for calibrating on the data you trained on.

    For each block (a time span, a repetition), fit on the other blocks and
    score this one. Blocks should be contiguous in time: EMG windows next to
    each other are nearly copies, so a random split would be too optimistic.
    Returns (decision scores, true class indices, classes); the final model
    should then be fitted on all the data.
    """
    Xa, ya, ba = np.asarray(X, dtype=np.float64), np.asarray(y), np.asarray(blocks)
    if not len(Xa) == len(ya) == len(ba):
        raise ValueError("X, y and blocks must have the same length")
    labels = np.unique(ba)
    if len(labels) < 2:
        raise ValueError("need at least two blocks")
    classes = np.unique(ya)
    decision = np.empty((len(ya), len(classes)))
    for b in labels:
        held = ba == b
        model = fit(Xa[~held], ya[~held])
        if not np.array_equal(np.asarray(model.classes), classes):
            raise ValueError(f"block {b!r} leaves a class with no training data")
        decision[held] = model.decision_function(Xa[held])
    return decision, np.searchsorted(classes, ya).astype(np.intp), classes


# --------------------------------------------------------------- the wrapper


def _streams(seed: int | None) -> tuple[np.random.Generator, np.random.Generator]:
    """Separate random streams for calibrating and predicting from one seed."""
    a, b = np.random.SeedSequence(seed).spawn(2)
    return np.random.default_rng(a), np.random.default_rng(b)


@dataclass
class ConformalGestures:
    """Turn a gesture model's scores into sets with a coverage promise.

    Build it with `calibrate` (from labelled data the model did not train
    on) or `from_scores`. The thresholds only depend on scores, so after
    `adapt()` gives you a new model you can keep the same calibration with
    `with_model`; that is the recipe `bench conformal` tests across days.

    Args:
        model: The classifier whose `decision_function` is used.
        thresholds: One threshold per class (all equal unless `mondrian`).
        alpha: The promised miss rate; sets contain the truth 1 - alpha of the time.
        temperature: Divides the decision scores before the softmax.
        score: "aps" or "lac" (see `nonconformity`).
        seed: Seeds the APS random share, so results repeat.
    """

    model: ScoringModel
    thresholds: FloatArray
    alpha: float
    temperature: float = 1.0
    score: str = "aps"
    seed: int | None = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.score not in SCORES:
            raise ValueError(f"score must be one of {SCORES}")
        self.thresholds = np.asarray(self.thresholds, dtype=np.float64)
        if self.thresholds.shape != (len(self.model.classes),):
            raise ValueError("need one threshold per class")
        self._rng = _streams(self.seed)[1]

    @classmethod
    def from_scores(
        cls,
        model: ScoringModel,
        decision: ArrayLike,
        y_index: ArrayLike,
        alpha: float = 0.1,
        temperature: float = 1.0,
        score: str = "aps",
        mondrian: bool = False,
        seed: int | None = 0,
    ) -> ConformalGestures:
        """Calibrate from decision scores you already have (e.g. `blocked_scores`).

        `mondrian=True` sets a separate threshold for each gesture, so the
        promise holds for every gesture on its own, not just on average.
        """
        rng = _streams(seed)[0]
        y = np.asarray(y_index, dtype=np.intp)
        s = nonconformity(softmax(decision, temperature), score, rng)
        true = s[np.arange(len(y)), y]
        k = len(model.classes)
        if mondrian:
            thresholds = np.array([conformal_quantile(true[y == c], alpha) for c in range(k)])
        else:
            thresholds = np.full(k, conformal_quantile(true, alpha))
        return cls(model, thresholds, alpha, temperature, score, seed)

    @classmethod
    def calibrate(
        cls,
        model: ScoringModel,
        X: ArrayLike,
        y: ArrayLike,
        alpha: float = 0.1,
        temperature: float = 1.0,
        score: str = "aps",
        mondrian: bool = False,
        seed: int | None = 0,
    ) -> ConformalGestures:
        """Calibrate on labelled feature rows the model was *not* trained on."""
        y_index = _label_index(model, y)
        return cls.from_scores(
            model, model.decision_function(X), y_index, alpha, temperature, score, mondrian, seed
        )

    def with_model(self, model: ScoringModel) -> ConformalGestures:
        """The same thresholds on another model with the same classes (e.g. after adapt)."""
        if not np.array_equal(np.asarray(model.classes), np.asarray(self.model.classes)):
            raise ValueError("the new model has different classes")
        return ConformalGestures(
            model, self.thresholds.copy(), self.alpha, self.temperature, self.score, self.seed
        )

    @property
    def classes(self) -> NDArray[Any]:
        return np.asarray(self.model.classes)

    @property
    def coverage(self) -> float:
        """The promised share of sets that contain the true gesture."""
        return 1.0 - self.alpha

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """The model's probabilities after temperature scaling."""
        return softmax(self.model.decision_function(X), self.temperature)

    def predict_sets(self, X: ArrayLike) -> BoolArray:
        """(n, k) booleans: column c is True if gesture c is in that row's set."""
        s = nonconformity(self.predict_proba(X), self.score, self._rng)
        out: BoolArray = s <= self.thresholds
        return out

    def predict(self, X: ArrayLike, unsure: Any = None) -> list[Any]:
        """The gesture when the set has exactly one, else `unsure`."""
        sets = self.predict_sets(X)
        single = sets.sum(axis=1) == 1
        best = np.argmax(sets, axis=1)
        return [self.classes[b] if one else unsure for b, one in zip(best, single, strict=True)]

    # ---------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        """Everything but the model, as JSON-friendly values."""
        return {
            "classes": [str(c) for c in self.classes],
            "thresholds": [None if not np.isfinite(t) else float(t) for t in self.thresholds],
            "alpha": self.alpha,
            "temperature": self.temperature,
            "score": self.score,
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> Path:
        """Write the calibration (not the model) to a JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path, model: ScoringModel) -> ConformalGestures:
        """Read a calibration written by `save` and attach it to `model`."""
        d = json.loads(Path(path).read_text())
        if d["classes"] != [str(c) for c in np.asarray(model.classes)]:
            raise ValueError("the calibration was made for a model with other classes")
        thresholds = np.array([np.inf if t is None else t for t in d["thresholds"]])
        return cls(model, thresholds, d["alpha"], d["temperature"], d["score"], d["seed"])


# --------------------------------------------------------- evaluation helpers


def set_metrics(
    sets: BoolArray, y_index: ArrayLike, top: ArrayLike | None = None
) -> dict[str, float]:
    """Coverage, mean set size, one-gesture share and accuracy when one gesture.

    `top` (the model's own best class per row) is only used for the last
    number; without it, the single member of a one-gesture set is used.
    """
    y = np.asarray(y_index, dtype=np.intp)
    rows = np.arange(len(y))
    size = sets.sum(axis=1)
    single = size == 1
    pick = np.argmax(sets, axis=1) if top is None else np.asarray(top)
    return {
        "coverage": float(np.mean(sets[rows, y])),
        "mean_set_size": float(np.mean(size)),
        "singleton_rate": float(np.mean(single)),
        "empty_rate": float(np.mean(size == 0)),
        "accuracy_when_singleton": float(np.mean(pick[single] == y[single]))
        if single.any()
        else float("nan"),
    }


class AdaptiveAlpha:
    """Adaptive conformal inference (Gibbs & Candès 2021) across sessions.

    After each session, feed back whether the sets missed on whatever labelled
    data that session had; the working miss rate moves so the long-run miss
    rate tracks the target even when the data drifts:

        alpha_next = alpha_now + gamma * (target - observed_miss_rate)
    """

    def __init__(self, target: float = 0.1, gamma: float = 0.05):
        if not 0.0 < target < 1.0 or gamma <= 0:
            raise ValueError("target must be in (0, 1) and gamma > 0")
        self.target = target
        self.gamma = gamma
        self.alpha = target
        self.history: list[float] = []

    @property
    def working_alpha(self) -> float:
        """The alpha to use now, clipped to (0, 1) so a threshold exists."""
        return float(np.clip(self.alpha, 1e-6, 1 - 1e-6))

    def update(self, observed_miss_rate: float) -> float:
        """Record one session's miss rate; returns the next working alpha."""
        self.history.append(self.working_alpha)
        self.alpha += self.gamma * (self.target - float(observed_miss_rate))
        return self.working_alpha
