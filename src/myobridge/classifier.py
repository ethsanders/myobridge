"""Linear Discriminant Analysis in plain numpy, plus calibration and smoothing.

LDA in one paragraph: model each gesture's feature vectors as a Gaussian blob
with its own mean but one *shared* covariance matrix. With a shared covariance
the log-probability of each class is a linear function of x, so prediction is
just one matrix multiply: scores = x @ W + b, pick the biggest. Fitting is
closed-form (means + one covariance + one linear solve), no iterations, which
is why it's the classic baseline for EMG gesture recognition.

Because the model is only "class means + one covariance", it is also easy to
*recalibrate*: keep the covariance learned from lots of old data and nudge the
means toward a few seconds of new data. That is what `adapt()` does, and what
the recalibration experiments in docs/RESULTS.md measure.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dsp import EmgFilter
from .features import extract_features, sliding_windows

FloatArray = NDArray[np.float64]


class LDAClassifier:
    def __init__(self, ridge: float = 1e-3):
        # Ridge adds a little to the covariance diagonal so it's always
        # invertible (e.g. a dead channel would otherwise make it singular).
        # It is scaled by the average variance, so it's unit-free.
        if ridge < 0:
            raise ValueError("ridge must be >= 0")
        self.ridge = ridge
        self.classes_: NDArray[Any] | None = None
        self.means_: FloatArray | None = None  # (n_classes, n_features)
        self.covariance_: FloatArray | None = None  # (n_features, n_features), before ridge
        self.priors_: FloatArray | None = None  # (n_classes,)
        self.weights_: FloatArray | None = None  # (n_features, n_classes)
        self.bias_: FloatArray | None = None  # (n_classes,)

    # ------------------------------------------------------------------ fitting

    def fit(self, X: ArrayLike, y: ArrayLike) -> LDAClassifier:
        Xa, ya = np.asarray(X, dtype=np.float64), np.asarray(y)
        if Xa.ndim != 2 or ya.ndim != 1 or len(Xa) != len(ya):
            raise ValueError("X must be (n_samples, n_features) and y (n_samples,)")
        classes = np.unique(ya)
        if len(classes) < 2:
            raise ValueError("need at least two classes")
        n, k = len(ya), len(classes)
        means = np.array([Xa[ya == c].mean(axis=0) for c in classes])  # (k, d)
        priors = np.array([np.mean(ya == c) for c in classes])

        # Pooled within-class covariance: scatter of each sample around ITS class mean.
        centered = Xa - means[np.searchsorted(classes, ya)]
        cov = centered.T @ centered / max(n - k, 1)
        return self.set_params(classes, means, cov, priors)

    def set_params(
        self, classes: ArrayLike, means: ArrayLike, covariance: ArrayLike, priors: ArrayLike
    ) -> LDAClassifier:
        """Build the linear decision rule from means, a shared covariance and priors.

        Separate from fit() so that adapt() (and tests) can supply the pieces.
        """
        self.classes_ = np.asarray(classes)
        self.means_ = np.asarray(means, dtype=np.float64)
        self.covariance_ = np.asarray(covariance, dtype=np.float64)
        self.priors_ = np.asarray(priors, dtype=np.float64)
        d = self.means_.shape[1]
        cov = self.covariance_ + self.ridge * np.trace(self.covariance_) / d * np.eye(d)
        if not np.all(np.isfinite(cov)) or np.trace(cov) == 0:
            # All features constant: fall back to identity so we still predict.
            cov = np.eye(d)

        # Gaussian log-likelihood with shared cov, dropping terms equal for all classes:
        #   score_c(x) = x^T S^-1 mu_c  -  0.5 mu_c^T S^-1 mu_c  +  log prior_c
        self.weights_ = np.linalg.solve(cov, self.means_.T)  # S^-1 mu_c for every class
        self.bias_ = -0.5 * np.sum(self.means_.T * self.weights_, axis=0) + np.log(self.priors_)
        return self

    # --------------------------------------------------------------- prediction

    def _check_fitted(self) -> tuple[FloatArray, FloatArray, NDArray[Any]]:
        if self.weights_ is None or self.bias_ is None or self.classes_ is None:
            raise RuntimeError("classifier is not fitted")
        return self.weights_, self.bias_, self.classes_

    def decision_function(self, X: ArrayLike) -> FloatArray:
        weights, bias, _ = self._check_fitted()
        scores: FloatArray = np.atleast_2d(np.asarray(X, dtype=np.float64)) @ weights + bias
        return scores

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)  # softmax, overflow-safe
        p = np.exp(scores)
        proba: FloatArray = p / p.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        _, _, classes = self._check_fitted()
        return classes[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Fraction of windows classified correctly."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        assert self.means_ is not None and self.covariance_ is not None and self.priors_ is not None
        assert self.classes_ is not None and self.weights_ is not None and self.bias_ is not None
        np.savez(
            path,
            classes=self.classes_,
            means=self.means_,
            covariance=self.covariance_,
            priors=self.priors_,
            ridge=np.array(self.ridge),
            weights=self.weights_,
            bias=self.bias_,
        )

    @classmethod
    def load(cls, path: str | Path) -> LDAClassifier:
        with np.load(path, allow_pickle=False) as data:
            if "means" in data:  # current format: rebuild from the parameters
                clf = cls(ridge=float(data["ridge"]))
                return clf.set_params(
                    data["classes"], data["means"], data["covariance"], data["priors"]
                )
            # v0.1 files only stored the decision rule; enough to predict, not to adapt.
            clf = cls()
            clf.classes_ = data["classes"]
            clf.weights_, clf.bias_ = data["weights"], data["bias"]
            return clf


# ------------------------------------------------------------------ recalibration


def adapt(
    base: LDAClassifier,
    X_new: ArrayLike,
    y_new: ArrayLike,
    mean_prior: float = 0.0,
    covariance_weight: float = 0.0,
) -> LDAClassifier:
    """Recalibrate a trained LDA with a small amount of new labelled data.

    Electrode shift, sweat and fatigue mostly *move* each gesture's cluster in
    feature space; the cluster *shape* (the shared covariance) changes less.
    So we keep the old covariance, which needs lots of data to estimate, and
    re-estimate the class means, which a few windows per class pin down well.

    Each class mean becomes a weighted average of the old mean and the mean of
    that class's new windows:

        mu_c = (n_c * newmean_c + mean_prior * oldmean_c) / (n_c + mean_prior)

    `mean_prior` is "how many windows the old mean is worth". 0 means: trust
    the new data completely (classes with no new data keep their old mean).
    `covariance_weight` in [0, 1] blends in the new data's pooled covariance;
    0 keeps the old one. This is the supervised version of the adaptive-LDA
    idea used in the myoelectric literature (e.g. Vidovic et al. 2016).
    """
    if base.means_ is None or base.covariance_ is None or base.classes_ is None:
        raise RuntimeError("base classifier must be fitted (and saved in the current format)")
    if mean_prior < 0 or not 0.0 <= covariance_weight <= 1.0:
        raise ValueError("mean_prior must be >= 0 and covariance_weight in [0, 1]")
    Xn, yn = np.asarray(X_new, dtype=np.float64), np.asarray(y_new)
    unknown = set(np.unique(yn).tolist()) - set(base.classes_.tolist())
    if unknown:
        raise ValueError(f"new data has classes the base model doesn't know: {unknown}")

    means = base.means_.copy()
    centered_rows: list[FloatArray] = []
    for i, c in enumerate(base.classes_):
        rows = Xn[yn == c]
        if len(rows) == 0:
            continue
        new_mean = rows.mean(axis=0)
        means[i] = (len(rows) * new_mean + mean_prior * base.means_[i]) / (len(rows) + mean_prior)
        centered_rows.append(rows - new_mean)

    cov = base.covariance_
    if covariance_weight > 0 and centered_rows:
        centered = np.vstack(centered_rows)
        dof = max(len(centered) - len(centered_rows), 1)
        cov = (1 - covariance_weight) * cov + covariance_weight * (centered.T @ centered / dof)

    assert base.priors_ is not None
    return LDAClassifier(ridge=base.ridge).set_params(base.classes_, means, cov, base.priors_)


# --------------------------------------------------------------------- smoothing


class MajorityVote:
    """Output the most common of the last k predictions, to kill 1-window flickers.

    Costs latency: a new gesture has to win k//2 + 1 windows before it shows.
    """

    def __init__(self, k: int = 5):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.history: deque[str] = deque(maxlen=k)

    def update(self, label: str) -> str:
        self.history.append(label)
        counts = Counter(self.history)
        # max() keeps the first best it sees; walking newest-first means a tie
        # goes to the most recent of the tied labels.
        return max(reversed(self.history), key=counts.__getitem__)


# ------------------------------------------------------------------ calibration


def recording_to_features(
    raw: ArrayLike,
    fs: int,
    window_s: float = 0.2,
    step_s: float = 0.05,
    settle_s: float = 0.25,
    mains_hz: float = 60.0,
) -> FloatArray:
    """Filter one held-gesture recording and turn it into feature vectors.

    The first `settle_s` seconds are skipped: the filters start from zero
    and need a moment to settle, and the user is still moving into the pose.
    """
    rawa = np.asarray(raw, dtype=np.float64)
    clean = EmgFilter(fs, rawa.shape[1], mains_hz=mains_hz).process(rawa).clean
    clean = clean[int(settle_s * fs) :]
    size, step = int(window_s * fs), int(step_s * fs)
    return np.array([extract_features(w) for _, w in sliding_windows(clean, size, step)])


def calibrate(
    record: Callable[[str, float], NDArray[Any]],
    gestures: Sequence[str],
    seconds: float,
    fs: int,
    mains_hz: float = 60.0,
) -> tuple[LDAClassifier, FloatArray, NDArray[Any]]:
    """Record `seconds` of each gesture via `record(gesture, seconds)`, then fit LDA.

    `record` is anything that returns an (n_samples, n_channels) array: the
    simulator in tests/demo, or a live UDP receiver with the real armband.
    """
    feats: list[FloatArray] = []
    labels: list[str] = []
    for g in gestures:
        X = recording_to_features(record(g, seconds), fs, mains_hz=mains_hz)
        feats.append(X)
        labels += [g] * len(X)
    X, y = np.vstack(feats), np.array(labels)
    return LDAClassifier().fit(X, y), X, y
