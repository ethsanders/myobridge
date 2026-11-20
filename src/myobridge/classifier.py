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
the recalibration experiments in docs/results.md measure.
"""

from __future__ import annotations

import json
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .dsp import EmgFilter
from .features import DEFAULT_FEATURE_SET, extract_features, feature_set_names, window_stack

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrainingSettings:
    """How the training features were made, so live use can make them the same way.

    `calibrate` fills this in and `LDAClassifier.save` keeps it, so
    `GesturePipeline.from_model` can rebuild a matching pipeline from a
    saved file alone.
    """

    fs: float
    window_s: float = 0.2
    step_s: float = 0.05
    mains_hz: float = 60.0
    zc_threshold: float = 5.0
    ssc_threshold: float = 25.0


class LDAClassifier:
    """Linear discriminant analysis with a scale-free ridge, in plain numpy.

    Works like scikit-learn's estimators (`fit`, `predict`, `predict_proba`,
    `score`) and adds what a wearable needs: `subset` to drop a broken
    channel without refitting, `save`/`load` without pickle, and `adapt` (a
    module function) to recalibrate from a few seconds of new data.

    Args:
        ridge: Added to each feature's variance as a fraction of itself, so
            the covariance is always invertible and units don't matter.
        feature_set: Which features the model was trained on (a key of
            `features.FEATURE_SETS`), so live use computes the same ones.
        settings: How those features were made; `calibrate` fills it in.
    """

    def __init__(
        self,
        ridge: float = 1e-3,
        feature_set: str | None = None,
        settings: TrainingSettings | None = None,
    ):
        # Ridge adds a little to the covariance diagonal so it's always
        # invertible (e.g. a dead channel would otherwise make it singular).
        # Each feature gets `ridge` times its own variance, so it's unit-free.
        # `feature_set` records which features the model was trained on
        # (see features.FEATURE_SETS) so a saved model is used with the same ones.
        if ridge < 0:
            raise ValueError("ridge must be >= 0")
        self.ridge = ridge
        self.feature_set = feature_set
        self.settings = settings
        self.classes_: NDArray[Any] | None = None
        self.means_: FloatArray | None = None  # (n_classes, n_features)
        self.covariance_: FloatArray | None = None  # (n_features, n_features), before ridge
        self.priors_: FloatArray | None = None  # (n_classes,)
        self.weights_: FloatArray | None = None  # (n_features, n_classes)
        self.bias_: FloatArray | None = None  # (n_classes,)

    # ------------------------------------------------------------------ fitting

    def fit(self, X: ArrayLike, y: ArrayLike) -> LDAClassifier:
        """Fit to feature rows `X` (n_samples, n_features) and labels `y`; returns self."""
        Xa, ya = np.asarray(X, dtype=np.float64), np.asarray(y)
        if Xa.ndim != 2 or ya.ndim != 1 or len(Xa) != len(ya):
            raise ValueError("X must be (n_samples, n_features) and y (n_samples,)")
        if not np.all(np.isfinite(Xa)):
            raise ValueError("X contains NaN or infinite values")
        if ya.dtype == object:  # e.g. labels from pandas; keep them saveable without pickle
            ya = ya.astype(str)
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
        # Ridge is relative to each feature's own variance, so rescaling a
        # feature (microvolts vs counts, AR coefficients vs zero-crossing
        # counts) never changes the predictions. A constant feature, such as
        # one from a dead channel, has no variance of its own, so it gets the
        # average one; that keeps cov invertible.
        var = np.diag(self.covariance_)
        fallback = float(np.mean(var)) if np.any(var > 0) else 1.0
        cov = self.covariance_ + self.ridge * np.diag(np.where(var > 0, var, fallback))
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
        """Unnormalised log-probability of each class, shape (n_samples, n_classes)."""
        weights, bias, _ = self._check_fitted()
        scores: FloatArray = np.atleast_2d(np.asarray(X, dtype=np.float64)) @ weights + bias
        return scores

    def predict_proba(self, X: ArrayLike) -> FloatArray:
        """Probability of each class (columns in the order of `classes`)."""
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)  # softmax, overflow-safe
        p = np.exp(scores)
        proba: FloatArray = p / p.sum(axis=1, keepdims=True)
        return proba

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """The most likely class for each row of `X`."""
        _, _, classes = self._check_fitted()
        return classes[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Fraction of windows classified correctly."""
        return float(np.mean(self.predict(X) == np.asarray(y)))

    @property
    def is_fitted(self) -> bool:
        """True once `fit`, `set_params` or `load` has given the model a decision rule."""
        return self.weights_ is not None

    @property
    def classes(self) -> NDArray[Any]:
        """The class labels, in the order of predict_proba's columns."""
        return self._check_fitted()[2]

    @property
    def n_features(self) -> int:
        """How many features each input row must have."""
        return int(self._check_fitted()[0].shape[0])

    def subset(self, columns: ArrayLike) -> LDAClassifier:
        """The model you'd get by training on only these feature columns.

        With a shared Gaussian covariance, dropping features is the same as
        taking the matching rows of the means and the matching block of the
        covariance; the ridge is per feature, so it carries over unchanged.
        So the live pipeline can stop using a broken electrode instantly,
        without keeping the training data or refitting (the sensor-fault
        tolerant LDA of Zhang & Huang, JNER 12:18, 2015).
        """
        if self.means_ is None or self.covariance_ is None or self.priors_ is None:
            raise RuntimeError("subset needs a fitted model (saved in the current format)")
        cols = np.asarray(columns, dtype=np.intp)
        clf = LDAClassifier(self.ridge, self.feature_set, self.settings)
        assert self.classes_ is not None
        return clf.set_params(
            self.classes_, self.means_[:, cols], self.covariance_[np.ix_(cols, cols)], self.priors_
        )

    # ------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> Path:
        """Write the model to an .npz file (the suffix is added if missing); returns the path."""
        self._check_fitted()
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        assert self.means_ is not None and self.covariance_ is not None and self.priors_ is not None
        assert self.classes_ is not None and self.weights_ is not None and self.bias_ is not None
        np.savez(
            path,
            classes=self.classes_,
            means=self.means_,
            covariance=self.covariance_,
            priors=self.priors_,
            ridge=np.array(self.ridge),
            feature_set=np.array(self.feature_set or ""),
            settings=np.array(json.dumps(asdict(self.settings)) if self.settings else ""),
            weights=self.weights_,
            bias=self.bias_,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> LDAClassifier:
        """Read a model written by `save` (the `.npz` suffix may be left off)."""
        path = Path(path)
        if not path.exists() and path.with_name(path.name + ".npz").exists():
            path = path.with_name(path.name + ".npz")  # save("model") wrote model.npz
        with np.load(path, allow_pickle=False) as data:
            # Files written before feature sets existed used the "td" features.
            feature_set = str(data["feature_set"]) if "feature_set" in data else "td"
            raw_settings = str(data["settings"]) if "settings" in data else ""
            settings = TrainingSettings(**json.loads(raw_settings)) if raw_settings else None
            if "means" in data:  # current format: rebuild from the parameters
                clf = cls(float(data["ridge"]), feature_set or None, settings)
                return clf.set_params(
                    data["classes"], data["means"], data["covariance"], data["priors"]
                )
            # v0.1 files only stored the decision rule; enough to predict, not to adapt.
            clf = cls(feature_set=feature_set)
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
    0 keeps the old one. This is the covariate-shift adaptation of Vidovic et
    al., "Improving the robustness of myoelectric pattern recognition for upper
    limb prostheses by covariate shift adaptation", IEEE TNSRE 24(9):961-970
    (2016); `mean_prior` states their mean weight as a number of windows.
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
    return LDAClassifier(base.ridge, base.feature_set, base.settings).set_params(
        base.classes_, means, cov, base.priors_
    )


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
        """Add the newest prediction and return the current majority."""
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
    feature_set: str = DEFAULT_FEATURE_SET,
) -> FloatArray:
    """Filter one held-gesture recording and turn it into feature vectors.

    The first `settle_s` seconds are skipped: the filters start from zero
    and need a moment to settle, and the user is still moving into the pose.
    """
    rawa = np.asarray(raw, dtype=np.float64)
    clean = EmgFilter(fs, rawa.shape[1], mains_hz=mains_hz).process(rawa).clean
    clean = clean[int(settle_s * fs) :]
    size, step = int(round(window_s * fs)), int(round(step_s * fs))
    windows = window_stack(clean, size, step)
    if len(windows) == 0:
        return np.zeros((0, len(feature_set_names(feature_set)) * rawa.shape[1]))
    return extract_features(windows, feature_set=feature_set)


def calibrate(
    record: Callable[[str, float], NDArray[Any]],
    gestures: Sequence[str],
    seconds: float,
    fs: int,
    mains_hz: float = 60.0,
    feature_set: str = DEFAULT_FEATURE_SET,
) -> tuple[LDAClassifier, FloatArray, NDArray[Any]]:
    """Record `seconds` of each gesture via `record(gesture, seconds)`, then fit LDA.

    `record` is anything that returns an (n_samples, n_channels) array: the
    simulator in tests/demo, or a live UDP receiver with the real armband.
    """
    feats: list[FloatArray] = []
    labels: list[str] = []
    for g in gestures:
        X = recording_to_features(
            record(g, seconds), fs, mains_hz=mains_hz, feature_set=feature_set
        )
        feats.append(X)
        labels += [g] * len(X)
    X, y = np.vstack(feats), np.array(labels)
    settings = TrainingSettings(fs=float(fs), mains_hz=mains_hz)
    return LDAClassifier(feature_set=feature_set, settings=settings).fit(X, y), X, y
