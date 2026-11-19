"""Linear Discriminant Analysis in plain numpy, plus calibration and smoothing.

LDA in one paragraph: model each gesture's feature vectors as a Gaussian blob
with its own mean but one *shared* covariance matrix. With a shared covariance
the log-probability of each class is a linear function of x, so prediction is
just one matrix multiply: scores = x @ W + b, pick the biggest. Fitting is
closed-form (means + one covariance + one linear solve), no iterations, which
is why it's the classic baseline for EMG gesture recognition.
"""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Callable, Sequence

import numpy as np

from .dsp import EmgFilter
from .features import extract_features, sliding_windows


class LDAClassifier:
    def __init__(self, ridge: float = 1e-3):
        # Ridge adds a little to the covariance diagonal so it's always
        # invertible (e.g. a dead channel would otherwise make it singular).
        self.ridge = ridge
        self.classes_: np.ndarray | None = None
        self.weights_: np.ndarray | None = None  # (n_features, n_classes)
        self.bias_: np.ndarray | None = None  # (n_classes,)

    def fit(self, X: np.ndarray, y: Sequence[str]) -> LDAClassifier:
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.classes_ = np.unique(y)
        n, d = X.shape
        k = len(self.classes_)
        means = np.array([X[y == c].mean(axis=0) for c in self.classes_])  # (k, d)
        priors = np.array([np.mean(y == c) for c in self.classes_])

        # Pooled within-class covariance: scatter of each sample around ITS class mean.
        centered = X - means[np.searchsorted(self.classes_, y)]
        cov = centered.T @ centered / max(n - k, 1)
        cov += self.ridge * np.trace(cov) / d * np.eye(d)

        # Gaussian log-likelihood with shared cov, dropping terms equal for all classes:
        #   score_c(x) = x^T S^-1 mu_c  -  0.5 mu_c^T S^-1 mu_c  +  log prior_c
        self.weights_ = np.linalg.solve(cov, means.T)  # S^-1 mu_c for every class
        self.bias_ = -0.5 * np.sum(means.T * self.weights_, axis=0) + np.log(priors)
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("classifier is not fitted")
        return np.atleast_2d(X) @ self.weights_ + self.bias_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self.decision_function(X)
        scores -= scores.max(axis=1, keepdims=True)  # softmax, overflow-safe
        p = np.exp(scores)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def save(self, path: str) -> None:
        np.savez(path, classes=self.classes_, weights=self.weights_, bias=self.bias_)

    @classmethod
    def load(cls, path: str) -> LDAClassifier:
        data = np.load(path, allow_pickle=False)
        clf = cls()
        clf.classes_ = data["classes"]
        clf.weights_, clf.bias_ = data["weights"], data["bias"]
        return clf


class MajorityVote:
    """Output the most common of the last k predictions, to kill 1-window flickers.

    Costs latency: a new gesture has to win k//2 + 1 windows before it shows.
    """

    def __init__(self, k: int = 5):
        self.history: deque[str] = deque(maxlen=k)

    def update(self, label: str) -> str:
        self.history.append(label)
        counts = Counter(self.history)
        # max() keeps the first best it sees; walking newest-first means a tie
        # goes to the most recent of the tied labels.
        return max(reversed(self.history), key=counts.__getitem__)


# ------------------------------------------------------------------ calibration


def recording_to_features(
    raw: np.ndarray,
    fs: int,
    window_s: float = 0.2,
    step_s: float = 0.05,
    settle_s: float = 0.25,
    mains_hz: float = 60.0,
) -> np.ndarray:
    """Filter one held-gesture recording and turn it into feature vectors.

    The first `settle_s` seconds are skipped: the filters start from zero
    and need a moment to settle, and the user is still moving into the pose.
    """
    clean = EmgFilter(fs, raw.shape[1], mains_hz=mains_hz).process(raw).clean
    clean = clean[int(settle_s * fs) :]
    size, step = int(window_s * fs), int(step_s * fs)
    return np.array([extract_features(w) for _, w in sliding_windows(clean, size, step)])


def calibrate(
    record: Callable[[str, float], np.ndarray],
    gestures: Sequence[str],
    seconds: float,
    fs: int,
    mains_hz: float = 60.0,
) -> tuple[LDAClassifier, np.ndarray, np.ndarray]:
    """Record `seconds` of each gesture via `record(gesture, seconds)`, then fit LDA.

    `record` is anything that returns an (n_samples, n_channels) array: the
    simulator in tests/demo, or a live UDP receiver with the real armband.
    """
    feats, labels = [], []
    for g in gestures:
        X = recording_to_features(record(g, seconds), fs, mains_hz=mains_hz)
        feats.append(X)
        labels += [g] * len(X)
    X, y = np.vstack(feats), np.array(labels)
    return LDAClassifier().fit(X, y), X, y
