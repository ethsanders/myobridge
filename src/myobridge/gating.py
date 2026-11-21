"""Decision policies: when to act on a gesture, and when to hold.

The decoder gives one answer per window, every 50 ms. What a user feels is
the *decision*: the gesture a prosthesis or game is told to perform. A
policy turns the stream of per-window outputs (class probabilities, and
optionally conformal gesture sets) into one decision per window: a class
index to act on, or `None` to hold.

Every policy works one window at a time (`step`), so it runs live, and
keeps a little state; give each stream its own policy, and `reset()` it
when a new stream starts.

- `VotePolicy`: the majority of the last k top-1 answers. This is what
  `GesturePipeline` reports as `Prediction.gesture`. It never holds.
- `ConfidenceGate`: the same vote, but hold while the newest window's top
  probability is below a cut-off.
- `SetGate`: act on a gesture only when at least m of the last n conformal
  sets hold exactly that one gesture.
- `EvidenceGate`: add up each gesture's log-probability over recent
  windows, with older windows fading, and act when the summed evidence
  makes one gesture likely enough. A leaky version of the sequential
  probability ratio test.

What "hold" means is up to the application. By default a gate's hold is
`None` ("do nothing"). With `latch=True` it repeats its last action
instead, so it only changes its output when it is sure of the new one.

`myobridge bench decision` measures these on real recordings; see
docs/decisions.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from numpy.typing import ArrayLike, NDArray

IntArray = NDArray[np.intp]

HOLD = -1  # what `DecisionPolicy.run` writes for a window that holds


class DecisionPolicy(ABC):
    """Turns one window's output at a time into an action or a hold.

    Subclasses implement `_decide`; this class adds the optional latch.

    Args:
        latch: When the policy would hold, repeat its last action instead
            (nothing before the first action).
    """

    needs_sets = False  # True if `step` must be given conformal sets

    def __init__(self, latch: bool = False):
        self.latch = latch
        self._last: int | None = None

    def step(self, proba: ArrayLike, gesture_set: ArrayLike | None = None) -> int | None:
        """Feed one window; returns the class index to act on, or None to hold.

        Args:
            proba: The window's class probabilities, shape (k,).
            gesture_set: The window's conformal set as k booleans, or None.
        """
        p = np.asarray(proba, dtype=np.float64).ravel()
        s = None if gesture_set is None else np.asarray(gesture_set, dtype=bool).ravel()
        if self.needs_sets and s is None:
            raise ValueError(f"{type(self).__name__} needs a conformal set for every window")
        decision = self._decide(p, s)
        if decision is not None:
            self._last = decision
            return decision
        return self._last if self.latch else None

    def reset(self) -> None:
        """Forget everything, as at the start of a new stream."""
        self._last = None
        self._reset()

    def run(self, proba: ArrayLike, sets: ArrayLike | None = None) -> IntArray:
        """`reset`, then `step` through a whole stream of windows.

        Returns one entry per window: the class index, or `HOLD` (-1).
        """
        p = np.atleast_2d(np.asarray(proba, dtype=np.float64))
        s = None if sets is None else np.atleast_2d(np.asarray(sets, dtype=bool))
        if s is not None and s.shape != p.shape:
            raise ValueError("proba and sets must have the same shape")
        self.reset()
        out = np.empty(len(p), dtype=np.intp)
        for i in range(len(p)):
            d = self.step(p[i], None if s is None else s[i])
            out[i] = HOLD if d is None else d
        return out

    @abstractmethod
    def _decide(
        self, proba: NDArray[np.float64], gesture_set: NDArray[np.bool_] | None
    ) -> int | None:
        """This window's own decision, before the latch."""

    @abstractmethod
    def _reset(self) -> None:
        """Clear the subclass's state."""


class _Vote:
    """The majority of the last k labels; a tie goes to the most recent of
    the tied labels (the same rule as `classifier.MajorityVote`)."""

    def __init__(self, k: int):
        if k < 1:
            raise ValueError("k must be >= 1")
        self.history: deque[int] = deque(maxlen=k)
        self.counts: dict[int, int] = {}

    def update(self, label: int) -> int:
        if len(self.history) == self.history.maxlen:
            old = self.history[0]
            self.counts[old] -= 1
        self.history.append(label)
        self.counts[label] = self.counts.get(label, 0) + 1
        best = max(self.counts.values())
        for lab in reversed(self.history):
            if self.counts[lab] == best:
                return lab
        raise AssertionError("unreachable")  # pragma: no cover

    def clear(self) -> None:
        self.history.clear()
        self.counts.clear()


class VotePolicy(DecisionPolicy):
    """The majority of the last `k` top-1 answers. Never holds.

    With the pipeline's `vote_k` this is exactly `Prediction.gesture`.
    """

    def __init__(self, k: int = 5):
        super().__init__(latch=False)
        self.k = k
        self._vote = _Vote(k)

    def _decide(self, proba: NDArray[np.float64], gesture_set: NDArray[np.bool_] | None) -> int:
        return self._vote.update(int(np.argmax(proba)))

    def _reset(self) -> None:
        self._vote.clear()

    def __repr__(self) -> str:
        return f"VotePolicy(k={self.k})"


class ConfidenceGate(DecisionPolicy):
    """The `k`-window vote, held while the newest window's top probability is below `cutoff`.

    Every window still goes into the vote; the cut-off only decides whether
    this window's vote is acted on. The probabilities are whatever you pass
    to `step`. In `GesturePipeline` with a conformal calibration they are
    temperature-scaled, which matters: raw LDA probabilities are nearly
    always 0 or 1.
    """

    def __init__(self, cutoff: float, k: int = 5, latch: bool = False):
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("cutoff must be in [0, 1]")
        super().__init__(latch)
        self.cutoff, self.k = cutoff, k
        self._vote = _Vote(k)

    def _decide(
        self, proba: NDArray[np.float64], gesture_set: NDArray[np.bool_] | None
    ) -> int | None:
        voted = self._vote.update(int(np.argmax(proba)))
        return voted if float(proba.max()) >= self.cutoff else None

    def _reset(self) -> None:
        self._vote.clear()

    def __repr__(self) -> str:
        return f"ConfidenceGate(cutoff={self.cutoff:g}, k={self.k}, latch={self.latch})"


class SetGate(DecisionPolicy):
    """Act on a gesture when at least `m` of the last `n` sets are exactly that gesture.

    `m` defaults to `n`: the last n sets must all be the same single
    gesture. If two gestures qualify (possible when m <= n / 2), the one
    seen most recently wins. Needs a conformal set for every window.
    """

    needs_sets = True

    def __init__(self, n: int = 3, m: int | None = None, latch: bool = False):
        m = n if m is None else m
        if n < 1 or not 1 <= m <= n:
            raise ValueError("need n >= 1 and 1 <= m <= n")
        super().__init__(latch)
        self.n, self.m = n, m
        self._singles: deque[int] = deque(maxlen=n)  # the gesture, or -1 if not one gesture

    def _decide(
        self, proba: NDArray[np.float64], gesture_set: NDArray[np.bool_] | None
    ) -> int | None:
        assert gesture_set is not None
        members = np.flatnonzero(gesture_set)
        self._singles.append(int(members[0]) if len(members) == 1 else -1)
        counts: dict[int, int] = {}
        for g in self._singles:
            if g >= 0:
                counts[g] = counts.get(g, 0) + 1
        for g in reversed(self._singles):
            if g >= 0 and counts[g] >= self.m:
                return g
        return None

    def _reset(self) -> None:
        self._singles.clear()

    def __repr__(self) -> str:
        return f"SetGate(n={self.n}, m={self.m}, latch={self.latch})"


class EvidenceGate(DecisionPolicy):
    """Leaky accumulation of log-probabilities; act when one gesture is likely enough.

    After each window, every gesture's evidence becomes
    `memory * evidence + log(max(p, floor))`. The softmax of the evidence is
    the posterior you'd get treating windows as independent and older ones
    as worth less; the gate acts on its top gesture once that posterior
    reaches `threshold`, and holds otherwise. Without the leak
    (`memory=1`) this is the multi-hypothesis sequential probability ratio
    test; the leak lets it change its mind when the gesture changes, and
    `floor` stops one window from ruling a gesture out for good.

    Overlapping windows are not independent, so `threshold` is a knob to
    set on held-out data, not a probability to trust.
    """

    def __init__(
        self, threshold: float, memory: float = 0.8, floor: float = 1e-4, latch: bool = False
    ):
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if not 0.0 <= memory <= 1.0:
            raise ValueError("memory must be in [0, 1]")
        if not 0.0 < floor < 1.0:
            raise ValueError("floor must be in (0, 1)")
        super().__init__(latch)
        self.threshold, self.memory, self.floor = threshold, memory, floor
        self._evidence: NDArray[np.float64] | None = None

    def _decide(
        self, proba: NDArray[np.float64], gesture_set: NDArray[np.bool_] | None
    ) -> int | None:
        log_p = np.log(np.maximum(proba, self.floor))
        if self._evidence is None or len(self._evidence) != len(log_p):
            self._evidence = log_p
        else:
            self._evidence = self.memory * self._evidence + log_p
        z = self._evidence - self._evidence.max()
        post = np.exp(z) / np.exp(z).sum()
        best = int(np.argmax(post))
        return best if post[best] >= self.threshold else None

    def _reset(self) -> None:
        self._evidence = None

    def __repr__(self) -> str:
        return (
            f"EvidenceGate(threshold={self.threshold:g}, memory={self.memory:g}, "
            f"floor={self.floor:g}, latch={self.latch})"
        )
