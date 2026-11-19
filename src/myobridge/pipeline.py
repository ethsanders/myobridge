"""Glue: raw blocks in, smoothed gesture predictions out. Used live and in the demo."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .classifier import LDAClassifier, MajorityVote
from .dsp import EmgFilter
from .features import StreamingWindower, extract_features


@dataclass
class Prediction:
    gesture: str  # after majority-vote smoothing
    raw_gesture: str  # what LDA said for this single window
    confidence: float  # LDA probability of `gesture` for this window
    sample_index: int  # stream position (samples) where the window ended


class GesturePipeline:
    def __init__(
        self,
        model: LDAClassifier,
        fs: int,
        n_channels: int,
        window_s: float = 0.2,
        step_s: float = 0.05,
        vote_k: int = 5,
        mains_hz: float = 60.0,
    ):
        self.model = model
        self.filter = EmgFilter(fs, n_channels, mains_hz=mains_hz)
        self.windower = StreamingWindower(int(window_s * fs), int(step_s * fs), n_channels)
        self.vote = MajorityVote(vote_k)
        self.samples_seen = 0
        self.last_envelope = np.zeros(n_channels)  # handy for live displays

    def process_block(self, raw: np.ndarray) -> list[Prediction]:
        """Feed one packet's worth of samples; returns 0 or more predictions."""
        out = self.filter.process(raw)
        self.last_envelope = out.envelope[-1]
        self.samples_seen += raw.shape[0]

        predictions = []
        windows = self.windower.push(out.clean)
        # Windows are `step` samples apart; the newest ended `since_last` samples ago.
        step = self.windower.step
        last_end = self.samples_seen - self.windower.since_last
        for i, window in enumerate(windows):
            proba = self.model.predict_proba(extract_features(window))[0]
            label = str(self.model.classes_[np.argmax(proba)])
            voted = self.vote.update(label)
            predictions.append(
                Prediction(
                    gesture=voted,
                    raw_gesture=label,
                    confidence=float(proba[list(self.model.classes_).index(voted)]),
                    sample_index=last_end - (len(windows) - 1 - i) * step,
                )
            )
        return predictions
