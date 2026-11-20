"""Publish the EMG stream and the decoded gestures on Lab Streaming Layer.

LSL is how neuroscience tools pass live signals to each other: recorders
(LabRecorder), analysis toolkits and game engines can all subscribe to a
stream by name. myobridge publishes two:

- an ``EMG`` stream with one float channel per electrode at the device's
  sample rate, carrying the raw samples as they arrive, and
- a ``Markers`` stream with one string per decision: the smoothed gesture.

Both share a source id, so recordings of the two line up. Needs the
optional dependency: ``pip install "myobridge[lsl]"``.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .pipeline import Prediction


def _pylsl() -> Any:
    try:
        import pylsl
    except ImportError as err:  # pragma: no cover - exercised only without pylsl
        raise ImportError('LSL output needs pylsl: pip install "myobridge[lsl]"') from err
    return pylsl


class LslOutlets:
    """Two LSL outlets: raw EMG and decoded gestures.

    Args:
        n_channels: Electrodes in the EMG stream.
        fs: Sample rate of the EMG stream in Hz.
        name: Stream name prefix; the streams are ``<name>-emg`` and
            ``<name>-gestures``.
        source_id: Identifies the device, so a restarted stream is
            recognised as the same one. Use something stable, like the
            armband's name.
    """

    def __init__(
        self,
        n_channels: int,
        fs: float,
        name: str = "myobridge",
        source_id: str = "myobridge",
    ):
        lsl = _pylsl()
        self.n_channels = n_channels
        emg = lsl.StreamInfo(
            f"{name}-emg", "EMG", n_channels, fs, lsl.cf_float32, f"{source_id}-emg"
        )
        chans = emg.desc().append_child("channels")
        for c in range(n_channels):
            ch = chans.append_child("channel")
            ch.append_child_value("label", f"ch{c}")
            ch.append_child_value("unit", "adc_counts")
            ch.append_child_value("type", "EMG")
        gestures = lsl.StreamInfo(
            f"{name}-gestures",
            "Markers",
            1,
            lsl.IRREGULAR_RATE,
            lsl.cf_string,
            f"{source_id}-gestures",
        )
        self.emg = lsl.StreamOutlet(emg)
        self.gestures = lsl.StreamOutlet(gestures)

    def push_emg(self, block: ArrayLike) -> None:
        """Publish raw samples, shape (n, n_channels)."""
        samples = np.asarray(block, dtype=np.float32)
        if samples.ndim != 2 or samples.shape[1] != self.n_channels:
            raise ValueError(f"expected samples shaped (n, {self.n_channels})")
        if len(samples):
            self.emg.push_chunk(samples.tolist())

    def push_gesture(self, prediction: Prediction) -> None:
        """Publish one decision as a JSON string marker."""
        marker = json.dumps(
            {
                "gesture": prediction.gesture,
                "confidence": round(prediction.confidence, 4),
                "sample_index": prediction.sample_index,
            }
        )
        self.gestures.push_sample([marker])
