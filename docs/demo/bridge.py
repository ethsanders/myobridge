"""Glue between the demo page and myobridge, run inside Pyodide.

The page fetches 3DC recordings straight from the dataset's public
repository, parses them into flat Float64Arrays, and hands them here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from myobridge.datasets import threedc
from myobridge.datasets.core import Recording
from myobridge.replay import Replay

_replay: Replay | None = None
_pending: dict[str, list[Recording]] = {"train": [], "test": []}


def _py(x: Any) -> Any:
    """JavaScript values arrive as proxies; turn them into Python ones."""
    return x.to_py() if hasattr(x, "to_py") else x


def add(part: str, values: Any, label: int, session: int, rep: int) -> None:
    """Queue one recording ("train" or "test"); `values` is a flat Float64Array."""
    emg = np.asarray(_py(values), dtype=np.float64).reshape(-1, threedc.N_CHANNELS)
    rec = Recording(emg, threedc.FS, threedc.GESTURES[label], label, 0, session, rep)
    _pending[part].append(rec)


def start(channels: Any = None) -> dict[str, Any]:
    """Calibrate on the queued train recordings and get ready to replay the test ones."""
    global _replay
    chans = _py(channels)
    train, test = _pending["train"], _pending["test"]
    _pending["train"], _pending["test"] = [], []
    _replay = Replay(
        train,
        test,
        threedc.GESTURES,
        mains_hz=threedc.MAINS_HZ,
        channels=list(chans) if chans else None,
        rail=threedc.ADC_RAIL,
    )
    return {
        "n_channels": _replay.n_channels,
        "duration_s": _replay.duration_s,
        "fs": _replay.fs,
        "display_fs": _replay.fs / _replay.display_every,
        "features": _replay.model.n_features,
        "trace_centre": _replay.trace_centre.tolist(),
        "trace_scale": _replay.trace_scale.tolist(),
    }


def step(seconds: float) -> dict[str, Any]:
    assert _replay is not None
    f = _replay.step(seconds)
    return {
        "t": f.t,
        "truth": f.truth,
        "prediction": f.prediction,
        "confidence": f.confidence,
        "accuracy": f.accuracy,
        "done": _replay.done,
        "channels": [
            {"issues": list(c.issues), "dropped": c.dropped, "fault": c.fault} for c in f.channels
        ],
        # flat, channel-interleaved samples: cheap to move to JavaScript
        "raw": f.raw.astype(np.float32).ravel().tolist(),
        "n_raw": int(f.raw.shape[0]),
    }


def inject(channel: int, fault: str | None) -> None:
    assert _replay is not None
    _replay.inject(channel, fault or None)
