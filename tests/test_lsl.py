"""Lab Streaming Layer output, round-tripped through a local inlet."""

import json
import time
import uuid

import numpy as np
import pytest

pylsl = pytest.importorskip("pylsl")

from myobridge.lsl import LslOutlets  # noqa: E402
from myobridge.pipeline import Prediction  # noqa: E402


def _inlet(name: str):
    found = pylsl.resolve_byprop("name", name, timeout=5)
    if not found:
        pytest.skip("LSL stream discovery doesn't work on this machine's network")
    inlet = pylsl.StreamInlet(found[0])
    inlet.open_stream(timeout=5)
    return inlet, found[0]


def test_emg_and_gestures_arrive_in_order():
    name = f"mbtest-{uuid.uuid4().hex[:8]}"
    out = LslOutlets(4, 1000.0, name=name, source_id=name)
    emg, info = _inlet(f"{name}-emg")
    gestures, _ = _inlet(f"{name}-gestures")
    assert info.type() == "EMG" and info.channel_count() == 4 and info.nominal_srate() == 1000
    time.sleep(0.3)  # let both connections finish before pushing
    block = np.arange(80, dtype=np.float32).reshape(20, 4)
    out.push_emg(block)
    out.push_emg(np.zeros((0, 4)))
    out.push_gesture(Prediction("fist", "fist", 0.912345, 200))
    got: list[list[float]] = []
    deadline = time.monotonic() + 5
    while len(got) < 20 and time.monotonic() < deadline:
        chunk, _ = emg.pull_chunk(timeout=0.5)
        got += chunk
    np.testing.assert_array_equal(np.array(got), block)
    marker, _ = gestures.pull_sample(timeout=5)
    assert json.loads(marker[0]) == {"gesture": "fist", "confidence": 0.9123, "sample_index": 200}
    with pytest.raises(ValueError, match=r"\(n, 4\)"):
        out.push_emg(np.zeros((5, 3)))
