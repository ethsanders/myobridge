"""The browser demo's Python side (docs/demo/bridge.py), run natively."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

BRIDGE = Path(__file__).resolve().parents[1] / "docs" / "demo" / "bridge.py"


@pytest.fixture
def bridge():
    spec = importlib.util.spec_from_file_location("demo_bridge", BRIDGE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flat(path: Path) -> np.ndarray:
    # what core.js's parseRecording hands over: every number, row by row
    return np.array(path.read_text().replace(",", " ").split(), dtype=np.float64)


def test_bridge_replays_with_faults(bridge, fake_threedc):
    root = fake_threedc / "Participant1"
    for session, folder, reps in ((0, "train", (0, 1)), (1, "test", (0,))):
        for rep in reps:
            for label in range(11):
                f = root / folder / "EMG" / f"3dc_EMG_gesture_{rep}_{label}.txt"
                bridge.add("train" if session == 0 else "test", _flat(f), label, session, rep)
    info = bridge.start([0, 2, 5, 7])
    assert info["n_channels"] == 4 and info["features"] == 60
    assert info["display_fs"] == 250.0
    assert len(info["trace_scale"]) == len(info["trace_centre"]) == 4
    assert min(info["trace_scale"]) > 0
    frames = [bridge.step(0.1) for _ in range(20)]
    bridge.inject(1, "lead_off")
    frames += [bridge.step(0.1) for _ in range(15)]
    last = frames[-1]
    assert last["channels"][1] == {"issues": ["flatline"], "dropped": True, "fault": "lead_off"}
    assert len(last["raw"]) == last["n_raw"] * 4 == 100
    assert isinstance(last["truth"], str) and 0.0 <= last["confidence"] <= 1.0
    bridge.inject(1, "")  # the page's "Working" option
    assert bridge.step(0.1)["channels"][1]["fault"] is None
    while not bridge.step(1.0)["done"]:
        pass
