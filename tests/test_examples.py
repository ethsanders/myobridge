"""The scripts in examples/ must keep running as the library changes."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _run(script: str, *args: str) -> str:
    done = subprocess.run(
        [sys.executable, str(ROOT / "examples" / script), *args],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    return done.stdout


def test_decode_simulated():
    out = _run("decode_simulated.py")
    assert "trained on" in out and out.count("holding") == 3


def test_replay_3dc_on_real_data():
    data = os.environ.get("MYOBRIDGE_DATA")
    if not data or not (Path(data) / "3dc").exists():
        pytest.skip("needs the 3DC dataset (myobridge fetch 3dc, then set MYOBRIDGE_DATA)")
    out = _run("replay_3dc.py", "--person", "7", "--data", data)
    assert "electrode 2 pulled off" in out and "ignoring channel [1]" in out
    assert out.strip().endswith("of decisions were right")
