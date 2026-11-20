"""Compile the firmware's pure C++ headers on this machine and compare them
with the Python implementation byte for byte. Skipped if there's no C++ compiler,
unless MYOBRIDGE_REQUIRE_CXX=1 (CI sets it, so the check can't silently vanish)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from myobridge.decimator import HALFBAND_Q15, decimate2
from myobridge.protocol import decode_packet, encode_packet

FIRMWARE = Path(__file__).resolve().parents[1] / "firmware"
CXX = shutil.which("g++") or shutil.which("clang++")


@pytest.fixture(scope="module")
def host(tmp_path_factory) -> Path:
    if CXX is None:
        if os.environ.get("MYOBRIDGE_REQUIRE_CXX") == "1":
            pytest.fail("MYOBRIDGE_REQUIRE_CXX=1 but no g++ or clang++ found")
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("fw") / "mb_host"
    subprocess.run(
        [
            CXX,
            "-std=c++11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-o",
            str(exe),
            str(FIRMWARE / "tests" / "mb_host.cpp"),
        ],
        check=True,
    )
    return exe


def run(host: Path, *args: str, stdin: str = "") -> str:
    done = subprocess.run(
        [str(host), *args], input=stdin, capture_output=True, text=True, check=True
    )
    return done.stdout


def test_selftest(host):
    assert run(host, "selftest").strip() == "ok"


@pytest.mark.parametrize(
    ("n_ch", "n_frames", "seq", "ts"),
    [
        (4, 20, 0, 0),
        (4, 20, 2**32 - 1, 2**63 + 12345),
        (1, 1, 7, 99),
        (8, 50, 123456, 1_700_000_000_000_000),
    ],
)
def test_packets_are_byte_identical(host, n_ch, n_frames, seq, ts):
    rng = np.random.default_rng(n_ch * 1000 + n_frames)
    samples = rng.integers(-32768, 32768, (n_frames, n_ch)).astype(np.int16)
    stdin = " ".join(map(str, samples.ravel().tolist()))
    cpp = bytes.fromhex(
        run(
            host, "packet", str(n_ch), str(n_frames), "1000", str(seq), str(ts), stdin=stdin
        ).strip()
    )
    assert cpp == encode_packet(samples, seq, ts, 1000)
    p = decode_packet(cpp)
    assert p.seq == seq and np.array_equal(p.samples, samples)


def test_decimator_is_bit_exact(host):
    rng = np.random.default_rng(0)
    # random full-scale noise plus a loud tone, to exercise rounding and saturation
    x = rng.integers(-32768, 32768, (2000, 4))
    x[:, 0] = np.clip(40000 * np.sin(np.arange(2000) * 0.3), -32768, 32767)
    x = x.astype(np.int16)
    out = run(host, "decimate", "2000", stdin=" ".join(map(str, x.ravel().tolist())))
    cpp = np.array([[int(v) for v in line.split()] for line in out.strip().splitlines()])
    assert np.array_equal(cpp, decimate2(x))


def test_header_holds_the_python_coefficients():
    text = (FIRMWARE / "esp32_emg_streamer" / "mb_decimator.h").read_text()
    body = re.search(r"HALFBAND_Q15_HALF\[12\] = \{([^}]*)\}", text)
    assert body is not None
    half = [int(v) for v in body[1].replace("\n", " ").split(",")]
    assert half == HALFBAND_Q15[:23:2].tolist()
    assert HALFBAND_Q15[23] == 16384


def test_sketch_uses_the_shared_headers():
    sketch = (FIRMWARE / "esp32_emg_streamer" / "esp32_emg_streamer.ino").read_text()
    assert '#include "mb_protocol.h"' in sketch and '#include "mb_decimator.h"' in sketch
    assert "crc16CcittFalse(const" not in sketch  # no second copy of the CRC
