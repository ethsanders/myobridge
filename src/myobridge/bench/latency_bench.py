"""How long does the live pipeline take per packet, on this computer?

Feeds 20-sample packets (what the firmware sends: 4 channels, 1 kHz, 50
packets per second) through GesturePipeline and times each call. The model
is an LDA fitted to random features, which costs exactly as much as a real one.
Needs no dataset. Timing numbers depend on the machine; the JSON records it.
"""

from __future__ import annotations

import platform
import time
from typing import Any

import numpy as np

from ..classifier import LDAClassifier, adapt
from ..features import extract_features
from ..pipeline import GesturePipeline
from .common import RunOptions, provenance, write_result

FS = 1000
N_CHANNELS = 4
PACKET = 20
WINDOW_S = 0.2
STEP_S = 0.05
VOTE_K = 5


def run(opts: RunOptions) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    n_classes = 6
    X = rng.standard_normal((600, 5 * N_CHANNELS))
    y = np.repeat(np.arange(n_classes), 100)
    model = LDAClassifier().fit(X, y)
    pipe = GesturePipeline(model, FS, N_CHANNELS, WINDOW_S, STEP_S, VOTE_K)
    n_packets = 500 if opts.quick else 15000
    data = rng.normal(0, 50, (n_packets, PACKET, N_CHANNELS))
    for block in data[:50]:  # warm-up
        pipe.process_block(block)
    times = np.empty(n_packets - 50)
    for i, block in enumerate(data[50:]):
        t0 = time.perf_counter()
        pipe.process_block(block)
        times[i] = time.perf_counter() - t0
    us = times * 1e6
    budget_us = PACKET / FS * 1e6

    # batch vs one-at-a-time features, the speed-up the benchmarks rely on
    windows = rng.standard_normal((2000, int(WINDOW_S * FS), N_CHANNELS))
    t0 = time.perf_counter()
    batch = extract_features(windows)
    t_batch = time.perf_counter() - t0
    t0 = time.perf_counter()
    loop = np.array([extract_features(w) for w in windows])
    t_loop = time.perf_counter() - t0

    # recalibration with adapt(): 11 gestures x 1 s of new windows, a 20-feature model
    Xb = rng.standard_normal((11 * 200, 5 * N_CHANNELS))
    yb = np.repeat(np.arange(11), 200)
    base = LDAClassifier().fit(Xb, yb)
    Xn, yn = Xb[::12] + 0.5, yb[::12]
    reps = 50 if opts.quick else 500
    t0 = time.perf_counter()
    for _ in range(reps):
        adapt(base, Xn, yn, mean_prior=20.0, covariance_weight=0.75)
    adapt_ms = (time.perf_counter() - t0) / reps * 1e3

    payload = {
        "provenance": provenance(
            None,
            {
                "fs": FS,
                "channels": N_CHANNELS,
                "packet_samples": PACKET,
                "window_s": WINDOW_S,
                "step_s": STEP_S,
                "vote_k": VOTE_K,
                "machine": platform.processor() or platform.machine(),
            },
        ),
        "per_packet_us": {
            "median": float(np.median(us)),
            "p99": float(np.percentile(us, 99)),
            "max": float(us.max()),
            "n": int(len(us)),
        },
        "packet_period_us": budget_us,
        "cpu_share_of_real_time": float(us.mean() / budget_us),
        "decision_delay_estimate_ms": {
            "packet_buffering": PACKET / FS * 1e3,
            "window_centre": WINDOW_S / 2 * 1e3,
            "majority_vote": (VOTE_K // 2) * STEP_S * 1e3,
            "note": "how long after a gesture starts the smoothed output can switch, "
            "ignoring network and muscle onset; computed from the settings, not measured",
        },
        "adapt_ms": {"mean": adapt_ms, "new_windows": len(yn), "classes": 11},
        "features_batch_vs_loop": {
            "windows": len(windows),
            "batch_s": t_batch,
            "loop_s": t_loop,
            "identical": bool(np.allclose(batch, loop)),
        },
    }
    print(f"  wrote {write_result(opts, 'latency', payload)}")
    return payload
