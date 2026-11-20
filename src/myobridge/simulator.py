"""A fake ESP32 + MyoWare armband, so everything runs without hardware.

Surface EMG is, to a first approximation, band-limited random noise whose
amplitude follows how hard the muscle under the electrode is working. So each
channel here is: white noise -> bandpass (EMG-like spectrum, ~20-250 Hz) ->
multiplied by a smooth activation level that depends on the current gesture.
On top we add the junk a real recording has: 60 Hz mains hum, slow baseline
drift (electrode/skin potential wandering), and ADC noise.

This is NOT real EMG. It's realistic enough to exercise the pipeline and the
classifier, but real data will be messier (see docs/HOW_IT_WORKS.md).
"""

from __future__ import annotations

import socket
import time
from collections.abc import Iterator

import numpy as np
from scipy import signal

from .protocol import encode_packet

# How strongly each gesture activates each of the 4 forearm sites (0..1).
# Think: ch0 = wrist flexors, ch1 = wrist extensors, ch2 = finger extensors,
# ch3 = thumb / brachioradialis. Made-up but plausible-looking patterns.
GESTURE_PATTERNS: dict[str, list[float]] = {
    "rest": [0.02, 0.02, 0.02, 0.02],
    "fist": [0.90, 0.80, 0.35, 0.60],
    "open": [0.20, 0.45, 0.90, 0.35],
    "flex": [0.95, 0.15, 0.20, 0.30],
    "extend": [0.15, 0.90, 0.40, 0.20],
}
GESTURES = tuple(GESTURE_PATTERNS)


class EmgSimulator:
    """Generates int16 "ADC counts" block by block, with continuous state."""

    def __init__(
        self,
        n_channels: int = 4,
        fs: int = 1000,
        seed: int = 0,
        mains_hz: float = 60.0,
        max_amplitude: float = 500.0,  # ADC counts at full contraction
        hum_amplitude: float = 60.0,
        noise_std: float = 4.0,
        variability: float = 0.3,  # how much effort wanders (std of log-gain)
    ):
        self.n_channels, self.fs = n_channels, fs
        self.rng = np.random.default_rng(seed)
        self.mains_hz, self.max_amplitude = mains_hz, max_amplitude
        self.hum_amplitude, self.noise_std, self.variability = hum_amplitude, noise_std, variability
        self.gesture = "rest"
        self.t = 0  # samples generated so far

        # Shapes white noise into an EMG-like spectrum (stateful, like dsp.py).
        self._shape_sos = signal.butter(4, [20, 250], btype="bandpass", fs=fs, output="sos")
        self._shape_zi = np.zeros((self._shape_sos.shape[0], 2, n_channels))
        # Activation follows the target with a ~60 ms time constant (muscles
        # don't switch on instantly). One-pole lowpass: y += a * (target - y).
        self._smooth_a = 1.0 - np.exp(-1.0 / (0.06 * fs))
        self._activation = np.zeros(n_channels)
        self._log_effort = np.zeros(n_channels)  # slowly wandering gain per channel
        self._drift_phase = self.rng.uniform(0, 2 * np.pi, n_channels)
        self._hum_phase = self.rng.uniform(0, 2 * np.pi)

    def set_gesture(self, gesture: str) -> None:
        if gesture not in GESTURE_PATTERNS:
            raise ValueError(f"unknown gesture {gesture!r}; choose from {GESTURES}")
        self.gesture = gesture

    def generate(self, n: int) -> np.ndarray:
        """Return the next `n` frames as an (n, n_channels) int16 array."""
        pattern = np.array(GESTURE_PATTERNS[self.gesture][: self.n_channels])
        # Pad if someone asks for more channels than the pattern table has.
        pattern = np.resize(pattern, self.n_channels)
        # Nobody holds a contraction perfectly steady: each channel's effort
        # wanders as a mean-reverting random walk (~1 s memory, std = variability).
        decay = np.exp(-1.0 / (1.0 * self.fs))
        kicks = self.rng.standard_normal((n, self.n_channels))
        kicks *= self.variability * np.sqrt(1 - decay**2)

        activation = np.empty((n, self.n_channels))
        for i in range(n):  # per-sample loop: slow-ish but obvious
            self._log_effort = decay * self._log_effort + kicks[i]
            target = pattern * np.exp(self._log_effort)
            self._activation += self._smooth_a * (target - self._activation)
            activation[i] = self._activation

        white = self.rng.standard_normal((n, self.n_channels))
        emg, self._shape_zi = signal.sosfilt(self._shape_sos, white, axis=0, zi=self._shape_zi)
        emg *= activation * self.max_amplitude * 2.0  # x2: shaped noise has std ~0.5

        t = (self.t + np.arange(n))[:, None] / self.fs
        hum = self.hum_amplitude * np.sin(2 * np.pi * self.mains_hz * t + self._hum_phase)
        drift = 80.0 * np.sin(2 * np.pi * 0.3 * t + self._drift_phase)
        noise = self.rng.normal(0.0, self.noise_std, (n, self.n_channels))
        self.t += n

        total = emg + hum + drift + noise
        out: np.ndarray = np.clip(np.round(total), -32768, 32767).astype(np.int16)
        return out

    def record(self, gesture: str, seconds: float) -> np.ndarray:
        """Hold one gesture for `seconds` and return the samples (calibration helper)."""
        self.set_gesture(gesture)
        return self.generate(int(seconds * self.fs))


def gesture_session(
    sim: EmgSimulator, gestures: tuple[str, ...], hold_s: float, repeats: int
) -> tuple[np.ndarray, np.ndarray]:
    """A continuous recording cycling through gestures, plus a per-sample label array."""
    chunks, labels = [], []
    for _ in range(repeats):
        for g in gestures:
            chunks.append(sim.record(g, hold_s))
            labels += [g] * chunks[-1].shape[0]
    return np.vstack(chunks), np.array(labels)


def packetize(
    sim: EmgSimulator, frames_per_packet: int = 20, drop_prob: float = 0.0
) -> Iterator[bytes]:
    """Endless stream of encoded packets, like the firmware would send.

    With drop_prob > 0 some packets are silently skipped but the sequence
    number still advances, exactly like a lost UDP datagram.
    """
    seq = 0
    while True:
        timestamp_us = sim.t * 1_000_000 // sim.fs
        block = sim.generate(frames_per_packet)
        if sim.rng.random() >= drop_prob:
            yield encode_packet(block, seq, timestamp_us, sim.fs)
        seq += 1


def run_udp(
    host: str,
    port: int,
    sim: EmgSimulator,
    hold_s: float = 3.0,
    frames_per_packet: int = 20,
    drop_prob: float = 0.0,
    duration_s: float | None = None,
    verbose: bool = True,
) -> int:
    """Send packets to host:port in real time, cycling gestures every `hold_s`.

    Returns the number of samples generated per channel.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    stream = packetize(sim, frames_per_packet, drop_prob)
    start = time.perf_counter()
    try:
        while duration_s is None or sim.t < duration_s * sim.fs:
            wanted = GESTURES[int(sim.t / sim.fs / hold_s) % len(GESTURES)]
            if wanted != sim.gesture or sim.t == 0:
                sim.set_gesture(wanted)
                if verbose:
                    print(f"[sim] gesture -> {wanted}", flush=True)
            sock.sendto(next(stream), (host, port))
            # Pace to real time: sleep until the device would have sampled this much.
            delay = start + sim.t / sim.fs - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    finally:
        sock.close()
    return sim.t
