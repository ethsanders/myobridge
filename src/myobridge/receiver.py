"""UDP receiver: datagrams in, validated packets out, with loss bookkeeping.

A plain blocking socket with a timeout, no threads or asyncio: the caller just
loops `for packet in receiver:`. The timeout means Ctrl-C is noticed promptly
and the caller gets a chance to redraw its display even when nothing arrives.
"""

from __future__ import annotations

import socket
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .protocol import LinkStats, Packet, ProtocolError, SequenceTracker, decode_packet

MAX_DATAGRAM = 65535


@dataclass
class ArrivalLog:
    """When each packet reached this computer, next to when the device sampled it.

    `Receiver.receive` adds every usable packet it reads while
    `Receiver.arrivals` is set; `Receiver.drain` doesn't, because packets
    that waited in the socket buffer come out in a burst that says nothing
    about the link. The two clocks aren't synchronised, so this measures how
    much the delay *varies* (jitter), not the delay itself.
    """

    max_packets: int = 1_000_000  # about 5.5 hours at 50 packets/s
    host_s: list[float] = field(default_factory=list)  # time.monotonic() on arrival
    device_us: list[int] = field(default_factory=list)  # the packet's timestamp_us
    duration_us: list[int] = field(default_factory=list)  # n_frames / sample_rate

    def add(self, host_s: float, packet: Packet) -> None:
        """Log one packet (ignored once `max_packets` are logged)."""
        if len(self.host_s) < self.max_packets:
            self.host_s.append(host_s)
            self.device_us.append(packet.timestamp_us)
            self.duration_us.append(packet.n_frames * 1_000_000 // packet.sample_rate)

    def summary(self) -> dict[str, Any]:
        """Jitter in milliseconds; only `packets_timed` with fewer than 10 packets.

        - `interval_error_ms`: arrival gap minus sampling gap, for packets that
          follow each other on the device with none missing in between.
        - `extra_delay_ms`: each packet's delay beyond the quickest packet's,
          after taking out the slow drift between the two clocks.
        """
        n = len(self.host_s)
        out: dict[str, Any] = {"packets_timed": n}
        if n < 10:
            return out
        host = np.asarray(self.host_s, dtype=np.float64)
        dev = np.asarray(self.device_us, dtype=np.float64) / 1e6
        step = np.asarray(self.duration_us[:-1], dtype=np.float64) / 1e6
        adjacent = np.isclose(np.diff(dev), step, rtol=0.0, atol=2e-4)
        err_ms = (np.diff(host) - np.diff(dev))[adjacent] * 1e3
        if len(err_ms):
            out["interval_error_ms"] = {
                "std": float(np.std(err_ms)),
                "p95_abs": float(np.percentile(np.abs(err_ms), 95)),
                "max_abs": float(np.max(np.abs(err_ms))),
                "pairs": len(err_ms),
            }
        # The delay is a floor (the quickest a packet can arrive) plus waiting.
        # The floor drifts with the two clocks, so fit a line through the
        # quickest packet of each of a few stretches of the recording.
        t, delay = dev - dev[0], host - dev
        quickest = [i[np.argmin(delay[i])] for i in np.array_split(np.arange(n), min(20, n // 5))]
        slope, offset = np.polyfit(t[quickest], delay[quickest], 1)
        extra = (delay - (offset + slope * t)) * 1e3
        extra -= extra.min()
        out["extra_delay_ms"] = {
            "p50": float(np.percentile(extra, 50)),
            "p95": float(np.percentile(extra, 95)),
            "p99": float(np.percentile(extra, 99)),
            "max": float(extra.max()),
        }
        out["clock_drift_ppm"] = float(slope * 1e6)
        return out


class Receiver:
    """Receives myobridge packets on one UDP port.

    Not thread-safe: use one Receiver from one thread. The socket, the
    sequence tracker and the stats are updated without locks.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5005, timeout_s: float = 0.5):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # No SO_REUSEADDR: UDP has no TIME_WAIT for it to help with, and with
            # it a second listener (another `myobridge listen`, say) can bind the
            # same port on Linux and macOS and silently take the packets. Without
            # it the second bind fails with "address in use". Windows lets
            # another socket take over the port anyway unless we claim it with
            # SO_EXCLUSIVEADDRUSE.
            if sys.platform == "win32":
                self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            self.sock.bind((host, port))
            self.sock.settimeout(timeout_s)
        except BaseException:
            self.sock.close()
            raise
        self.tracker = SequenceTracker()
        self.last_error: str | None = None
        self.invalid: dict[str, int] = {}  # invalid datagrams by ProtocolError.kind
        self.arrivals: ArrivalLog | None = None  # set one to time packets (jitter)

    @property
    def stats(self) -> LinkStats:
        return self.tracker.stats

    @property
    def port(self) -> int:
        """Actual bound port (useful when you bind to port 0 = "pick any free one")."""
        return int(self.sock.getsockname()[1])

    def handle_datagram(self, data: bytes) -> Packet | None:
        """Validate one datagram. Returns the packet, or None if it must be ignored."""
        try:
            packet = decode_packet(data)
        except ProtocolError as err:
            self.stats.corrupt += 1
            self.invalid[err.kind] = self.invalid.get(err.kind, 0) + 1
            self.last_error = f"{err.kind}: {err}"
            return None
        return packet if self.tracker.accept(packet.seq, packet.timestamp_us) else None

    def receive(self) -> Packet | None:
        """Wait up to the timeout for one usable packet. None on timeout/bad packet."""
        try:
            data, _addr = self.sock.recvfrom(MAX_DATAGRAM)
        except TimeoutError:  # socket.timeout is an alias of this since 3.10
            return None
        arrived = time.monotonic()
        packet = self.handle_datagram(data)
        if packet is not None and self.arrivals is not None:
            self.arrivals.add(arrived, packet)
        return packet

    def link_summary(self) -> dict[str, Any]:
        """The link stats so far as plain JSON: the counts, the loss, invalid
        datagrams by kind (`crc`, `truncated`, ...) and, while `arrivals` is
        set, the jitter (`ArrivalLog.summary`)."""
        out: dict[str, Any] = asdict(self.stats)
        out["loss_percent"] = self.stats.loss_percent
        out["invalid_by_kind"] = dict(sorted(self.invalid.items()))
        if self.arrivals is not None:
            out["timing"] = self.arrivals.summary()
        return out

    def __iter__(self) -> Iterator[Packet | None]:
        """Yield packets forever; yields None on timeouts so callers can do housekeeping."""
        while True:
            yield self.receive()

    def drain(self) -> int:
        """Discard everything already queued in the socket buffer, without waiting.

        The packets still go through the sequence tracker, so the stats stay
        right. Returns how many datagrams were discarded.
        """
        timeout = self.sock.gettimeout()
        self.sock.setblocking(False)
        count = 0
        try:
            while True:
                try:
                    data, _addr = self.sock.recvfrom(MAX_DATAGRAM)
                except BlockingIOError:
                    return count
                self.handle_datagram(data)
                count += 1
        finally:
            self.sock.settimeout(timeout)

    def record(
        self, seconds: float, fs: int | None = None, *, timeout_s: float | None = 5.0
    ) -> np.ndarray:
        """Collect `seconds` of samples (used for live calibration).

        `fs` defaults to the sample rate the packets report. Raises
        TimeoutError if no usable packet arrives for `timeout_s` seconds
        (None waits forever), and ValueError if the channel count changes.
        """
        blocks: list[np.ndarray] = []
        total, needed = 0, None if fs is None else seconds * fs
        last_packet = time.monotonic()
        while needed is None or total < needed:
            packet = self.receive()
            if packet is None:
                if timeout_s is not None and time.monotonic() - last_packet > timeout_s:
                    raise TimeoutError(self._no_data_message(timeout_s))
                continue
            last_packet = time.monotonic()
            if needed is None:
                needed = seconds * packet.sample_rate
            if blocks and packet.n_channels != blocks[0].shape[1]:
                raise ValueError(
                    f"channel count changed from {blocks[0].shape[1]} to "
                    f"{packet.n_channels} during a recording"
                )
            blocks.append(packet.samples)
            total += packet.n_frames
        return np.vstack(blocks)

    def _no_data_message(self, timeout_s: float) -> str:
        message = f"no usable packets on UDP port {self.port} for {timeout_s:g} s"
        if self.last_error is not None:
            message += f" ({self.stats.corrupt} invalid, last: {self.last_error})"
        return message

    def close(self) -> None:
        self.sock.close()

    def __enter__(self) -> Receiver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
