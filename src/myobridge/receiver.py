"""UDP receiver: datagrams in, validated packets out, with loss bookkeeping.

A plain blocking socket with a timeout, no threads or asyncio: the caller just
loops `for packet in receiver:`. The timeout means Ctrl-C is noticed promptly
and the caller gets a chance to redraw its display even when nothing arrives.
"""

from __future__ import annotations

import socket
from collections.abc import Iterator

import numpy as np

from .protocol import LinkStats, Packet, ProtocolError, SequenceTracker, decode_packet

MAX_DATAGRAM = 65535


class Receiver:
    def __init__(self, host: str = "0.0.0.0", port: int = 5005, timeout_s: float = 0.5):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.settimeout(timeout_s)
        self.tracker = SequenceTracker()
        self.last_error: str | None = None

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
            self.last_error = f"{err.kind}: {err}"
            return None
        return packet if self.tracker.accept(packet.seq) else None

    def receive(self) -> Packet | None:
        """Wait up to the timeout for one usable packet. None on timeout/bad packet."""
        try:
            data, _addr = self.sock.recvfrom(MAX_DATAGRAM)
        except TimeoutError:  # socket.timeout is an alias of this since 3.10
            return None
        return self.handle_datagram(data)

    def __iter__(self) -> Iterator[Packet | None]:
        """Yield packets forever; yields None on timeouts so callers can do housekeeping."""
        while True:
            yield self.receive()

    def record(self, seconds: float, fs: int) -> np.ndarray:
        """Collect `seconds` of samples (used for live calibration)."""
        blocks, total = [], 0
        while total < seconds * fs:
            packet = self.receive()
            if packet is not None:
                blocks.append(packet.samples)
                total += packet.n_frames
        return np.vstack(blocks)

    def close(self) -> None:
        self.sock.close()

    def __enter__(self) -> Receiver:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
