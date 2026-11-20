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

import numpy as np

from .protocol import LinkStats, Packet, ProtocolError, SequenceTracker, decode_packet

MAX_DATAGRAM = 65535


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
        return packet if self.tracker.accept(packet.seq, packet.timestamp_us) else None

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
