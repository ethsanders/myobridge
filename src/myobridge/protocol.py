"""Binary wire protocol shared by the ESP32 firmware and the Python receiver.

One UDP datagram = one packet. Everything is little-endian. See docs/PROTOCOL.md.

    offset  size  field
    0       2     magic          b"MB"
    2       1     version        1
    3       1     n_channels     e.g. 4
    4       2     sample_rate    Hz, e.g. 1000
    6       4     seq            packet counter, wraps at 2**32
    10      8     timestamp_us   device time of the first sample (microseconds)
    18      2     n_frames       samples per channel in this packet
    20      ...   samples        int16, interleaved: f0c0 f0c1 .. f0cN f1c0 ...
    end-2   2     crc16          CRC-16/CCITT-FALSE over every byte before it
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"MB"
VERSION = 1
HEADER = struct.Struct("<2sBBHIQH")  # 20 bytes, matches the table above
CRC_SIZE = 2
SEQ_MODULO = 2**32


class ProtocolError(ValueError):
    """Raised when a datagram is not a valid packet. `kind` is used for stats."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind  # "truncated", "magic", "version" or "crc"


# --------------------------------------------------------------------------- CRC


def _make_crc_table() -> list[int]:
    """Precompute the CRC of every possible byte (the standard table trick).

    The firmware uses the plain bit-by-bit loop instead; both give identical
    results, the table is just ~8x faster in Python.
    """
    table = []
    for byte in range(256):
        crc = byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
        table.append(crc & 0xFFFF)
    return table


_CRC_TABLE = _make_crc_table()


def crc16_ccitt_false(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection, no final XOR."""
    crc = 0xFFFF
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[(crc >> 8) ^ byte]
    return crc


# ----------------------------------------------------------------- encode/decode


@dataclass(frozen=True)
class Packet:
    seq: int
    timestamp_us: int
    sample_rate: int
    samples: np.ndarray  # shape (n_frames, n_channels), dtype int16

    @property
    def n_channels(self) -> int:
        return int(self.samples.shape[1])

    @property
    def n_frames(self) -> int:
        return int(self.samples.shape[0])


def encode_packet(samples: np.ndarray, seq: int, timestamp_us: int, sample_rate: int) -> bytes:
    """Build one packet from an (n_frames, n_channels) int16 array."""
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.dtype != np.int16:
        raise ValueError("samples must be a 2-D int16 array (n_frames, n_channels)")
    n_frames, n_channels = samples.shape
    header = HEADER.pack(
        MAGIC, VERSION, n_channels, sample_rate, seq % SEQ_MODULO, timestamp_us, n_frames
    )
    # "<i2" forces little-endian int16 regardless of the host CPU.
    body = header + samples.astype("<i2").tobytes()
    return body + struct.pack("<H", crc16_ccitt_false(body))


def decode_packet(data: bytes) -> Packet:
    """Parse and validate one datagram. Raises ProtocolError if anything is off."""
    if len(data) < HEADER.size + CRC_SIZE:
        raise ProtocolError("truncated", f"packet too short ({len(data)} bytes)")
    magic, version, n_channels, sample_rate, seq, ts, n_frames = HEADER.unpack_from(data)
    if magic != MAGIC:
        raise ProtocolError("magic", f"bad magic {magic!r}")
    if version != VERSION:
        raise ProtocolError("version", f"unsupported version {version}")
    expected_len = HEADER.size + 2 * n_channels * n_frames + CRC_SIZE
    if len(data) != expected_len:
        raise ProtocolError("truncated", f"length {len(data)} != expected {expected_len}")
    (crc_received,) = struct.unpack_from("<H", data, expected_len - CRC_SIZE)
    if crc16_ccitt_false(data[: expected_len - CRC_SIZE]) != crc_received:
        raise ProtocolError("crc", "CRC mismatch")
    flat = np.frombuffer(data, dtype="<i2", count=n_channels * n_frames, offset=HEADER.size)
    samples = flat.reshape(n_frames, n_channels).astype(np.int16)
    return Packet(seq=seq, timestamp_us=ts, sample_rate=sample_rate, samples=samples)


# ------------------------------------------------------------- loss detection


@dataclass
class LinkStats:
    received: int = 0  # valid packets accepted in order
    dropped: int = 0  # packets that never arrived (gaps in seq)
    corrupt: int = 0  # bad magic / CRC / length / version
    reordered: int = 0  # arrived late (or duplicated) and were discarded
    restarts: int = 0  # device rebooted (seq jumped far backwards)

    @property
    def loss_percent(self) -> float:
        total = self.received + self.dropped
        return 100.0 * self.dropped / total if total else 0.0


@dataclass
class SequenceTracker:
    """Detects gaps and out-of-order packets using the 32-bit sequence number.

    All arithmetic is modulo 2**32 so that seq 0 right after 4294967295 is just
    "the next packet", not a jump backwards.
    """

    restart_window: int = 1000  # a jump back further than this = device reboot
    stats: LinkStats = field(default_factory=LinkStats)
    expected: int | None = None

    def accept(self, seq: int) -> bool:
        """Update stats for `seq`. Returns True if the packet should be used."""
        if self.expected is None:  # very first packet
            self.expected = (seq + 1) % SEQ_MODULO
            self.stats.received += 1
            return True

        ahead = (seq - self.expected) % SEQ_MODULO  # how far ahead of what we wanted
        if ahead < SEQ_MODULO // 2:
            # On time (ahead == 0) or after a gap: everything in between was lost.
            self.stats.dropped += ahead
            self.stats.received += 1
            self.expected = (seq + 1) % SEQ_MODULO
            return True

        behind = SEQ_MODULO - ahead  # 1 means "the packet just before expected"
        if behind > self.restart_window:
            # Too far back to be a late packet: the device restarted at seq 0.
            self.stats.restarts += 1
            self.stats.received += 1
            self.expected = (seq + 1) % SEQ_MODULO
            return True

        # A late packet. We already counted it as dropped when we saw the gap,
        # so undo that. Streaming filters can't go back in time: discard it.
        self.stats.reordered += 1
        self.stats.dropped = max(0, self.stats.dropped - 1)
        return False
