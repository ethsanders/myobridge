import numpy as np
import pytest

from myobridge.protocol import (
    HEADER,
    ProtocolError,
    SequenceTracker,
    crc16_ccitt_false,
    decode_packet,
    encode_packet,
)


def make_samples(frames=20, channels=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(-32768, 32767, size=(frames, channels), dtype=np.int16)


def test_crc_known_answer():
    # The standard check value for CRC-16/CCITT-FALSE.
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_crc_empty_is_init_value():
    assert crc16_ccitt_false(b"") == 0xFFFF


def test_round_trip():
    samples = make_samples()
    data = encode_packet(samples, seq=42, timestamp_us=123_456_789_000, sample_rate=1000)
    assert len(data) == HEADER.size + samples.size * 2 + 2
    pkt = decode_packet(data)
    assert pkt.seq == 42
    assert pkt.timestamp_us == 123_456_789_000
    assert pkt.sample_rate == 1000
    assert (pkt.n_frames, pkt.n_channels) == (20, 4)
    np.testing.assert_array_equal(pkt.samples, samples)


def test_header_layout_is_little_endian():
    data = encode_packet(
        np.array([[1, -2]], dtype=np.int16), seq=0x01020304, timestamp_us=5, sample_rate=1000
    )
    assert data[:2] == b"MB"
    assert data[6:10] == bytes([4, 3, 2, 1])  # seq, least significant byte first
    assert data[20:24] == bytes([1, 0, 0xFE, 0xFF])  # 1 and -2 as int16 LE


def test_seq_wraps_when_encoding():
    pkt = decode_packet(encode_packet(make_samples(), 2**32 + 5, 0, 1000))
    assert pkt.seq == 5


@pytest.mark.parametrize("bad_dtype", [np.int32, np.float32])
def test_encode_rejects_wrong_dtype(bad_dtype):
    with pytest.raises(ValueError):
        encode_packet(np.zeros((2, 2), dtype=bad_dtype), 0, 0, 1000)


def test_bad_magic_rejected():
    data = bytearray(encode_packet(make_samples(), 0, 0, 1000))
    data[0:2] = b"XX"
    with pytest.raises(ProtocolError) as err:
        decode_packet(bytes(data))
    assert err.value.kind == "magic"


def test_bad_version_rejected():
    data = bytearray(encode_packet(make_samples(), 0, 0, 1000))
    data[2] = 99
    with pytest.raises(ProtocolError) as err:
        decode_packet(bytes(data))
    assert err.value.kind == "version"


@pytest.mark.parametrize("position", [3, 10, 25, -1])
def test_any_bit_flip_fails_crc_or_length(position):
    data = bytearray(encode_packet(make_samples(), 7, 0, 1000))
    data[position] ^= 0x10
    with pytest.raises(ProtocolError):
        decode_packet(bytes(data))


def test_payload_flip_is_crc_error():
    data = bytearray(encode_packet(make_samples(), 7, 0, 1000))
    data[30] ^= 0x01
    with pytest.raises(ProtocolError) as err:
        decode_packet(bytes(data))
    assert err.value.kind == "crc"


@pytest.mark.parametrize("cut", [0, 5, 21, 50, 1])
def test_truncated_rejected(cut):
    data = encode_packet(make_samples(), 0, 0, 1000)
    with pytest.raises(ProtocolError) as err:
        decode_packet(data[: len(data) - cut] if cut else data[:10])
    assert err.value.kind == "truncated"


def test_tracker_in_order():
    t = SequenceTracker()
    assert all(t.accept(s) for s in range(100))
    assert (t.stats.received, t.stats.dropped, t.stats.reordered) == (100, 0, 0)


def test_tracker_gap_counts_drops():
    t = SequenceTracker()
    for s in [0, 1, 2, 6, 7]:
        t.accept(s)
    assert t.stats.dropped == 3  # 3, 4, 5 missing
    assert t.stats.received == 5
    assert t.stats.loss_percent == pytest.approx(100 * 3 / 8)


def test_tracker_late_packet_is_reordered_not_dropped():
    t = SequenceTracker()
    for s in [0, 1, 3]:
        assert t.accept(s)
    assert t.stats.dropped == 1
    assert not t.accept(2)  # arrives late: counted, but discarded
    assert (t.stats.dropped, t.stats.reordered) == (0, 1)
    assert t.accept(4)


def test_tracker_wraparound_is_not_a_gap():
    t = SequenceTracker()
    for s in [2**32 - 2, 2**32 - 1, 0, 1]:
        assert t.accept(s)
    assert (t.stats.dropped, t.stats.reordered) == (0, 0)


def test_tracker_gap_across_wraparound():
    t = SequenceTracker()
    t.accept(2**32 - 2)
    t.accept(1)  # skipped 2**32-1 and 0
    assert t.stats.dropped == 2


def test_tracker_late_packet_across_wraparound():
    t = SequenceTracker()
    t.accept(2**32 - 2)
    t.accept(0)  # 2**32-1 missing
    assert not t.accept(2**32 - 1)
    assert (t.stats.dropped, t.stats.reordered) == (0, 1)


def test_tracker_device_restart():
    t = SequenceTracker()
    for s in range(5000, 5010):
        t.accept(s)
    assert t.accept(0)  # device rebooted
    assert t.stats.restarts == 1
    assert t.stats.dropped == 0
    assert t.accept(1)
