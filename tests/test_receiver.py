import gc
import socket
import threading
import time

import numpy as np
import pytest

from myobridge.protocol import encode_packet
from myobridge.receiver import Receiver
from myobridge.simulator import EmgSimulator, run_udp


def pkt(seq, channels=4, fs=1000):
    return encode_packet(np.full((20, channels), seq, dtype=np.int16), seq, seq * 20_000, fs)


def send(port, datagrams):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
        for datagram in datagrams:
            tx.sendto(datagram, ("127.0.0.1", port))


def test_receiver_over_localhost_udp():
    with Receiver(host="127.0.0.1", port=0, timeout_s=1.0) as rx:
        send(rx.port, [pkt(0), pkt(1), b"garbage!" * 5, pkt(3), pkt(2), pkt(4)])
        got = [rx.receive() for _ in range(6)]
        seqs = [p.seq for p in got if p is not None]
        assert seqs == [0, 1, 3, 4]  # garbage rejected, late seq 2 discarded
        s = rx.stats
        assert (s.received, s.corrupt, s.dropped, s.reordered) == (4, 1, 0, 1)
        assert rx.last_error.startswith("magic")


def test_receiver_timeout_returns_none():
    with Receiver(host="127.0.0.1", port=0, timeout_s=0.05) as rx:
        assert rx.receive() is None


def test_simulator_to_receiver_end_to_end():
    with Receiver(host="127.0.0.1", port=0, timeout_s=2.0) as rx:
        sim = EmgSimulator(seed=0)
        sender = threading.Thread(
            target=run_udp,
            args=("127.0.0.1", rx.port, sim),
            kwargs={"duration_s": 0.3, "verbose": False},
            daemon=True,
        )
        sender.start()
        data = rx.record(0.3, fs=1000, timeout_s=5.0)
        sender.join(timeout=5.0)
        assert not sender.is_alive()
    assert data.shape == (300, 4)
    assert rx.stats.dropped == 0


def test_record_gives_up_when_nothing_arrives():
    with Receiver(host="127.0.0.1", port=0, timeout_s=0.05) as rx:
        start = time.monotonic()
        with pytest.raises(TimeoutError, match=f"port {rx.port}"):
            rx.record(1.0, fs=1000, timeout_s=0.3)
        assert time.monotonic() - start < 2.0


def test_record_timeout_mentions_invalid_packets():
    with Receiver(host="127.0.0.1", port=0, timeout_s=0.05) as rx:
        send(rx.port, [b"garbage!" * 5])
        with pytest.raises(TimeoutError, match="1 invalid, last: magic"):
            rx.record(1.0, fs=1000, timeout_s=0.3)


def test_record_reads_the_sample_rate_from_the_packets():
    with Receiver(host="127.0.0.1", port=0, timeout_s=0.5) as rx:
        send(rx.port, [pkt(s, fs=500) for s in range(10)])
        data = rx.record(0.2, timeout_s=2.0)  # 0.2 s at 500 Hz = 100 samples = 5 packets
    assert data.shape == (100, 4)


def test_record_rejects_a_channel_change():
    with Receiver(host="127.0.0.1", port=0, timeout_s=0.5) as rx:
        send(rx.port, [pkt(0), pkt(1, channels=3)])
        with pytest.raises(ValueError, match="channel count changed from 4 to 3"):
            rx.record(1.0, fs=1000, timeout_s=2.0)


def test_drain_discards_queued_packets_but_counts_them():
    with Receiver(host="127.0.0.1", port=0, timeout_s=0.5) as rx:
        send(rx.port, [pkt(s) for s in range(100)])  # queued while nobody reads
        time.sleep(0.1)  # let loopback deliver them all
        assert rx.drain() == 100
        assert rx.stats.received == 100
        assert rx.sock.gettimeout() == 0.5  # back to blocking with a timeout
        send(rx.port, [pkt(s) for s in range(100, 110)])
        data = rx.record(0.2, fs=1000, timeout_s=2.0)
    assert data[0, 0] == 100  # the first packet after the drain, not a stale one
    assert rx.stats.dropped == 0


def test_drain_with_nothing_queued():
    with Receiver(host="127.0.0.1", port=0, timeout_s=0.5) as rx:
        assert rx.drain() == 0
        assert rx.sock.gettimeout() == 0.5


def test_second_receiver_on_the_same_port_is_refused():
    with Receiver(host="127.0.0.1", port=0) as rx, pytest.raises(OSError):
        Receiver(host="127.0.0.1", port=rx.port)


def test_failed_bind_closes_the_socket(recwarn):
    with Receiver(host="127.0.0.1", port=0) as rx:
        with pytest.raises(OSError):
            Receiver(host="127.0.0.1", port=rx.port)
        with pytest.raises(OSError):
            Receiver(host="not-a-host.invalid", port=0)
        gc.collect()
    assert not [w for w in recwarn if issubclass(w.category, ResourceWarning)]
