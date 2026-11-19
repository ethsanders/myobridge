import socket
import threading

import numpy as np

from myobridge.protocol import encode_packet
from myobridge.receiver import Receiver
from myobridge.simulator import EmgSimulator, run_udp


def pkt(seq):
    return encode_packet(np.full((20, 4), seq, dtype=np.int16), seq, seq * 20_000, 1000)


def test_receiver_over_localhost_udp():
    with Receiver(host="127.0.0.1", port=0, timeout_s=1.0) as rx:
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        for datagram in [pkt(0), pkt(1), b"garbage!" * 5, pkt(3), pkt(2), pkt(4)]:
            tx.sendto(datagram, ("127.0.0.1", rx.port))
        tx.close()
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
        )
        sender.start()
        data = rx.record(0.3, fs=1000)
        sender.join()
    assert data.shape == (300, 4)
    assert rx.stats.dropped == 0
