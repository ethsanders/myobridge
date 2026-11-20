import json
import queue
import socket
import threading
import time
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

import myobridge
import myobridge.cli as cli
from myobridge.cli import build_parser, main
from myobridge.protocol import encode_packet
from myobridge.simulator import GESTURES, EmgSimulator, run_udp


class Background:
    """Runs main(argv) in a daemon thread. join() fails the test if it hangs and
    re-raises whatever the command raised, so a failure can't go unnoticed."""

    def __init__(self, argv):
        self.error = None
        self.thread = threading.Thread(target=self._run, args=(argv,), daemon=True)
        self.thread.start()

    def _run(self, argv):
        try:
            main(argv)
        except BaseException as err:  # includes SystemExit from a CLI error
            self.error = err

    def join(self, timeout=20.0):
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "command did not finish"
        if self.error is not None:
            raise self.error


@pytest.fixture
def receivers(monkeypatch):
    """Makes the CLI bind 127.0.0.1 with a short timeout. `next()` returns the
    command's Receiver once it is bound (use its port: tests pass --port 0), and
    `stop.set()` makes the command see Ctrl-C."""
    made = queue.Queue()
    stop = threading.Event()
    drained = []

    class TestReceiver(cli.Receiver):
        def __init__(self, host="0.0.0.0", port=5005, timeout_s=0.5):
            super().__init__("127.0.0.1", port, 0.1)
            made.put(self)

        def __iter__(self):
            for item in super().__iter__():
                if stop.is_set():
                    raise KeyboardInterrupt
                yield item

        def drain(self):
            drained.append(super().drain())
            return drained[-1]

    monkeypatch.setattr(cli, "Receiver", TestReceiver)
    return SimpleNamespace(next=lambda: made.get(timeout=10), stop=stop, drained=drained)


def stream(port, seconds, fs=1000, seed=0):
    """Simulated EMG to `port` in real time, from a daemon thread."""
    sim = EmgSimulator(fs=fs, seed=seed)
    t = threading.Thread(
        target=run_udp,
        args=("127.0.0.1", port, sim),
        kwargs={"duration_s": seconds, "verbose": False},
        daemon=True,
    )
    t.start()
    return t


def send(port, datagrams):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
        for datagram in datagrams:
            tx.sendto(datagram, ("127.0.0.1", port))


def wait_until(condition, timeout=10.0):
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.02)


def noise_packets(n, fs, channels=4, first_seq=0, frames=20):
    rng = np.random.default_rng(first_seq)
    for seq in range(first_seq, first_seq + n):
        block = rng.normal(0, 50, (frames, channels)).astype(np.int16)
        yield encode_packet(block, seq, seq * 20 * 1_000_000 // fs, fs)


def wait_for_stderr(capsys, text):
    err = ""

    def seen():
        nonlocal err
        err += capsys.readouterr().err
        return text in err

    wait_until(seen)
    return err


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "m.npz"
    main(["calibrate", "--sim", "--seconds", "2", "--out", str(path)])
    return path


def test_demo_runs_end_to_end(capsys):
    main(["demo", "--train-seconds", "4", "--repeats", "1"])
    out = capsys.readouterr().out
    assert "accuracy, every window" in out
    assert "confusion matrix" in out
    assert "latency" in out


def test_calibrate_sim_writes_model(tmp_path, capsys):
    path = tmp_path / "m.npz"
    main(["calibrate", "--sim", "--seconds", "2", "--out", str(path)])
    assert path.exists()
    assert "training accuracy" in capsys.readouterr().out


@pytest.fixture
def game():
    """A pretend Unity app listening for gesture JSON."""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        s.settimeout(10.0)
        yield s


def test_forward_sends_gesture_json(model, game, receivers):
    to = f"127.0.0.1:{game.getsockname()[1]}"
    cmd = Background(["forward", "--port", "0", "--model", str(model), "--to", to])
    sender = stream(receivers.next().port, 0.6, seed=5)
    msg = json.loads(game.recv(1024))
    sender.join(timeout=10)
    receivers.stop.set()
    cmd.join()
    assert msg["type"] == "gesture"
    assert msg["gesture"] in {"rest", "fist", "open", "flex", "extend"}
    assert 0.0 <= msg["confidence"] <= 1.0


def test_forward_works_below_900_hz(tmp_path, game, receivers):
    sim = EmgSimulator(fs=500, seed=3)
    clf, _, _ = myobridge.calibrate(sim.record, ["rest", "fist"], 2.0, 500)
    path = tmp_path / "m500.npz"
    clf.save(path)
    to = f"127.0.0.1:{game.getsockname()[1]}"
    cmd = Background(["forward", "--port", "0", "--model", str(path), "--to", to])
    rx = receivers.next()
    send(rx.port, noise_packets(30, fs=500))
    msg = json.loads(game.recv(1024))
    receivers.stop.set()
    cmd.join()
    assert msg["type"] == "gesture"


def test_forward_refuses_a_model_trained_at_another_rate(model, receivers):
    cmd = Background(["forward", "--port", "0", "--model", str(model), "--to", "127.0.0.1:9"])
    rx = receivers.next()
    send(rx.port, noise_packets(3, fs=500))
    with pytest.raises(SystemExit) as exit_:
        cmd.join()
    assert "trained at 1000 Hz but the device sends 500 Hz" in str(exit_.value)


def test_forward_publishes_on_lsl(model, game, receivers):
    pylsl = pytest.importorskip("pylsl")
    name = f"mbcli-{uuid.uuid4().hex[:8]}"
    to = f"127.0.0.1:{game.getsockname()[1]}"
    argv = ["forward", "--port", "0", "--model", str(model), "--to", to, "--lsl", name]
    cmd = Background(argv)
    rx = receivers.next()
    send(rx.port, noise_packets(1, fs=1000))  # the outlets open on the first packet
    found = pylsl.resolve_byprop("name", f"{name}-gestures", timeout=5)
    if not found:
        receivers.stop.set()
        cmd.join()
        pytest.skip("LSL stream discovery doesn't work on this machine's network")
    inlet = pylsl.StreamInlet(found[0])
    inlet.open_stream(timeout=5)
    time.sleep(0.3)
    send(rx.port, noise_packets(30, fs=1000, first_seq=1))
    marker, _ = inlet.pull_sample(timeout=5)
    receivers.stop.set()
    cmd.join()
    assert marker is not None and json.loads(marker[0])["gesture"] in GESTURES


def test_listen_shows_channel_health(receivers, capsys):
    cmd = Background(["listen", "--port", "0", "--rest-seconds", "0.5"])
    port = receivers.next().port
    stream(port, 2.5, seed=2).join(timeout=10)
    receivers.stop.set()
    cmd.join()
    out = capsys.readouterr().out
    assert f"waiting for packets on udp://0.0.0.0:{port}" in out
    assert "receiving 4 channels at 1000 Hz" in out
    assert "ch0 |" in out
    assert "ok" in out or "CHECK:" in out  # a health report was drawn


def test_listen_at_500_hz_with_empty_packets(receivers, capsys):
    cmd = Background(["listen", "--port", "0", "--rest-seconds", "0.5", "--mains", "50"])
    rx = receivers.next()
    empty = [*noise_packets(1, fs=500, frames=0), *noise_packets(1, 500, first_seq=51, frames=0)]
    send(rx.port, [empty[0], *noise_packets(50, fs=500, first_seq=1), empty[1]])
    wait_until(lambda: rx.stats.received >= 52)
    receivers.stop.set()
    cmd.join()
    out = capsys.readouterr().out
    assert "receiving 4 channels at 500 Hz" in out
    assert "ch3 |" in out


def test_listen_stops_cleanly_when_the_stream_changes(receivers, capsys):
    cmd = Background(["listen", "--port", "0"])
    rx = receivers.next()
    send(rx.port, [*noise_packets(5, fs=1000), *noise_packets(1, 1000, channels=3, first_seq=5)])
    with pytest.raises(SystemExit) as exit_:
        cmd.join()
    assert "the stream changed from 4 channels at 1000 Hz to 3" in str(exit_.value)


def test_channel_status_text():
    from myobridge.cli import channel_status
    from myobridge.quality import ChannelReport, Issue, QualityReport

    healthy = ChannelReport(0, 10.0, 0.0, 0.01, 0.1, ())
    bad = ChannelReport(1, 0.1, 0.0, 0.01, 0.1, (Issue.FLATLINE,))
    report = QualityReport(channels=(healthy, bad))
    assert channel_status(None, 0) == ""
    assert channel_status(report, 0) == "ok"
    assert channel_status(report, 1) == "CHECK: flatline"


def test_record_saves_samples_and_reports_noise(tmp_path, receivers, capsys):
    out = tmp_path / "rest.npz"
    cmd = Background(["record", "--port", "0", "--seconds", "1", "--out", str(out)])
    sender = stream(receivers.next().port, 1.5, seed=3)
    cmd.join()
    sender.join(timeout=10)
    saved = np.load(out)
    assert saved["samples"].shape == (1000, 4) and saved["samples"].dtype == np.int16
    assert int(saved["fs"]) == 1000
    text = capsys.readouterr().out
    assert text.count("20-450 Hz RMS") == 4


def test_record_hints_when_nothing_arrives(receivers, capsys, monkeypatch):
    monkeypatch.setattr(cli, "NO_DATA_HINT_S", 0.2)
    cmd = Background(["record", "--port", "0"])
    rx = receivers.next()
    wait_for_stderr(capsys, "no packets on UDP port")
    receivers.stop.set()
    cmd.join()
    assert rx.stats.received == 0


def test_listen_hints_when_only_invalid_packets_arrive(receivers, capsys, monkeypatch):
    monkeypatch.setattr(cli, "NO_DATA_HINT_S", 0.2)
    cmd = Background(["listen", "--port", "0"])
    send(receivers.next().port, [b"garbage!" * 5])
    err = wait_for_stderr(capsys, "not valid myobridge packets")
    receivers.stop.set()
    cmd.join()
    assert "last: magic" in err


def test_live_calibrate_uses_the_device_rate_and_fresh_data(
    tmp_path, receivers, capsys, monkeypatch
):
    def slow_user(prompt):
        time.sleep(0.5)  # packets pile up in the socket meanwhile
        return ""

    monkeypatch.setattr("builtins.input", slow_user)
    path = tmp_path / "live.npz"
    argv = ["calibrate", "--port", "0", "--gestures", "rest,fist", "--seconds", "1"]
    cmd = Background([*argv, "--rate", "1000", "--out", str(path)])
    sender = stream(receivers.next().port, 5.0, fs=2000)
    cmd.join()
    sender.join(timeout=10)
    captured = capsys.readouterr()
    assert "receiving 4 channels at 2000 Hz" in captured.out
    assert "the device sends 2000 Hz, not 1000 Hz" in captured.err
    assert len(receivers.drained) == 2 and min(receivers.drained) > 0
    assert path.exists()


def test_live_calibrate_times_out_when_the_device_stops(tmp_path, receivers, monkeypatch):
    monkeypatch.setattr(cli, "RECORD_TIMEOUT_S", 0.3)
    monkeypatch.setattr("builtins.input", lambda prompt: "")
    cmd = Background(["calibrate", "--port", "0", "--out", str(tmp_path / "m.npz")])
    send(receivers.next().port, noise_packets(1, fs=1000))  # one packet, then silence
    with pytest.raises(SystemExit) as exit_:
        cmd.join()
    assert "no usable packets" in str(exit_.value)


def test_version(capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["--version"])
    assert exit_.value.code == 0
    assert capsys.readouterr().out.strip() == f"myobridge {myobridge.__version__}"


@pytest.mark.parametrize(
    ("argv", "mains"),
    [
        (["listen"], 60.0),
        (["--mains", "50", "listen"], 50.0),
        (["listen", "--mains", "50"], 50.0),
        (["--mains", "50", "record", "--mains", "60"], 60.0),
        (["forward", "--mains", "50"], 50.0),
        (["simulate", "--mains", "50"], 50.0),
        (["calibrate", "--sim", "--mains", "50"], 50.0),
    ],
)
def test_mains_works_before_or_after_the_subcommand(argv, mains):
    assert build_parser().parse_args(argv).mains == mains


def test_help_shows_defaults(capsys):
    with pytest.raises(SystemExit):
        main(["listen", "--help"])
    out = capsys.readouterr().out
    assert "(default: 5005)" in out
    assert "(default: None)" not in out and "(default: False)" not in out


@pytest.mark.parametrize(
    ("argv", "code", "message"),
    [
        (["forward", "--model", "missing.npz"], 1, "no model at missing.npz"),
        (["forward", "--to", "localhost"], 2, "expected host:port"),
        (["forward", "--to", "localhost:99999"], 2, "must be 1..65535"),
        (["listen", "--port", "70000"], 2, "must be 0..65535"),
        (["listen", "--port", "-1"], 2, "must be 0..65535"),
        (["fetch", "foo"], 1, "unknown dataset 'foo'; choose from 3dc, multiday"),
        (["bench", "foo"], 1, "unknown benchmark 'foo'"),
        (["simulate", "--rate", "500"], 2, "must be 501..65535"),
        (["simulate", "--channels", "300"], 2, "must be 1..255"),
        (["simulate", "--channels", "0"], 2, "must be 1..255"),
        (["calibrate", "--sim", "--gestures", "rest"], 2, "at least two different gestures"),
        (["calibrate", "--sim", "--gestures", "rest,wave"], 1, "no gesture 'wave'"),
        (["demo", "--repeats", "0"], 2, "must be 1..1000"),
    ],
)
def test_bad_input_is_a_one_line_error(argv, code, message, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    status, err = run_failing(argv, capsys)
    assert status == code
    assert message in err.splitlines()[-1]
    assert "Traceback" not in err


def run_failing(argv, capsys):
    """(exit status, stderr) of a command that should fail."""
    with pytest.raises(SystemExit) as exit_:
        main(argv)
    code = exit_.value.code
    if isinstance(code, str):  # sys.exit(message) prints it and exits with status 1
        return 1, code
    return code, capsys.readouterr().err


def test_port_in_use_is_a_one_line_error(capsys):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as other:
        other.bind(("0.0.0.0", 0))
        port = other.getsockname()[1]
        status, err = run_failing(["record", "--port", str(port)], capsys)
    assert status == 1
    assert err.startswith(f"myobridge record: error: can't listen on UDP port {port}")
