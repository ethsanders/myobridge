import json
import socket
import threading

from myobridge.cli import main
from myobridge.simulator import EmgSimulator, run_udp


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


def free_udp_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_forward_sends_gesture_json(tmp_path, monkeypatch):
    model = tmp_path / "m.npz"
    main(["calibrate", "--sim", "--seconds", "2", "--out", str(model)])

    game = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # pretend Unity app
    game.bind(("127.0.0.1", 0))
    game.settimeout(5.0)
    emg_port = free_udp_port()

    # Stop forward's receive loop after a while by making it see Ctrl-C.
    import myobridge.cli as cli

    real_receiver = cli.Receiver

    class StoppingReceiver(real_receiver):
        def __iter__(self):
            for i, item in enumerate(super().__iter__()):
                if i > 40:
                    raise KeyboardInterrupt
                yield item

    monkeypatch.setattr(cli, "Receiver", StoppingReceiver)
    fwd = threading.Thread(
        target=main,
        args=(
            [
                "forward",
                "--port",
                str(emg_port),
                "--model",
                str(model),
                "--to",
                f"127.0.0.1:{game.getsockname()[1]}",
            ],
        ),
    )
    fwd.start()
    run_udp("127.0.0.1", emg_port, EmgSimulator(seed=5), duration_s=0.6, verbose=False)
    msg = json.loads(game.recv(1024))
    fwd.join(timeout=10)
    game.close()
    assert msg["type"] == "gesture"
    assert msg["gesture"] in {"rest", "fist", "open", "flex", "extend"}
    assert 0.0 <= msg["confidence"] <= 1.0


def test_listen_shows_channel_health(monkeypatch, capsys):
    import myobridge.cli as cli

    real_receiver = cli.Receiver

    class StoppingReceiver(real_receiver):
        def __iter__(self):
            for i, item in enumerate(super().__iter__()):
                if i > 100:  # ~2 s of 20 ms packets, well before the simulator stops
                    raise KeyboardInterrupt
                yield item

    monkeypatch.setattr(cli, "Receiver", StoppingReceiver)
    port = free_udp_port()
    rx = threading.Thread(
        target=main, args=(["listen", "--port", str(port), "--rest-seconds", "0.5"],)
    )
    rx.start()
    run_udp("127.0.0.1", port, EmgSimulator(seed=2), duration_s=2.5, verbose=False)
    rx.join(timeout=10)
    out = capsys.readouterr().out
    assert "ch0 |" in out
    assert "ok" in out or "CHECK:" in out  # a health report was drawn


def test_channel_status_text():
    from myobridge.cli import channel_status
    from myobridge.quality import ChannelReport, Issue, QualityReport

    healthy = ChannelReport(0, 10.0, 0.0, 0.01, 0.1, ())
    bad = ChannelReport(1, 0.1, 0.0, 0.01, 0.1, (Issue.FLATLINE,))
    report = QualityReport(channels=(healthy, bad))
    assert channel_status(None, 0) == ""
    assert channel_status(report, 0) == "ok"
    assert channel_status(report, 1) == "CHECK: flatline"


def test_record_saves_samples_and_reports_noise(tmp_path, capsys):
    import numpy as np

    port = free_udp_port()
    out = tmp_path / "rest.npz"
    rx = threading.Thread(
        target=main, args=(["record", "--port", str(port), "--seconds", "1", "--out", str(out)],)
    )
    rx.start()
    run_udp("127.0.0.1", port, EmgSimulator(seed=3), duration_s=2.0, verbose=False)
    rx.join(timeout=10)
    saved = np.load(out)
    assert saved["samples"].shape == (1000, 4) and saved["samples"].dtype == np.int16
    assert int(saved["fs"]) == 1000
    text = capsys.readouterr().out
    assert text.count("20-450 Hz RMS") == 4
