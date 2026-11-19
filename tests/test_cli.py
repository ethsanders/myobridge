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
