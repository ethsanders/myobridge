"""Command line entry point: `myobridge <simulate|listen|calibrate|forward|demo>`."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

import numpy as np

from .classifier import LDAClassifier, calibrate
from .dsp import EmgFilter
from .pipeline import GesturePipeline
from .protocol import SequenceTracker, decode_packet
from .receiver import Receiver
from .simulator import GESTURES, EmgSimulator, gesture_session, packetize, run_udp

DEFAULT_PORT = 5005  # ESP32 -> PC
DEFAULT_FORWARD_PORT = 5006  # PC -> game engine


# ------------------------------------------------------------------- simulate


def cmd_simulate(args) -> None:
    sim = EmgSimulator(n_channels=args.channels, fs=args.rate, seed=args.seed, mains_hz=args.mains)
    print(f"Simulating {args.channels} ch @ {args.rate} Hz -> udp://{args.host}:{args.port}")
    run_udp(args.host, args.port, sim, args.hold, drop_prob=args.drop, duration_s=args.duration)


# --------------------------------------------------------------------- listen


def bar(value: float, full_scale: float, width: int = 40) -> str:
    filled = int(min(value / full_scale, 1.0) * width)
    return "#" * filled + "." * (width - filled)


def cmd_listen(args) -> None:
    print(f"Listening on udp://0.0.0.0:{args.port} (Ctrl-C to stop)")
    emg_filter, last_draw, lines_drawn = None, 0.0, 0
    with Receiver(port=args.port) as rx:
        for packet in rx:
            if packet is not None:
                if emg_filter is None:  # build filters once we know the stream format
                    emg_filter = EmgFilter(
                        packet.sample_rate, packet.n_channels, mains_hz=args.mains
                    )
                envelope = emg_filter.process(packet.samples).envelope[-1]
            if emg_filter is None or time.monotonic() - last_draw < 0.1:
                continue
            last_draw = time.monotonic()
            s = rx.stats
            rows = [f"ch{i} |{bar(v, args.scale)}| {v:7.1f}" for i, v in enumerate(envelope)]
            rows.append(
                f"rx {s.received}  dropped {s.dropped} ({s.loss_percent:.2f}%)  "
                f"corrupt {s.corrupt}  reordered {s.reordered}  restarts {s.restarts}"
            )
            # ANSI "cursor up N lines" so the display redraws in place.
            if lines_drawn:
                sys.stdout.write(f"\x1b[{lines_drawn}F")
            sys.stdout.write("\n".join(r + "\x1b[K" for r in rows) + "\n")
            sys.stdout.flush()
            lines_drawn = len(rows)


# ------------------------------------------------------------------ calibrate


def cmd_calibrate(args) -> None:
    gestures = args.gestures.split(",")
    if args.sim:
        sim = EmgSimulator(seed=args.seed, mains_hz=args.mains)
        fs, record, rx = sim.fs, sim.record, None
    else:
        rx = Receiver(port=args.port)
        fs = args.rate

        def record(gesture: str, seconds: float) -> np.ndarray:
            input(f"Get ready to hold '{gesture}' for {seconds:.0f} s, then press Enter...")
            rx.record(0.5, fs)  # throw away whatever queued up while we waited
            print("  recording...", flush=True)
            return rx.record(seconds, fs)

    clf, X, y = calibrate(record, gestures, args.seconds, fs, mains_hz=args.mains)
    if rx is not None:
        rx.close()
    train_acc = np.mean(clf.predict(X) == y)
    clf.save(args.out)
    print(f"Trained on {len(y)} windows, training accuracy {train_acc:.1%}. Saved {args.out}")


# -------------------------------------------------------------------- forward


def cmd_forward(args) -> None:
    """Live: UDP EMG in -> gesture JSON out (e.g. to a Unity app on a Quest)."""
    model = LDAClassifier.load(args.model)
    host, port = args.to.rsplit(":", 1)
    out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pipeline, msg_seq = None, 0
    print(f"udp://0.0.0.0:{args.port} -> gestures -> udp://{host}:{port}")
    with Receiver(port=args.port) as rx:
        for packet in rx:
            if packet is None:
                continue
            if pipeline is None:
                pipeline = GesturePipeline(
                    model, packet.sample_rate, packet.n_channels, mains_hz=args.mains
                )
            for pred in pipeline.process_block(packet.samples):
                message = {
                    "type": "gesture",
                    "gesture": pred.gesture,
                    "confidence": round(pred.confidence, 3),
                    "seq": msg_seq,
                    "t": round(time.time(), 3),
                }
                out.sendto(json.dumps(message).encode(), (host, int(port)))
                msg_seq += 1
                if args.verbose:
                    print(message)


# ----------------------------------------------------------------------- demo


def link_check(seed: int, n_packets: int = 2000, drop_prob: float = 0.01) -> str:
    """Push packets through encode -> lossy 'network' -> decode, entirely in memory."""
    rng = np.random.default_rng(seed)
    tracker, corrupt = SequenceTracker(), 0
    stream = packetize(EmgSimulator(seed=seed), drop_prob=drop_prob)
    for _ in range(n_packets):
        data = bytearray(next(stream))
        if rng.random() < 0.005:  # flip a random bit now and then
            data[rng.integers(len(data))] ^= 1 << int(rng.integers(8))
        try:
            tracker.accept(decode_packet(bytes(data)).seq)
        except ValueError:
            corrupt += 1
    s = tracker.stats
    # A corrupt packet is rejected before the tracker sees it, so its seq shows
    # up as a gap too: "missing" = lost in transit + thrown away for bad CRC.
    return (
        f"received {s.received}, corrupt {corrupt} (caught by CRC), "
        f"missing seq numbers {s.dropped} ({s.loss_percent:.2f}%), reordered {s.reordered}"
    )


def cmd_demo(args) -> None:
    fs, block = 1000, 20  # 20 samples per packet = one packet every 20 ms
    print("myobridge demo: simulated 4-channel EMG, nothing leaves this process\n")

    print("[1/4] Wire protocol: 2000 packets, 1% simulated loss + random bit flips")
    print("      " + link_check(args.seed) + "\n")

    print(f"[2/4] Calibration: {args.train_seconds:.0f} s per gesture ({', '.join(GESTURES)})")
    trainer = EmgSimulator(seed=args.seed)
    clf, X, y = calibrate(trainer.record, GESTURES, args.train_seconds, fs)
    print(
        f"      {len(y)} windows x {X.shape[1]} features, LDA training accuracy "
        f"{np.mean(clf.predict(X) == y):.1%}\n"
    )

    hold = 3.0
    print(
        f"[3/4] Held-out session (new seed): {args.repeats} x {len(GESTURES)} gestures, "
        f"{hold:.0f} s each, streamed in {block}-sample packets"
    )
    tester = EmgSimulator(seed=args.seed + 1000)
    raw, labels = gesture_session(tester, GESTURES, hold, args.repeats)
    pipeline = GesturePipeline(clf, fs, raw.shape[1])
    preds, latencies_us = [], []
    for start in range(0, len(raw), block):
        t0 = time.perf_counter_ns()
        preds += pipeline.process_block(raw[start : start + block])
        latencies_us.append((time.perf_counter_ns() - t0) / 1000)

    ends = np.array([p.sample_index for p in preds]) - 1
    truth = labels[ends]
    raw_pred = np.array([p.raw_gesture for p in preds])
    smooth_pred = np.array([p.gesture for p in preds])
    window = pipeline.windower.size
    steady = np.array([labels[max(0, e - window - 100)] == labels[e] for e in ends])
    print(f"      windows: {len(preds)}")
    print(f"      accuracy, every window:          {np.mean(raw_pred == truth):.1%}")
    print(
        f"      accuracy, majority vote (k=5):   {np.mean(smooth_pred == truth):.1%}"
        "  (steadier output, but lags ~100 ms at each switch)"
    )
    print(
        f"      accuracy, steady-state windows:  {np.mean(raw_pred[steady] == truth[steady]):.1%}"
        "  (excludes ~300 ms after each switch)"
    )
    print("\n      confusion matrix (rows = true, cols = predicted, every window)")
    names = list(clf.classes_)
    print("      " + " " * 8 + "".join(f"{n:>8}" for n in names))
    for t in names:
        row = [np.sum((truth == t) & (raw_pred == p)) for p in names]
        print(f"      {t:>8}" + "".join(f"{c:>8}" for c in row))

    lat = np.array(latencies_us)
    print(f"\n[4/4] Processing latency per {block}-sample block (filter+features+LDA+vote)")
    print(
        f"      mean {lat.mean():.0f} us, median {np.median(lat):.0f} us, "
        f"p99 {np.percentile(lat, 99):.0f} us, max {lat.max():.0f} us "
        f"(budget: {1e6 * block / fs:.0f} us per block)"
    )


# ----------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="myobridge", description=__doc__)
    p.add_argument("--mains", type=float, default=60.0, help="powerline frequency (50 or 60 Hz)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("simulate", help="fake ESP32: stream simulated EMG over UDP")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--channels", type=int, default=4)
    s.add_argument("--rate", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--hold", type=float, default=3.0, help="seconds per gesture")
    s.add_argument("--drop", type=float, default=0.0, help="packet loss probability to inject")
    s.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    s.set_defaults(func=cmd_simulate)

    s = sub.add_parser("listen", help="show live per-channel RMS and link stats")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--scale", type=float, default=600.0, help="RMS value for a full bar")
    s.set_defaults(func=cmd_listen)

    s = sub.add_parser("calibrate", help="record each gesture and train an LDA model")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--rate", type=int, default=1000, help="device sample rate")
    s.add_argument("--gestures", default=",".join(GESTURES))
    s.add_argument("--seconds", type=float, default=5.0, help="recording length per gesture")
    s.add_argument("--out", default="model.npz")
    s.add_argument("--sim", action="store_true", help="calibrate from the simulator instead")
    s.add_argument("--seed", type=int, default=0, help="simulator seed (with --sim)")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("forward", help="classify live EMG and send gesture JSON over UDP")
    s.add_argument("--port", type=int, default=DEFAULT_PORT)
    s.add_argument("--model", default="model.npz")
    s.add_argument("--to", default=f"127.0.0.1:{DEFAULT_FORWARD_PORT}", help="host:port")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_forward)

    s = sub.add_parser("demo", help="offline end-to-end run: simulate, train, evaluate")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--train-seconds", type=float, default=8.0)
    s.add_argument("--repeats", type=int, default=3)
    s.set_defaults(func=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:  # Ctrl-C is the normal way to stop the live commands
        print()


if __name__ == "__main__":
    main()
