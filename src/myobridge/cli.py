"""Command line entry point: `myobridge <subcommand>`, see `myobridge --help`."""

from __future__ import annotations

import argparse
import errno
import json
import socket
import sys
import time
from collections.abc import Callable, Iterable, Iterator

import numpy as np

from . import __version__
from .classifier import LDAClassifier, calibrate
from .dsp import EmgFilter
from .lsl import LslOutlets
from .pipeline import GesturePipeline
from .protocol import Packet, SequenceTracker, decode_packet
from .quality import QualityReport
from .receiver import Receiver
from .simulator import GESTURES, EmgSimulator, gesture_session, packetize, run_udp

DEFAULT_PORT = 5005  # ESP32 -> PC
DEFAULT_FORWARD_PORT = 5006  # PC -> game engine
NO_DATA_HINT_S = 3.0  # silence before the live commands print a troubleshooting hint
RECORD_TIMEOUT_S = 5.0  # live calibrate gives up after this long without packets


class CliError(Exception):
    """An error caused by the input or the setup: printed as one line, exit status 1."""


# ----------------------------------------------------------------- live input


def open_receiver(port: int) -> Receiver:
    try:
        return Receiver(port=port)
    except OSError as err:
        hint = ""
        if err.errno == errno.EADDRINUSE:
            hint = " (is another myobridge already listening there?)"
        raise CliError(f"can't listen on UDP port {port}: {err.strerror or err}{hint}") from None


def no_data_hint(rx: Receiver) -> str:
    if rx.stats.corrupt:
        return (
            f"{rx.stats.corrupt} datagram(s) on UDP port {rx.port} were not valid myobridge "
            f"packets (last: {rx.last_error}); is the firmware the same version as myobridge?"
        )
    return (
        f"no packets on UDP port {rx.port} after {NO_DATA_HINT_S:g} s: check HOST_IP and "
        "HOST_PORT in config.h, that the device is on this computer's network, and that "
        f"the firewall lets UDP port {rx.port} in (Ctrl-C to stop)"
    )


def live_packets(rx: Receiver) -> Iterator[Packet | None]:
    """`rx`'s packets, with a hint if none arrive and a check that the format stays put.

    Yields None on timeouts, like iterating `rx` itself. Packets with no
    samples are passed on as None too.
    """
    print(f"waiting for packets on udp://0.0.0.0:{rx.port} ...", flush=True)
    start, hinted = time.monotonic(), False
    first: Packet | None = None
    for item in rx:
        packet = item if item is not None and item.n_frames > 0 else None
        if packet is not None:
            if first is None:
                first = packet
                print(
                    f"receiving {packet.n_channels} channels at {packet.sample_rate} Hz", flush=True
                )
            elif (packet.n_channels, packet.sample_rate) != (first.n_channels, first.sample_rate):
                raise CliError(
                    f"the stream changed from {first.n_channels} channels at "
                    f"{first.sample_rate} Hz to {packet.n_channels} at {packet.sample_rate} Hz; "
                    "restart the command"
                )
        elif first is None and not hinted and time.monotonic() - start >= NO_DATA_HINT_S:
            print(no_data_hint(rx), file=sys.stderr, flush=True)
            hinted = True
        yield packet


def stream_error(packet: Packet, err: ValueError) -> CliError:
    return CliError(f"can't process {packet.n_channels} channels at {packet.sample_rate} Hz: {err}")


# ------------------------------------------------------------------- simulate


def cmd_simulate(args: argparse.Namespace) -> None:
    sim = EmgSimulator(n_channels=args.channels, fs=args.rate, seed=args.seed, mains_hz=args.mains)
    print(f"Simulating {args.channels} ch @ {args.rate} Hz -> udp://{args.host}:{args.port}")
    run_udp(args.host, args.port, sim, args.hold, drop_prob=args.drop, duration_s=args.duration)


# --------------------------------------------------------------------- listen


def bar(value: float, full_scale: float, width: int = 40) -> str:
    filled = int(min(value / full_scale, 1.0) * width)
    return "#" * filled + "." * (width - filled)


def channel_status(report: QualityReport | None, channel: int) -> str:
    """One-word health note for the live display ("" until the first report)."""
    if report is None:
        return ""
    issues = report.channels[channel].issues
    return "ok" if not issues else "CHECK: " + ", ".join(x.value for x in issues)


def cmd_listen(args: argparse.Namespace) -> None:
    from .quality import QualityConfig, QualityMonitor, rest_floor

    print("Showing live per-channel RMS and link stats (Ctrl-C to stop).")
    print(
        f"Relax your arm for the first {args.rest_seconds:g} s (sets each channel's noise floor)."
    )
    emg_filter, monitor, last_draw, lines_drawn = None, None, 0.0, 0
    rest: list[np.ndarray] = []
    report: QualityReport | None = None
    with open_receiver(args.port) as rx:
        for packet in live_packets(rx):
            if packet is not None:
                if emg_filter is None:  # build filters once we know the stream format
                    try:
                        emg_filter = EmgFilter(
                            packet.sample_rate, packet.n_channels, mains_hz=args.mains
                        )
                        qcfg = QualityConfig(
                            fs=packet.sample_rate,
                            mains_hz=args.mains,
                            clip_low=-args.rail,
                            clip_high=args.rail,
                        )
                    except ValueError as err:
                        raise stream_error(packet, err) from None
                    monitor = QualityMonitor(qcfg, packet.n_channels)
                envelope = emg_filter.process(packet.samples).envelope[-1]
                assert monitor is not None
                raw = packet.samples.astype(np.float64)
                if monitor.rest_floor is None:
                    rest.append(raw)
                    if sum(len(r) for r in rest) >= args.rest_seconds * packet.sample_rate:
                        monitor.rest_floor = rest_floor(np.vstack(rest), monitor.cfg)
                report = monitor.push(raw) or report
            if emg_filter is None or time.monotonic() - last_draw < 0.1:
                continue
            last_draw = time.monotonic()
            s = rx.stats
            rows = []
            for i, v in enumerate(envelope):
                status = channel_status(report, i)
                rows.append(f"ch{i} |{bar(v, args.scale)}| {v:7.1f}  {status}")
            rows.append(
                f"rx {s.received}  dropped {s.dropped} ({s.loss_percent:.2f}%)  "
                f"corrupt {s.corrupt}  reordered {s.reordered}  duplicates {s.duplicates}  "
                f"restarts {s.restarts}"
            )
            # ANSI "cursor up N lines" so the display redraws in place.
            if lines_drawn:
                sys.stdout.write(f"\x1b[{lines_drawn}F")
            sys.stdout.write("\n".join(r + "\x1b[K" for r in rows) + "\n")
            sys.stdout.flush()
            lines_drawn = len(rows)


# --------------------------------------------------------------------- record


def noise_summary(raw: np.ndarray, fs: float, mains_hz: float) -> list[str]:
    """Per-channel band RMS (20-450 Hz at 1 kHz) and health of a recording, one line each."""
    from .quality import QualityConfig, assess

    emg_filter = EmgFilter(fs, raw.shape[1], mains_hz=mains_hz)
    clean = emg_filter.process(raw.astype(np.float64)).clean
    low, high = emg_filter.band
    settle = clean[int(0.25 * fs) :]  # skip the filters' start-up transient
    rms = np.sqrt(np.mean(settle**2, axis=0))
    report = assess(raw.astype(np.float64), QualityConfig(fs=fs, mains_hz=mains_hz))
    return [
        f"ch{i}  {low:g}-{high:g} Hz RMS {rms[i]:8.2f} counts   mains share "
        f"{c.line_noise_share:5.1%}   {channel_status(report, i)}"
        for i, c in enumerate(report.channels)
    ]


def cmd_record(args: argparse.Namespace) -> None:
    """Save raw samples to .npz (for bring-up measurements and offline analysis)."""
    blocks: list[np.ndarray] = []
    fs, total = None, 0
    print(f"Recording {args.seconds:g} s.", flush=True)
    with open_receiver(args.port) as rx:
        for packet in live_packets(rx):
            if packet is None:
                continue
            fs = packet.sample_rate
            blocks.append(packet.samples)
            total += packet.n_frames
            if total >= args.seconds * fs:
                break
        stats = rx.stats
    assert fs is not None
    raw = np.vstack(blocks)[: int(args.seconds * fs)]
    np.savez(args.out, samples=raw, fs=fs, dropped_packets=stats.dropped)
    print(f"Saved {len(raw)} samples x {raw.shape[1]} channels at {fs} Hz to {args.out}")
    print(f"packets dropped {stats.dropped} ({stats.loss_percent:.2f}%)")
    try:
        summary = noise_summary(raw, fs, args.mains)
    except ValueError as err:
        raise CliError(f"saved, but can't summarise a {fs} Hz recording: {err}") from None
    for line in summary:
        print("  " + line)


# ------------------------------------------------------------------ calibrate


def cmd_calibrate(args: argparse.Namespace) -> None:
    gestures: list[str] = args.gestures
    if args.sim:
        unknown = sorted(set(gestures) - set(GESTURES))
        if unknown:
            raise CliError(
                f"the simulator has no gesture {unknown[0]!r}; choose from {', '.join(GESTURES)}"
            )
        sim = EmgSimulator(seed=args.seed, mains_hz=args.mains)
        clf, X, y = calibrate(sim.record, gestures, args.seconds, sim.fs, mains_hz=args.mains)
    else:
        with open_receiver(args.port) as rx:
            clf, X, y = calibrate_live(rx, gestures, args)
    train_acc = np.mean(clf.predict(X) == y)
    clf.save(args.out)
    print(f"Trained on {len(y)} windows, training accuracy {train_acc:.1%}. Saved {args.out}")


def calibrate_live(
    rx: Receiver, gestures: list[str], args: argparse.Namespace
) -> tuple[LDAClassifier, np.ndarray, np.ndarray]:
    """Prompt for each gesture and record it from the device, at the device's rate."""
    first = next(p for p in live_packets(rx) if p is not None)
    fs = first.sample_rate
    if args.rate is not None and args.rate != fs:
        print(f"note: the device sends {fs} Hz, not {args.rate} Hz; using {fs} Hz", file=sys.stderr)

    def record(gesture: str, seconds: float) -> np.ndarray:
        input(f"Get ready to hold '{gesture}' for {seconds:g} s, then press Enter...")
        rx.drain()  # drop what queued up while we waited for Enter
        print("  recording...", flush=True)
        try:
            return rx.record(seconds, fs, timeout_s=RECORD_TIMEOUT_S)
        except (TimeoutError, ValueError) as err:
            raise CliError(str(err)) from None

    return calibrate(record, gestures, args.seconds, fs, mains_hz=args.mains)


# -------------------------------------------------------------------- forward


def forward_pipeline(model: LDAClassifier, packet: Packet, mains_hz: float) -> GesturePipeline:
    """A live pipeline for `packet`'s stream, set up the way `model` was trained.

    A model trained at another sample rate would see different features, so
    that is an error with a way out, not a silent drop in accuracy. Models
    saved before 0.2.0 don't record their settings and use the defaults.
    """
    trained = model.settings
    if trained is not None and not np.isclose(trained.fs, packet.sample_rate):
        raise CliError(
            f"the model was trained at {trained.fs:g} Hz but the device sends "
            f"{packet.sample_rate} Hz; run `myobridge calibrate` again with this device"
        )
    try:
        if trained is not None:
            return GesturePipeline.from_model(model, packet.n_channels)
        return GesturePipeline(model, packet.sample_rate, packet.n_channels, mains_hz=mains_hz)
    except ValueError as err:
        raise stream_error(packet, err) from None


def open_lsl(name: str, packet: Packet) -> LslOutlets:
    """LSL outlets for `packet`'s stream, or a CLI error saying what to install."""
    try:
        outlets = LslOutlets(packet.n_channels, packet.sample_rate, name=name, source_id=name)
    except ImportError as err:
        raise CliError(str(err)) from None
    print(f"Publishing LSL streams {name}-emg and {name}-gestures", flush=True)
    return outlets


def cmd_forward(args: argparse.Namespace) -> None:
    """Live: UDP EMG in -> gesture JSON out (e.g. to a Unity app on a Quest)."""
    try:
        model = LDAClassifier.load(args.model)
    except FileNotFoundError:
        raise CliError(
            f"no model at {args.model}; train one with `myobridge calibrate --out {args.model}`"
        ) from None
    host, port = args.to
    print(f"Sending gestures to udp://{host}:{port}")
    with open_receiver(args.port) as rx, socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as out:
        pipeline, msg_seq = None, 0
        lsl: LslOutlets | None = None
        for packet in live_packets(rx):
            if packet is None:
                continue
            if pipeline is None:
                pipeline = forward_pipeline(model, packet, args.mains)
                if args.lsl is not None:
                    lsl = open_lsl(args.lsl, packet)
            if lsl is not None:
                lsl.push_emg(packet.samples)
            for pred in pipeline.process_block(packet.samples):
                if lsl is not None:
                    lsl.push_gesture(pred)
                message = {
                    "type": "gesture",
                    "gesture": pred.gesture,
                    "confidence": round(pred.confidence, 3),
                    "seq": msg_seq,
                    "t": round(time.time(), 3),
                }
                out.sendto(json.dumps(message).encode(), (host, port))
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
            packet = decode_packet(bytes(data))
        except ValueError:
            corrupt += 1
            continue
        tracker.accept(packet.seq, packet.timestamp_us)
    s = tracker.stats
    # A corrupt packet is rejected before the tracker sees it, so its seq shows
    # up as a gap too: "missing" = lost in transit + thrown away for bad CRC.
    return (
        f"received {s.received}, corrupt {corrupt} (caught by CRC), "
        f"missing seq numbers {s.dropped} ({s.loss_percent:.2f}%), reordered {s.reordered}"
    )


def cmd_demo(args: argparse.Namespace) -> None:
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
    assert clf.classes_ is not None
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


# ------------------------------------------------------------- fetch / bench


def check_names(names: list[str], known: Iterable[str], what: str) -> None:
    choices = sorted(known)
    for name in names:
        if name not in choices:
            raise CliError(f"unknown {what} {name!r}; choose from {', '.join(choices)}")


def cmd_fetch(args: argparse.Namespace) -> None:
    from . import datasets

    check_names(args.names, datasets.REGISTRY, "dataset")
    for name in args.names or sorted(datasets.REGISTRY):
        module = datasets.REGISTRY[name]
        try:
            path = datasets.fetch(name, module.SOURCE, args.data)
        except datasets.DatasetError as err:
            raise CliError(str(err)) from None
        print(f"{name}: {path} @ {module.SOURCE.commit[:12]}")
        print(f"  please cite: {module.SOURCE.citation}")


def cmd_bench(args: argparse.Namespace) -> None:
    from pathlib import Path

    from . import bench
    from .datasets import DatasetError

    check_names(args.names, bench.EXPERIMENTS, "benchmark")
    names = args.names or list(bench.EXPERIMENTS)
    opts = bench.RunOptions(
        data_root=Path(args.data) if args.data else None,
        out_dir=Path(args.out),
        figures_dir=None if args.no_figures else Path(args.figures),
        quick=args.quick,
        download=not args.no_download,
    )
    try:
        bench.run(names, opts)
    except DatasetError as err:
        raise CliError(str(err)) from None


# --------------------------------------------------------- argument parsing


def int_in(low: int, high: int, why: str = "") -> Callable[[str], int]:
    """argparse type: a whole number in [low, high]."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"expected a whole number, got {text!r}") from None
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"must be {low}..{high}{why}, got {value}")
        return value

    return parse


def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None


def positive(text: str) -> float:
    """argparse type: a number > 0."""
    value = _number(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {text}")
    return value


def probability(text: str) -> float:
    """argparse type: a number in [0, 1]."""
    value = _number(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError(f"must be between 0 and 1, got {text}")
    return value


listen_port = int_in(0, 65535)  # 0 = any free port
send_port = int_in(1, 65535)


def host_port(text: str) -> tuple[str, int]:
    """argparse type: "host:port"."""
    host, sep, port = text.rpartition(":")
    if not sep or not host:
        raise argparse.ArgumentTypeError(f"expected host:port, e.g. 127.0.0.1:5006, got {text!r}")
    return host, send_port(port)


def gesture_list(text: str) -> list[str]:
    """argparse type: comma-separated gesture names, at least two different ones."""
    names = [g.strip() for g in text.split(",") if g.strip()]
    if len(set(names)) < 2:
        raise argparse.ArgumentTypeError(f"need at least two different gestures, got {text!r}")
    return names


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
    """Shows each option's default, except None and False (they say nothing)."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


DESCRIPTION = """\
Stream multi-channel EMG from an ESP32 over UDP, check the signal, train a
gesture model and send its predictions to a game engine."""

EPILOG = f"""\
quickstart, no hardware needed:
  myobridge demo        offline: simulate, train, evaluate
  myobridge simulate    terminal 1: a fake armband streaming to port {DEFAULT_PORT}
  myobridge listen      terminal 2: live levels, channel health and link stats

Run `myobridge <command> --help` for a command's options."""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="myobridge", description=DESCRIPTION, epilog=EPILOG, formatter_class=HelpFormatter
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    mains_help = "mains (powerline) frequency in Hz: 50 or 60"
    p.add_argument("--mains", type=positive, default=60.0, help=mains_help)
    # The same option after the subcommand. SUPPRESS keeps it from overwriting
    # a global --mains when it isn't given.
    mains = argparse.ArgumentParser(add_help=False)
    mains.add_argument(
        "--mains",
        type=positive,
        default=argparse.SUPPRESS,
        help=mains_help + " (default: 60, or the global --mains)",
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="<command>")

    def command(
        name: str, about: str, func: Callable[[argparse.Namespace], None], *, uses_mains: bool
    ) -> argparse.ArgumentParser:
        s = sub.add_parser(
            name,
            help=about,
            description=about,
            formatter_class=HelpFormatter,
            parents=[mains] if uses_mains else [],
        )
        s.set_defaults(func=func)
        return s

    port_help = "UDP port the device sends to (0 = any free port)"

    s = command(
        "simulate", "fake ESP32: stream simulated EMG over UDP", cmd_simulate, uses_mains=True
    )
    s.add_argument("--host", default="127.0.0.1", help="where to send the packets")
    s.add_argument("--port", type=send_port, default=DEFAULT_PORT, help="UDP port to send to")
    s.add_argument("--channels", type=int_in(1, 255), default=4, help="channels")
    s.add_argument(
        "--rate",
        type=int_in(501, 65535, " (the simulated EMG reaches 250 Hz)"),
        default=1000,
        help="sample rate in Hz",
    )
    s.add_argument("--seed", type=int, default=0, help="random seed")
    s.add_argument("--hold", type=positive, default=3.0, help="seconds per gesture")
    s.add_argument("--drop", type=probability, default=0.0, help="packet loss to inject (0-1)")
    s.add_argument("--duration", type=positive, help="stop after N seconds")

    s = command("listen", "show live per-channel RMS and link stats", cmd_listen, uses_mains=True)
    s.add_argument("--port", type=listen_port, default=DEFAULT_PORT, help=port_help)
    s.add_argument("--scale", type=positive, default=600.0, help="RMS for a full bar")
    s.add_argument(
        "--rest-seconds",
        type=positive,
        default=3.0,
        help="rest period at the start that sets each channel's noise floor",
    )
    s.add_argument("--rail", type=float, default=2040.0, help="|sample| counted as clipped")

    s = command(
        "record",
        "save raw samples to .npz and print each channel's noise",
        cmd_record,
        uses_mains=True,
    )
    s.add_argument("--port", type=listen_port, default=DEFAULT_PORT, help=port_help)
    s.add_argument("--seconds", type=positive, default=10.0, help="how long")
    s.add_argument("--out", default="recording.npz", help="output file")

    s = command(
        "calibrate", "record each gesture and train an LDA model", cmd_calibrate, uses_mains=True
    )
    s.add_argument("--port", type=listen_port, default=DEFAULT_PORT, help=port_help)
    s.add_argument(
        "--rate",
        type=int_in(1, 65535),
        help="sample rate you expect; the device's own rate is used, this warns if they differ",
    )
    s.add_argument(
        "--gestures",
        type=gesture_list,
        default=",".join(GESTURES),
        help="comma-separated gesture names",
    )
    s.add_argument("--seconds", type=positive, default=5.0, help="recording length per gesture")
    s.add_argument("--out", default="model.npz", help="where to save the model")
    s.add_argument("--sim", action="store_true", help="calibrate from the simulator instead")
    s.add_argument("--seed", type=int, default=0, help="simulator seed (with --sim)")

    s = command(
        "forward", "classify live EMG and send gesture JSON over UDP", cmd_forward, uses_mains=True
    )
    s.add_argument("--port", type=listen_port, default=DEFAULT_PORT, help=port_help)
    s.add_argument("--model", default="model.npz", help="model from `myobridge calibrate`")
    s.add_argument(
        "--to",
        type=host_port,
        default=f"127.0.0.1:{DEFAULT_FORWARD_PORT}",
        help="host:port to send the gesture JSON to",
    )
    s.add_argument(
        "--lsl",
        nargs="?",
        const="myobridge",
        metavar="NAME",
        help="also publish the EMG and gestures on Lab Streaming Layer as NAME-emg and "
        'NAME-gestures (default name: myobridge; needs pip install "myobridge[lsl]")',
    )
    s.add_argument("--verbose", action="store_true", help="print every message sent")

    s = command(
        "demo", "offline end-to-end run: simulate, train, evaluate", cmd_demo, uses_mains=False
    )
    s.add_argument("--seed", type=int, default=0, help="random seed")
    s.add_argument(
        "--train-seconds",
        type=positive,
        default=8.0,
        help="calibration seconds per gesture",
    )
    s.add_argument(
        "--repeats",
        type=int_in(1, 1000),
        default=3,
        help="times through every gesture in the test session",
    )

    s = command(
        "fetch", "download the public EMG datasets (pinned versions)", cmd_fetch, uses_mains=False
    )
    s.add_argument("names", nargs="*", help="3dc, multiday (default: both)")
    s.add_argument("--data", help="folder for datasets (default: $MYOBRIDGE_DATA)")

    s = command(
        "bench", "run the benchmarks on real EMG and write results/", cmd_bench, uses_mains=False
    )
    s.add_argument(
        "names",
        nargs="*",
        help="multiday threedc aliasing quality latency conformal partial_calibration"
        " (default: all)",
    )
    s.add_argument("--data", help="folder for datasets (default: $MYOBRIDGE_DATA)")
    s.add_argument("--out", default="results", help="where to write the JSON results")
    s.add_argument("--figures", default="docs/figures", help="where to draw the figures")
    s.add_argument("--no-figures", action="store_true", help="skip the figures")
    s.add_argument("--quick", action="store_true", help="tiny subset: a smoke test, not results")
    s.add_argument("--no-download", action="store_true", help="use the data folder as it is")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except KeyboardInterrupt:  # Ctrl-C is the normal way to stop the live commands
        print()
    except (CliError, OSError) as err:
        sys.exit(f"myobridge {args.command}: error: {err}")


if __name__ == "__main__":
    main()
