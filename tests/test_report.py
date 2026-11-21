"""`myobridge report`: hardware evidence from sessions, checked without hardware."""

import json

import numpy as np
import pytest

from myobridge.cli import ESP32_RAIL
from myobridge.faults import lead_off, mains_hum
from myobridge.protocol import SequenceTracker, decode_packet, encode_packet
from myobridge.receiver import ArrivalLog, Receiver
from myobridge.report import (
    Recording,
    SignalChain,
    analyse,
    filtered_rms,
    load_inputs,
    render_markdown,
    write_report,
)
from myobridge.sessions import Session
from myobridge.simulator import EmgSimulator

GESTURES = ("rest", "fist", "open")
FS = 1000


def sim_session(seed, seconds=4.0, gestures=GESTURES, **sim):
    s = EmgSimulator(seed=seed, **sim)
    return Session(float(FS), tuple(gestures), tuple(s.record(g, seconds) for g in gestures))


def checks_session(seed=7):
    """rest, then a fist with ch0's electrode lifted, then rest with strong hum."""
    s = EmgSimulator(seed=seed, hum_amplitude=0.0)
    rest = s.record("rest", 4.0)
    lifted = s.record("fist", 4.0).astype(np.float64)
    lifted[:, 0] = lead_off(lifted[:, 0], 0.003)
    hummed = s.record("rest", 4.0).astype(np.float64)
    band = filtered_rms(rest, FS, 60.0)
    for ch in range(hummed.shape[1]):
        hummed[:, ch] = mains_hum(hummed[:, ch], band[ch], FS, 60.0, 8.0, np.random.default_rng(ch))
    holds = (rest, lifted.astype(np.int16), hummed.astype(np.int16))
    return Session(float(FS), ("rest", "lifted", "mains"), holds)


@pytest.fixture(scope="module")
def days():
    return [sim_session(seed) for seed in range(3)]


@pytest.fixture(scope="module")
def written(tmp_path_factory, days):
    out = tmp_path_factory.mktemp("report")
    free = EmgSimulator(seed=9)
    free_raw = np.vstack([free.record(g, 1.5) for g in ("rest", "fist", "open", "rest")])
    recording = Recording(float(FS), free_raw, "free.npz")
    result = write_report([*days, checks_session()], out, [recording], SignalChain(gain=1000))
    return out, result


def test_the_report_writes_every_file(written):
    out, result = written
    for name in ("report.md", "report.json", "traces.png", "spectra.png", "gesture_rms.png"):
        assert (out / name).stat().st_size > 1000, name
    assert (out / "confusion.png").exists()
    assert json.loads((out / "report.json").read_text())["decoding"] == result["decoding"]
    text = (out / "report.md").read_text()
    for heading in (
        "## Recordings",
        "## Noise floor at rest",
        "## Deliberate faults",
        "## Free use",
        "## Link",
        "## For the README",
    ):
        assert heading in text
    assert "![Raw traces](traces.png)" in text


def test_the_numbers_are_sane(written):
    _, result = written
    floors = [c for s in result["signal"] for c in s.get("rest", [])]
    assert len(floors) == 4 * 4  # three days and the checks session, 4 channels each
    for c in floors:
        assert 1 < c["rms_counts"] < 100  # the simulator's rest is tens of counts
        assert c["rms_uv_at_electrodes"] == pytest.approx(c["rms_counts"] * 3.3 / 4096 / 1000 * 1e6)
        assert 0 <= c["hum_share"] <= 1
    dec = result["decoding"]
    assert dec["gestures"] == sorted(GESTURES) and dec["chance"] == pytest.approx(1 / 3)
    assert dec["within_session"]["accuracy"] > 0.8
    assert len(dec["within_session"]["sessions"]) == 3  # the checks session isn't decoded
    assert len(dec["leave_one_session_out"]["sessions"]) == 3
    across = dec["across_days"]
    assert len(across["train"]) == 2
    assert across["with_adapt"]["accuracy"] >= across["without_adapt"]["accuracy"] - 0.05
    m = np.asarray(across["with_adapt"]["confusion"])
    assert m.sum() == across["with_adapt"]["windows"]
    rms = np.asarray(result["gesture_rms"]["rms_counts"])
    rest_row = result["gesture_rms"]["gestures"].index("rest")
    assert (rms[rest_row] < rms.max(axis=0)).all()  # every channel is quietest at rest


def test_the_monitor_catches_the_deliberate_faults(written):
    _, result = written
    holds = {h["label"]: h for h in result["signal"][-1]["holds"]}
    lifted = holds["lifted"]["flags"]["flatline"]
    assert lifted[0] > 0.9 and max(lifted[1:]) < 0.1
    assert min(holds["mains"]["flags"]["line_noise"]) > 0.9
    assert min(holds["mains"]["hum_share"]) > 0.6


def test_free_use_is_scanned_against_a_rest_floor(written):
    _, result = written
    (free,) = result["free_use"]
    assert free["seconds"] == pytest.approx(6.0)
    assert "flatline compared with the rest hold" in free["note"]
    assert all(0 <= v <= 1 for v in free["flags"]["any"])


def test_microvolts_by_hand():
    """A 100 Hz sine of 400 counts peak: RMS 400/sqrt(2) = 282.8 counts.
    At 3.3 V over 4096 counts that is 227.9 mV at the ADC, and with a gain of
    500, 455.8 µV at the electrodes."""
    t = np.arange(5 * FS) / FS
    rest = (37 + 400 * np.sin(2 * np.pi * 100 * t))[:, None].repeat(2, axis=1)
    fist = rest * 3
    session = Session(float(FS), ("rest", "fist"), (rest, fist))
    chain = SignalChain(gain=500.0)
    assert chain.mv_per_count == pytest.approx(3300 / 4096)
    assert chain.to_uv(1.0) == pytest.approx(3.3 / 4096 / 500 * 1e6)  # 1.611 µV per count
    (c, _) = analyse([session], chain=chain)["signal"][0]["rest"]
    assert c["rms_counts"] == pytest.approx(282.84, rel=0.01)
    assert c["rms_mv_at_adc"] == pytest.approx(227.88, rel=0.01)
    assert c["rms_uv_at_electrodes"] == pytest.approx(455.8, rel=0.01)
    assert c["mean_counts"] == pytest.approx(37, abs=0.5)
    assert c["hum_share"] < 0.01  # 100 Hz isn't mains


def test_without_a_gain_there_are_no_microvolts(days):
    result = analyse(days[:1])
    assert result["signal_chain"]["gain"] is None
    assert all(c["rms_uv_at_electrodes"] is None for c in result["signal"][0]["rest"])
    text = render_markdown(result, {})
    assert "µV at electrodes" not in text and "No microvolts" in text
    with pytest.raises(ValueError, match="gain"):
        SignalChain(gain=0)


def test_the_rail_matches_the_cli():
    assert SignalChain().rail == ESP32_RAIL


def test_a_session_saved_before_link_stats_still_loads(tmp_path, days):
    """The format before link stats: meta without a "link" key."""
    meta = {"fs": 1000.0, "labels": list(GESTURES), "mains_hz": 60.0, "recorded_at": "2026-11-01"}
    path = tmp_path / "old.npz"
    holds = {f"hold_{i}": h for i, h in enumerate(days[0].holds)}
    np.savez(path, meta=np.array(json.dumps(meta)), **holds)
    old = Session.load(path)
    assert old.link is None and old.labels == GESTURES
    result = analyse([old])
    assert result["link"] == [] and result["files"][0]["link_recorded"] is False
    assert "Not recorded." in render_markdown(result, {}).split("## Link")[1]


def test_link_stats_round_trip_and_reach_the_report(tmp_path, days):
    link = {
        "received": 990,
        "dropped": 10,
        "gaps": 3,
        "longest_gap": 6,
        "corrupt": 2,
        "reordered": 1,
        "duplicates": 0,
        "restarts": 0,
        "loss_percent": 1.0,
        "invalid_by_kind": {"crc": 2},
        "timing": {"packets_timed": 900, "extra_delay_ms": {"p95": 4.0, "p99": 9.5}},
    }
    s = Session(1000.0, GESTURES, days[0].holds, link=link)
    back = Session.load(s.save(tmp_path / "with_link"))
    assert back.link == link
    sessions, recordings = load_inputs([tmp_path / "with_link.npz"])
    assert recordings == [] and sessions[0].link == link
    text = render_markdown(analyse(sessions), {})
    assert "| with_link.npz | 990 | 10 (1.00%) | 3 | 6 | 2 of 2 |" in text
    assert "1.00% of 1000 packets lost, 99% within 10 ms of the quickest" in text


def test_a_record_file_from_before_link_stats(tmp_path):
    path = tmp_path / "rec.npz"
    np.savez(path, samples=np.zeros((2000, 4), np.int16), fs=1000, dropped_packets=3)
    rec = Recording.load(path)
    assert rec.link == {"dropped": 3, "partial": True}
    with pytest.raises(ValueError, match="at least one session"):
        analyse([], [rec])
    np.savez(tmp_path / "junk.npz", other=np.zeros(3))
    with pytest.raises(ValueError, match="not a myobridge recording"):
        load_inputs([tmp_path / "junk.npz"])


def test_short_or_mismatched_sessions_are_explained():
    short = sim_session(1, seconds=2.0)
    dec = analyse([short])["decoding"]
    assert "within_session" not in dec and "3 s" in dec["skipped"]["why"]
    other = sim_session(2, gestures=("rest", "fist"))
    dec = analyse([sim_session(1), other])["decoding"]
    assert "holds" in dec["across_sessions_skipped"]


def test_a_saved_model_is_scored_on_every_session(tmp_path, days):
    from myobridge.sessions import fit_sets

    model_path, _ = fit_sets(days).save(tmp_path / "model")
    scores = analyse(days, model=model_path)["model"]
    assert len(scores["sessions"]) == 3
    for s in scores["sessions"]:
        assert 0 <= s["without_adapt"]["accuracy"] <= 1
        assert s["sets_after_recalibration"]["promised"] == pytest.approx(0.9)


# ------------------------------------------------------------- link capture


def test_arrival_log_measures_jitter():
    log = ArrivalLog()
    rng = np.random.default_rng(0)
    jitter = rng.exponential(0.002, 500)  # seconds of extra delay, mostly small
    for i in range(500):
        packet = encode_packet(np.zeros((20, 4), np.int16), i, i * 20_000, 1000)
        log.add(100.0 + i * 0.020 * (1 + 50e-6) + 0.005 + jitter[i], decode_packet(packet))
    s = log.summary()
    assert s["packets_timed"] == 500
    assert s["clock_drift_ppm"] == pytest.approx(50, abs=5)
    assert s["extra_delay_ms"]["p50"] == pytest.approx(
        np.median(jitter - jitter.min()) * 1e3, abs=0.3
    )
    assert s["interval_error_ms"]["pairs"] == 499
    assert ArrivalLog().summary() == {"packets_timed": 0}


def test_receiver_counts_crc_errors_and_times_packets():
    with Receiver("127.0.0.1", 0, timeout_s=0.1) as rx:
        good = encode_packet(np.zeros((20, 4), np.int16), 0, 0, 1000)
        bad = bytearray(good)
        bad[25] ^= 1
        assert rx.handle_datagram(bytes(bad)) is None
        rx.arrivals = ArrivalLog()
        rx.sock.sendto(good, ("127.0.0.1", rx.port))
        assert rx.receive() is not None
        link = rx.link_summary()
    assert link["invalid_by_kind"] == {"crc": 1} and link["corrupt"] == 1
    assert link["received"] == 1 and link["timing"] == {"packets_timed": 1}


def test_sequence_gaps_are_counted():
    tracker = SequenceTracker()
    for seq in (0, 1, 4, 5, 6, 10, 11):
        tracker.accept(seq)
    assert tracker.stats.dropped == 5
    assert tracker.stats.gaps == 2 and tracker.stats.longest_gap == 3
