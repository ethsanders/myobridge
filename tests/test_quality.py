import numpy as np
import pytest

from myobridge.quality import (
    Issue,
    QualityConfig,
    QualityMonitor,
    assess,
    rest_floor,
    scan_recording,
)

FS = 1000.0


def emg(seconds=1.0, n_ch=3, seed=0, scale=50.0):
    """Band-limited noise standing in for clean EMG."""
    from scipy import signal

    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (int(seconds * FS), n_ch))
    sos = signal.butter(4, [30, 300], btype="band", fs=FS, output="sos")
    return signal.sosfilt(sos, x, axis=0) * scale


CFG = QualityConfig(fs=FS, mains_hz=60.0, clip_low=-2048, clip_high=2047)


def test_clean_signal_passes():
    x = emg(seed=1)
    floor = rest_floor(emg(seed=2, scale=10.0), CFG)
    report = assess(x, CFG, floor)
    assert report.ok and report.bad_channels == []
    assert all(c.line_noise_share < 0.1 and c.motion_share < 0.05 for c in report.channels)


def test_each_fault_raises_its_own_check():
    floor = rest_floor(emg(seed=2, scale=10.0), CFG)
    x = emg(seed=3)
    t = np.arange(len(x)) / FS
    x[:, 0] *= 0.001  # lead off
    x[:, 1] += 200 * np.sin(2 * np.pi * 60 * t)  # strong hum
    x[:, 2] += 2200 * np.sin(2 * np.pi * 1.5 * t)  # cable tug, and it clips
    x = np.clip(x, -2048, 2047)
    r = assess(x, CFG, floor)
    assert r.channels[0].issues == (Issue.FLATLINE,)
    assert r.channels[1].issues == (Issue.LINE_NOISE,)
    assert set(r.channels[2].issues) == {Issue.CLIPPING, Issue.MOTION}
    assert r.bad_channels == [0, 1, 2] and not r.ok


def test_line_noise_counts_only_the_peak_over_the_emg_floor():
    # broadband EMG has power at 60 Hz too; that must not count as hum
    r = assess(emg(seconds=2, seed=4), CFG)
    assert max(c.line_noise_share for c in r.channels) < 0.1


def test_50hz_mains_and_harmonics():
    cfg = QualityConfig(fs=FS, mains_hz=50.0)
    x = emg(seed=5)
    t = np.arange(len(x)) / FS
    x[:, 0] += 150 * np.sin(2 * np.pi * 150 * t)  # 3rd harmonic only
    r = assess(x, cfg)
    assert Issue.LINE_NOISE in r.channels[0].issues and r.channels[1].ok


def test_flatline_needs_a_rest_floor():
    x = emg() * 0.0001
    assert assess(x, CFG).ok  # without a floor we can't tell quiet from dead
    assert not assess(x, CFG, rest_floor(emg(scale=10.0), CFG)).ok


def test_config_validation():
    with pytest.raises(ValueError):
        QualityConfig(fs=0)
    with pytest.raises(ValueError):
        QualityConfig(fs=FS, clip_low=0)
    with pytest.raises(ValueError):
        assess(np.zeros((8, 2)), CFG)


def test_streaming_monitor_reports_every_hop():
    mon = QualityMonitor(CFG, n_channels=3, hop_s=0.25)
    x = emg(seconds=3)
    reports = [mon.push(x[i : i + 20]) for i in range(0, len(x), 20)]
    got = [r for r in reports if r is not None]
    # first report once a full 1 s window exists, then one per 250 ms
    assert len(got) == 9 and all(r.ok for r in got)
    assert mon.window_samples == 1000


def test_scan_recording_uses_only_the_past():
    x = emg(seconds=3)
    x[2000:, 1] += 300 * np.sin(2 * np.pi * 60 * np.arange(1000) / FS)
    times, flagged = scan_recording(x, CFG, hop_s=0.25)
    assert times[0] == 0.25 and np.allclose(np.diff(times), 0.25)
    assert not flagged[times <= 2.0].any()  # nothing before the hum starts
    assert flagged[-1, 1] and not flagged[:, [0, 2]].any()


def test_scan_recording_can_count_only_some_issues():
    from myobridge.quality import DROP_CHANNEL_ON

    x = emg(seconds=2)
    x[:, 1] += 300 * np.sin(2 * np.pi * 60 * np.arange(len(x)) / FS)
    _, any_flag = scan_recording(x, CFG)
    _, drop = scan_recording(x, CFG, only=DROP_CHANNEL_ON)
    assert any_flag[-1, 1] and not drop.any()  # hum is reported but isn't a reason to drop


def test_rest_floor_ignores_one_noisy_second():
    quiet = emg(seconds=5.0, seed=7, scale=10.0)
    loud = quiet.copy()
    loud[2000:3000] *= 20.0  # a twitch in the middle of "rest"
    a, b = rest_floor(quiet, CFG), rest_floor(loud, CFG)
    assert np.allclose(a, b, rtol=0.2)
    # a short rest recording still works (one piece)
    assert rest_floor(quiet[:500], CFG).shape == (3,)
