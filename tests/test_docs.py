"""The headline numbers in the docs must match the benchmark results.

The docs quote numbers from ``results/*.json``. If a benchmark is rerun and a
number moves, these tests fail until the docs are updated, so the README can't
quietly drift away from what the code produces. A results file that isn't
there (a fresh clone before ``myobridge bench``) skips its tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from myobridge.quality import QualityConfig

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DOCS = ROOT / "docs"


def _load(name: str) -> dict[str, Any]:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not generated")
    data: dict[str, Any] = json.loads(path.read_text())
    if data.get("quick"):
        pytest.skip(f"{path.name} is from a --quick run")
    return data


def _text(*names: str) -> str:
    return "\n".join((ROOT / n).read_text(encoding="utf-8") for n in names)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def ci(stats: dict[str, float]) -> str:
    return f"[{100 * stats['ci95_low']:.1f}, {100 * stats['ci95_high']:.1f}]"


def points(stats: dict[str, float]) -> str:
    return f"{100 * stats['mean']:+.1f}"


def assert_quoted(text: str, *snippets: str) -> None:
    missing = [s for s in snippets if s not in text]
    assert not missing, f"not found in docs: {missing}"


README = "README.md"
RESULTS_MD = "docs/results.md"
QUALITY_MD = "docs/signal-quality.md"


def test_within_session_and_channels() -> None:
    d = _load("threedc")
    within = d["within_subject"]["accuracy"]
    even = d["channels"]["even_4"]["accuracy"]
    for doc in (README, RESULTS_MD):
        assert_quoted(_text(doc), pct(within["mean"]), ci(within), pct(even["mean"]), ci(even))
    by_k = d["channels"]["by_k"]
    assert_quoted(_text(RESULTS_MD), *(pct(by_k[k]["mean_over_subsets"]) for k in ("1", "4", "8")))


def test_cross_subject() -> None:
    cal = _load("threedc")["cross_subject"]["by_calibration_seconds"]["2.0"]
    none = cal["methods"]["none"]["accuracy"]
    adapt = cal["methods"]["adapt"]["accuracy"]
    gain = cal["paired_differences"]["adapt_minus_new_only"]
    for doc in (README, RESULTS_MD):
        assert_quoted(_text(doc), pct(none["mean"]), pct(adapt["mean"]))
    assert_quoted(_text(README), f"{100 * gain['mean']:.1f} points", ci(gain))


def test_drift_and_recalibration() -> None:
    d = _load("multiday")
    cal = d["recalibration"]["by_calibration_seconds"]["1.0"]
    none = cal["methods"]["none"]["accuracy"]
    adapt = cal["methods"]["adapt"]["accuracy"]
    gain = cal["paired_differences"]["adapt_minus_none"]
    assert_quoted(_text(README), pct(none["mean"]), ci(none), pct(adapt["mean"]), ci(adapt))
    assert_quoted(_text(README), f"{points(gain)} points {ci(gain)}")
    decay = d["decay_by_gap_days"]
    assert_quoted(_text(RESULTS_MD), *(pct(decay[g]["mean"]) for g in ("1", "30", "90")))
    assert_quoted(_text(RESULTS_MD), pct(d["recalibration"]["same_day_only"]["accuracy"]["mean"]))


def test_aliasing() -> None:
    cases = _load("aliasing")["white_noise_model"]["cases"]
    before = f"+{cases['order1_fs1000']['noise_floor_increase_db']:.1f} dB"
    after = f"+{cases['order1_fs2000_halfband']['noise_floor_increase_db']:.1f} dB"
    for doc in (README, RESULTS_MD, "docs/aliasing.md"):
        text = _text(doc)
        assert_quoted(text, before)
        assert after in text or after.replace(" dB", "") in text


def test_latency() -> None:
    d = _load("latency")
    median = f"{d['per_packet_us']['median']:.0f} µs"
    p99 = f"{d['per_packet_us']['p99']:.0f} µs"
    share = f"{100 * d['cpu_share_of_real_time']:.1f}%"
    adapt = f"{d['adapt_ms']['mean']:.2f} ms"
    assert_quoted(_text(RESULTS_MD), median, p99, share, adapt)
    assert_quoted(_text(README), median, p99, share)


def test_fault_tolerance() -> None:
    res = _load("quality")["fault_tolerance"]["results"]
    lead = res["lead_off"]
    for doc in (README, QUALITY_MD):
        assert_quoted(_text(doc), pct(lead["no_monitor"]["mean"]), pct(lead["monitor"]["mean"]))
    assert_quoted(_text(README), ci(lead["no_monitor"]), ci(lead["monitor"]))
    rows = [
        pct(res[f][k]["mean"])
        for f in ("lead_off", "clipping", "line_noise", "motion")
        for k in ("no_monitor", "monitor", "monitor_drop_on_any_flag")
    ]
    assert_quoted(_text(QUALITY_MD), *rows)


def test_false_alarms() -> None:
    fa = _load("quality")["false_alarms_clean"]
    assert_quoted(_text(QUALITY_MD), pct(fa["any_check"]), pct(fa["per_check"]["flatline"]))


def test_quality_defaults_match_benchmark() -> None:
    chosen = _load("quality")["thresholds"]["chosen"]
    cfg = QualityConfig(fs=1000.0, mains_hz=60.0)
    for name, value in chosen.items():
        assert getattr(cfg, name) == pytest.approx(value, abs=0.025), name
    table = _text(QUALITY_MD)
    assert_quoted(table, *(f"{value:.3f}" for value in chosen.values()))


def test_readme_example_runs() -> None:
    """The quick-start code in the README is real code."""
    text = _text(README)
    start = text.index("```python\n") + len("```python\n")
    code = text[start : text.index("```", start)]
    exec(compile(code, "README.md", "exec"), {})
