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


CONFORMAL_MD = "docs/conformal.md"


def test_conformal() -> None:
    d = _load("conformal")
    md = d["multiday"]["rows"]
    head = md["adapt_temp_aps"]["0.1"]
    textbook = md["static_validation_days"]["0.1"]
    for doc in (README, RESULTS_MD, CONFORMAL_MD):
        assert_quoted(_text(doc), pct(head["coverage"]["mean"]), ci(head["coverage"]))
    assert_quoted(_text(README), pct(head["accuracy_when_singleton"]))
    for doc in (README, CONFORMAL_MD, "docs/index.md"):
        assert_quoted(_text(doc), pct(head["coverage"]["min"]))
    assert_quoted(_text("docs/index.md"), pct(head["coverage"]["mean"]))
    for doc in (RESULTS_MD, CONFORMAL_MD):
        assert_quoted(_text(doc), pct(textbook["coverage"]["mean"]), ci(textbook["coverage"]))
    page = _text(CONFORMAL_MD)
    for a in ("0.05", "0.1", "0.2"):
        r = md["adapt_temp_aps"][a]
        assert_quoted(
            page,
            f"{pct(r['coverage']['mean'])} {ci(r['coverage'])}",
            f"{r['mean_set_size']:.2f}",
            pct(r["singleton_rate"]),
            pct(r["accuracy_when_singleton"]),
        )
    assert_quoted(page, pct(md["adapt_validation_days"]["0.1"]["coverage"]["mean"]))
    assert_quoted(page, pct(md["static_temp_aps"]["0.1"]["coverage"]["mean"]))
    within = d["threedc"]["within_sitting"]["rows"]["temp_aps"]["0.1"]
    cross = d["threedc"]["cross_person"]["adapt"]["rows"]["temp_aps"]["0.1"]
    lead = d["threedc"]["faults"]["results"]["lead_off"]
    for doc in (RESULTS_MD, CONFORMAL_MD):
        text = _text(doc)
        for r in (within, cross, lead["monitor"]["0.1"], lead["no_monitor"]["0.1"]):
            assert_quoted(text, pct(r["coverage"]["mean"]), ci(r["coverage"]))
    shared = lead["monitor_shared_threshold"]["0.1"]
    assert_quoted(
        page,
        f"{pct(shared['coverage']['mean'])} {ci(shared['coverage'])}",
        f"{shared['mean_set_size']:.2f} gestures",
    )
    tail = d["multiday"]["without_last_second_of_hold"]
    for doc in (RESULTS_MD, CONFORMAL_MD):
        assert_quoted(
            _text(doc),
            pct(tail["share_of_test_windows_dropped"]),
            pct(tail["accuracy"]["adapted"]),
        )
    assert_quoted(
        page,
        pct(tail["rows"]["adapt_temp_aps"]["0.1"]["coverage"]["mean"]),
        pct(tail["rows"]["adapt_temp_aps_all_trimmed"]["0.1"]["coverage"]["mean"]),
    )
    # the comparison with simpler rules (README and the conformal page)
    base90 = head["baselines"]
    base95 = md["adapt_temp_aps"]["0.05"]["baselines"]
    top1 = base90["top1"]["coverage"]
    for doc in (README, CONFORMAL_MD):
        assert_quoted(
            _text(doc), pct(top1["min"]), pct(base90["max_prob_rejection"]["accuracy_when_named"])
        )
    assert_quoted(_text(README), pct(top1["mean"]), pct(base90["top1"]["share_within_5_points"]))
    assert base90["fixed_top_k"]["k"]["mean"] == 1 and base95["fixed_top_k"]["k"]["mean"] == 3
    assert_quoted(_text(README), pct(base95["fixed_top_k"]["coverage"]["mean"]))
    nested = base90["max_prob_rejection_nested"]
    assert_quoted(
        page,
        pct(top1["max"]),
        pct(head["coverage"]["max"]),
        pct(base95["fixed_top_k"]["coverage"]["mean"]),
        pct(base95["fixed_top_k"]["coverage"]["min"]),
        pct(md["adapt_temp_aps"]["0.05"]["coverage"]["min"]),
        pct(nested["named_one_share"]),
        pct(nested["accuracy_when_named"]),
    )
    frozen = d["multiday"]["confidence_rejection"]["adapt_temperature"]["0.95"]
    for doc in (README, CONFORMAL_MD):
        assert_quoted(_text(doc), pct(frozen["accuracy_when_answering"]["min"]))
    assert_quoted(page, pct(frozen["accuracy_when_answering"]["mean"]))
    chosen = d["multiday"]["selection_check"]["rows"]
    assert chosen["adapt_temp_aps"]["0.1"]["share_within_5_points"] == 1.0
    assert all(
        r["0.1"]["share_within_5_points"] < 1.0 for k, r in chosen.items() if k != "adapt_temp_aps"
    )
    assert_quoted(page, pct(chosen["adapt_temp_aps"]["0.1"]["coverage"]))
    few = d["multiday"]["few_sessions"]["0.1"]
    for doc in (CONFORMAL_MD, "docs/your-promise.md"):
        assert_quoted(
            _text(doc), pct(few["coverage_over_runs"]["mean"]), pct(few["worst_day_over_runs"])
        )
    assert_quoted(
        page,
        f"{pct(few['coverage_over_runs']['min'])} to {pct(few['coverage_over_runs']['max'])}",
        f"{few['coverage_over_runs']['n']} runs",
    )
    seeds = d["multiday"]["seed_spread_adapt_temp_aps"]["by_alpha"]["0.1"]["coverage"]
    assert_quoted(page, f"{pct(seeds['min'])} to {pct(seeds['max'])}")


def test_partial_calibration() -> None:
    d = _load("partial_calibration")
    one = d["by_calibration_seconds"]["1.0"]
    share = d["shared_translation_share"]
    rest = one["shift_rest"]["minus_none"]
    assert_quoted(
        _text(RESULTS_MD),
        f"{100 * rest['mean']:.1f} points {ci(rest)}",
        f"{100 * one['adapt_all']['minus_none']['mean']:.1f}",
        pct(one["hybrid_rest+2"]["test"]["mean"]),
        pct(one["shift_all"]["test"]["mean"]),
        f"{pct(share['mean'])} {ci(share)}",
    )


def test_readme_example_runs() -> None:
    """The quick-start code in the README is real code."""
    text = _text(README)
    start = text.index("```python\n") + len("```python\n")
    code = text[start : text.index("```", start)]
    exec(compile(code, "README.md", "exec"), {})


def test_demo_payload_matches_the_benchmark() -> None:
    """The demo page reads docs/demo/days.json, a link to results/demo.json."""
    link = DOCS / "demo" / "days.json"
    assert link.resolve() == (ROOT / "results" / "demo.json").resolve()
    demo = json.loads(link.read_text())
    conf = _load("conformal")
    md, h = conf["multiday"]["rows"], demo["headline"]
    for key, row in (
        ("adapt-myobridge", "adapt_temp_aps"),
        ("adapt-textbook", "adapt_validation_days"),
        ("static-myobridge", "static_temp_aps"),
        ("static-textbook", "static_validation_days"),
    ):
        assert h["multiday"][key]["mean"] == md[row]["0.1"]["coverage"]["mean"]
    lead = conf["threedc"]["faults"]["results"]["lead_off"]
    assert h["threedc_lead_off"]["no_monitor"] == lead["no_monitor"]["0.1"]["coverage"]["mean"]
    assert h["threedc_temperature"] == conf["threedc"]["within_sitting"]["temperature"]
    assert demo["typical_day"] in [d["day"] for d in demo["days"]]
    # the textbook rows don't depend on a random draw, so they match day by day
    for d in demo["days"]:
        n = len(d["truth"])
        for variant, row in (
            ("static-textbook-ok", "static_validation_days"),
            ("adapt-textbook-ok", "adapt_validation_days"),
        ):
            got = d["variants"][variant]["check"]["covered"] / n
            assert abs(got - md[row]["0.1"]["per_group"][str(d["day"])]["coverage"]) < 1e-6
        # the page quotes the bench's own day beside its count
        own = h["bench_per_day"][str(d["day"])]
        assert (
            own["adapt-myobridge"]
            == md["adapt_temp_aps"]["0.1"]["per_group"][str(d["day"])]["coverage"]
        )
        # myobridge's random tie-breaks come from another stream here, so the
        # page's count may move, but only by about a point (the page says so)
        got = d["variants"]["adapt-myobridge-ok"]["check"]["covered"] / n
        assert abs(got - own["adapt-myobridge"]) < 0.025
    within = md["adapt_validation_days"]["0.1"]["share_within_5_points"]
    assert h["spread"]["adapt-textbook"]["within_5_points"] == pytest.approx(within, abs=1e-6)
