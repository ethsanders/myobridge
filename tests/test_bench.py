"""The benchmarks, end to end, on tiny fake datasets (the real ones aren't in CI).

These check that every experiment runs, respects its splits, and writes a
complete results file. The numbers themselves only mean something on the
real data; see docs/results.md.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from fakes import write_multiday, write_threedc

from myobridge import bench
from myobridge.bench import common, multiday_bench
from myobridge.classifier import LDAClassifier
from myobridge.cli import main


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("data")
    days = (
        multiday_bench.TRAIN_DAYS[:3] + multiday_bench.VAL_DAYS[:3] + multiday_bench.TEST_DAYS[:6]
    )
    write_multiday(root / "multiday", days=days, seconds=4.0)
    write_threedc(root / "3dc", subjects=[1, 2, 7, 8], seconds=4.0)
    return root


def opts(data, tmp_path):
    return common.RunOptions(data, tmp_path / "results", None, quick=True, download=False)


def test_multiday_quick(data, tmp_path):
    out = bench.EXPERIMENTS["multiday"](opts(data, tmp_path))
    rc = out["recalibration"]["by_calibration_seconds"]
    assert set(rc) == {"0.25", "0.5", "1.0"}
    assert set(rc["1.0"]["methods"]) == set(common.METHODS)
    assert out["recalibration"]["test_days"] == multiday_bench.TEST_DAYS[:6]
    assert out["recalibration"]["same_day_only"]["calibration_seconds"] == 1.5
    saved = json.loads((tmp_path / "results" / "multiday.quick.json").read_text())
    assert saved["quick"] is True
    assert (
        saved["provenance"]["dataset"]["commit"] == multiday_bench.datasets.multiday.SOURCE.commit
    )


def test_threedc_quick(data, tmp_path):
    out = bench.EXPERIMENTS["threedc"](opts(data, tmp_path))
    assert out["within_subject"]["accuracy"]["n"] == 4
    assert out["within_subject"]["accuracy"]["mean"] > 0.5  # the fake data is separable
    assert set(out["windows"]) == {"0.1", "0.15", "0.2", "0.25", "0.3"}
    assert (
        out["cross_subject"]["by_calibration_seconds"]["2.0"]["methods"]["adapt"]["accuracy"]["n"]
        == 2
    )


def test_quality_quick(data, tmp_path):
    out = bench.EXPERIMENTS["quality"](opts(data, tmp_path))
    assert set(out["detection"]) == {"lead_off", "line_noise", "motion", "clipping"}
    assert out["detection"]["lead_off"]["x0.01"]["detected"] == 1.0
    tol = out["fault_tolerance"]["results"]["lead_off"]
    assert set(tol) == {"clean", "no_monitor", "monitor", "monitor_drop_on_any_flag", "oracle_drop"}
    assert tol["monitor"]["mean"] >= tol["no_monitor"]["mean"]


def test_aliasing_quick(data, tmp_path):
    out = bench.EXPERIMENTS["aliasing"](opts(data, tmp_path))
    assert set(out["emulated_1024hz"]) == {"ideal", "rc", "rc_ideal", "none"}
    assert out["white_noise_model"]["cases"]["order1_fs1000"]["noise_floor_increase_db"] > 2.5


def test_latency_quick(tmp_path):
    out = bench.EXPERIMENTS["latency"](common.RunOptions(None, tmp_path, None, quick=True))
    assert out["features_batch_vs_loop"]["identical"]
    assert out["per_packet_us"]["median"] > 0
    assert 0 < out["adapt_ms"]["mean"] < 100


def test_cli_bench_and_unknown_names(data, tmp_path):
    main(["bench", "latency", "--quick", "--out", str(tmp_path), "--no-figures"])
    assert (tmp_path / "latency.quick.json").exists()
    with pytest.raises(KeyError):
        bench.run(["nope"], opts(data, tmp_path))


def test_recalibrate_methods():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 3)) + np.repeat(np.eye(3) * 4, 100, axis=0)
    y = np.repeat([0, 1, 2], 100)
    base = LDAClassifier().fit(X, y)
    cal_X, cal_y = X[::10] + 1, y[::10]
    assert common.recalibrate("none", base, X, y, cal_X, cal_y) is base
    for m in ("new_only", "pooled"):
        assert common.recalibrate(m, base, X, y, cal_X, cal_y).score(cal_X, cal_y) > 0.9
    setting = common.AdaptSetting(5.0, 0.0)
    assert common.recalibrate("adapt", base, X, y, cal_X, cal_y, setting).means_ is not None
    with pytest.raises(ValueError):
        common.recalibrate("adapt", base, X, y, cal_X, cal_y)
    with pytest.raises(KeyError):
        common.recalibrate("magic", base, X, y, cal_X, cal_y)


def test_best_setting_breaks_ties_by_grid_order():
    a, b = common.ADAPT_GRID[3], common.ADAPT_GRID[1]
    assert common.best_setting({a: [0.5, 0.7], b: [0.6, 0.6]}) == b
    assert common.best_setting({a: [0.9], b: [0.1]}) == a


def test_results_are_plain_json(tmp_path):
    o = common.RunOptions(None, tmp_path, None)
    path = common.write_result(o, "x", {"a": np.float64(1 / 3), "b": np.arange(3), "c": (1, 2)})
    assert json.loads(path.read_text()) == {
        "quick": False,
        "a": 0.333333,
        "b": [0, 1, 2],
        "c": [1, 2],
    }


def test_figures_draw_from_results(data, tmp_path):
    """Every figure renders from a complete set of results files."""
    pytest.importorskip("matplotlib")
    from myobridge.bench.figures import make_all

    results = tmp_path / "results"
    for name in ("multiday", "threedc", "aliasing", "quality"):
        bench.EXPERIMENTS[name](opts(data, tmp_path))
        (results / f"{name}.quick.json").rename(results / f"{name}.json")
    paths = make_all(results, tmp_path / "figures")
    assert {p.name for p in paths} == {
        "multiday_recalibration.png",
        "threedc_channels_cross_subject.png",
        "threedc_confusion.png",
        "aliasing_noise.png",
        "quality.png",
    }
    assert all(p.stat().st_size > 10_000 for p in paths)


def test_figures_skip_missing_results(tmp_path):
    pytest.importorskip("matplotlib")
    from myobridge.bench.figures import make_all

    assert make_all(tmp_path, tmp_path / "figures") == []


def test_thresholds_ignore_test_data(data):
    """Wrecking every test-session recording leaves training features unchanged."""
    from myobridge import datasets
    from myobridge.bench import threedc_bench
    from myobridge.evaluation import FeatureConfig

    ds = datasets.load("3dc", root=data, download=False, subjects=[1])
    p = threedc_bench.Prepared(ds, FeatureConfig())
    train_idx = p.indices([1], [threedc_bench.TRAIN])
    before = p.table(train_idx, train_idx).X.copy()
    for i in p.indices([1], [threedc_bench.TEST]):
        p.clean[i] = p.clean[i] * 1000.0 + 5.0
    after = p.table(train_idx, train_idx).X
    assert np.array_equal(before, after)
    both = p.table(p.indices([1]), train_idx)
    assert np.array_equal(both.where(sessions=[threedc_bench.TRAIN]).X, before)


def test_conformal_quick(data, tmp_path):
    from myobridge.bench import conformal_bench as cb

    out = bench.EXPERIMENTS["conformal"](opts(data, tmp_path))
    rows = out["multiday"]["rows"]
    assert set(rows) == {
        "static_training_days",
        "static_validation_days",
        "adapt_validation_days",
        "adapt_same_day",
        "adapt_blocked_cross",
        "adapt_aci",
        "static_temp_aps",
        "adapt_temp_aps",
        "adapt_temp_aps_mondrian",
        "adapt_temp_aps_swapped",
        "adapt_temp_lac",
    }
    test_days = multiday_bench.TEST_DAYS[:6]
    assert out["multiday"]["test_days"] == test_days
    # the temperature and threshold days are disjoint and both before the test days
    t_days, q_days = out["multiday"]["temperature_days"], out["multiday"]["threshold_days"]
    assert t_days and q_days and not set(t_days) & set(q_days)
    assert max(t_days + q_days) < min(test_days)
    row = rows["adapt_temp_aps"]
    for a in cb.ALPHAS:
        r = row[str(a)]
        assert r["target"] == pytest.approx(1 - a)
        assert set(r["per_group"]) == {str(d) for d in test_days}
        assert 0.0 <= r["coverage"]["mean"] <= 1.0
        assert len(r["coverage_by_gesture"]) == 11
        assert sum(b["n_windows"] for b in r["by_time_into_hold"].values()) > 0
    spread = out["multiday"]["seed_spread_adapt_temp_aps"]
    assert spread["seeds"] == cb.SEEDS
    assert spread["by_alpha"]["0.1"]["coverage"]["n"] == len(cb.SEEDS)
    tail = out["multiday"]["without_last_second_of_hold"]
    assert 0.0 < tail["share_of_test_windows_dropped"] < 1.0
    assert set(tail["rows"]["adapt_temp_aps"]["0.1"]["per_group"]) == {str(d) for d in test_days}
    within = out["threedc"]["within_sitting"]
    assert within["rows"]["temp_aps"]["0.1"]["coverage"]["n"] == 2  # persons 7 and 8
    assert out["threedc"]["cross_person"]["adapt"]["rows"]["lac"]["0.1"]["coverage"]["n"] == 2
    flt = out["threedc"]["faults"]["results"]["lead_off"]
    assert set(flt) == {"clean", "no_monitor", "monitor", "monitor_shared_threshold", "oracle_drop"}
    saved = json.loads((tmp_path / "results" / "conformal.quick.json").read_text())
    assert saved["datasets"]["3dc"]["commit"] == out["datasets"]["3dc"]["commit"]


def test_partial_calibration_quick(data, tmp_path):
    from myobridge.bench import partial_bench as pb

    out = bench.EXPERIMENTS["partial_calibration"](opts(data, tmp_path))
    by_s = out["by_calibration_seconds"]
    assert set(by_s) == {str(s) for s in pb.CAL_SECONDS}
    methods = by_s["1.0"]
    assert set(methods) == {
        "none",
        "adapt_all",
        "adapt_rest",
        "shift_rest",
        "gain_rest",
        "shift_all",
        "gain_all",
        "hybrid_rest+1",
        "hybrid_rest+2",
    }
    for name, r in methods.items():
        assert len(r["per_day"]) == len(out["test_days"]) == 6
        assert 0.0 <= r["test"]["mean"] <= 1.0
        if name != "none":
            assert r["minus_none"]["n"] == 6
    extra = methods["hybrid_rest+2"]["extra_gestures"]
    assert len(set(extra)) == 2 and pb.REST not in extra
    assert extra[0] == methods["hybrid_rest+1"]["extra_gestures"][0]
    assert out["end_of_hold"]["non_rest_holds"] > 0
    assert 0.0 <= out["end_of_hold"]["share_predicted_rest"] <= 1.0
    assert (tmp_path / "results" / "partial_calibration.quick.json").exists()


def test_partial_shift_and_hybrid_by_hand():
    from myobridge.bench import partial_bench as pb
    from myobridge.evaluation import FeatureTable

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(c * 3.0, 1.0, size=(50, 2)) for c in range(3)])
    y = np.repeat([0, 1, 2], 50)
    base = LDAClassifier().fit(X, y)
    n = 30
    offset = np.array([1.0, -2.0])
    cal_X = np.vstack([np.tile(base.means_[0] + offset, (n, 1)), base.means_[1] + 5])
    cal_y = np.r_[np.zeros(n, int), np.ones(1, int)]
    z = np.zeros(len(cal_y))
    zi = np.zeros(len(cal_y), dtype=np.intp)
    cal = FeatureTable(cal_X, cal_y, zi, zi, zi, zi, z, z + 0.2, n_channels=1)
    # all classes move by rest's offset, shrunk by n / (n + prior)
    moved = pb.shared_shift(base, cal, [0], prior=10.0)
    np.testing.assert_allclose(moved.means_, base.means_ + offset * n / (n + 10.0))
    # the calibrated classes get their own means, the others the shared shift
    hyb = pb.hybrid(base, cal, [0, 1], prior=0.0)
    np.testing.assert_allclose(hyb.means_[0], base.means_[0] + offset)
    np.testing.assert_allclose(hyb.means_[1], base.means_[1] + 5)
    shared = (n * offset + 1 * np.array([5.0, 5.0])) / (n + 1)
    np.testing.assert_allclose(hyb.means_[2], base.means_[2] + shared)


def test_demo_quick(data, tmp_path):
    from myobridge.bench import demo_bench

    o = opts(data, tmp_path)
    with pytest.raises(FileNotFoundError, match="bench conformal"):
        bench.EXPERIMENTS["demo"](o)
    bench.EXPERIMENTS["conformal"](o)
    out = bench.EXPERIMENTS["demo"](o)
    saved = json.loads((tmp_path / "results" / "demo.quick.json").read_text())
    assert saved["days"][0]["day"] == out["days"][0]["day"]
    days = out["days"]
    assert [d["day"] for d in days] == multiday_bench.TEST_DAYS[:2]
    k = len(out["gestures"])
    for d in days:
        n = len(d["truth"])
        assert n == len(d["t"]) > 0
        assert all(b >= a for a, b in zip(d["t"], d["t"][1:], strict=False))  # play order
        assert len(d["variants"]) == 12
        for name, v in d["variants"].items():
            assert len(v["sets"]) == len(v["top"]) == len(v["dropped"]) == n
            assert all(0 <= s < 2**k for s in v["sets"])
            if not name.endswith("off_monitor"):
                assert not any(v["dropped"])
        # the electrode lifts halfway: the first half of each variant is untouched
        first = [i for i, t in enumerate(d["t"]) if t <= d["fault_from"]]
        ok, off = d["variants"]["adapt-myobridge-ok"], d["variants"]["adapt-myobridge-off"]
        assert first and [ok["top"][i] for i in first] == [off["top"][i] for i in first]
        seg = d["segments"]
        assert [s["gesture"] for s in seg] == sorted(s["gesture"] for s in seg)
        assert seg[-1]["start"] + seg[-1]["length"] >= d["t"][-1] - 1e-6
    h = out["headline"]
    conf = json.loads((tmp_path / "results" / "conformal.quick.json").read_text())
    assert h["multiday"]["adapt-myobridge"]["mean"] == pytest.approx(
        conf["multiday"]["rows"]["adapt_temp_aps"]["0.1"]["coverage"]["mean"]
    )
    assert h["threedc_temperature"] == conf["threedc"]["within_sitting"]["temperature"]
    assert set(out["temperatures"]) == {
        "static-textbook",
        "static-myobridge",
        "adapt-textbook",
        "adapt-myobridge",
    }
    assert out["temperatures"]["adapt-textbook"] == 1.0
    assert demo_bench._encode(np.array([[True, False, True], [False, False, False]])) == [5, 0]
