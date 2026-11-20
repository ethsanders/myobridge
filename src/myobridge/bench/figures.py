"""Figures for the README and docs/, drawn only from the results/*.json files.

Every number on a figure comes from a results file, so re-running the
benchmarks and then this module regenerates the docs exactly. The tables in
docs/results.md carry the same numbers for anyone who can't see the colours.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.ticker import MultipleLocator, NullLocator

# Categorical slots in fixed order (checked for colour-blind separation).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METHOD_COLORS = {"none": INK2, "new_only": ORANGE, "pooled": YELLOW, "adapt": BLUE}
METHOD_LABELS = {
    "none": "no recalibration",
    "new_only": "retrain on new data only",
    "pooled": "retrain on old + new",
    "adapt": "adapt old model (myobridge)",
}


def _plt() -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": INK2,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )
    return plt


def _load(results: Path, name: str) -> dict[str, Any] | None:
    path = results / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _pct(ax: Any) -> None:
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")


def multiday(results: Path, out: Path) -> Path | None:
    r = _load(results, "multiday")
    if r is None:
        return None
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.15]})

    decay = r["decay_by_gap_days"]
    gaps = [int(g) for g in decay]
    mean = [decay[g]["mean"] for g in decay]
    lo = [decay[g]["ci95_low"] for g in decay]
    hi = [decay[g]["ci95_high"] for g in decay]
    same = r["recalibration"]["same_day_only"]
    ceiling = same["accuracy"]["mean"]
    a.fill_between(gaps, lo, hi, color=BLUE, alpha=0.15, linewidth=0)
    a.plot(gaps, mean, color=BLUE, marker="o", markersize=5)
    a.axhline(ceiling, color=INK2, linestyle="--", linewidth=1)
    a.annotate(
        f"trained on the day itself ({same['calibration_seconds']:g} s/gesture)",
        (gaps[-1], ceiling),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color=INK2,
        fontsize=9,
    )
    a.set_xscale("log")
    a.set_xticks(gaps, [str(g) for g in gaps])
    a.set_xlabel("days between training and use")
    a.set_ylabel("accuracy, 11 gestures")
    a.set_title("A model trained on one day goes stale")
    a.yaxis.set_major_locator(MultipleLocator(0.05))
    a.grid(axis="x", visible=False)
    _pct(a)

    rc = r["recalibration"]["by_calibration_seconds"]
    secs = list(rc)
    x = np.arange(len(secs))
    width = 0.2
    for i, m in enumerate(METHOD_COLORS):
        vals = [rc[s]["methods"][m]["accuracy"]["mean"] for s in secs]
        bars = b.bar(
            x + (i - 1.5) * width,
            vals,
            width - 0.02,
            color=METHOD_COLORS[m],
            label=METHOD_LABELS[m],
        )
        if m == "adapt":
            b.bar_label(bars, [f"{v:.0%}" for v in vals], fontsize=8, color=INK, padding=2)
    b.axhline(ceiling, color=INK2, linestyle="--", linewidth=1)
    b.annotate(
        "same-day model",
        (-0.45, ceiling),
        xytext=(0, 3),
        textcoords="offset points",
        color=INK2,
        fontsize=8,
    )
    b.set_xticks(x, [f"{s} s" for s in secs])
    b.set_xlabel("new data per gesture on the day of use")
    b.set_ylim(0.5, 1.0)
    b.set_title("A second per gesture brings it back")
    b.legend(fontsize=8, loc="upper left", ncol=2)
    b.grid(axis="x", visible=False)
    _pct(b)
    fig.tight_layout()
    path = out / "multiday_recalibration.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def threedc(results: Path, out: Path) -> list[Path]:
    r = _load(results, "threedc")
    if r is None:
        return []
    plt = _plt()
    paths = []

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    by_k = r["channels"]["by_k"]
    ks = [int(k) for k in by_k]
    a.fill_between(
        ks,
        [by_k[str(k)]["worst"]["accuracy"] for k in ks],
        [by_k[str(k)]["best"]["accuracy"] for k in ks],
        color=BLUE,
        alpha=0.15,
        linewidth=0,
        label="best to worst channel subset",
    )
    a.plot(
        ks,
        [by_k[str(k)]["mean_over_subsets"] for k in ks],
        color=BLUE,
        marker="o",
        markersize=5,
        label="average subset",
    )
    even = r["channels"]["even_4"]["accuracy"]["mean"]
    a.plot(
        [4],
        [even],
        marker="D",
        markersize=8,
        color=ORANGE,
        linestyle="none",
        label="4 evenly spaced (like myobridge)",
    )
    a.annotate(
        f"{even:.0%}", (4, even), xytext=(8, -12), textcoords="offset points", color=INK, fontsize=9
    )
    a.set_xticks(ks)
    a.set_xlabel("number of forearm channels")
    a.set_ylabel("accuracy, 11 gestures")
    a.set_title("What fewer electrodes cost (22 people)")
    a.legend(fontsize=8, loc="lower right")
    _pct(a)

    cs = r["cross_subject"]["by_calibration_seconds"]
    secs = list(cs)
    x = np.arange(len(secs))
    width = 0.2
    for i, m in enumerate(METHOD_COLORS):
        vals = [cs[s]["methods"][m]["accuracy"]["mean"] for s in secs]
        bars = b.bar(
            x + (i - 1.5) * width,
            vals,
            width - 0.02,
            color=METHOD_COLORS[m],
            label=METHOD_LABELS[m],
        )
        if m == "adapt":
            b.bar_label(bars, [f"{v:.0%}" for v in vals], fontsize=8, color=INK, padding=2)
    within = r["within_subject"]["accuracy"]["mean"]
    b.axhline(within, color=INK2, linestyle="--", linewidth=1)
    b.annotate(
        "trained on that person",
        (x[-1] + 0.4, within),
        xytext=(0, 4),
        textcoords="offset points",
        ha="right",
        color=INK2,
        fontsize=9,
    )
    b.set_xticks(x, [f"{s} s" for s in secs])
    b.set_xlabel("calibration data per gesture from the new person")
    b.set_ylim(0, 1)
    b.set_title("A new person, with a short calibration")
    b.legend(fontsize=8, loc="upper left")
    _pct(b)
    fig.tight_layout()
    path = out / "threedc_channels_cross_subject.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    conf = np.asarray(r["within_subject"]["confusion_counts"], dtype=float)
    conf = conf / conf.sum(axis=1, keepdims=True)
    names = [g.replace("_", " ") for g in r["within_subject"]["gestures"]]
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.grid(False)
    im = ax.imshow(conf, cmap="Blues", vmin=0, vmax=1)
    for i in range(len(names)):
        for j in range(len(names)):
            if conf[i, j] >= 0.02:
                ax.text(
                    j,
                    i,
                    f"{conf[i, j]:.0%}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if conf[i, j] > 0.5 else INK,
                )
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("performed")
    ax.set_title("Which gestures get confused (3DC, all 22 people)")
    fig.colorbar(im, ax=ax, fraction=0.046, format=lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    path = out / "threedc_confusion.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def aliasing(results: Path, out: Path) -> Path | None:
    r = _load(results, "aliasing")
    if r is None:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    all_cases = list(r["white_noise_model"]["cases"].values())
    cases = [c for c in all_cases if "decimator" not in c]  # ideal decimation
    for order, color in zip((1, 2, 4), (ORANGE, BLUE, AQUA), strict=True):
        pts = sorted((c["fs"], c["noise_floor_increase_db"]) for c in cases if c["order"] == order)
        xs, ys = zip(*pts, strict=True)
        ax.plot(
            xs,
            ys,
            color=color,
            marker="o",
            markersize=5,
            label=f"order {order} low-pass" + (" (MyoWare-like)" if order == 1 else ""),
        )
        ax.annotate(
            f"+{ys[0]:.1f} dB",
            (xs[0], ys[0]),
            xytext=(6, 2),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    fw = [c for c in all_cases if "decimator" in c and c["order"] == 1]
    if fw:
        y = fw[0]["noise_floor_increase_db"]
        ax.plot([2000], [y], marker="*", markersize=13, color=INK, linestyle="none", zorder=5)
        ax.annotate(
            f"firmware OVERSAMPLE 2\n(real halfband filter): +{y:.1f} dB",
            (2000, y),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=8,
            color=INK,
        )
    ax.set_xscale("log")
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks([1000, 2000, 4000], ["1 kHz", "2 kHz", "4 kHz"])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("ADC sample rate")
    ax.set_ylabel("in-band noise floor increase (dB)")
    ax.set_title("Wideband noise folded into 20-450 Hz")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "aliasing_noise.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def quality(results: Path, out: Path) -> Path | None:
    r = _load(results, "quality")
    if r is None:
        return None
    plt = _plt()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    det = r["detection"]
    faults = list(det)
    x = np.arange(len(faults))
    width = 0.26
    for i, color in enumerate((YELLOW, ORANGE, BLUE)):
        levels = [list(det[f])[i] for f in faults]
        vals = [det[f][lv]["detected"] for f, lv in zip(faults, levels, strict=True)]
        a.bar(
            x + (i - 1) * width,
            vals,
            width - 0.03,
            color=color,
            label=("mild", "medium", "severe")[i],
        )
    a.set_xticks(x, [f.replace("_", " ") for f in faults])
    a.set_ylim(0, 1.22)
    a.set_yticks(np.linspace(0, 1, 6))
    a.set_ylabel("faults caught")
    fa = r["false_alarms_clean"]["any_check"]
    a.set_title(f"Injected faults caught (clean data flagged: {fa:.1%})")
    a.legend(fontsize=8, loc="upper center", ncol=3)
    a.grid(axis="x", visible=False)
    _pct(a)

    results_ft = r["fault_tolerance"]["results"]
    scen = [sc for sc in ("lead_off", "clipping") if sc in results_ft]
    keys = [("clean", "no fault", INK2), ("no_monitor", "fault, no monitor", ORANGE)]
    keys.append(("monitor", "fault, monitor on", BLUE))
    xb = np.arange(len(scen))
    wb = 0.26
    for i, (k, label, color) in enumerate(keys):
        vals = [results_ft[sc][k]["mean"] for sc in scen]
        bars = b.bar(xb + (i - 1) * wb, vals, wb - 0.03, color=color, label=label)
        b.bar_label(bars, [f"{v:.0%}" for v in vals], fontsize=8, padding=2)
    names = {"lead_off": "electrode off", "clipping": "channel clipping"}
    b.set_xticks(xb, [names[sc] for sc in scen])
    b.set_ylim(0, 1.22)
    b.set_yticks(np.linspace(0, 1, 6))
    b.set_ylabel("accuracy, 11 gestures")
    b.set_title("Decoding with one bad channel")
    b.legend(fontsize=8, loc="upper center", ncol=3)
    b.grid(axis="x", visible=False)
    _pct(b)
    fig.tight_layout()
    path = out / "quality.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def make_all(results: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path | None] = [
        multiday(results, out),
        aliasing(results, out),
        quality(results, out),
    ]
    return [p for p in [*paths, *threedc(results, out)] if p is not None]
