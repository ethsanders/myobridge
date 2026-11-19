"""Generate docs/demo.png: raw vs filtered vs envelope, with predicted gestures.

pip install -e ".[plot]"
python scripts/make_demo_plot.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to a file, no window needed
import matplotlib.pyplot as plt
import numpy as np

from myobridge.classifier import calibrate
from myobridge.dsp import EmgFilter
from myobridge.pipeline import GesturePipeline
from myobridge.simulator import GESTURES, EmgSimulator, gesture_session

FS, BLOCK = 1000, 20
OUT = Path(__file__).resolve().parent.parent / "docs" / "demo.png"

# One fixed color per gesture (categorical, in a fixed order).
GESTURE_COLORS = dict(
    zip(GESTURES, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"], strict=True)
)
RAW, CLEAN, ENVELOPE, INK = "#b9b8b2", "#2a78d6", "#0b0b0b", "#52514e"


def spans(labels: np.ndarray, t: np.ndarray):
    """Yield (start_time, end_time, label) for each run of identical labels."""
    edges = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts, ends = np.r_[0, edges], np.r_[edges, len(labels)]
    for s, e in zip(starts, ends, strict=True):
        yield t[s], t[e - 1], labels[s]


def main() -> None:
    clf, _, _ = calibrate(EmgSimulator(seed=0).record, GESTURES, 8.0, FS)
    raw, truth = gesture_session(EmgSimulator(seed=7), GESTURES, 2.0, 1)
    filtered = EmgFilter(FS, raw.shape[1]).process(raw)

    pipeline = GesturePipeline(clf, FS, raw.shape[1])
    predicted = np.full(len(raw), "", dtype=object)
    last = "rest"
    for start in range(0, len(raw), BLOCK):
        for p in pipeline.process_block(raw[start : start + BLOCK]):
            last = p.gesture
        predicted[start : start + BLOCK] = last  # hold the latest prediction

    t = np.arange(len(raw)) / FS
    n_ch = raw.shape[1]
    fig, axes = plt.subplots(
        n_ch + 1,
        2,
        figsize=(12, 8.5),
        gridspec_kw={"height_ratios": [0.9] + [1] * n_ch, "width_ratios": [1, 3]},
    )
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "myobridge on simulated 4-channel EMG: raw -> filtered -> RMS envelope -> LDA",
        x=0.01,
        ha="left",
        fontsize=12,
        color="#0b0b0b",
    )

    # Top right: true vs predicted gesture lanes, sharing the time axis below.
    lanes = axes[0, 1]
    for row, labels in enumerate([truth, predicted]):
        y = 1 - row
        for a, b, g in spans(labels, t):
            lanes.barh(
                y,
                b - a,
                left=a,
                height=0.8,
                color=GESTURE_COLORS[g],
                edgecolor="white",
                linewidth=2,
            )
            if b - a > 0.35:
                lanes.text(
                    (a + b) / 2,
                    y,
                    g,
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
    lanes.set_yticks([1, 0], ["true", "predicted"])
    lanes.set_ylim(-0.6, 1.6)
    lanes.set_xlim(t[0], t[-1])

    # Left column: a 300 ms zoom during "rest", where hum and drift dominate.
    zoom = (t >= 1.0) & (t < 1.3)
    for ch in range(n_ch):
        ax = axes[ch + 1, 0]
        ax.plot(t[zoom], raw[zoom, ch], color=RAW, lw=1.2, label="raw ADC counts")
        ax.plot(
            t[zoom],
            filtered.clean[zoom, ch],
            color=CLEAN,
            lw=1.2,
            label="filtered (20-450 Hz bandpass + 60 Hz notch)",
        )
        ax.set_ylabel(f"ch{ch}", rotation=0, labelpad=18, color=INK)
    axes[-1, 0].set_xlabel("zoom at rest (s)", color=INK)

    # Right column: the whole session, with the envelope the features "see".
    for ch in range(n_ch):
        ax = axes[ch + 1, 1]
        for a, b, g in spans(predicted, t):  # predicted gesture as a faint background
            ax.axvspan(a, b, color=GESTURE_COLORS[g], alpha=0.10, linewidth=0)
        ax.plot(t, filtered.clean[:, ch], color=CLEAN, lw=0.4, alpha=0.45)
        ax.plot(
            t, filtered.envelope[:, ch], color=ENVELOPE, lw=2, label="RMS envelope (100 ms window)"
        )
        ax.set_ylim(-1500, 1500)
        ax.set_xlim(t[0], t[-1])
        if ch < n_ch - 1:
            ax.set_xticklabels([])
    axes[-1, 1].set_xlabel("time (s)  -  background = predicted gesture", color=INK)

    # Top left: just the legend.
    legend_ax = axes[0, 0]
    legend_ax.axis("off")
    handles = [axes[1, 0].lines[0], axes[1, 0].lines[1], axes[1, 1].lines[-1]]
    labels = ["raw ADC counts (hum + drift)", "filtered: bandpass + notch", "RMS envelope (100 ms)"]
    legend_ax.legend(handles, labels, loc="center left", fontsize=9, frameon=False)

    for ax in axes.flat:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color("#c3c2b7")
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(colors=INK, labelsize=9)
        if ax is not lanes:
            ax.grid(axis="y", color="#ecebe6", lw=0.8)
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
