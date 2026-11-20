"""Render docs/figures/demo.gif: the live demo, played offline on real 3DC data.

    python scripts/render_demo_gif.py [--data DIR] [--person 7]

It runs exactly what the browser demo runs (myobridge.replay.Replay) on one
person with the myobridge armband's four channels, and pulls one electrode
off part-way through so the monitor's reaction is visible. Needs the 3DC
dataset (`myobridge fetch 3dc`) and matplotlib.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from myobridge import datasets  # noqa: E402
from myobridge.bench.threedc_bench import EVEN_4  # noqa: E402
from myobridge.datasets import threedc  # noqa: E402
from myobridge.replay import Frame, Replay  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OK, BAD, TRACE = "#1baf7a", "#d2452f", "#36454f"
STEP_S = 0.1
FRAME_EVERY = 2  # one GIF frame per 0.2 s of data, shown at 10 fps: 2x speed
START_S, END_S = 9.5, 33.5  # wrist flexion to pronation
FAULT_AT_S, FAULT_CHANNEL = 21.5, 2
HISTORY_S = 1.5
DPI = 90
COLOURS = 24  # a small shared palette keeps the GIF small


def pretty(name: str | None) -> str:
    return name.replace("_", " ") if name else "-"


def run(data: Path, person: int) -> tuple[Replay, list[Frame]]:
    ds = datasets.load("3dc", root=data, download=False, subjects=[person])
    train = ds.select(sessions=[0], reps=[0, 1])
    test = ds.select(sessions=[1], reps=[0])
    r = Replay(train, test, ds.gestures, ds.mains_hz, channels=EVEN_4, rail=threedc.ADC_RAIL)
    frames = []
    while not r.done and r.position / r.fs < END_S:
        if abs(r.position / r.fs - FAULT_AT_S) < STEP_S / 2:
            r.inject(FAULT_CHANNEL, "lead_off")
        frames.append(r.step(STEP_S))
    return r, frames


def render(r: Replay, frames: list[Frame], out: Path) -> None:
    raw = np.vstack([f.raw for f in frames])
    disp_fs = r.fs / r.display_every
    t = np.arange(len(raw)) / disp_fs
    centre, scale = r.trace_centre, r.trace_scale  # the same scale the browser demo uses

    fig = plt.figure(figsize=(8.0, 3.6), dpi=DPI, facecolor=SURFACE)
    grid = fig.add_gridspec(4, 2, width_ratios=[2.3, 1.0], wspace=0.08, hspace=0.15)
    lanes = [fig.add_subplot(grid[i, 0]) for i in range(4)]
    panel = fig.add_subplot(grid[:, 1])
    panel.axis("off")
    lines = []
    for i, ax in enumerate(lanes):
        ax.set_facecolor(SURFACE)
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_ylim(-1.15, 1.15)
        ax.text(
            -0.02,
            0.5,
            f"ch {EVEN_4[i] + 1}",
            transform=ax.transAxes,
            ha="right",
            va="center",
            color=INK2,
            fontsize=9,
        )
        (line,) = ax.plot([], [], color=TRACE, lw=0.7, antialiased=False)
        lines.append(line)
    status = [
        lanes[i].text(
            1.0, 0.92, "", transform=lanes[i].transAxes, ha="right", va="top", fontsize=8.5
        )
        for i in range(4)
    ]
    kw = {"transform": panel.transAxes, "ha": "left"}
    panel.text(0.08, 0.86, "HOLDING", color=INK2, fontsize=8.5, **kw)
    truth = panel.text(0.08, 0.74, "", color=INK, fontsize=15, weight="bold", **kw)
    panel.text(0.08, 0.55, "DECODED", color=INK2, fontsize=8.5, **kw)
    pred = panel.text(0.08, 0.43, "", fontsize=15, weight="bold", **kw)
    score = panel.text(0.08, 0.22, "", color=INK2, fontsize=9, **kw)
    panel.text(0.08, 0.04, "3DC person 7, 4 electrodes, 2x speed", color=INK2, fontsize=7.5, **kw)

    shown = [i for i, f in enumerate(frames) if f.t >= START_S][::FRAME_EVERY]
    per_frame = frames[0].raw.shape[0]

    def draw(k: int) -> list[object]:
        i = shown[k]
        f = frames[i]
        end = (i + 1) * per_frame
        begin = max(0, end - int(HISTORY_S * disp_fs))
        for c, line in enumerate(lines):
            y = np.clip((raw[begin:end, c] - centre[c]) / scale[c], -1.1, 1.1)
            line.set_data(t[begin:end] - t[end - 1], y)
            lanes[c].set_xlim(-HISTORY_S, 0)
            ch = f.channels[c]
            if ch.dropped:
                status[c].set_text("electrode off: dropped")
                status[c].set_color(BAD)
                line.set_color(BAD)
            else:
                status[c].set_text("")
                line.set_color(TRACE)
        truth.set_text(pretty(f.truth))
        pred.set_text(pretty(f.prediction))
        pred.set_color(OK if f.prediction == f.truth else BAD)
        if f.accuracy is not None:
            score.set_text(f"{100 * f.accuracy:.0f}% of decisions correct so far")
        return [*lines, *status, truth, pred, score]

    images = []
    for k in range(len(shown)):
        draw(k)
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        images.append(Image.fromarray(rgb))
    plt.close(fig)
    # One palette for every frame, taken from a frame with both trace colours.
    palette = images[-1].quantize(COLOURS, method=Image.Quantize.MEDIANCUT)
    gif = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    out.parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(
        out, save_all=True, append_images=gif[1:], duration=100, loop=0, optimize=True, disposal=1
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(shown)} frames)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=datasets.data_root())
    p.add_argument("--person", type=int, default=7)
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "demo.gif")
    args = p.parse_args()
    r, frames = run(args.data, args.person)
    render(r, frames, args.out)


if __name__ == "__main__":
    main()
