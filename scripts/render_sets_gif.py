"""Render docs/figures/sets.gif: the demo's across-days panel, in four scenes.

    python scripts/render_sets_gif.py [--day 61]

It draws the same data the demo page plays (results/demo.json, written by
`myobridge bench demo`) and counts it the same way, on one MultiDay day:

1. the myobridge sets, recalibrated that day
2. the textbook recipe: the usual score, threshold from days 11-30, no recalibration
3. electrode 1 lifted halfway through, with no quality monitor
4. the same, with the monitor dropping the electrode

The day defaults to the one the page opens on (`typical_day`): the demo
day whose recipe and textbook coverage are, together, closest to their
91-day benchmark means, so the GIF shows an ordinary day, not the best one.
Needs matplotlib and Pillow.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
KIND = {"sure": "#2f8f4e", "unsure": "#9aa5ad", "wrong": "#c2412d", "missed": "#d08a1e"}
FAULT = "#c2412d"
FPS = 10
SPEED = 6.0  # seconds of the day per second of GIF
HOLD_S = 1.6  # pause on each finished scene
DPI = 90
COLOURS = 32
SCENES = (
    ("adapt-myobridge-ok", "myobridge sets, recalibrated with 1 s per gesture"),
    ("static-textbook-ok", "The textbook recipe, same 90% promise, no recalibration"),
    ("adapt-myobridge-off", "Electrode 1 comes off halfway, no quality monitor"),
    ("adapt-myobridge-off_monitor", "The same, with the quality monitor"),
)


def kinds(sets: list[int], truth: list[int]) -> list[str]:
    out = []
    for s, y in zip(sets, truth, strict=True):
        hit = bool(s >> y & 1)
        single = s != 0 and s & (s - 1) == 0
        out.append(("sure" if hit else "wrong") if single else ("unsure" if hit else "missed"))
    return out


def pretty(name: str) -> str:
    return name.replace("_", " ")


def render(data: dict[str, Any], day: dict[str, Any], out: Path) -> None:
    gestures = data["gestures"]
    t = np.array(day["t"])
    truth = day["truth"]
    total = day["segments"][-1]["start"] + day["segments"][-1]["length"]
    fig = plt.figure(figsize=(8.0, 3.4), dpi=DPI, facecolor=SURFACE)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    images: list[Image.Image] = []
    promised = data["headline"]["promised"]

    def frame(key: str, title: str, now: float) -> None:
        ax.clear()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        v = day["variants"][key]
        n = int(np.searchsorted(t, now, side="right"))
        ks = kinds(v["sets"][:n], truth[:n])
        ax.text(0.03, 0.9, title, fontsize=13, weight="bold", color=INK)
        seg = next(
            (s for s in day["segments"] if now < s["start"] + s["length"]), day["segments"][-1]
        )
        ax.text(0.03, 0.74, "HOLDING", fontsize=8, color=INK2)
        ax.text(0.03, 0.64, pretty(gestures[seg["gesture"]]), fontsize=13, color=INK)
        ax.text(0.36, 0.74, "DECODER'S ANSWER", fontsize=8, color=INK2)
        if n:
            members = [c for c in range(len(gestures)) if v["sets"][n - 1] >> c & 1]
            members.sort(key=lambda c: c != v["top"][n - 1])
            x = 0.36
            if len(members) != 1:
                ax.text(x, 0.64, "not sure:", fontsize=11, color=INK2, weight="bold")
                x += 0.11
            shown = 0
            for c in members:
                label = pretty(gestures[c])
                w = 0.012 * len(label) + 0.03
                if x + w > 0.93:
                    break
                shown += 1
                sure = len(members) == 1
                colour = KIND["sure"] if ks[-1] == "sure" else KIND["wrong"]
                ax.add_patch(
                    FancyBboxPatch(
                        (x, 0.615),
                        w,
                        0.075,
                        boxstyle="round,pad=0.005,rounding_size=0.035",
                        facecolor=colour if sure else SURFACE,
                        edgecolor=colour if sure else KIND["unsure"],
                        lw=1.5,
                    )
                )
                ax.text(
                    x + w / 2,
                    0.652,
                    label,
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    weight="bold",
                    color=SURFACE if sure else INK,
                )
                x += w + 0.012
            if len(members) > shown:
                ax.text(x, 0.64, f"+{len(members) - shown}", fontsize=10, color=INK2)
        cov = np.mean([k in ("sure", "unsure") for k in ks]) if n else None
        one = np.mean([k in ("sure", "wrong") for k in ks]) if n else None
        line = f"Promised {promised:.0%}   ·   delivered so far: "
        ax.text(0.03, 0.47, line, fontsize=11.5, color=INK2)
        short = cov is not None and cov < promised - 0.05
        ax.text(
            0.03 + 0.0105 * len(line),
            0.47,
            "-" if cov is None else f"{100 * cov:.1f}%",
            fontsize=15,
            weight="bold",
            color=KIND["wrong"] if short else INK,
        )
        ax.text(
            0.63,
            0.47,
            "" if one is None else f"·   answered with one gesture: {100 * one:.0f}%",
            fontsize=11.5,
            color=INK2,
        )
        # the strip: every decision so far, coloured by how its set did
        x0, x1, y0, h = 0.03, 0.97, 0.18, 0.17
        for s in day["segments"]:
            ax.add_patch(
                Rectangle((x0 + (x1 - x0) * s["start"] / total, y0), 0.0012, h, color=GRID)
            )
        if key.endswith("off") or key.endswith("off_monitor"):
            f0 = x0 + (x1 - x0) * day["fault_from"] / total
            ax.add_patch(Rectangle((f0, y0 + h + 0.015), x1 - f0, 0.018, color=FAULT, lw=0))
            ax.text(f0, y0 + h + 0.045, "electrode 1 off", fontsize=8, color=FAULT)
        # one pixel column per 1/600 of the day, coloured by the latest window in it
        cols = 600
        row = np.full((1, cols, 3), matplotlib.colors.to_rgb(SURFACE))
        for i, k in enumerate(ks):
            a = int((t[i] - 0.2) / total * cols)
            b = max(a + 1, int(t[i] / total * cols))
            row[0, max(a, 0) : min(b, cols)] = matplotlib.colors.to_rgb(KIND[k])
        ax.imshow(row, extent=(x0, x1, y0, y0 + h), aspect="auto", interpolation="nearest")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.add_patch(
            Rectangle(
                (x0 + (x1 - x0) * min(now, total) / total, y0 - 0.02), 0.002, h + 0.04, color=INK
            )
        )
        ax.text(
            0.03,
            0.05,
            f"MultiDay day {day['day']}: one person, electrodes re-applied daily, decoder "
            f"trained on days 1-10. Green: sure and right; grey: not sure, right one in the "
            f"set; red: sure and wrong; amber: missed.",
            fontsize=7.2,
            color=INK2,
            wrap=True,
        )
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        images.append(Image.fromarray(rgb))

    for key, title in SCENES:
        steps = int(np.ceil(total / SPEED * FPS))
        for i in range(steps + 1):
            frame(key, title, min(total, i * SPEED / FPS))
        images.extend([images[-1]] * int(HOLD_S * FPS))
    plt.close(fig)
    palette = images[-1].quantize(COLOURS, method=Image.Quantize.MEDIANCUT)
    gif = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
    out.parent.mkdir(parents=True, exist_ok=True)
    gif[0].save(
        out,
        save_all=True,
        append_images=gif[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
        disposal=1,
    )
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(gif)} frames, day {day['day']})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", type=Path, default=ROOT / "results" / "demo.json")
    p.add_argument("--day", type=int, help="a demo day (default: the most typical one)")
    p.add_argument("--out", type=Path, default=ROOT / "docs" / "figures" / "sets.gif")
    args = p.parse_args()
    data = json.loads(args.data.read_text())
    wanted = data["typical_day"] if args.day is None else args.day
    day = next((d for d in data["days"] if d["day"] == wanted), None)
    if day is None:
        raise SystemExit(f"day {wanted} isn't a demo day: {[d['day'] for d in data['days']]}")
    render(data, day, args.out)


if __name__ == "__main__":
    main()
