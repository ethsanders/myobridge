"""Render docs/figures/social-preview.png (1280x640), for the repo's Settings > Social preview.

    python scripts/render_social_preview.py

Uses a frame of docs/figures/demo.gif, so render that first.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
GIF = ROOT / "docs" / "figures" / "demo.gif"
OUT = ROOT / "docs" / "figures" / "social-preview.png"
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
FRAME = 60  # electrodes all healthy, decoder correct


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    import matplotlib

    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
    return ImageFont.truetype(str(path), size)


def main() -> None:
    gif = Image.open(GIF)
    gif.seek(FRAME)
    frame = gif.convert("RGB")
    frame = frame.crop((0, 30, frame.width, frame.height - 30))  # trim empty margins
    canvas = Image.new("RGB", (1280, 640), SURFACE)
    height = 400
    width = round(frame.width * height / frame.height)
    frame = frame.resize((width, height), Image.Resampling.LANCZOS)
    canvas.paste(frame, ((1280 - width) // 2, 640 - height - 24))
    draw = ImageDraw.Draw(canvas)
    draw.text((64, 40), "myobridge", fill=INK, font=font(64, bold=True))
    draw.text(
        (66, 118),
        "Forearm EMG to hand gestures, live: ESP32 firmware, a Python decoder,\n"
        "and benchmarks on public data",
        fill=INK2,
        font=font(28),
        spacing=8,
    )
    canvas.save(OUT, optimize=True)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
