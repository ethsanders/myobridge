"""Build the myobridge wheel the browser demo installs, next to the demo page.

    python scripts/build_demo.py          # writes docs/demo/<wheel> and docs/demo/wheel.json

The docs workflow runs this before `mkdocs build`. To try the demo locally:

    python scripts/build_demo.py && python -m http.server -d docs/demo 8000
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "docs" / "demo"


def main() -> None:
    for old in DEMO.glob("myobridge-*.whl"):
        old.unlink()
    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", str(ROOT), "--no-deps", "-q", "-w", str(DEMO)],
        check=True,
    )
    (wheel,) = DEMO.glob("myobridge-*.whl")
    (DEMO / "wheel.json").write_text(json.dumps({"file": wheel.name}) + "\n")
    print(f"built {wheel.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
