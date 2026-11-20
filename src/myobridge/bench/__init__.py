"""Benchmarks on public real-EMG datasets. Run with `myobridge bench <name>`.

Each experiment is a module with `run(RunOptions) -> dict` that writes
results/<name>.json; `figures.make_all` then draws docs/figures/ from those
files. docs/results.md explains every experiment's splits and numbers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from . import (
    aliasing_bench,
    conformal_bench,
    latency_bench,
    multiday_bench,
    partial_bench,
    quality_bench,
    threedc_bench,
)
from .common import RunOptions

EXPERIMENTS: dict[str, Callable[[RunOptions], dict[str, Any]]] = {
    "multiday": multiday_bench.run,
    "threedc": threedc_bench.run,
    "aliasing": aliasing_bench.run,
    "quality": quality_bench.run,
    "latency": latency_bench.run,
    "conformal": conformal_bench.run,
    "partial_calibration": partial_bench.run,
}

__all__ = ["EXPERIMENTS", "RunOptions", "run"]


def run(names: list[str], opts: RunOptions) -> None:
    for name in names:
        if name not in EXPERIMENTS:
            raise KeyError(f"unknown benchmark {name!r}; choose from {sorted(EXPERIMENTS)}")
    for name in names:
        print(f"[{name}]", flush=True)
        EXPERIMENTS[name](opts)
    if opts.figures_dir is not None and not opts.quick:
        from .figures import make_all

        for path in make_all(opts.out_dir, opts.figures_dir):
            print(f"  figure {path}")
