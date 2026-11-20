"""Plumbing shared by every benchmark: where results go and what they record.

Each experiment writes one JSON file to results/. Besides the numbers, the
file records everything needed to reproduce them: the dataset's pinned
commit, the feature settings, the software versions and the exact splits.
docs/RESULTS.md and the README quote these files; a test checks they agree.
"""

from __future__ import annotations

import json
import platform
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .. import __version__
from ..classifier import LDAClassifier, adapt
from ..datasets.core import Dataset

RESULTS_SCHEMA = 1


@dataclass(frozen=True)
class RunOptions:
    data_root: Path | None
    out_dir: Path
    figures_dir: Path | None
    quick: bool = False  # fewer subjects/days: a smoke test, numbers NOT for reporting
    download: bool = True  # False: use what's in data_root as-is (tests, offline use)


def _jsonable(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        return _jsonable(obj.item())
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, Path):
        return obj.as_posix()
    return obj


def provenance(dataset: Dataset | None, settings: dict[str, Any]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "schema": RESULTS_SCHEMA,
        "myobridge": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "settings": settings,
    }
    if dataset is not None:
        info["dataset"] = {
            "name": dataset.name,
            "url": dataset.source.url,
            "commit": dataset.source.commit,
            "citation": dataset.source.citation,
            "fs": dataset.fs,
            "n_channels": dataset.n_channels,
            "n_recordings": len(dataset.recordings),
        }
    return info


def write_result(opts: RunOptions, name: str, payload: dict[str, Any]) -> Path:
    """Write results/<name>.json (or <name>.quick.json for a --quick run)."""
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".quick.json" if opts.quick else ".json"
    path = opts.out_dir / f"{name}{suffix}"
    body = {"quick": opts.quick, **payload}
    path.write_text(json.dumps(_jsonable(body), indent=2, sort_keys=False) + "\n")
    return path


@contextmanager
def timed(label: str) -> Iterator[None]:
    start = time.perf_counter()
    yield
    print(f"  {label}: {time.perf_counter() - start:.1f} s", flush=True)


# --------------------------------------------------------- model selection


@dataclass(frozen=True)
class AdaptSetting:
    mean_prior: float
    covariance_weight: float


ADAPT_GRID = tuple(
    AdaptSetting(t, b)
    for t in (0.0, 2.0, 5.0, 10.0, 20.0, 50.0)
    for b in (0.0, 0.25, 0.5, 0.75, 1.0)
)


def recalibrate(
    method: str,
    base: LDAClassifier,
    X_base: np.ndarray,
    y_base: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    setting: AdaptSetting | None = None,
) -> LDAClassifier:
    """Ways of using a little new data from the session we're about to decode.

    - "none":     ignore it, keep the old model.
    - "new_only": throw the old model away, train on the new data alone.
    - "pooled":   retrain on old + new data together.
    - "adapt":    move the old model's class means (and optionally its
                  covariance) towards the new data; see classifier.adapt.
    """
    if method == "none":
        return base
    if method == "new_only":
        return LDAClassifier().fit(X_cal, y_cal)
    if method == "pooled":
        return LDAClassifier().fit(np.vstack([X_base, X_cal]), np.concatenate([y_base, y_cal]))
    if method == "adapt":
        if setting is None:
            raise ValueError("adapt needs a setting")
        return adapt(base, X_cal, y_cal, setting.mean_prior, setting.covariance_weight)
    raise KeyError(f"unknown method {method!r}")


METHODS = ("none", "new_only", "pooled", "adapt")


def best_setting(scores: Mapping[AdaptSetting, Sequence[float]]) -> AdaptSetting:
    """The setting with the highest mean validation score (ties: first in grid order)."""
    return max(scores, key=lambda s: (float(np.mean(scores[s])), -ADAPT_GRID.index(s)))
