"""Public sEMG datasets used to benchmark myobridge on real muscles.

from myobridge import datasets
ds = datasets.load("3dc")          # clones the pinned mirror on first use
ds = datasets.load("multiday", days=[1, 2, 3])
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import multiday, threedc
from .core import Dataset, DatasetError, Recording, Source, data_root, fetch

REGISTRY = {"3dc": threedc, "multiday": multiday}

__all__ = [
    "REGISTRY",
    "Dataset",
    "DatasetError",
    "Recording",
    "Source",
    "data_root",
    "fetch",
    "load",
]


def load(
    name: str, root: str | Path | None = None, download: bool = True, **kwargs: Any
) -> Dataset:
    """Load a dataset by name, downloading the pinned version first if needed.

    Extra keyword arguments go to the dataset's own loader (`subjects=` for
    3DC, `days=` for MultiDay, `cache=`).
    """
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(REGISTRY)}")
    module = REGISTRY[name]
    folder = fetch(name, module.SOURCE, root) if download else data_root(root) / name
    if not folder.exists():
        raise DatasetError(
            f"{folder} not found; run with download=True or `myobridge fetch {name}`"
        )
    dataset: Dataset = module.load(folder, **kwargs)
    return dataset
