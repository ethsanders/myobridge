"""Shared fixtures (the fake datasets themselves are written by fakes.py)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fakes import write_multiday, write_threedc


@pytest.fixture
def fake_threedc(tmp_path: Path) -> Path:
    return write_threedc(tmp_path / "3dc")


@pytest.fixture
def fake_multiday(tmp_path: Path) -> Path:
    return write_multiday(tmp_path / "multiday")
