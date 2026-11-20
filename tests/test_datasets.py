import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from fakes import write_multiday, write_threedc

from myobridge import datasets
from myobridge.datasets import core, multiday, threedc
from myobridge.datasets.core import DatasetError, Source


def test_threedc_loader_reads_the_layout(fake_threedc):
    ds = threedc.load(fake_threedc, subjects=[1, 2])
    assert ds.n_channels == 10 and ds.fs == 1000.0
    assert len(ds.recordings) == 2 * 2 * 4 * 11
    assert ds.subjects == [1, 2] and ds.sessions == [0, 1]
    r = ds.select(subjects=[2], sessions=[1], reps=[3])
    assert len(r) == 11 and {x.label for x in r} == set(range(11))
    assert r[4].gesture == "wrist_extension"
    assert r[0].duration_s == pytest.approx(3.0)


def test_threedc_refuses_an_incomplete_copy(fake_threedc):
    (fake_threedc / "Participant1" / "test" / "EMG" / "3dc_EMG_gesture_0_5.txt").unlink()
    with pytest.raises(DatasetError, match="expected 176"):
        threedc.load(fake_threedc, subjects=[1, 2])
    with pytest.raises(DatasetError, match="expected 1936"):  # all 22 people by default
        threedc.load(fake_threedc)


def test_multiday_loader(fake_multiday):
    ds = multiday.load(fake_multiday, days=[1, 2, 3])
    assert ds.fs == 2048.0 and ds.n_channels == 4 and ds.mains_hz == 50.0
    assert ds.sessions == [1, 2, 3] and len(ds.recordings) == 33
    assert len(multiday.load(fake_multiday, days=[2]).recordings) == 11
    with pytest.raises(DatasetError):
        multiday.load(fake_multiday, days=[1, 4])


def test_cache_is_used_and_invalidated(fake_multiday):
    path = fake_multiday / "S0_D1_C3.csv"
    a = core.load_matrix(path, None, 4, cache=True)
    cached = path.parent / ".myobridge-cache" / (path.name + ".npy")
    assert cached.exists()
    np.save(cached, np.zeros_like(a))  # tamper with the cache...
    assert np.all(core.load_matrix(path, None, 4, cache=True) == 0)  # ...it's used
    path.write_text(path.read_text())  # newer source file -> cache rebuilt
    import os

    os.utime(path, (cached.stat().st_mtime + 10, cached.stat().st_mtime + 10))
    assert np.allclose(core.load_matrix(path, None, 4, cache=True), a)


def test_bad_files_are_rejected(tmp_path):
    p = tmp_path / "x.csv"
    p.write_text("1 2 3\n4 5 6\n")
    with pytest.raises(DatasetError, match="expected n x 4"):
        core.load_matrix(p, None, 4, cache=False)
    p.write_text("1 2 3 nan\n")
    with pytest.raises(DatasetError, match="NaN"):
        core.load_matrix(p, None, 4, cache=False)


def test_scan_rejects_two_files_with_the_same_key(tmp_path):
    (tmp_path / "a_1.txt").write_text("")
    (tmp_path / "a_01.txt").write_text("")
    with pytest.raises(DatasetError, match="two files"):
        core.scan(tmp_path, re.compile(r"a_(\d+)\.txt"), lambda m: (int(m[1]), 0, 0, 0))


def test_data_root(monkeypatch, tmp_path):
    monkeypatch.delenv(core.ENV_VAR, raising=False)
    assert core.data_root() == Path.home() / ".cache" / "myobridge"
    monkeypatch.setenv(core.ENV_VAR, str(tmp_path))
    assert core.data_root() == tmp_path
    assert core.data_root(tmp_path / "x") == tmp_path / "x"


def test_load_unknown_or_missing(tmp_path):
    with pytest.raises(KeyError):
        datasets.load("nope")
    with pytest.raises(DatasetError, match="not found"):
        datasets.load("3dc", root=tmp_path, download=False)


# --------------------------------------------------- pinned-version download


def _git(*args, cwd):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def mirror(tmp_path):
    """A local git repo standing in for the GitHub mirror, with two commits."""
    if shutil.which("git") is None:
        pytest.skip("git not installed")
    repo = tmp_path / "mirror"
    write_multiday(repo, days=[1])
    _git("init", "-q", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "add", ".", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "data", cwd=repo)
    pinned = _git("rev-parse", "HEAD", cwd=repo).stdout.strip()
    (repo / "README.md").write_text("later change")
    _git("-c", "user.name=t", "-c", "user.email=t@t", "add", ".", cwd=repo)
    _git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "later", cwd=repo)
    _git("config", "uploadpack.allowReachableSHA1InWant", "true", cwd=repo)
    return repo, pinned


def test_fetch_checks_out_the_pinned_commit(mirror, tmp_path):
    repo, pinned = mirror
    src = Source(url=repo.as_uri(), commit=pinned, citation="test")
    dest = core.fetch("multiday", src, tmp_path / "data")
    assert _git("rev-parse", "HEAD", cwd=dest).stdout.strip() == pinned
    assert not (dest / "README.md").exists()  # the later commit is not what we use
    assert core.check_version(dest, src) == pinned
    assert core.fetch("multiday", src, tmp_path / "data") == dest  # second call is a no-op


def test_fetch_and_check_refuse_other_versions(mirror, tmp_path):
    repo, _ = mirror
    bad = Source(url=repo.as_uri(), commit="0" * 40, citation="test")
    with pytest.raises(DatasetError, match="pinned commit"):
        core.fetch("multiday", bad, tmp_path / "data")
    with pytest.raises(DatasetError, match="expected"):
        core.check_version(tmp_path / "data" / "multiday", bad)
    assert core.check_version(tmp_path, bad) is None  # not a git checkout at all


def test_fetch_wont_overwrite_a_non_git_folder(tmp_path):
    (tmp_path / "data" / "3dc").mkdir(parents=True)
    (tmp_path / "data" / "3dc" / "mine.txt").write_text("x")
    with pytest.raises(DatasetError, match="not a git clone"):
        core.fetch("3dc", threedc.SOURCE, tmp_path / "data")


def test_write_threedc_helper_matches_real_names(tmp_path):
    root = write_threedc(tmp_path / "t", subjects=[3], seconds=0.5)
    assert (root / "Participant3" / "test" / "EMG" / "3dc_EMG_gesture_3_10.txt").exists()
