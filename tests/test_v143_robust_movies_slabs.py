"""v0.1.43: temporary MP4 publication, native-file checks, adaptive LOS slabs."""

from __future__ import annotations

import os
import shutil

import matplotlib

matplotlib.use("Agg", force=True)
import numpy as np
import pytest
from test_engine import make_data
from test_v142_review_fixes import _run

from bifrost_proto import plotting
from bifrost_proto.movie import movie
from bifrost_proto.plotting import _integrated_plane


def test_los_slab_adapts_to_budget(monkeypatch):
    dd = make_data(19)
    selections = {"x": None, "y": None, "z": None}
    expected, _ = _integrated_plane(
        dd, "modb", 0, los="z", selection=selections, units="solar", low_memory=None
    )
    monkeypatch.setattr(plotting, "_LOS_TARGET_MIB", 0.004)
    monkeypatch.setattr(plotting, "_LOS_SLAB", 64)
    original = dd.get_var
    widths = []

    def spy(*args, **kwargs):
        widths.append(kwargs["iy"].stop - kwargs["iy"].start)
        return original(*args, **kwargs)

    monkeypatch.setattr(dd, "get_var", spy)
    actual, _ = _integrated_plane(
        dd,
        "modb",
        0,
        los="z",
        selection=selections,
        units="solar",
        low_memory=None,
        stream=True,
    )
    assert max(widths) <= 2  # floor(0.004 MiB / (19*19*4 bytes)) == 2
    assert len(widths) >= 10
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-8)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_truncated_native_snap_preflight_and_retry(tmp_path):
    dd = _run(tmp_path)
    path = tmp_path / "demo_001.snap"
    original_size = path.stat().st_size
    with path.open("r+b") as fp:
        fp.truncate(original_size // 2)
    with pytest.raises(ValueError, match=r"Incomplete .*demo_001.snap"):
        dd.movie("r", [0, 1, 2], y=1.0, progress=False, cbar=False)
    movies = tmp_path / "movies"
    assert not movies.exists() or not list(movies.iterdir())
    with path.open("ab") as fp:
        fp.truncate(original_size)
    result = dd.movie("r", [0, 1, 2], y=1.0, progress=False, cbar=False)
    assert result.stat().st_size > 200
    assert list(movies.iterdir()) == [result]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_atomic_no_overwrite_if_competitor_publishes(tmp_path, monkeypatch):
    dd = _run(tmp_path)
    dest = tmp_path / "collision.mp4"
    original_link = os.link

    def competitor(source, destination):
        dest.write_bytes(b"other concurrent process")
        return original_link(source, destination)  # must fail with FileExistsError

    monkeypatch.setattr(os, "link", competitor)
    with pytest.raises(FileExistsError):
        movie(dd, "r", [0, 1, 2], y=1.0, cbar=False, output=dest, progress=False)
    assert dest.read_bytes() == b"other concurrent process"
    assert not list(tmp_path.glob(".*.partial.mp4"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_spatial_scan_checks_snap_and_leaves_no_mp4(tmp_path):
    dd = _run(tmp_path, missing_snap=2)
    with pytest.raises(FileNotFoundError, match="demo_002.snap"):
        dd.movie("r", 2, scan="y", progress=False)
    assert not (tmp_path / "movies").exists()
