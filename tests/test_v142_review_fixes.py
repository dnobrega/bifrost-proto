"""Review fixes on a native on-disk run: movies, slab LOS, contours, colorbar."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import shutil

import numpy as np
import pytest

from bifrost_proto import BifrostData


def _run(tmp_path, nsnap=3, missing_snap=None):
    n = 8
    coord = np.arange(n, dtype=np.float32) * 0.5
    with (tmp_path / "mesh.mesh").open("w", encoding="utf-8") as stream:
        for _ in range(3):
            stream.write(f"{n}\n")
            for values in (coord, coord - 0.25, np.ones(n), np.ones(n)):
                stream.write(" ".join(map(str, values)) + "\n")
    for k in range(nsnap):
        (tmp_path / f"demo_{k:03d}.idl").write_text(
            "snapname='demo'\nmx=8\nmy=8\nmz=8\ndx=0.5\ndy=0.5\ndz=0.5\n"
            f"tsnap={10.0 * k}\ndo_mhd=.true.\ndo_hion=.false.\n"
            "do_helium=.false.\naux=''\nmeshfile='mesh.mesh'\n",
            encoding="utf-8",
        )
        if k == missing_snap:
            continue
        data = np.memmap(
            tmp_path / f"demo_{k:03d}.snap",
            dtype="<f4",
            mode="w+",
            shape=(n, n, n, 8),
            order="F",
        )
        data[..., 0] = 1.0 + k
        data.flush()
        del data
    return BifrostData(folder=tmp_path)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_incomplete_snapshot_fails_before_encoding(tmp_path):
    dd = _run(tmp_path, missing_snap=2)
    assert dd.snaps == [0, 1]  # dd.snaps requires .idl and .snap
    with pytest.raises(FileNotFoundError, match=r"2: .*demo_002\.snap"):
        dd.movie("r", [0, 1, 2], y=1.0, progress=False)
    assert not (tmp_path / "movies").exists() or not any(
        (tmp_path / "movies").iterdir()
    )
    with pytest.raises(FileNotFoundError, match="demo_002.snap"):
        dd.get_var("r", 2)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_failed_encoding_leaves_no_file_and_retry_works(tmp_path, monkeypatch):
    dd = _run(tmp_path)
    original = dd.get_var
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated read failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(dd, "get_var", flaky)
    with pytest.raises(RuntimeError, match="simulated"):
        dd.movie("r", dd.snaps, y=1.0, progress=False)
    assert list((tmp_path / "movies").iterdir()) == []
    monkeypatch.setattr(dd, "get_var", original)
    path = dd.movie("r", dd.snaps, y=1.0, progress=False)
    assert path.is_file() and path.name == "r_0000_0002_0001_XZ_y=1.00.mp4"


def test_streamed_los_matches_full_cube_on_slabs(tmp_path, monkeypatch):
    """Slab streaming is exact, including ROI, and never asks for a full cube."""
    import matplotlib.pyplot as plt

    from bifrost_proto import plotting

    dd = _run(tmp_path)
    monkeypatch.setattr(plotting, "_LOS_SLAB", 3)  # force several slabs
    for los in "xyz":
        for sel in ({}, {"x": slice(0.5, 3.0), "z": slice(-3.0, -0.5)}):
            fig, _, full = dd.plot("r", 1, los=los, cbar=False, **sel)
            expected = full.get_array().copy()
            plt.close(fig)
            fig, _, streamed = dd.plot(
                "r", 1, los=los, los_stream=True, cbar=False, **sel
            )
            np.testing.assert_array_equal(streamed.get_array(), expected)
            plt.close(fig)


def test_contour_units_and_constrained_colorbar(tmp_path):
    import matplotlib.pyplot as plt

    dd = _run(tmp_path)
    fig, _, _ = dd.plot("r", 0, y=1.0, contours={"var": "r", "units": "si"}, cbar=False)
    plt.close(fig)
    fig = plt.figure(figsize=(4, 4), layout="constrained")
    ax = fig.add_subplot()
    dd.plot("r", 0, y=1.0, ax=ax)
    fig.canvas.draw()
    assert fig.axes[-1].xaxis.label.get_window_extent().y1 <= fig.bbox.height
    plt.close(fig)
