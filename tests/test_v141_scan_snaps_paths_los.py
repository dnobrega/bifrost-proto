"""v0.1.41 regressions, extended to the v0.1.42 .idl + .snap requirement."""

from __future__ import annotations

import shutil
import subprocess

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest

from bifrost_proto import BifrostData
from bifrost_proto.movie import _scan_positions, movie
from test_engine import make_data


def _data(n=20):
    dd = make_data(n=n)
    dd._meta_cache[0].params["tsnap"] = 120.0
    return dd


def test_snaps_requires_idl_and_snap_and_rescans(tmp_path):
    dd = BifrostData("my_run", folder=tmp_path)
    assert dd.snaps == []
    for filename in (
        "my_run_000.idl",
        "my_run_002.idl",
        "my_run_010.idl",
        "my_run_002.snap",
        "other_run_004.idl",
        "my_run_001.snap",
        "my_run.idl.scr",
        "my_run_020.idl.scr",
        "my_run_bad.idl",
    ):
        (tmp_path / filename).touch()
    folder = tmp_path / "subfolder"
    folder.mkdir()
    (folder / "my_run_007.idl").touch()
    (folder / "my_run_007.snap").touch()
    assert dd.snaps == [2]
    (tmp_path / "my_run_010.snap").touch()
    assert dd.snaps == [2, 10]
    (tmp_path / "my_run_000.snap").touch()
    assert dd.snaps == [0, 2, 10]
    (tmp_path / "my_run_002.idl").unlink()
    assert dd.snaps == [0, 10]
    (tmp_path / "my_run.idl").touch()
    assert dd.snaps == [0, 10]


def test_snaps_legacy_zero(tmp_path):
    dd = BifrostData("old", folder=tmp_path)
    (tmp_path / "old.idl").touch()
    (tmp_path / "old.snap").touch()
    assert dd.snaps == [0]
    (tmp_path / "old_000.idl").touch()
    assert dd.snaps == [0]


def test_scan_positions_respect_stagger_and_native_z():
    dd = _data(n=23)
    for var, axis in [("bx", "x"), ("by", "y"), ("bz", "z"), ("r", "z")]:
        full = np.asarray(dd.get_grid(location=dd.get_location(var))["xyz".index(axis)])
        selected = _scan_positions(
            dd, var, 0, axis, {"x": None, "y": None, "z": None}, 3
        )
        np.testing.assert_array_equal(selected, full[::3])


def test_scan_coordinate_physical_roi_and_every():
    dd = _data(n=24)
    xx, yy, zz = dd.get_grid(location=dd.get_location("r"))
    roi = {
        "x": slice(float(xx[2]), float(xx[20])),
        "y": slice(float(yy[3]), float(yy[19])),
        "z": slice(float(zz[4]), float(zz[18])),
    }
    expected = dd.get_grid(location="center", **roi)[1][::4]
    np.testing.assert_array_equal(_scan_positions(dd, "r", 0, "y", roi, 4), expected)


@pytest.mark.parametrize("axis", ["x", "y", "z"])
@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
def test_real_reader_spatial_movie_reads_one_plane_per_frame(tmp_path, axis):
    dd = _data(n=20)
    dd.fdir = str(tmp_path)
    dd._meta_cache[0].params["tsnap"] = 120.0
    coords = dd.get_grid(location=dd.get_location("r"))["xyz".index(axis)]
    roi = slice(float(coords[2]), float(coords[12]))
    wanted = dd.get_grid(location="center", **{axis: roi})["xyz".index(axis)][::3]
    calls = []
    original_get_var = dd.get_var

    def track(var, snap, **kwargs):
        calls.append((snap, kwargs.copy()))
        return original_get_var(var, snap, **kwargs)

    dd.get_var = track
    target = movie(
        dd,
        "r",
        0,
        scan=axis,
        every=3,
        **{axis: roi},
        fps=12,
        dpi=72,
        figsize=(4, 3),
        progress=False,
        cbar=False,
    )
    assert target.is_file() and target.parent == tmp_path / "movies"
    assert target.name.startswith(
        f"r_0000_{dict(x='YZ', y='XZ', z='XY')[axis]}_scan-{axis}="
    )
    assert f"_every=0003.mp4" in target.name
    assert len(calls) == len(wanted)  # No re-read of first plane.
    assert all(int(snap) == 0 for snap, _ in calls)
    for (_, kwargs), pos in zip(calls, wanted):
        assert kwargs[axis] == float(pos)
        assert kwargs["uniform_z"] == (axis != "z")
        assert all(kwargs[other] is None for other in "xyz" if other != axis)
    if shutil.which("ffprobe"):
        count = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(target),
            ],
            text=True,
        ).strip()
        assert int(count) == len(wanted)
    with pytest.raises(FileExistsError):
        movie(dd, "r", 0, scan=axis, every=3, **{axis: roi}, progress=False)


def test_invalid_scan_modes_rejected_before_encoding(tmp_path):
    dd = _data(n=20)
    dd.fdir = str(tmp_path)
    for kwargs, message in [
        ({"scan": "bad"}, "scan must"),
        ({"scan": "x", "every": 0}, "every"),
        ({"scan": "x", "every": 1.5}, "every"),
        ({"scan": "x", "y": 3.0}, "second fixed axis"),
        ({"scan": "x", "x": slice(0, 10, 2)}, "slice.step"),
    ]:
        with pytest.raises((ValueError, TypeError), match=message):
            movie(dd, "r", 0, **kwargs)
    with pytest.raises(ValueError, match="one nonnegative integer"):
        movie(dd, "r", [0], scan="x")
    with pytest.raises(ValueError, match="only supported"):
        movie(dd, "r", [0, 2], y=4.0, every=2)
    assert not (tmp_path / "movies").exists()


def test_png_paths_use_simulation_directory_and_absolute_works(tmp_path, monkeypatch):
    dd = _data(n=16)
    sim = tmp_path / "simulation"
    dd.fdir = str(sim)
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    y = float(dd.get_y()[6])
    for name, expected in [
        (True, sim / "plots"),
        ("my_plot.png", sim / "plots"),
        ("reports/my_plot.png", sim / "reports"),
        (str(tmp_path / "absolute.png"), tmp_path),
    ]:
        fig, _, _ = dd.plot("r", 0, y=y, png=name, cbar=False, figsize=(2, 2))
        plt.close(fig)
        assert len(list(expected.glob("*.png"))) > 0
    assert not (working / "plots").exists()


def test_los_periodic_constant_face_integrals_agree_with_center(tmp_path):
    dd = _data(n=24)
    # Fully periodic X and Y. Test constants to catch the missing-cell error.
    for var, axis, value in [("bx", "x", 3.0), ("by", "y", 4.0)]:
        dd.arrays[var][:] = value
        fig, _, im = dd.plot(var, 0, los=axis, cbar=False)
        try:
            widths = getattr(dd, f"get_d{axis}1d")()
            face = dd.get_var(var, 0, units="solar").astype(np.float64)
            expected = np.mean(face, axis="xyz".index(axis)) * widths.sum() * 1e8
            np.testing.assert_allclose(im.get_array(), expected, rtol=1e-5)
        finally:
            plt.close(fig)


def test_los_face_full_physical_slice_and_nonperiodic_fallback():
    dd = _data(n=20)
    var, axis = "bx", "x"
    coords = dd.get_grid(location=dd.get_location(var))[0]
    full_slice = slice(float(coords[0]), float(coords[-1]))
    fig, _, full = dd.plot(var, 0, los=axis, cbar=False)
    fig2, _, equivalent = dd.plot(var, 0, los=axis, x=full_slice, cbar=False)
    try:
        np.testing.assert_allclose(full.get_array(), equivalent.get_array())
    finally:
        plt.close(fig)
        plt.close(fig2)
    dd._meta_cache[0].periodic = (False, True, False)
    fig, _, im = dd.plot(var, 0, los=axis, cbar=False)
    try:
        plane = dd.get_var(var, 0, units="solar").astype(np.float64)
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        expected = trapz(plane, x=coords * 1e8, axis=0)
        np.testing.assert_allclose(im.get_array(), expected.T, rtol=1e-6)
    finally:
        plt.close(fig)


def test_los_face_cropped_retains_trapezoidal_rule():
    dd = _data(n=22)
    coords = dd.get_grid(location=dd.get_location("bx"))[0]
    part = slice(float(coords[3]), float(coords[14]))
    fig, _, im = dd.plot("bx", 0, los="x", x=part, cbar=False)
    try:
        face_pos = dd.get_grid(location="xface", x=part)[0]
        values = dd.get_var("bx", 0, x=part).astype(np.float64)
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        expected = trapz(values, x=face_pos * 1e8, axis=0)
        np.testing.assert_allclose(im.get_array(), expected.T, rtol=1e-6)
    finally:
        plt.close(fig)


def test_los_periodic_nonconstant_matches_conservative_centering():
    """For periodic fields the integral must agree across stagger positions."""
    dd = _data(n=32)
    i = np.arange(32, dtype=np.float32)
    wave = (3 + np.sin(2 * np.pi * i / 32) + 0.23 * np.cos(8 * np.pi * i / 32)).astype(
        np.float32
    )
    for var, axis in (("bx", "x"), ("by", "y")):
        dd.arrays[var][:] = wave[:, None, None] if axis == "x" else wave[None, :, None]
        face = dd.get_var(var, 0, units="solar")
        center = dd.get_var(var, 0, centered=True, units="solar")
        widths = getattr(dd, f"get_d{axis}1d")() * 1e8
        if axis == "x":
            face_integral = np.einsum("ijk,i->jk", face, widths, dtype=np.float64)
            center_integral = np.einsum("ijk,i->jk", center, widths, dtype=np.float64)
        else:
            face_integral = np.einsum("ijk,j->ik", face, widths, dtype=np.float64)
            center_integral = np.einsum("ijk,j->ik", center, widths, dtype=np.float64)
        fig, _, image = dd.plot(var, 0, los=axis, cbar=False)
        try:
            np.testing.assert_allclose(image.get_array(), face_integral.T, rtol=1e-6)
            np.testing.assert_allclose(face_integral, center_integral, rtol=2e-6)
        finally:
            plt.close(fig)


def test_movie_relative_folder_path_uses_run_root(tmp_path):
    from bifrost_proto.movie import _output_path, _scan_output_path

    dd = BifrostData("run", folder=tmp_path)
    seq = np.array([0, 2, 4])
    assert (
        _output_path(dd, "r", seq, 2, "y", 1.0, "XZ", "reports/out.mp4", None)
        == tmp_path / "reports/out.mp4"
    )
    assert (
        _scan_output_path(
            dd,
            "r",
            0,
            "z",
            np.array([0.0, 1.0]),
            2,
            "reports/out.mp4",
            None,
            {"x": None, "y": None, "z": None},
        )
        == tmp_path / "reports/out.mp4"
    )
