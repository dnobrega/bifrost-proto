"""v0.1.42 regressions for streamed LOS, movie overlays, centering and filenames."""

from __future__ import annotations

import shutil
import subprocess

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes

from bifrost_proto.plot_naming import _coordinate_label
from bifrost_proto.movie import movie, _scan_output_path
from bifrost_proto.plotting import _integrated_plane
from test_engine import ArrayData, make_data


def _data(n=18):
    dd = make_data(n)
    for snap in (0, 2, 4):
        dd._meta_cache[snap] = dd._meta_cache[0]
    dd._meta_cache[0].params["tsnap"] = 120.0
    return dd


def test_filename_float32_scan_and_roi(tmp_path):
    dd = _data()
    dd.fdir = str(tmp_path)
    pos = np.array([0.0, np.float32(51.1)], dtype=np.float64)
    p = _scan_output_path(dd, "modb", 299, "y", pos, 1, None, None, {
        "x": slice(np.float32(0), np.float32(18.1)), "y": None, "z": None,
    })
    assert p.name == "modb_0299_XZ_scan-y=0.00-51.10_every=0001_x=0.00-18.10.mp4"
    assert _coordinate_label(-0.0) == "0.00"
    assert _coordinate_label(-0.00001) == "0.00"


def test_centered_contours_use_center_coordinates(monkeypatch):
    dd = _data(22)
    xx, yy, zz = dd.get_grid(location="center")
    selection = {"y": float(yy[8]), "x": slice(float(xx[3]), float(xx[16])),
                 "z": slice(float(zz[3]), float(zz[18]))}
    captured = {}
    real = Axes.contour

    def record(self, x, z, value, **opts):
        captured.update(x=np.array(x), z=np.array(z), value=np.array(value), opts=opts)
        return real(self, x, z, value, **opts)

    monkeypatch.setattr(Axes, "contour", record)
    fig, _, _ = dd.plot("r", 0, **selection, cbar=False, contours={
        "var": "bx", "centered": True, "levels": 4, "colors": "white",
    })
    try:
        expected = dd.get_var("bx", 0, **selection, uniform_z=True, centered=True)
        xc, _, zc = dd.get_grid(location="center", **selection, uniform_z=True)
        np.testing.assert_allclose(captured["value"], expected.T)
        np.testing.assert_allclose(captured["x"], xc)
        np.testing.assert_allclose(captured["z"], zc)
        assert "centered" not in captured["opts"]
    finally:
        plt.close(fig)


def test_contour_default_aligns_with_centered_background(monkeypatch):
    dd = _data(20)
    _, yy, zz = dd.get_grid(location="center")
    selection = dict(y=float(yy[9]), z=slice(float(zz[2]), float(zz[15])))
    records = []
    real = Axes.contour

    def record(self, x, z, value, **opts):
        records.append((np.asarray(x).copy(), np.asarray(z).copy()))
        return real(self, x, z, value, **opts)

    monkeypatch.setattr(Axes, "contour", record)
    for centered in (None, False):
        mapping = {"var": "bx", "levels": 4}
        if centered is False:
            mapping["centered"] = False
        fig, _, _ = dd.plot("r", 0, **selection, cbar=False, contours=mapping)
        plt.close(fig)
    x_center, _, z_center = dd.get_grid(location="center", **selection, uniform_z=True)
    x_face, _, z_face = dd.get_grid(location="xface", **selection, uniform_z=True)
    np.testing.assert_allclose(records[0][0], x_center)
    np.testing.assert_allclose(records[0][1], z_center)
    np.testing.assert_allclose(records[1][0], x_face)
    np.testing.assert_allclose(records[1][1], z_face)
    assert not np.array_equal(records[0][0], records[1][0])


@pytest.mark.parametrize("var,los", [
    ("r", "x"), ("r", "y"), ("r", "z"),
    ("bx", "x"), ("by", "y"), ("bz", "z"),
])
def test_los_stream_matches_volume_and_never_reads_cube(var, los, monkeypatch):
    from bifrost_proto import plotting

    monkeypatch.setattr(plotting, "_LOS_SLAB", 4)
    dd = _data(17)
    sel = {"x": None, "y": None, "z": None}
    volume, _ = _integrated_plane(dd, var, 0, los=los, selection=sel,
                                  units="solar", low_memory=None)
    calls = []
    native = dd.get_var

    def track(name, snap, **kwargs):
        result = native(name, snap, **kwargs)
        calls.append((result.ndim, kwargs))
        return result

    dd.get_var = track
    streamed, _ = _integrated_plane(dd, var, 0, los=los, selection=sel,
                                    units="solar", low_memory=None, stream=True)
    np.testing.assert_allclose(streamed, volume, rtol=1e-6, atol=1e-8)
    # Slabs of at most 4 planes on a non-LOS axis (z for LOS y, y otherwise).
    cut = "z" if los == "y" else "y"
    assert len(calls) == -(-17 // 4)
    assert all(ndim == 3 for ndim, _ in calls)
    assert all(kwargs[f"i{cut}"].stop - kwargs[f"i{cut}"].start <= 4 for _, kwargs in calls)
    assert all(cut not in kwargs for _, kwargs in calls)
    assert all(kwargs["uniform_z"] is False for _, kwargs in calls)


def test_los_stream_nonuniform_z_roi():
    n = 18
    base = make_data(n)
    axis = np.arange(n, dtype=np.float32)
    z = -(np.arange(n, dtype=np.float32) ** 1.3)[::-1].copy()
    dd = ArrayData(base.arrays, axis, axis, z)
    zz = dd.get_z()
    sl = {"x": slice(3, 13), "y": slice(2, 15),
          "z": slice(float(zz[3]), float(zz[15]))}
    for los in "xyz":
        full, _ = _integrated_plane(dd, "r", 0, los=los, selection=sl,
                                    units="solar", low_memory=True)
        streamed, _ = _integrated_plane(dd, "r", 0, los=los, selection=sl,
                                        units="solar", low_memory=True, stream=True)
        np.testing.assert_allclose(streamed, full, rtol=1e-6)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg not installed")
@pytest.mark.parametrize("mode", ["temporal", "spatial", "los"])
def test_movies_with_real_reader_and_overlay_or_stream(tmp_path, mode):
    dd = _data(18)
    dd.fdir = str(tmp_path)
    calls = []
    original = dd.get_var

    def track(var, snap, **kwargs):
        arr = original(var, snap, **kwargs)
        calls.append((var, snap, kwargs, arr.ndim))
        return arr

    dd.get_var = track
    options = dict(dpi=60, figsize=(3, 2), cbar=False, progress=False,
                   output=tmp_path / f"{mode}.mp4")
    if mode == "temporal":
        result = movie(dd, "r", [0, 2, 4], y=7.0,
                       contours={"var": "bx", "centered": True,
                                 "levels": [0.05, 0.1], "colors": "white"},
                       **options)
        assert len([c for c in calls if c[0] == "r"]) == 3
        assert len([c for c in calls if c[0] == "bx"]) == 3
        assert all(c[2].get("centered") is True for c in calls if c[0] == "bx")
    elif mode == "spatial":
        zz = dd.get_z()
        result = movie(dd, "r", 0, scan="z", z=slice(float(zz[3]), float(zz[13])),
                       every=4, contours={"var": "bx", "centered": True,
                                          "levels": [0.05, 0.1]}, **options)
        assert len([c for c in calls if c[0] == "bx"]) == len([c for c in calls if c[0] == "r"])
        assert all(c[3] == 2 for c in calls)
    else:
        result = movie(dd, "r", [0, 2, 4], los="z", **options)
        assert len(calls) == 3  # one slab per frame: 18 planes < _LOS_SLAB
    assert result.exists() and result.stat().st_size > 500
    if shutil.which("ffprobe"):
        frames = int(subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1",
            str(result),
        ], text=True).strip())
        assert frames == (3 if mode in ("temporal", "los") else 3)


def test_movie_rejects_incompatible_modes(tmp_path):
    dd = _data()
    dd.fdir = str(tmp_path)
    with pytest.raises(ValueError, match="scan and los"):
        movie(dd, "r", 0, scan="z", los="x")
    with pytest.raises(ValueError, match="contours are not supported"):
        movie(dd, "r", [0, 2], los="x", contours={"var": "bx"})
    with pytest.raises(ValueError, match="slice or None"):
        movie(dd, "r", [0, 2], los="z", x=4.0)
    with pytest.raises(TypeError, match="contours"):
        movie(dd, "r", [0, 2], y=4.0, contours=["bx"])
    assert not (tmp_path / "movies").exists()
