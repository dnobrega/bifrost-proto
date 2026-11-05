"""LOS quadrature, physical units, ROI and automatic PNG regressions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bifrost_proto.plotting import _colorbar_label, _integrated_unit_plain
from test_engine import ArrayData, make_data


def _data(n=16):
    dd = make_data(n)
    dd._meta_cache[0].params["tsnap"] = 1.5
    return dd


@pytest.mark.parametrize("los,expected_shape", [
    ("x", (16, 16)), ("y", (16, 16)), ("z", (16, 16)),
])
@pytest.mark.parametrize("units,expected", [
    ("solar", 320.0), ("cgs", 320.0), ("si", 3200.0), ("bifrost", 32.0),
])
def test_center_constant_density_column_all_axes(los, expected_shape, units, expected):
    """Center-cell widths cover the full 16-cell domain in each direction."""
    dd = _data()
    fig, ax, im = dd.plot("r", 0, los=los, units=units, cbar=False)
    try:
        assert im.get_array().shape == expected_shape
        np.testing.assert_allclose(im.get_array(), expected, rtol=2e-6)
        assert ax.get_title(loc="left") == f"LOS={los.upper()}"
    finally:
        plt.close(fig)


def test_center_integral_uses_nonuniform_z_widths_and_roi():
    """Use native nonuniform widths rather than uniform-Z quadrature."""
    n = 21
    base = make_data(n)
    arrays = base.arrays
    axis = np.arange(n, dtype=np.float32)
    native_z = np.arange(n, dtype=np.float32) ** 1.35
    dd = ArrayData(arrays, axis, axis, native_z)
    dd._meta_cache[0].params["tsnap"] = 1.5
    z = dd.get_z()
    z_slice = slice(float(z[3]), float(z[17]))
    widths = dd.get_dz1d(z=z_slice, units="solar")
    expected = 2.0e-7 * 1e8 * widths.sum()
    fig, ax, im = dd.plot("r", 0, los="z", z=z_slice, cbar=False)
    try:
        np.testing.assert_allclose(im.get_array(), expected, rtol=5e-6)
        assert not np.isclose(expected, 2.0e-7 * (z[17] - z[3]) * 1e8)
    finally:
        plt.close(fig)


def test_staggered_integral_uses_periodic_rectangle_on_xy_and_trapezoid_on_z():
    """A complete periodic axis wraps around; native Z is nonperiodic."""
    dd = _data(20)
    for var, los in [("bx", "x"), ("by", "y"), ("bz", "z")]:
        coords = dd.get_grid(location=dd.get_location(var))
        plane = dd.get_var(var, 0, units="solar").astype(np.float64)
        axis = "xyz".index(los)
        positions = np.asarray(coords[axis]) * 1e8
        trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        if los in ("x", "y"):
            weights = getattr(dd, f"get_d{los}1d")() * 1e8
            expected = np.einsum(
                "ijk,i->jk" if los == "x" else "ijk,j->ik",
                plane, weights, dtype=np.float64,
            )
        else:
            expected = trapz(plane, x=positions, axis=axis)
        fig, ax, im = dd.plot(var, 0, los=los, cbar=False)
        try:
            np.testing.assert_allclose(im.get_array(), expected.T, rtol=1e-6, atol=1e-7)
        finally:
            plt.close(fig)


def test_los_roi_and_vertical_remesh_take_place_after_integral():
    dd = _data(24)
    x, y, z = dd.get_grid()
    selection = {"x": slice(float(x[4]), float(x[17])),
                 "y": slice(float(y[2]), float(y[10])),
                 "z": slice(float(z[3]), float(z[18]))}
    fig, ax, im = dd.plot("r", 0, los="y", **selection, cbar=False)
    try:
        raw = dd.get_var("r", 0, **selection, units="solar", uniform_z=False)
        widths = dd.get_dy1d(y=selection["y"])
        expected = np.einsum("ijk,j->ik", raw, widths * 1e8, dtype=np.float64)
        np.testing.assert_allclose(im.get_array(), expected.T)
        assert im.get_array().shape[1] < 24 and im.get_array().shape[0] < 24
    finally:
        plt.close(fig)


def test_unit_simplification_and_registered_names():
    dd = _data()
    for mode, expected in [("solar", "g cm$^{-2}$"),
                           ("cgs", "g cm$^{-2}$"),
                           ("si", "kg m$^{-2}$"),
                           ("bifrost", "bifrost Mm")]:
        label = _colorbar_label(dd, "r", mode, los="y")
        assert r"\int \rho\,\mathrm{d}y" in label
        assert expected in label
    assert _integrated_unit_plain("G Mm^-1", "cm", solar=True) == ("G", 1e-8)
    assert _integrated_unit_plain("K", "cm", solar=True) == ("K cm", 1.0)
    assert _integrated_unit_plain("erg cm^-3 s^-1", "cm")[0] == "erg cm^-2 s^-1"
    assert _integrated_unit_plain("cm^-1", "cm")[0] == ""


def test_solar_Mm_inverse_conversion_is_numerically_consistent():
    dd = _data()

    @dd.register_variable("test_gradient", location="center", unit_type="magnetic_gradient")
    def test_gradient(data, snap, selection, context):
        return data._getvar("r", snap, selection, context)

    solar_gradient = dd.get_var("test_gradient", 0, units="solar")
    fig, ax, im = dd.plot("test_gradient", 0, los="y", cbar=False)
    try:
        expected = np.sum(solar_gradient, axis=1) * dd.get_dy()  # G / Mm * Mm = G
        np.testing.assert_allclose(im.get_array(), expected.T, rtol=1e-6)
        assert _colorbar_label(dd, "test_gradient", "solar", los="y").endswith("(G)")
    finally:
        plt.close(fig)


def test_los_invalid_selections_and_contours():
    dd = _data()
    with pytest.raises(ValueError, match="los must"):
        dd.plot("r", 0, los="q", cbar=False)
    with pytest.raises(ValueError, match="must be a physical slice"):
        dd.plot("r", 0, los="x", x=4.0, cbar=False)
    with pytest.raises(ValueError, match="contours are not supported"):
        dd.plot("r", 0, los="y", contours={"var": "bz"})
    with pytest.raises(ValueError, match="slice.step"):
        dd.plot("r", 0, los="z", x=slice(1, 3, 1))


def test_png_auto_path_dpi_and_overwrite_protection(tmp_path, monkeypatch):
    dd = _data(12)
    dd.fdir = str(tmp_path / "simulation")
    monkeypatch.chdir(tmp_path)
    fig, ax, im = dd.plot("r", 0, los="y", x=slice(2, 8), png=True,
                          cbar=False, figsize=(2, 2))
    try:
        pngs = list((tmp_path / "simulation" / "plots").glob("r_0000_XZ_los=y_x=2.00-8.00.png"))
        assert len(pngs) == 1
        assert pngs[0].stat().st_size > 200
    finally:
        plt.close(fig)
    with pytest.raises(FileExistsError, match="PNG already exists"):
        fig, _, _ = dd.plot("r", 0, los="y", x=slice(2, 8), png=True, cbar=False,
                            figsize=(2, 2))
        plt.close(fig)
    fig, _, _ = dd.plot("r", 0, los="y", x=slice(2, 8), png=True,
                        overwrite=True, cbar=False, figsize=(2, 2))
    plt.close(fig)


def test_png_custom_filename_and_regular_plot(tmp_path, monkeypatch):
    dd = _data(12)
    dd.fdir = str(tmp_path / "simulation")
    monkeypatch.chdir(tmp_path)
    fig, _, _ = dd.plot("r", 0, y=float(dd.get_y()[5]), png="custom.png",
                        cbar=False, figsize=(2, 2), dpi=150)
    plt.close(fig)
    assert (tmp_path / "simulation" / "plots" / "custom.png").is_file()
    fig, _, _ = dd.plot("r", 0, y=float(dd.get_y()[5]), png="output/custom.png",
                        cbar=False, figsize=(2, 2))
    plt.close(fig)
    assert (tmp_path / "simulation" / "output" / "custom.png").is_file()
    with pytest.raises(ValueError, match=".png"):
        dd.plot("r", 0, los="y", png="bad.jpg", cbar=False)


def test_png_default_dpi_and_auto_name_for_single_cut(tmp_path, monkeypatch):
    """A single-cut PNG uses the same filename components as an MP4 cut."""
    from PIL import Image

    dd = _data(12)
    dd.fdir = str(tmp_path / "simulation")
    monkeypatch.chdir(tmp_path)
    fig, _, _ = dd.plot("r", 0, y=float(dd.get_y()[5]), png=True,
                        cbar=False, figsize=(2, 2))
    plt.close(fig)
    filename = next((tmp_path / "simulation" / "plots").glob("r_0000_XZ_y=*.png"))
    with Image.open(filename) as image:
        assert image.format == "PNG"
        assert image.info["dpi"][0] == pytest.approx(300, abs=1)
        assert image.info["dpi"][1] == pytest.approx(300, abs=1)


def test_png_rejects_existing_file_before_variable_load(tmp_path, monkeypatch):
    """Existing output protection must not trigger unnecessary cube evaluation."""
    dd = _data(12)
    dd.fdir = str(tmp_path / "simulation")
    monkeypatch.chdir(tmp_path)
    fig, _, _ = dd.plot("r", 0, los="y", png=True, cbar=False, figsize=(2, 2))
    plt.close(fig)
    dd.read_counts.clear()
    with pytest.raises(FileExistsError, match="PNG already exists"):
        dd.plot("r", 0, los="y", png=True, cbar=False, figsize=(2, 2))
    assert dd.read_counts == {}
