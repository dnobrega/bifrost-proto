"""Regression tests for v0.1.31 mesh-coordinate selection helpers."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from test_engine import ArrayData, make_data


def _nonuniform_data(n=24):
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = (np.arange(n, dtype=np.float32) ** np.float32(1.25)).astype(np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r = np.ones((n, n, n), dtype=np.float32)
    arrays = {
        "r": r,
        "e": r,
        "px": r,
        "py": r,
        "pz": r,
        "bx": xx.astype(np.float32),
        "by": yy.astype(np.float32),
        "bz": zz.astype(np.float32),
    }
    return ArrayData(arrays, x, y, z)


def test_center_coordinate_physical_slices_match_get_var_shape():
    dd = make_data(n=24)
    xfull, yfull, zfull = dd.get_grid()
    xs = slice(float(xfull[4]), float(xfull[10]))
    ys = slice(float(yfull[5]), float(yfull[12]))
    zs = slice(float(zfull[6]), float(zfull[15]))

    x = dd.get_x(x=xs)
    y = dd.get_y(y=ys)
    z = dd.get_z(z=zs)

    np.testing.assert_array_equal(x, xfull[4:11])
    np.testing.assert_array_equal(y, yfull[5:13])
    np.testing.assert_array_equal(z, zfull[6:16])

    cube = dd.get_var("r", 0, x=xs, y=ys, z=zs, units="bifrost")
    assert cube.shape == (x.size, y.size, z.size)


def test_face_coordinates_apply_same_axis_selection_as_face_variables():
    dd = make_data(n=24)

    xf = dd.get_xf()
    yf = dd.get_yf()
    zf = dd.get_zf()
    xs = slice(float(xf[3]), float(xf[11]))
    ys = slice(float(yf[4]), float(yf[13]))
    zs = slice(float(zf[5]), float(zf[14]))

    xsel = dd.get_xf(x=xs)
    ysel = dd.get_yf(y=ys)
    zsel = dd.get_zf(z=zs)

    bx = dd.get_var("bx", 0, x=xs, units="bifrost")
    by = dd.get_var("by", 0, y=ys, units="bifrost")
    bz = dd.get_var("bz", 0, z=zs, units="bifrost")

    assert bx.shape[0] == xsel.size
    assert by.shape[1] == ysel.size
    assert bz.shape[2] == zsel.size


def test_public_and_bifrost_index_selectors_match_coordinate_orientation():
    dd = make_data(n=24)

    np.testing.assert_array_equal(dd.get_x(ix=slice(2, 13, 2)), dd.get_x()[2:13:2])
    np.testing.assert_array_equal(dd.get_y(iy=slice(3, 14, 3)), dd.get_y()[3:14:3])
    np.testing.assert_array_equal(dd.get_z(iz=slice(4, 17, 2)), dd.get_z()[4:17:2])

    # Native-index integers map through the same Y/Z public-orientation reversal
    # used by get_var().
    assert dd.get_x(ix_br=4) == dd.get_x()[4]
    assert dd.get_y(iy_br=4) == dd.get_y()[-5]
    assert dd.get_z(iz_br=4) == dd.get_z()[-5]

    # In native-output mode *_br selectors are direct aliases, including slices.
    np.testing.assert_array_equal(
        dd.get_z(iz_br=slice(3, 12, 2), br=True),
        dd.get_z(br=True)[3:12:2],
    )


def test_scalar_and_linear_coordinate_selection_returns_matching_coordinate():
    dd = make_data(n=24)
    z = dd.get_z()
    target = 0.5 * (z[8] + z[9])

    nearest = dd.get_z(z=target, method="nearest")
    linear = dd.get_z(z=target, method="linear")

    assert np.isscalar(nearest)
    assert np.isscalar(linear)
    assert nearest in (z[8], z[9])
    assert np.isclose(linear, target)


def test_selected_uniform_z_matches_get_var_and_dz1d():
    dd = _nonuniform_data()
    zfull = dd.get_z()
    zs = slice(float(zfull[4]), float(zfull[17]))

    znative = dd.get_z(z=zs)
    zuniform = dd.get_z(z=zs, uniform_z=True)
    expected = np.linspace(znative[0], znative[-1], znative.size)
    np.testing.assert_allclose(zuniform, expected)

    cube = dd.get_var("r", 0, z=zs, uniform_z=True, units="bifrost")
    assert cube.shape[2] == zuniform.size

    dz = dd.get_dz1d(z=zs, uniform_z=True)
    assert dz.shape == zuniform.shape
    np.testing.assert_allclose(dz, np.gradient(zuniform))
    assert np.isclose(dd.get_dz(z=zs, uniform_z=True), np.mean(dz))


def test_spacing_selection_matches_coordinate_size_and_has_no_location_argument():
    dd = _nonuniform_data()
    x = dd.get_x()
    y = dd.get_y()
    z = dd.get_z()
    xs = slice(float(x[2]), float(x[12]))
    ys = slice(float(y[3]), float(y[13]))
    zs = slice(float(z[4]), float(z[14]))

    assert dd.get_dx1d(x=xs).shape == dd.get_x(x=xs).shape
    assert dd.get_dy1d(y=ys).shape == dd.get_y(y=ys).shape
    assert dd.get_dz1d(z=zs).shape == dd.get_z(z=zs).shape

    for name in ("get_dx1d", "get_dy1d", "get_dz1d", "get_dx", "get_dy", "get_dz"):
        assert "location" not in inspect.signature(getattr(dd, name)).parameters

    with pytest.raises(TypeError):
        dd.get_dz1d(location="zface")


def test_get_grid_accepts_full_get_var_style_selection_and_uniform_z():
    dd = _nonuniform_data()
    z = dd.get_z()
    zs = slice(float(z[5]), float(z[16]))

    gx, gy, gz = dd.get_grid(
        location="center",
        ix=slice(2, 11, 2),
        iy=7,
        z=zs,
        uniform_z=True,
    )
    cube = dd.get_var(
        "r",
        0,
        ix=slice(2, 11, 2),
        iy=7,
        z=zs,
        uniform_z=True,
        units="bifrost",
    )

    assert np.isscalar(gy)
    assert cube.shape == (gx.size, gz.size)
    np.testing.assert_allclose(gz, np.linspace(gz[0], gz[-1], gz.size))


def test_physical_selector_stays_in_mm_when_coordinate_output_units_change():
    dd = make_data(n=24)
    x = dd.get_x()
    xs = slice(float(x[4]), float(x[9]))

    solar = dd.get_x(x=xs, units="solar")
    cgs = dd.get_x(x=xs, units="cgs")
    si = dd.get_x(x=xs, units="si")

    np.testing.assert_allclose(cgs, solar * 1.0e8)
    np.testing.assert_allclose(si, solar * 1.0e6)
