"""Regression tests for absolute values and the 3-D electric diagnostics."""

import numpy as np
import pytest
from matplotlib.colors import LogNorm
from test_engine import ArrayData, make_data

from bifrost_proto.plot_defaults import get_plot_preset


def test_absolute_aliases_preserve_mesh_and_units_without_component_sign():
    dd = make_data(n=16)
    dd.arrays["by"] -= 8
    for units in ("bifrost", "solar", "si"):
        actual = dd.get_var("absby", 0, units=units)
        expected = np.abs(dd.get_var("by", 0, units=units))
        np.testing.assert_array_equal(actual, expected)
    assert dd.variables("absby") == ["absb_y"]
    assert dd.get_location("absby") == dd.get_location("by")
    assert dd.get_unit_type("absby") == dd.get_unit_type("by")
    assert dd.get_var_name("absby", latex=True) == r"$\left|B_y\right|$"
    assert "absmodb" not in dd.variables()
    cmap, norm = get_plot_preset(dd, "abseparb")
    assert cmap == "jet"
    assert isinstance(norm, LogNorm)
    assert (norm.vmin, norm.vmax) == (1e-5, 1e-3)


def test_large_absolute_preserves_native_layout_and_cached_source():
    shape = (128, 128, 64)
    values = np.full(shape, -2, dtype=np.float32, order="F")
    values[0, 0, :4] = [-0.0, np.nan, -np.inf, np.inf]
    coords = [np.arange(length, dtype=np.float32) for length in shape]
    dd = ArrayData({"r": values, "by": values}, *coords)
    with dd.cache():
        source = dd.get_var("by", 0, br=True, units="bifrost")
        result = dd.get_var("absby", 0, br=True, units="bifrost")
        np.testing.assert_array_equal(result, np.abs(source))
        assert result.flags.f_contiguous
        assert not np.signbit(result[0, 0, 0])
        assert not np.shares_memory(source, result)
        assert dd.read_counts["by"] == 1
        assert np.signbit(source[0, 0, 0])
        assert source[-1, -1, -1] == -2


def make_eparb_data(n=40, *, constant=False):
    x = np.arange(n, dtype=np.float32) / 4
    y = np.arange(n, dtype=np.float32) / 2
    z = np.arange(n, dtype=np.float32)
    shape = (n, n, n)
    one = np.ones(shape, dtype=np.float32)
    zero = np.zeros(shape, dtype=np.float32)
    if constant:
        bz = zero
        ex, ey, ez = zero, one, zero
    else:
        # B=(1,0,x+y), E=-B; sample E_z at its native (xdown, ydown) edge.
        bz = (x[:, None, None] + y[None, :, None]) * one
        ex, ey = -one, zero
        ez = -(bz - (0.125 + 0.25))
    arrays = {"r": one, "bx": one, "by": zero, "bz": bz, "ex": ex, "ey": ey, "ez": ez}
    return ArrayData(arrays, x, y, z)


def test_eparb_diagnostics_follow_staggered_curl_and_convert_units():
    dd = make_eparb_data()
    interior = (slice(15, 25),) * 3
    vertical = dd.arrays["bz"][interior]
    # Curl(E_parallel)=(-1,1,0), so B dot curl=-1 and |B cross curl| is below.
    criterion = np.sqrt(1 + 2 * vertical**2)
    squared_field = 1 + vertical**2
    expected = {
        "keparb": criterion,
        "taurec3d": squared_field / criterion,
        "taudif3d": squared_field,
    }
    with dd.cache():
        for name, reference in expected.items():
            native = dd.get_var(name, 0, br=True, units="bifrost")
            sampled = dd.get_var(
                name,
                0,
                br=True,
                units="bifrost",
                ix=interior[0],
                iy=interior[1],
                iz=interior[2],
            )
            np.testing.assert_allclose(sampled, native[interior], rtol=2e-6)
            np.testing.assert_allclose(sampled, reference, rtol=2e-5)
            if name == "keparb":
                u = dd._grid_meta().units
                factors = {
                    "solar": u.u_B * u.u_el / u.length_to_Mm,
                    "cgs": u.u_B * u.u_el / u.u_l,
                    "si": (u.u_B * 1e-4) * (u.u_el * 29979.2458) / (u.u_l * 1e-2),
                }
            else:
                factors = dict.fromkeys(
                    ("solar", "cgs", "si"), dd._grid_meta().units.u_t
                )
            for units, factor in factors.items():
                converted = dd.get_var(name, 0, br=True, units=units)
                np.testing.assert_allclose(
                    converted[interior], reference * factor, rtol=2e-5
                )


def test_zero_curl_and_characteristic_time_plot_defaults():
    dd = make_eparb_data(constant=True)
    assert np.all(dd.get_var("keparb", 0) == 0)
    for name in ("taurec3d", "taudif3d"):
        assert np.all(np.isposinf(dd.get_var(name, 0)))
        assert dd.get_units(name) == "s"
        cmap, norm = get_plot_preset(dd, name)
        assert cmap == "red_terrain_r"
        assert isinstance(norm, LogNorm)
        assert (norm.vmin, norm.vmax) == (0.1, 1000)
    cmap, norm = get_plot_preset(dd, "keparb")
    assert cmap == "jet"
    assert isinstance(norm, LogNorm)
    assert 0 < norm.vmin < norm.vmax


@pytest.mark.parametrize("name", ["absby", "keparb", "taurec3d"])
def test_empty_logarithmic_los_colorbar(name):
    import matplotlib.pyplot as plt

    dd = make_eparb_data(constant=True)
    fig, _, artist = dd.plot(name, 0, los="z", title=False)
    try:
        fig.canvas.draw()
        assert isinstance(artist.norm, LogNorm)
        assert 0 < artist.norm.vmin < artist.norm.vmax
    finally:
        plt.close(fig)


def test_absolute_label_requires_generated_variant():
    from bifrost_proto.labels import latex_varname

    assert latex_varname("absb_y") == r"$\left|B_y\right|$"
    # User quantities that merely start with "abs" keep the generic label.
    assert latex_varname("absr") == r"$\mathrm{absr}$"
    assert latex_varname("absorb") == r"$\mathrm{absorb}$"
    assert latex_varname("abs") == r"$\mathrm{abs}$"


def test_fused_cross_norm_matches_numpy_on_strided_views():
    from bifrost_proto.quantities import _cross_norm_arrays

    rng = np.random.default_rng(3)
    base = rng.normal(size=(6, 9, 11, 7)).astype(np.float32)
    parts = [np.asfortranarray(base[i])[1:8:2, :, 2:] for i in range(6)]
    a, b = np.stack(parts[:3], axis=-1), np.stack(parts[3:], axis=-1)
    expected = np.linalg.norm(np.cross(a, b), axis=-1)
    np.testing.assert_allclose(
        _cross_norm_arrays(*parts), expected, rtol=2e-6, atol=1e-6
    )
    contiguous = [np.asfortranarray(p) for p in parts]
    np.testing.assert_allclose(
        _cross_norm_arrays(*contiguous), expected, rtol=2e-6, atol=1e-6
    )


def _wrapped_eparb_run(tmp_path, n=72, nz=16):
    """On-disk run with n > 64 so X/Y halos use unwrapped periodic windows."""
    from test_api_io_v127 import _write_mesh

    from bifrost_proto import BifrostData

    x = np.arange(n, dtype=np.float32)
    z = np.arange(nz, dtype=np.float32)
    _write_mesh(tmp_path / "mesh.mesh", x, x, z)
    (tmp_path / "w_000.idl").write_text(
        f"mx={n}\nmy={n}\nmz={nz}\ndx=1\ndy=1\ndz=1\n"
        "do_mhd=.true.\ndo_hion=.false.\ndo_helium=.false.\n"
        "aux='ex ey ez'\nmeshfile='mesh.mesh'\nu_l=1e8\nu_t=1e2\nu_r=1e-7\n"
    )
    shape = (n, n, nz)
    kx = 2 * np.pi * x[:, None, None] / n
    ky = 2 * np.pi * x[None, :, None] / n
    kz = z[None, None, :] / nz
    one = np.ones(shape, dtype=np.float32)
    snap = np.memmap(
        tmp_path / "w_000.snap", dtype="<f4", mode="w+", shape=(*shape, 8), order="F"
    )
    snap[..., 0] = one
    snap[..., 1:4] = 0
    snap[..., 4] = 2 * one
    snap[..., 5] = one + 0.5 * np.sin(ky) * np.cos(kz)
    snap[..., 6] = 0.3 * np.cos(kx) * one
    snap[..., 7] = one + 0.4 * np.sin(kx + ky) * one
    snap.flush()
    aux = np.memmap(
        tmp_path / "w_000.aux", dtype="<f4", mode="w+", shape=(*shape, 3), order="F"
    )
    aux[..., 0] = 0.1 * np.sin(2 * kx) * np.cos(ky) * one
    aux[..., 1] = 0.1 * np.cos(kx) * np.sin(kz + ky)
    aux[..., 2] = 0.1 * np.sin(kx - ky) * one + 0.05 * kz
    aux.flush()
    del snap, aux
    return BifrostData("w", fdir=tmp_path)


def test_eparb_diagnostics_match_full_crop_across_periodic_edges(tmp_path):
    dd = _wrapped_eparb_run(tmp_path)
    for name in ("keparb", "taurec3d", "taudif3d", "abseparb_x"):
        full = dd.get_var(name, 0, br=True, units="bifrost")
        for axis in "xy":
            for index in (0, 1, 70, 71):
                plane = dd.get_var(
                    name, 0, br=True, units="bifrost", **{"i" + axis: index}
                )
                np.testing.assert_array_equal(
                    plane, np.take(full, index, axis="xyz".index(axis))
                )
        window = dd.get_var(
            name, 0, br=True, units="bifrost", ix=slice(0, 60), iz=slice(0, 5)
        )
        np.testing.assert_array_equal(window, full[:60, :, :5])
