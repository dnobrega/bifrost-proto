"""Regression tests for spreadsheet-based presets and local solar colormaps."""

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import LogNorm, Normalize, SymLogNorm

from bifrost_proto import BifrostData  # noqa: F401  populate real registry
from bifrost_proto.colormaps import _idl3_rgb, make_colormaps, register_colormaps
from bifrost_proto.plot_defaults import PLOT_PRESETS, get_plot_preset
from bifrost_proto.registry import GLOBAL_REGISTRY
from bifrost_proto.units import BifrostUnits


class DummyData:
    registry = GLOBAL_REGISTRY

    def get_unit_type(self, var):
        return self.registry[var].unit_type

    def _grid_meta(self):
        class Meta:
            units = BifrostUnits()

        return Meta()


def test_all_registry_quantities_have_valid_defaults():
    assert set(PLOT_PRESETS) == set(GLOBAL_REGISTRY), (
        set(GLOBAL_REGISTRY) - set(PLOT_PRESETS),
        set(PLOT_PRESETS) - set(GLOBAL_REGISTRY),
    )
    dd = DummyData()
    for var in GLOBAL_REGISTRY:
        cmap, norm = get_plot_preset(dd, var)
        assert cmap is not None
        assert isinstance(norm, Normalize)
        assert norm.vmin < norm.vmax


def test_user_corrections_and_alias():
    dd = DummyData()
    cmap, norm = get_plot_preset(dd, "lf_h")
    assert cmap == "jet"
    assert isinstance(norm, LogNorm)
    assert (norm.vmin, norm.vmax) == (1e-11, 1)
    cmap, norm = get_plot_preset(dd, "uperpb")
    assert cmap == "jet"
    assert type(norm) is Normalize
    assert (norm.vmin, norm.vmax) == (0, 50)
    assert get_plot_preset(dd, "bx")[0] == get_plot_preset(dd, "b_x")[0]


def test_instrument_local_los_norms():
    dd = DummyData()
    for var in ["aia94", "aia171", "aia304", "eui174", "xrt_cpoly"]:
        _cmap, norm = get_plot_preset(dd, var)
        assert isinstance(norm, LogNorm)
        assert (norm.vmin, norm.vmax) == (1e-8, 1e-4)
        _, los_norm = get_plot_preset(dd, var, los="z")
        assert isinstance(los_norm, LogNorm)
        assert (los_norm.vmin, los_norm.vmax) == (
            (0.1, 100) if var.startswith("xrt_") else (1, 20000)
        )
    assert get_plot_preset(dd, "tg", los="z")[1] is None


def test_unit_scaling_of_norm_and_los():
    dd = DummyData()
    _, solar = get_plot_preset(dd, "u_z")
    _, si = get_plot_preset(dd, "u_z", units="si")
    assert si.vmax == pytest.approx(solar.vmax * 1000)
    _, local_si = get_plot_preset(dd, "aia171", units="si")
    assert local_si.vmin == pytest.approx(1e-6)
    _, los_si = get_plot_preset(dd, "aia171", los="z", units="si")
    assert los_si.vmin == pytest.approx(1.0)


def test_terrain_and_instrument_colors_with_reversed_versions():
    maps = make_colormaps()
    assert len(maps) == 24
    for name in (
        "red_terrain",
        "green_terrain",
        "divergent_terrain",
        "aia94",
        "aia193",
        "aia304",
        "eui174",
        "xrt",
    ):
        assert name in maps and name + "_r" in maps
        x = np.linspace(0, 1, 256)
        np.testing.assert_allclose(maps[name](x)[::-1], maps[name + "_r"](x), atol=1e-9)
    np.testing.assert_allclose(
        maps["eui174"](np.linspace(0, 1, 256)), maps["aia171"](np.linspace(0, 1, 256))
    )
    np.testing.assert_allclose(
        maps["xrt"](np.linspace(0, 1, 256)), maps["aia304"](np.linspace(0, 1, 256))
    )
    r, g, b = _idl3_rgb()
    assert (r[0], g[0], b[0]) == (0, 0, 0)
    assert (r[120], g[120], b[120]) == (173, 0, 0)
    assert (r[176], g[176], b[176]) == (255, 105, 0)
    assert (r[255], g[255], b[255]) == (255, 255, 255)
    register_colormaps()
    assert "red_terrain_r" in colormaps
    assert "eui174" in colormaps
    register_colormaps()  # idempotent


def test_sym_log_for_signed_quantities():
    dd = DummyData()
    for name, entry in PLOT_PRESETS.items():
        if entry[0] == "SymLogNorm":
            _, norm = get_plot_preset(dd, name)
            assert isinstance(norm, SymLogNorm)


def test_real_plot_applies_and_overrides_presets():
    """Plotting integration uses the same defaults, unless explicitly overridden."""
    import matplotlib.pyplot as plt
    from test_engine import make_data

    dd = make_data(n=12)
    y = float(dd.get_y()[5])
    fig, _ax, artist = dd.plot("r", 0, y=y, cbar=False, title=False)
    try:
        assert isinstance(artist.norm, LogNorm)
        assert artist.get_cmap().name == "terrain_r"
    finally:
        plt.close(fig)

    manual = Normalize(-7, 5)
    fig, _ax, artist = dd.plot(
        "r", 0, y=y, norm=manual, cmap="red_terrain_r", cbar=False, title=False
    )
    try:
        assert artist.norm is manual
        assert artist.get_cmap().name == "red_terrain_r"
    finally:
        plt.close(fig)
