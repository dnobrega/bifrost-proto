"""Regression tests for the integrated quick-look plotting API."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from test_engine import make_data


def test_plot_scalar_cut_uses_canonical_grid_api():
    """A scalar cut must work with get_x/get_y/get_z having no location kwarg."""
    dd = make_data(n=16)
    dd._meta_cache[0].params["tsnap"] = 3.0
    norm = Normalize(vmin=0.0, vmax=2.0)

    fig, ax, im = dd.plot(
        "r",
        0,
        y=float(dd.get_y()[7]),
        norm=norm,
        cbar=False,
    )
    try:
        assert im.norm is norm
        assert np.asarray(im.get_array()).shape == (16, 16)
        assert ax.get_xlabel() == "x (Mm)"
        assert ax.get_ylabel() == "z (Mm)"
        assert "Y=" in ax.get_title(loc="left")
        assert "min" in ax.get_title(loc="right")
    finally:
        plt.close(fig)


def test_plot_uses_registered_face_coordinates():
    """A staggered quantity must use coordinates matching its registered location."""
    dd = make_data(n=16)
    dd._meta_cache[0].params["tsnap"] = 0.0

    fig, ax, im = dd.plot(
        "bz",
        0,
        x=float(dd.get_x()[6]),
        cbar=False,
        title=False,
    )
    try:
        extent = np.asarray(im.get_extent(), dtype=float)
        y = np.asarray(dd.get_y(), dtype=float)
        zf = np.asarray(dd.get_zf(units="solar"), dtype=float)
        zf_uniform = np.linspace(zf[0], zf[-1], zf.size)
        expected = np.array([
            y[0] - 0.5 * (y[1] - y[0]),
            y[-1] + 0.5 * (y[-1] - y[-2]),
            zf_uniform[0] - 0.5 * (zf_uniform[1] - zf_uniform[0]),
            zf_uniform[-1] + 0.5 * (zf_uniform[-1] - zf_uniform[-2]),
        ])
        np.testing.assert_allclose(extent, expected)
    finally:
        plt.close(fig)


def test_plot_roi_slices_read_only_selected_data():
    """Physical slice selectors crop the plotted image rather than start a slider."""
    dd = make_data(n=24)
    dd._meta_cache[0].params["tsnap"] = 1.5
    x, y, z = dd.get_grid()
    selection = {
        "y": float(y[10]),
        "x": slice(float(x[4]), float(x[16])),
        "z": slice(float(z[3]), float(z[15])),
    }
    fig, ax, im = dd.plot("r", 0, **selection, cbar=False)
    try:
        raw = dd.get_var("r", 0, **selection, uniform_z=True)
        cx, _, cz = dd.get_grid(**selection, uniform_z=True)
        np.testing.assert_allclose(im.get_array(), raw.T)
        assert np.asarray(im.get_array()).shape == (len(cz), len(cx))
        assert len(cx) < 24 and len(cz) < 24
    finally:
        plt.close(fig)


def test_plot_external_axes_preserve_return_contract():
    """Automatic single plot and multi-panel embedding share one drawing path."""
    dd = make_data(n=16)
    dd._meta_cache[0].params["tsnap"] = 1.5
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    try:
        for ax, var in zip(axes, ("r", "bz")):
            fig_out, ax_out, im = dd.plot(
                var, 0, y=float(dd.get_y()[7]), ax=ax, cbar=True,
            )
            assert fig_out is fig and ax_out is ax
            assert im.axes is ax
        assert len(fig.axes) == 4  # Two data axes and two colorbar axes.
    finally:
        plt.close(fig)


def test_plot_contour_uses_same_roi_and_its_own_stagger_grid(monkeypatch):
    """Contour geometry comes from the contour variable, not image pixel indices."""
    dd = make_data(n=24)
    dd._meta_cache[0].params["tsnap"] = 1.5
    x, y, z = dd.get_grid()
    selections = {
        "y": float(y[10]),
        "x": slice(float(x[4]), float(x[16])),
        "z": slice(float(z[3]), float(z[15])),
    }
    captured = {}
    from matplotlib.axes import Axes
    old_contour = Axes.contour

    def record_contour(self, xx, zz, values, **kwargs):
        captured.update(x=np.asarray(xx), z=np.asarray(zz),
                        values=np.asarray(values), kwargs=kwargs)
        return old_contour(self, xx, zz, values, **kwargs)

    monkeypatch.setattr(Axes, "contour", record_contour)
    fig, ax, _ = dd.plot(
        "r", 0, **selections, cbar=False,
        contours={"var": "bz", "centered": False, "levels": [1e-3, 2e-3],
                  "colors": "white", "linewidths": 0.8},
    )
    try:
        expected = dd.get_var("bz", 0, **selections, uniform_z=True)
        gx, _, gz = dd.get_grid(location=dd.get_location("bz"),
                                **selections, uniform_z=True)
        np.testing.assert_allclose(captured["values"], expected.T)
        np.testing.assert_allclose(captured["x"], gx)
        np.testing.assert_allclose(captured["z"], gz)
        assert captured["kwargs"]["linewidths"] == 0.8
        assert "var" not in captured["kwargs"]
    finally:
        plt.close(fig)


def test_plot_validates_cut_and_roi_slices():
    """Exactly one scalar is required, and step-bearing slices are rejected."""
    import pytest

    dd = make_data(n=12)
    dd._meta_cache[0].params["tsnap"] = 1.5
    for selections in ({"z": slice(0, 2)}, {"x": 1.0, "y": 2.0}, {}):
        with pytest.raises(ValueError, match="exactly one scalar"):
            dd.plot("r", 0, cbar=False, **selections)
    with pytest.raises(ValueError, match="slice.step"):
        dd.plot("r", 0, y=1.0, x=slice(0, 2, 1), cbar=False)
    with pytest.raises(ValueError, match="start"):
        dd.plot("r", 0, y=1.0, z=slice(5, 2), cbar=False)
    with pytest.raises(TypeError, match="contours"):
        dd.plot("r", 0, y=1.0, contours=["bz"], cbar=False)


def test_plot_all_three_planes_with_crop():
    """XY, XZ and YZ crops agree with get_var in public orientation."""
    dd = make_data(n=22)
    dd._meta_cache[0].params["tsnap"] = 1.5
    x, y, z = dd.get_grid()
    selections = [
        {"z": float(z[7]), "x": slice(float(x[4]), float(x[14])),
         "y": slice(float(y[5]), float(y[16]))},
        {"y": float(y[7]), "x": slice(float(x[4]), float(x[14])),
         "z": slice(float(z[5]), float(z[16]))},
        {"x": float(x[7]), "y": slice(float(y[4]), float(y[14])),
         "z": slice(float(z[5]), float(z[16]))},
    ]
    for sel in selections:
        fig, ax, im = dd.plot("r", 0, **sel, cbar=False)
        try:
            np.testing.assert_allclose(im.get_array(), dd.get_var(
                "r", 0, **sel, uniform_z=(sel.get("z") is not None and
                                                  isinstance(sel["z"], slice)),
            ).T)
        finally:
            plt.close(fig)


def test_plot_low_memory_forwarded(monkeypatch):
    """The plotting wrapper propagates the optional resolver cache policy."""
    dd = make_data(n=14)
    dd._meta_cache[0].params["tsnap"] = 1.5
    original_get_var = dd.get_var
    seen = []

    def spy(var, snap, **kwargs):
        seen.append(kwargs["low_memory"])
        return original_get_var(var, snap, **kwargs)

    monkeypatch.setattr(dd, "get_var", spy)
    fig, ax, im = dd.plot("r", 0, y=float(dd.get_y()[6]),
                          contours={"var": "bz", "levels": 4},
                          low_memory=True, cbar=False)
    try:
        assert seen == [True, True]
    finally:
        plt.close(fig)
