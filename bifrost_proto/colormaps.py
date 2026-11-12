"""Small, dependency-free (except NumPy/Matplotlib) solar display colormaps.

The AIA/XRT/EUI color tables follow SunPy 8.0.0's AIA/IDL 3 RGB tables;
see resources/colormaps/SUNPY_LICENSE.txt.  SunPy and Astropy are NOT needed.
This module deliberately registers maps lazily, only on the first plot or
explicit ``register_colormaps()`` call.
"""

from __future__ import annotations

import colorsys
from functools import lru_cache

import numpy as np
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

TERRAIN_HSV = {
    "red_terrain": [
        (0.0002, 0, 0, 100),
        (0.0002666, 36, 60, 40),
        (0.0004, 48, 71, 100),
        (0.0005333, 359, 70, 87),
        (0.0006, 359, 100, 75),
    ],
    "green_terrain": [
        (0.0002, 0, 0, 100),
        (0.0002666, 36, 50, 70),
        (0.0004, 48, 71, 100),
        (0.0005333, 109, 70, 87),
        (0.0006, 129, 80, 90),
    ],
    "divergent_terrain": [
        (-0.00060, 229, 100, 75),
        (-0.00053, 189, 70, 87),
        (-0.00040, 97, 70, 100),
        (-0.0002666, 107, 60, 40),
        (0, 0, 0, 100),
        (0.0002666, 36, 60, 40),
        (0.00040, 48, 71, 100),
        (0.00053, 359, 70, 87),
        (0.00060, 359, 100, 75),
    ],
}


def _terrain(name, points):
    """Convert user-defined HSV control points into a 256-color map."""
    low, high = points[0][0], points[-1][0]
    colors = [
        (
            (value - low) / (high - low),
            colorsys.hsv_to_rgb(hue / 360, saturation / 100, brightness / 100),
        )
        for value, hue, saturation, brightness in points
    ]
    return LinearSegmentedColormap.from_list(name, colors, N=256)


def _idl3_rgb():
    """IDL Red Temperature table underlying AIA 304 and all Hinode/XRT.

    Equivalent to SunPy 8.0.0 ``data/idl_3.csv`` (the original 256 RGB
    integers are obtained by integer truncation of the three linear ramps).
    """
    i = np.arange(256, dtype=np.float64)
    r = np.minimum(255, (i * 255 / 176).astype(np.int64))
    g = np.clip(((i - 120) * 255 / 135).astype(np.int64), 0, 255)
    b = np.clip(((i - 190) * 255 / 65).astype(np.int64), 0, 255)
    return r, g, b


def _instrument_rgb():
    """Reproduce the AIA, EUI and XRT RGB arrays without importing SunPy."""
    r0, g0, b0 = _idl3_rgb()
    c0 = np.arange(256, dtype=np.float32)
    c1 = (np.sqrt(c0) * np.sqrt(np.float32(255.0))).astype(np.float32)
    c2 = (np.arange(256) ** 2 / 255.0).astype(np.float32)
    c3 = ((c1 + c2 / 2.0) * 255.0 / (c1.max() + c2.max() / 2.0)).astype(np.float32)
    return {
        "aia94": (c2, c3, c0),
        "aia131": (g0, r0, r0),
        "aia171": (r0, c0, b0),
        "aia193": (c1, c0, c2),
        "aia211": (c1, c0, c3),
        "aia304": (r0, g0, b0),
        "aia335": (c2, c0, c1),
        "eui174": (r0, c0, b0),  # SunPy: Solar Orbiter HRI 174 = AIA 171
        "xrt": (r0, g0, b0),  # SunPy: all XRT filters = IDL red table
    }


def make_colormaps():
    """Return six user terrain maps and 18 instrument maps (incl. _r)."""
    maps = {name: _terrain(name, points) for name, points in TERRAIN_HSV.items()}
    for name, rgb in _instrument_rgb().items():
        maps[name] = ListedColormap(np.column_stack(rgb) / 255.0, name=name)
    return {
        key: cm
        for name, cmap in maps.items()
        for key, cm in ((name, cmap), (f"{name}_r", cmap.reversed(name=f"{name}_r")))
    }


@lru_cache(maxsize=1)
def register_colormaps():
    """Register the package names with Matplotlib, idempotently and lazily."""
    maps = make_colormaps()
    for name, cmap in maps.items():
        if name not in colormaps:
            colormaps.register(cmap, name=name)
    return maps
