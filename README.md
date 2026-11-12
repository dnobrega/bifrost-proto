# Bifrost Proto

**A Python toolkit for reading, analyzing, and visualizing Bifrost radiative-MHD simulations.**

Bifrost Proto provides efficient access to native and derived variables, staggered-grid operations, physical unit conversions, synthetic solar observations, 2D plots, and MP4 animations.

## License and citation

Bifrost Proto is released under the **[BSD 3-Clause License](LICENSE)**.

If you use Bifrost Proto in scientific research, please cite the repository. Citation information is available in [`CITATION.cff`](CITATION.cff). Third-party attribution and licensing details are in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/dnobrega/bifrost-proto.git
```

Or clone the repository for development:

```bash
git clone https://github.com/dnobrega/bifrost-proto.git
cd bifrost-proto
pip install -e .
```

## Quick start

### Read simulation data

```python
from bifrost_proto import BifrostData

dd = BifrostData("run_name", fdir="/path/to/simulation")
snap = dd.snaps[0]  # First available snapshot

# Coordinates in Mm
xx = dd.get_x()
yy = dd.get_y()
zz = dd.get_z()

# Read physical quantities
tg = dd.get_var("tg", snap)  # Full 3D cube
bz = dd.get_var("b_z", snap, x=slice(2, 12), y=slice(0, 10), z=2)  # 2D subdomain
uz = dd.get_var("u_z", snap, x=25.5, centered=True, units="cgs")  # Centered YZ plane
```

Spatial coordinates use Mm. By default, variables are returned in **`solar` units**; `bifrost`, `cgs`, and `si` are also supported.

### Find variables and inspect units

```python
# Get variable information and physical units
print(dd.info("b_z"))
print("----")
print(dd.get_units("j_x"))

# List registered variables
print(dd.variables())
print("----")

# Search names, aliases, and descriptions (e.g., pressure)
print(dd.variables("pressure"))
```

### Plot simulation quantities

```python
# Default colormap and normalization
dd.plot("tg", snap, z=2)

# Overlay a |B| = 100 G contour and save the PNG
dd.plot(
    "tg",
    snap,
    z=1.5,
    contours={
        "var": "modb",
        "levels": [1e2],
        "colors": "white",
        "linewidths": 0.7,
    },
    png=True,
    overwrite=True,
)
```

With `png=True`, plots are saved under `<simulation>/plots/`. `overwrite=True` allows replacing an existing PNG; omit it to protect existing files. You can also provide a filename, e.g. `png="tg_contours.png"`.

### Synthetic EUV and X-ray observations

Synthetic channels are available for SDO/AIA, Solar Orbiter/EUI, and Hinode/XRT.

```python
from matplotlib.colors import LogNorm

# Local emissivity on an XZ slice; automatically scale positive values
dd.plot("aia193", snap, y=2, norm=LogNorm(clip=True))

# LOS-integrated intensity, with an explicit color scale
dd.plot(
    "aia193",
    snap,
    los="z",
    cmap="aia193",
    norm=LogNorm(vmin=1, vmax=2e4),
)
```

Local emissivity and LOS-integrated intensity have different units and visualization scales.

### Create movies

```python
# Temporal evolution at a fixed height (equally spaced snapshots)
dd.movie("tg", range(0, 100, 2), z=2)

# Spatial scan along X in a single snapshot (successive YZ slices)
dd.movie("tg", snap, scan="x", every=2)
```

For spatial scans, `scan="x"` moves through successive YZ planes at a fixed snapshot; `every=2` uses every second X position. Scans also support `"y"` and `"z"`. Movies use consistent color scales and are saved under `<simulation>/movies/` by default. MP4 encoding requires FFmpeg.

## Documentation

See [`docs/`](docs/) for further details on variables, staggered grids, diagnostics, EOS, and file formats, and [`CHANGELOG.md`](CHANGELOG.md) for changes between versions.

## Acknowledgements

This research has been supported by the European Research Council through the
Synergy Grant number 810218 ("The Whole Sun", ERC-2018-SyG); by the Centre
National d'Etudes Spatiales (CNES) Solar Orbiter project; by the Research
Council of Norway through its Centres of Excellence scheme, project number
262622; and by grant CEX2025-001609-S, awarded to the Instituto de Astrofísica
de Canarias under the Severo Ochoa Centre of Excellence program and funded by
MICIU/AEI/10.13039/501100011033.
