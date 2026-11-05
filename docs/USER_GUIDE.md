# Bifrost Proto — User Guide

This guide covers the main workflows in **Bifrost Proto v0.1.47**. For a shorter introduction, start with the [README](../README.md). For implementation details, see [developer notes](DEVELOPER_NOTES.md).

## 1. Open a simulation

Bifrost Proto reads native Bifrost snapshots using their IDL metadata and binary files. A readable numbered snapshot needs a matching `.idl` and `.snap` file; `.aux` is optional but required for quantities stored only there (such as `qrad` and `qgenrad`).

```python
from bifrost_proto import BifrostData

dd = BifrostData("run_name", fdir="/path/to/simulation")
snap = dd.snaps[0]  # First available snapshot
```

If the directory contains `.idl` files identifying exactly one run through `snapname`, the run name can be detected automatically:

```python
dd = BifrostData(fdir="/path/to/simulation")
print(dd.snaps)  # Snapshots with matching .idl and .snap files
```

If several run names are present, select one explicitly, particularly in scripts or batch jobs. Snapshot availability depends on the files in the run directory; `.aux` is not required for a snapshot to appear in `dd.snaps`.

## 2. Coordinates, selections, and units

```python
# Physical coordinates (Mm by default)
xx = dd.get_x()
yy = dd.get_y()
zz = dd.get_z()

# Read one variable in three different ways
tg = dd.get_var("tg", snap)                                       # Full 3-D cube
bz = dd.get_var("b_z", snap, x=slice(2, 12), y=slice(0, 10), z=2) # XY subdomain
uz = dd.get_var("u_z", snap, x=25.5, centered=True, units="cgs") # YZ plane
```

The physical selectors `x=`, `y=`, and `z=` use coordinates in **Mm**. A scalar selects a position; a `slice(start, stop)` selects a physical interval. `get_var` also supports index selectors `ix=`, `iy=`, `iz=`. Use `method="linear"` for interpolation at physical coordinates instead of the default `"nearest"` sampling. `centered=True` moves a staggered quantity to cell centers before selection.

You may use `uniform_z=True` to resample the selected Z extent to a uniform grid. The native Bifrost Z grid need not be uniform.

Bifrost Proto uses **`units="solar"` by default**, independently of the input file's Bifrost normalization. The available systems are `"solar"`, `"cgs"`, `"si"`, and `"bifrost"` (native code units). Some common `solar` units are:

| Quantity | Example variable | `solar` unit |
| --- | --- | --- |
| Coordinates / characteristic lengths | `get_x()`, `lcurlb` | Mm |
| Temperature | `tg` | K |
| Mass density | `r` | g cm^-3 |
| Velocity | `u_z` | km s^-1 |
| Magnetic field | `b_z` | G |
| Gas pressure | `pg` | dyn cm^-2 |
| Current density | `j_x` | statA cm^-2 |
| Volumetric heating/cooling | `qrad`, `qjoule` | erg cm^-3 s^-1 |
| Heating/cooling per unit mass | `qrador` | erg g^-1 s^-1 |
| Characteristic times | `tauqrad` | s |

In v0.1.47, `ldivb`, `lcurlb`, `ldivu`, and `lcurlu` are **lengths**, not inverse lengths. For an exact label in a chosen system, ask the package rather than hard-coding it:

```python
print(dd.get_units("j_x"))
print(dd.get_units("u_z", units="si"))
print(dd.get_unit_type("qrad"))
```

## 3. Discover variables and diagnostics

The registry contains native quantities, aliases and derived MHD diagnostics. You do not need to memorize the variable names.

```python
print(dd.variables())           # All registered canonical variables
print(dd.variables("pressure")) # Search names, aliases and descriptions
print(dd.info("b_z"))           # Description, location and units
print(dd.info("qrad", snap))    # Also report availability/source for this snapshot
```

Examples of derived families include `modb`, `beta`, `modj`, `theta_jb`, `divu`, `modcurlu`, `modlf`, and the signed energy terms `qjoule`, `qpdv`, `qrad` and `qgenrad`. The latter two are **AUX-only** inputs, not reconstructed from thermodynamic variables. Their specific rates divide by density (`qrador`, `qgenrador`), while characteristic times use `e / abs(q)` (`tauqrad`, `tauqgenrad`). The sign of the heating/cooling terms is preserved.

Availability of a particular native/AUX field varies between runs. Use `dd.info(variable, snap)` to check it.

## 4. Plot cuts, contours and LOS-integrated images

For a 2-D cut, specify **one** fixed physical coordinate. The other two axes can be cropped with slices.

```python
# Apply the default colormap and normalization for temperature
dd.plot("tg", snap, z=2)

# Overlay |B| = 100 G and save a PNG in <simulation>/plots/
dd.plot(
    "tg", snap, z=1.5,
    contours={
        "var": "modb",
        "levels": [1e2],
        "colors": "white",
        "linewidths": 0.7,
    },
    png=True, overwrite=True,
)
```

`png=True` generates an automatic filename. Use `png="my_plot.png"` for a chosen filename. Files are not overwritten by default; `overwrite=True` explicitly permits replacement.

A line-of-sight (LOS) integral instead uses `los="x"`, `"y"` or `"z"`. It integrates over the native grid in physical units. For large simulations, `los_stream=True` reduces the peak memory needed for the integration. A scalar cut and `los` cannot be combined, and contour overlays are not supported on LOS plots.

```python
from matplotlib.colors import LogNorm

# Local synthetic contribution per unit path length
dd.plot("aia193", snap, y=2, norm=LogNorm(clip=True))

# Synthetic LOS-integrated signal
dd.plot("aia193", snap, los="z",
        cmap="aia193", norm=LogNorm(vmin=1, vmax=2e4))
```

Synthetic channels cover SDO/AIA, Hinode/XRT, and Solar Orbiter/EUI. **Local** values are in `DN cm^-1 s^-1 pix^-1` for `solar`/CGS; after LOS integration they are in `DN s^-1 pix^-1`. Therefore, local and integrated images need different color-scale limits. Instrumental colormaps are included without requiring SunPy at runtime. The response tables retain their original source metadata; see [third-party notices](../THIRD_PARTY_NOTICES.md).

## 5. Create MP4 movies

Install FFmpeg (with H.264/libx264 support) and ensure `ffmpeg` is on `PATH`. The default output folder is `<simulation>/movies/`.

```python
# Temporal movie: equally spaced snapshots at a fixed height
dd.movie("tg", range(0, 100, 2), z=2)

# Spatial movie: scan X in one snapshot, using every second grid position
dd.movie("tg", snap, scan="x", every=2)

# Temporal movie of LOS-integrated emission
dd.movie("aia193", range(0, 100, 2), los="z")
```

Scans also support `scan="y"` or `scan="z"`. Temporal movies require equally spaced snapshot numbers. Spatial scanning cannot be combined with LOS integration. Movie rendering keeps the color scale consistent across frames; existing MP4s are not overwritten unless `overwrite=True` is given. Movies can also draw contours on individual cuts.

## 6. Work efficiently with large simulations

Prefer reading **2-D cuts or small physical subdomains** rather than entire 3-D cubes when exploring a simulation. Derived quantities are computed with the required stagger halos before the requested region is returned.

```python
# Share computed intermediates while reading related diagnostics
with dd.cache():
    bz = dd.get_var("b_z", snap, z=2)
    beta = dd.get_var("beta", snap, z=2)

# Optional reduced-memory evaluation for larger derived quantities
field = dd.get_var("modb", snap, z=2, low_memory=True)

# Save an expensive native quantity for reuse in later sessions
# (only when it makes sense for the selected simulation)
dd.savevar("modb", snap)
```

Saved quantities live in `<simulation>/saved/` as an `.npy` array plus JSON metadata. The package checks metadata before reusing saved arrays; source files and saved quantities are not interchangeable. Use `dd.info()` and `dd.get_units()` when in doubt about the physical interpretation of an output.

## Further reading

- [README](../README.md) — installation and quick start
- [Developer notes](DEVELOPER_NOTES.md) — implementation details and conventions
- [Changelog](../CHANGELOG.md) — release history and compatibility changes
- [Citation](../CITATION.cff), [License](../LICENSE), [Third-party notices](../THIRD_PARTY_NOTICES.md)
