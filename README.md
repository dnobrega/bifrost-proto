# bifrost-proto v0.1.47

Compact Bifrost-only analysis prototype with exact legacy stagger arithmetic,
partial-domain halo evaluation, EOS fallbacks, unit conversion, and a growing
registry of commonly used derived quantities.

Implementation conventions that must remain stable when extending the package
are collected in [`docs/DEVELOPER_NOTES.md`](docs/DEVELOPER_NOTES.md).

## License and scientific citation

Bifrost Proto is released under the [BSD 3-Clause License](LICENSE): you may
use, modify, and redistribute it as long as you retain the legally required
copyright and license notices. Please **cite Bifrost Proto in scientific
publications** that use this software. The exact release citation is in
[`CITATION.cff`](CITATION.cff); after the repository is published, GitHub's
**Cite this repository** menu can generate an APA or BibTeX reference.

A GitHub repository URL and Zenodo DOI have not been filled into `CITATION.cff`
because they have not yet been provided. Once published, add `repository-code`
and optionally the Zenodo DOI there; no fictional URL is used.

Third-party source and copyright attributions are in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), including SunPy colormaps.
The permissive BSD license does not *legally* require a bibliography citation
from someone who only uses the library; the requested scientific citation is
an academic attribution request.

## Numerical reproducibility

EOS interpolation and `gamma1` evaluate intermediate values in `float64`
before returning `float32` arrays. This makes Numba JIT and pure-Python
(`NUMBA_DISABLE_JIT=1`) paths numerically consistent with NumPy 2, without
changing the stored EOS tables or final output type. Independent precision
regressions are in `tests/test_v147_eos_precision.py`.

## Release versioning

The authoritative version is `bifrost_proto/version.py`. Setuptools reads it
automatically to generate package metadata, so no version number needs to
be duplicated in `pyproject.toml` or in tests. When releasing, update the
README and CHANGELOG descriptions separately.

## v0.1.47 — Radiative heating diagnostics and publication metadata

- Register native **AUX-only** `qrad` and `qgenrad`, both signed volumetric
  radiative heating/cooling rates in `erg cm^-3 s^-1` (`units="solar"`). They
  retain the sign and are unavailable if the simulation did not save them in
  the AUX variable list. No approximate or reconstructed fallback is used.
- Add `qrador = qrad / r` and `qgenrador = qgenrad / r` (`erg g^-1 s^-1`), plus
  `tauqrad = e / abs(qrad)` and `tauqgenrad = e / abs(qgenrad)` (seconds). As in
  other heating families, a zero source yields an infinite timescale.
- Plot presets match existing `qspitz`, `qspitzor` and `tauqspitz` settings:
  signed `SymLogNorm` with `RdBu_r` for source rates, `LogNorm` with
  `red_terrain_r` for timescales.
- Add `LICENSE` (BSD 3-Clause), `CITATION.cff`, and third-party notices.

**Radiation-module distinction:** `qrad` comes from the full radiative-transfer
solver, while `qgenrad` is the output of the configurable generalized
radiative-heating/cooling module (e.g. approximate chromospheric radiation,
coronal absorption and other recipes). They are *not interchangeable*.
`qgenrad` can overlap the processes represented by other radiation terms,
including `qthin`, depending on the modules enabled for your run. Do not
blindly add them without inspecting the model's configuration.

## v0.1.46 — Velocity palettes, characteristic lengths and safe MP4 cleanup

- Use `bwr_r` for signed velocity vector components `u_x/y/z`, `uparb_x/y/z`,
  and `uperpb_x/y/z`. Blue represents positive values (upflows for `u_z`).
- **Breaking change:** `ldivb`, `lcurlb`, `ldivu`, and `lcurlu` are now
  characteristic **lengths** (`|field| / |gradient|`), not inverse lengths.
  Their solar/CGS/SI units are Mm/cm/m; all labels and metadata reflect this.
  Zero denominator with positive magnitude produces infinity; `0/0` is NaN.
- Preserve completed MP4 temporary encodings on interruptions during the
  publication stage. A cleanup failure of the already-published MP4's temp
  logs a warning instead of making `movie()` appear to fail, including under
  `-W error`.

## v0.1.45 — Robust movie publication, centered LOS and Matplotlib compatibility

- MP4 files now follow normal file creation permissions (e.g., `0644` under
  `umask 022`, `0600` under `umask 077`). The secure temporary is created
  exclusively in the final directory, and retains its mode on publication.
- `overwrite=False` still uses atomic hardlinks when available. exFAT, SMB,
  FUSE, and other filesystems rejecting hardlinks use an exclusive-create
  copy fallback. The fallback never replaces an existing destination, but
  **the destination may become visible during copying**. If publication
  fails, the complete encoded `.partial.mp4` is preserved and its path is
  included in the error. Encoding/read failures still clean up temporaries.
- `dd.plot(..., los=...)` uses a data-driven `CenteredNorm(vcenter=0)` for
  signed variables whose *preset* uses a diverging color map (for example
  `u_z` and `j_z`). This guarantees the neutral color is at zero; a movie
  retains the scale established by its first frame. Explicit `norm=` is
  always respected. This does not modify the user-selected gray-scale
  `b_z` preset, or positive quantities that use a diverging map.
- `b_h` now defaults to `LogNorm(1, 1000)` with `jet`, matching `modb`.
- Instrument maps no longer use the deprecated `ListedColormap(N=256)`
  argument. The color arrays still contain exactly 256 samples.


## Automatic simulation detection (v0.1.40)

Omitting the run root makes `BifrostData` look for `snapname` in the `*.idl`
files of the simulation directory (non-recursively). If every valid file
contains the same name, it is used automatically; with multiple names, a
numbered interactive menu shows each name and how many files contain it.

```python
from bifrost_proto import BifrostData

dd = BifrostData(folder="/path/to/simulation")
# Equivalent:
dd = BifrostData(fdir="/path/to/simulation")

# Known run: skip directory scanning and interactive selection.
dd = BifrostData("run_name", fdir="/path/to/simulation")
dd = BifrostData(folder="/path/to/simulation", idlparam="run_name")
```

If several run names are found but interactive input is unavailable, the
constructor raises an informative error listing the alternatives. Files
without a valid `snapname` are ignored; if none is found, an explicit run
root is required. Detection occurs only during object initialization.

The read-only ``dd.snaps`` property lists sorted, nonnegative snapshot numbers
having **both** a matching ``<snapname>_NNN.idl`` **and** a
``<snapname>_NNN.snap`` file (or the legacy unnumbered pair for snapshot zero).
An ``.aux`` file is not required. The property does not descend into
subfolders, ignores restart files and rescans whenever accessed:

```python
print(dd.snaps)                       # e.g. [0, 2, 4, 6, 8]
dd.movie("tg", dd.snaps, y=25.5)      # if the sequence is equispaced
```

As before, temporal movies require an equispaced integer sequence; ``dd.snaps``
may contain gaps, in which case choose an equispaced subset.

## Install for development

```bash
python -m pip install -e .
```

## Quick-look 2-D plotting

``BifrostData.plot`` draws a single XY, XZ or YZ physical cut using the
registered mesh location and partial-domain ``get_var`` reads. Exactly one
coordinate must be scalar (the fixed plane). Optional physical ``slice``
selections on the other axes crop the calculation, rather than loading a
whole plane and merely changing the display limits. All spatial values
are in Mm, with automatic uniform-Z remeshing for XZ/YZ cuts.

```python
from matplotlib.colors import LogNorm, TwoSlopeNorm

# One-line quick look (creates its own figure, axes and colorbar).
fig, ax, im = dd.plot("tg", snap, y=25.5, cmap="magma", norm=LogNorm())

# Crop the physically evaluated region and overlay another variable.
fig, ax, im = dd.plot(
    "tg", snap, y=25.5, x=slice(5, 20), z=slice(0, 10),
    cmap="turbo", norm=LogNorm(vmin=1e3, vmax=2e6),
    contours={"var": "modb", "levels": [100, 300, 500],
              "colors": "white", "linewidths": 0.8},
)

# Existing Matplotlib figures are supported without changing quick looks.
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
dd.plot("tg", snap, y=25.5, ax=axes[0])
dd.plot("modb", snap, y=25.5, ax=axes[1])

# Weak resolver cache only when explicitly needed.
dd.plot("uperpb", snap, y=25.5, low_memory=True)
```

With ``ax=None``, a figure is created automatically, as before; providing
``ax=`` embeds the plot in a pre-existing Matplotlib figure. In both modes the
return value is ``(fig, ax, im)``. The ``contours`` mapping requires ``var``
and forwards Matplotlib-specific entries to ``Axes.contour``. By default,
contours over a *cell-centered image* are interpolated to cell centers
by default, preventing a spurious half-cell visual offset. A staggered
background retains native contour coordinates by default. To select
explicitly, use ``centered=True`` or ``centered=False`` in the mapping:


```python
dd.plot("tg", 536, y=25.5, contours={
    "var": "bx", "centered": True, "levels": [-100, 0, 100],
    "colors": "white",
})
```

This computes the contour variable on the center grid *before* the final
physical selection. It is not a cosmetic half-pixel shift. For physical
native-grid contours over a centered image, use ``centered=False``. Scalar cut coordinates use nearest-grid
sampling by default.

### PNG output and LOS integration (v0.1.39)

``dd.plot`` can save high-resolution PNG images in the simulation's
``plots/`` directory (the same convention as movies in ``movies/``):

```python
# <simulation>/plots/tg_0230_XZ_y=25.50_x=5.00-20.00.png (300 DPI)
dd.plot("tg", 230, y=25.5, x=slice(5, 20), png=True)

# Bare names go under <simulation>/plots. Relative paths containing folders
# are relative to <simulation>; absolute paths are respected.
dd.plot("r", 230, z=1.5, png="density.png", dpi=200)

# Existing outputs are protected unless overwrite=True is requested.
dd.plot("r", 230, z=1.5, png="density.png", overwrite=True)
```

``los="x"``, ``los="y"`` or ``los="z"`` integrates a registered 3-D
quantity along the chosen physical axis, producing respectively a YZ, XZ or
XY map. All physical-coordinate arguments may be ``None`` or ``slice``; a
scalar coordinate is not allowed with ``los``. The LOS-axis slice limits the
integration domain, while the other slices crop the displayed region:

```python
# Column density: integrate rho along Y, display an XZ map of x=5..20 Mm.
dd.plot("r", 230, los="y", x=slice(5, 20), png=True)

# Partial Z integral, displayed as an XY map.
dd.plot("r", 230, los="z", z=slice(0, 10), png=True)
```

On the LOS axis, center-grid variables use the local center-cell widths
(``get_dx1d``/``get_dy1d``/``get_dz1d``). On a **complete periodic X or Y**
axis, a variable staggered to faces along the LOS uses a full-period sum with
all N cell widths (equivalent to trapezoids including the periodic closing
interval). For nonperiodic axes (including Z), or cropped face-axis LOS
selections, it uses trapezoids between the selected face coordinates. The
reduction happens on the **original nonuniform Z mesh**;
vertical 2-D maps are interpolated uniformly in Z **after** reduction.
NumPy ``einsum`` accumulates into float64 without allocating a weighted
3-D temporary array. By default, ``dd.plot(los=...)`` reads the selected 3-D
field; ``dd.plot(los=..., los_stream=True)`` instead integrates small
transverse 3-D slabs that span the entire LOS, accumulating float64 maps
without loading the full cube. The slab size adapts to a target of about
64 MiB per native input array, capped at 64 transverse samples (this is
not a strict bound on intermediate memory for derived quantities). This
is also the memory-saving integration mode used by LOS movies.

Spatial plots continue to use Mm on their axes. LOS numerical lengths and
colorbar units instead use cm (solar/CGS), m (SI) and Mm (native Bifrost).
Solar units involving explicit Mm powers are converted algebraically to cm
(including the numerical scaling) before simplifying. Thus ``r`` becomes
``$\int \rho\,\mathrm{d}y$`` in ``g cm^-2``, with no mismatch between
pixel values and the displayed units. LOS plots remain incompatible
with the ``contours`` argument; overlays can be added manually using ``ax``.
``dd.movie`` can animate LOS integrals in time with the same quadrature as
``dd.plot`` and its slab-wise accumulation. Movies preflight `.idl` and `.snap`
files (including binary sizes) and publish a complete MP4 atomically; failed
encodes leave no obstructing partial file.

Contour overlays accept their own physical unit system without changing the
background image units, e.g. ``contours={"var": "uz", "units": "si",
"levels": [10000, 30000], "colors": "white"}``. For custom axes created
with ``layout="constrained"``, colorbars use the figure-managed layout so
that their labels fit within the figure bounds.

The interactive ``PlaneViewer`` and slider/sweep functionality have been
removed in v0.1.38. Use ``dd.movie`` for temporal or spatial animations,
or request a scalar cut with ``dd.plot``. A lone ``z=slice(...)`` is not a valid plot:
provide exactly one scalar fixed axis, e.g. ``y=25.5, z=slice(0, 10)``.

The plotting layer uses ``get_grid(location=...)`` rather than passing a
``location`` argument to ``get_x``, ``get_y`` or ``get_z``.

## Snapshot time and interactive stagger

Snapshot time is read from the Bifrost IDL ``tsnap`` parameter:

```python
t = dd.get_t(snap)                    # solar units: seconds
t_cgs = dd.get_t(snap, units="cgs")   # seconds
t_si = dd.get_t(snap, units="si")     # seconds
t_br = dd.get_t(snap, units="bifrost") # native tsnap
```

The public ``dd.stagger.*`` helpers operate on full-domain 3-D arrays only.
They preserve F-contiguous layout in both physical and native orientations and
use the same parallel Y/Z orientation transform as ``get_var``. For cropped
subdomains, request the centered/derived quantity through ``get_var`` so the
resolver can expand the required stagger halo and retain the correct global
periodic/nonuniform geometry.

## Composition and synthetic instrument responses

The elemental mixture is read directly from the run's `tabparam.in`:

```python
print(dd.elements)
print(dd.abundances)      # logarithmic abundances, H=12
print(dd.atomic_weights)  # amu
print(dd.mass_fractions)
print(dd.grams_per_hydrogen)
```

The derived center quantity `nh` is the hydrogen-nucleus number density in
`cm^-3`, computed from `r` and the same `abund`/`aweight` mixture.

Synthetic local instrument signals are registered under the historical compact
IDL names:

```text
AIA: aia94 aia131 aia171 aia193 aia211 aia304 aia335
XRT: xrt_almed xrt_almesh xrt_alpoly xrt_althick xrt_bemed
     xrt_bethick xrt_bethin xrt_cpoly xrt_tipoly
EUI: eui174
```

All channels use one runtime definition:

```text
local signal = ne * nh * G(T, ne)
```

and return `DN cm^-1 s^-1 pix^-1` in solar/CGS units. These are local
contributions per unit path length, not line-of-sight integrated images. All
AIA, XRT, and EUI emissivities use the same explicit coronal validity mask:
values are set to exactly zero where `T < 1e4 K` or where the physical
cell-center height is `z < 0 Mm`. The mask is applied on the selected native
domain before public reorientation/resampling, so full cubes and subdomains
share identical semantics.

Response resources live under:

```text
bifrost_proto/resources/instrument_responses/<name>/
    table.npy
    logt.npy
    logne.npy
    metadata.json
```

The canonical table shape is `(n_logne, n_logt)`. `metadata.json` records the
abundance set, ionization-equilibrium provenance, source file/hash, fill value,
and normalization. Runtime interpolation is implemented inside
`bifrost_proto` and requires no IDL, SolarSoft, CHIANTI, Astropy, or SunPy.

A custom response root can be supplied when constructing a reader:

```python
dd = BifrostData(root, fdir=folder, responses_dir="/scratch/my_responses")
```

and the provenance of an installed response can be inspected with:

```python
dd.response_info("aia193")
dd.info("aia193")
```

If the four response files are absent, `info()` reports
`response_table_available=False`; evaluation raises a clear `FileNotFoundError`
rather than silently substituting a different response.

Legacy tables are converted once with the scripts under
`tools/build_instrument_responses/`. `convert_all.py` converts the complete
seven-channel AIA and nine-filter XRT sets in one pass, with a preflight check
for all expected source files; passing `--nh-over-ne` also converts EUI/HRI 174.
AIA and XRT `.sav` files are read with SciPy. EUI/HRI 174 uses its legacy FITS
plus the exact `proton_dens(T)` helper array used by the IDL recipe; Astropy is
needed only for that conversion tool, not at package runtime. Response-axis
origins and spacings always come from the source axes/header and are never
hardcoded.


## EOS interpolation

The Bifrost rho-e EOS tables are uniform in `ln(rho)` and `ln(ee)`.  Runtime
interpolation of `tg_eos`, `ne_eos`, `pg_eos`, and `kr_eos` therefore uses a
fused parallel bilinear Numba kernel instead of the former general-purpose
`scipy.ndimage.map_coordinates()` path.  Inputs outside either EOS axis raise a clear `ValueError` instead of
silently returning the boundary values. Both axes are validated before
calling the Numba kernel; `gamma1` uses a fused one-output Numba kernel.

C- and Fortran-contiguous 3-D arrays use copy-free 1-D memory views; strided
3-D views are evaluated directly without contiguizing the inputs. Unusual
broadcast/lower-dimensional inputs use a vectorized NumPy fallback. The former
`EOSTable.interpolate_scipy()` runtime method has been removed; the SciPy
definition is retained only inside validation tests/benchmarks.

An isolated benchmark is available with:

```bash
python benchmarks/benchmark_eos_interpolation.py scipy --n 256 --order F --output ne
python benchmarks/benchmark_eos_interpolation.py numba --n 256 --order F --output ne
```

## Canonical vector naming

Vector components now follow one rule everywhere:

```text
<quantity>_x
<quantity>_y
<quantity>_z
```

The historical compact Bifrost/IDL names remain aliases, so old analysis code
continues to work:

```python
bx = dd.get_var("bx", snap)      # alias of b_x
ux = dd.get_var("ux", snap)      # alias of u_x
jx = dd.get_var("jx", snap)      # alias of j_x
```

The canonical/raw mapping is explicit in the registry.  For example `b_x` is
the public quantity while `bx` is its Bifrost on-disk storage name.  Aliases
resolve to the same cache node rather than duplicating a calculation.

## Mesh and native Bifrost orientation

One location vocabulary is used everywhere:

`center`, `xface`, `yface`, `zface`, `xedge`, `yedge`, `zedge`.

```python
x = dd.get_x()
y = dd.get_y()
z = dd.get_z()

xf = dd.get_xf()
yf = dd.get_yf()
zf = dd.get_zf()

# Physical slices use Mm, exactly as in get_var().
z_sub = dd.get_z(z=slice(2.5, 10.0))
zf_sub = dd.get_zf(z=slice(2.5, 10.0))
dz_sub = dd.get_dz1d(z=slice(2.5, 10.0))

# Public and native index selectors follow get_var() semantics.
x_sub = dd.get_x(ix=slice(10, 30))
z_native_index = dd.get_z(iz_br=20)

# Match get_var(..., uniform_z=True) on the same selected Z interval.
z_uniform = dd.get_z(z=slice(2.5, 10.0), uniform_z=True)
dz_uniform = dd.get_dz1d(z=slice(2.5, 10.0), uniform_z=True)

# Coordinates use the standard length-unit system. Selectors still use Mm.
z_cgs = dd.get_z(z=slice(2.5, 10.0), units="cgs")
z_si = dd.get_z(z=slice(2.5, 10.0), units="si")

# Full topological locations and complete 3-D selections are available through get_grid().
xg, yg, zg = dd.get_grid(
    location="xedge", ix=slice(10, 30), z=slice(2.5, 10.0), uniform_z=True
)

bx_br = dd.get_var("b_x", snap, units="bifrost", br=True)
y_br = dd.get_y(br=True)
```

Scalar coordinate/index selections return a scalar coordinate on that axis;
slices return a 1-D coordinate array. The public physical Y coordinate starts
at zero; `br=True` keeps the original Bifrost coordinates, orientation, and
component signs untouched. The spacing helpers `get_dx/get_dy/get_dz` and
`get_dx1d/get_dy1d/get_dz1d` are defined on the cell-center mesh and therefore
no longer expose a `location=` argument.

For scalar physical-coordinate requests, `method="nearest"` picks the nearest
sample on the variable's own center/face grid. At a staggered boundary, it also
accepts the adjacent cell-center coordinate (for example, `get_var("uz", snap,
z=-6)` when the first Z face lies slightly above that height). It does not
accept coordinates arbitrarily outside the center/face extent. `method="linear"`
never extrapolates outside the selected variable's sample coordinates. Physical
coordinate `slice(...)` requests do not use either sampling method.

## Public API naming and native writers

Canonical public getters use Python-style underscores:

```python
dd.get_var("bz", snap, center=True)
dd.get_x()
dd.get_xf()
dd.get_t(snap)
dd.get_var_name("bz", latex=True)
```

Only the underscored `get_*` spellings are part of the public API. This keeps
interactive tab completion compact and unambiguous. `center=True` remains an
explicit alias for `centered=True`; arbitrary keyword abbreviations are not
accepted.

Native Bifrost binary files can be written without constructing a 4-D
temporary array:

```python
dd.write_snap(filename, r, px, py, pz, e, bx, by, bz)
dd.write_aux(filename_aux, p, tg, extra=[("qjoule", qjoule)])
```

Writer inputs must already be full-domain arrays in native Bifrost orientation
and Bifrost units. `write_snap` fixes the required block order to
`r, px, py, pz, e, bx, by, bz`. `write_aux` always writes gas pressure `p`
first and temperature `tg` second; any additional AUX quantities are supplied
as an ordered list of `(name, array)` pairs, and the matching IDL `aux` list
must use the same order. If the destination already exists, the writer
raises `FileExistsError` unless `overwrite=True` is explicitly supplied.
Replacing an existing file emits a warning but never prompts interactively.
The writer first creates a temporary file in the destination directory and
atomically replaces the target only after all native blocks have been written.

Snapshot zero follows the numbered Bifrost convention (`_000.idl`, `_000.snap`,
`_000.aux`); legacy unnumbered snapshot-zero files remain readable as a
fallback.

## Stagger

```python
xup = dd.stagger.xup
yup = dd.stagger.yup
zup = dd.stagger.zup

xdn = dd.stagger.xdn
ydn = dd.stagger.ydn
zdn = dd.stagger.zdn
```

X/Y retain the exact grouped float32 arithmetic of the legacy IDL/C operators;
Z uses the geometry-aware non-uniform coefficients.  Derived quantities use the
same engine with a five-cell computational halo.

## Sampling and output-finalization performance

`uniform_z=True` keeps the selected native Z extent and number of points but
interpolates directly into one output array with a fused parallel Numba kernel.
The implementation is layout-aware for the F-order physical/native arrays returned
by the API; C-order inputs produced by unusual post-processing remain supported.
Independent outer iterations are distributed with `prange` while the contiguous
inner axis is preserved. It avoids the two complete `np.take()` arrays and float64
broadcast temporaries required by the former vectorized path. A standalone comparison is available with:

```bash
python benchmarks/benchmark_uniform_z.py --n 256 --order C
python benchmarks/benchmark_uniform_z.py --n 256 --order F
```

The v0.1.19 algebra audit also includes:

```bash
python benchmarks/benchmark_algebra_v119.py --n 192
```

The default physical orientation is materialized directly as an F-contiguous
array with a parallel Numba copy. Native Y/Z reversal and the sign change of Y/Z
vector components are fused into that single pass. This avoids the former
negative-stride-view -> C-contiguous conversion, which was extremely expensive
for multi-GiB cubes. Native `br=True` output remains F-contiguous as well.

The final `get_var()` unit conversion also skips exact factor-one conversions
(e.g. temperature, number density, dimensionless quantities, and instrument
emissivities in solar units). For the normal physical orientation, nontrivial
unit factors are applied in-place to the already-detached public output buffer;
resolver/cache arrays are never modified.

`nh` likewise uses one combined density-to-number-density factor, avoiding an
intermediate full 3-D CGS density array.

## Numba parallel execution

The large element-wise kernels that have no cross-cell dependencies use
`parallel=True`/`prange`: uniform-Z interpolation, vector magnitudes, fused dot
and cross algebra, `theta_jb`, `tauq*`, Poynting-flux algebra, rho-e EOS
interpolation, and the fused instrument-response interpolation. `fastmath` is
deliberately not enabled; the parallel versions preserve the per-cell arithmetic
order used by the validated serial kernels.

Pointwise quantity kernels use a flat `prange(N)` path when all participating
arrays share a contiguous C or Fortran layout. Arbitrary 3-D strided views use
`prange(ny*nz)` over collapsed Y-Z rows while retaining X as the innermost loop.
The stagger kernels use the same collapsed Y-Z strategy. This avoids the old
`prange(nz)` bottleneck for horizontal cuts and thin Z slabs (`nz=1`, `2`, ...)
without copying strided inputs or changing the validated stagger arithmetic.
A small reproducible benchmark is provided in
`benchmarks/benchmark_thin_slab_parallel_v132.py`.

`bifrost-proto` never fixes the Numba thread count. Numba therefore follows the
process/environment configuration. On shared/HPC nodes it can be limited, for
example, with:

```bash
export NUMBA_NUM_THREADS=8
```

This is especially important when several Python or MPI processes run on the
same node, to avoid oversubscribing CPU cores.

## Instrument-response performance

Synthetic AIA, XRT, and EUI variables still represent the local contribution

```text
ne * nH * G(T, ne)  [DN cm^-1 s^-1 pix^-1]
```

but canonical response tables are now evaluated with a fused parallel Numba
kernel. The bilinear interpolation in `log10(T)`/`log10(ne)` and the final
`ne*nH` multiplication happen in one pass, so the runtime no longer creates
full-cube arrays for logarithms, masks, interpolation indices, table corners, or
the intermediate response `G`. Resolver/cache inputs are never modified in
place. C/F-contiguous cubes use copy-free flat views, while arbitrary 3-D
strided views are evaluated directly without contiguizing the inputs.

`InstrumentResponseTable.evaluate()` remains available as the fully general
NumPy interpolation API and as a fallback for genuinely non-uniform response
axes. Uniformity detection explicitly tolerates the small interval jitter
introduced when legacy uniformly spaced axes were stored as `float32`; the
actual axis origin and legacy first-interval spacing are still read from each
table. The public Bifrost recipes (`aia193`, `xrt_alpoly`, `eui174`, etc.) use
the fused `evaluate_emissivity()` path for these canonical tables.

A standalone benchmark compares both paths:

```bash
python benchmarks/benchmark_instrument_response.py numpy --n 256 --order F
python benchmarks/benchmark_instrument_response.py numba --n 256 --order F
```

The first Numba call in a fresh environment may include JIT compilation; the
benchmark warms the kernel before timing.

## Vector helpers and memory behavior

Common vector diagnostics are registered declaratively from their explicit
components, for example:

```python
_register_magnitude(
    "modb",
    ("b_x", "b_y", "b_z"),
    "magnetic_field",
    "Magnitude of the cell-centered magnetic field, |B|.",
)
```

The helper reads each component's registered stagger location and moves it to
cell centers through the shared grid-geometry map. Magnitudes use fused parallel
Numba kernels with one owned output buffer, so resolver/cache arrays are never
modified in place and squared temporary cubes are avoided. C- and
Fortran-contiguous inputs use copy-free 1-D memory views; arbitrary 3-D strided
views are evaluated directly by stride-aware kernels rather than being copied to
contiguous storage. Divergence is intentionally restricted to face vectors. Curl
validates topology and uses the compatible mapping:

```text
face vector -> curl on edges, using dd?dn derivatives
edge vector -> curl on faces, using dd?up derivatives
```

`benchmarks/benchmark_magnitude_memory.py` compares direct NumPy, buffered
`np.hypot`, and the exact fused Numba implementation used by the package on
large synthetic cubes. Use `--order F` for the layout most representative of
Bifrost arrays; `--stride 2` (or larger) exercises the non-contiguous
stride-aware kernel and makes hidden-copy regressions visible in peak RSS.

## Magnetic quantities

```text
modb        |B| at cell centers
b_h         horizontal |B| at cell centers (alias: bh)
bhat_x/y/z  components of B/|B| on their native faces
            legacy aliases: bxob/byob/bzob

divb        div(B), center
curlb_x/y/z curl(B) components, edges
modcurlb    |curl(B)|, center
ldivb       |B| / |div(B)| [characteristic length]
lcurlb      |B| / |curl(B)| [characteristic length]
beta        pg / pb (equivalent to 2 pg / |B|^2 in Bifrost units)
```

The four `ldiv*`/`lcurl*` diagnostics are **lengths**, not inverse
lengths. `ldivb` is associated with numerical solenoidality errors; it is not
a physical magnetic-gradient scale. Very small denominators can make lengths
huge even for smooth fields. The default color limits for these four variables
remain provisional until validated with simulation plots.

The historical face definition is preserved exactly, e.g.

```text
bhat_x = b_x / xdn(modb)
```

## Velocity quantities

```text
modu             |u|, center
u_h              horizontal |u|, center (alias: uh)
uparb            (u.B)/|B|, signed scalar at center
uparb_x/y/z      parallel-velocity vector components on faces
uperpb_x/y/z     perpendicular-velocity vector components on faces
uperpb           |u_perp,B|, center

divu             div(u), center
curlu_x/y/z      curl(u) components, edges
modcurlu          |curl(u)|, center
lcurlu            |u| / |curl(u)| [characteristic length]
ldivu             |u| / |div(u)| [characteristic length]
```

`uparb` preserves the legacy IDL staggering order:

```text
[xup(u_x*b_x) + yup(u_y*b_y) + zup(u_z*b_z)] / modb
```

The product is formed on the halo-expanded face domain *before* applying the
`up` operator.  It is therefore not replaced by `xup(u_x)*xup(b_x)`.

## Current and electric field

`j_x/y/z` are the current-density components; legacy aliases `jx/jy/jz` remain.
Numerically their native values are `curl(B)`, while their physical conversion
uses the Bifrost current-density factor `u_i`.

Total `e_x/y/z` (`ex/ey/ez`) are stored-only because the non-ideal/hyperdiffusive
part cannot generally be reconstructed from a snapshot.  The ideal field is
separate:

```text
e_ideal_x/y/z = -u x B
legacy aliases: ex_ideal/ey_ideal/ez_ideal
```

When the stored total electric field is available, the following diagnostics are
also available:

```text
mode              |E|, center
eparb             signed E.B/|B|, center
eparb_x/y/z       vector E_parallel components on x/y/z edges
eperpb_x/y/z      vector E_perp components on x/y/z edges
eperpb            |E_perp,B|, center
```

The parallel vector is formed at cell centers and moved back to the native edge
locations; the perpendicular vector is then `E - E_parallel`.  These total-field
diagnostics intentionally fail when `ex/ey/ez` are unavailable.

Current magnitude and the acute J-B alignment angle are also registered:

```text
modj       |J|, center
theta_jb   arccos(|J.B|/(|J||B|)), center
```

`theta_jb` is stored internally in radians, returned in degrees in the default
`solar` unit system, clipped safely before `acos`, and is `NaN` where `|J||B|=0`.

## Pressure-gradient and Lorentz-force diagnostics

The pressure-gradient diagnostics use the physical force sign, `-grad(pg)`,
so their signed components can be compared directly with the Lorentz force.
In the public physical frame, for example, positive `gradpg_z` points upward.

```text
gradpg_x/y/z      -grad(pg) force-density components on faces
gradpg_h          horizontal |-grad(pg)|, center
modgradpg         |-grad(pg)|, center

gradpgor_x/y/z   -grad(pg)/r acceleration components on faces
gradpgor_h        horizontal |-grad(pg)/r|, center
modgradpgor       |-grad(pg)/r|, center
```

The Lorentz force follows the legacy Bifrost/IDL stagger path exactly rather
than forming a separately centered `J x B`:

```text
lf_x/y/z          J x B components on faces
lf_h              horizontal |J x B|, center
modlf             |J x B|, center

lfor_x/y/z        (J x B)/r components on faces
lfor_h            horizontal |(J x B)/r|, center
modlfor           |(J x B)/r|, center
```

`gradpg*` and `lf*` use force-density units (`dyn cm^-3` in solar/CGS display),
while the `*or`/`lfor*` families are accelerations (`km s^-2` in solar units).

## Energy diagnostics

Energy reservoirs at cell centers:

```text
eb   magnetic energy density = 0.5 * modb^2
ek   kinetic energy density  = 0.5 * r * modu^2
```

The volumetric source terms recorded by Bifrost in the AUX file include:

```text
qjoule    Joule dissipation
qvisc     Viscous heating
qthin     Optically thin radiation
qspitz    Spitzer thermal conduction
qrad      Full radiative-transfer heating/cooling (if saved)
qgenrad   Generalized radiation module heating/cooling (if saved)
```

Each requires an actual AUX field with that name; no reconstruction fallback is
provided. The suffix `or` means *over density* and the prefixed `tau` denotes
the internal-energy timescale:

```text
<qname>or  = <qname> / r
 tau<qname> = e / abs(<qname>)
```

These definitions apply to all six stored heating terms (e.g. `qrador`,
`qgenrador`, `tauqrad`, `tauqgenrad`). `q/r` retains its original sign;
`e/abs(q)` is nonnegative. Exactly zero source leads to `+inf`, not an
arbitrary epsilon. `qrad` and `qgenrad` are *distinct* source modules;
inspect the run configuration before combining radiation terms.

Compression/expansion work is derived from the gas pressure and velocity
divergence:

```text
qpdv    = -pg * divu
qpdvor  = qpdv / r
tauqpdv = e / abs(qpdv)
```

Lorentz-force work is kept separate from the internal-energy heating family:

```text
qlf   = u . (J x B) = J . E_ideal
qlfor = qlf / r
```

No generic `tauqlf` is defined.  `qlf` preserves the historical IDL staggering
order when centering the three edge contributions.

## Poynting flux

The total Poynting flux uses the stored total electric field, including any
non-ideal contribution available in the Bifrost output:

```text
sp_x/y/z    total Poynting-flux components = E x B, on faces
sp_h        horizontal magnitude of the total Poynting flux, center
modsp       magnitude of the total Poynting flux, center
```

The native Bifrost normalization of `E x B` corresponds to an energy-flux unit
`u_e*u_l/u_t`.  Solar/CGS output is in `erg cm^-2 s^-1`; SI output is in
`W m^-2`.  The total `sp_*` family is unavailable when the stored total electric
field `e_x/e_y/e_z` is unavailable; it never falls back silently to the ideal
field.

The ideal-MHD family is derived directly from velocity and magnetic field:

```text
sp_ideal_x/y/z   B^2 u - (u.B) B, on faces
sp_ideal_h       horizontal magnitude, center
modsp_ideal      magnitude, center
```

The vertical ideal flux is also decomposed into emergence and shear terms:

```text
sp_ideal_emerg_z = u_z * (B_x^2 + B_y^2)
sp_ideal_shear_z = -B_z * (u_x*B_x + u_y*B_y)
sp_ideal_z       = sp_ideal_emerg_z + sp_ideal_shear_z
```

All products and stagger operations are evaluated in native Bifrost orientation.
The normal public orientation transform is applied only at the `get_var()`
boundary, so a positive public `sp_z` points toward physical +Z.

## Thermodynamics

Canonical quantities use stored simulation values first and EOS only as a
fallback:

```text
tg : stored tg -> hiontg -> tg_eos
ne : stored ne -> hionne -> ne_eos
pg : stored AUX p -> raw pg compatibility -> pg_eos
kr : stored kr -> kr_eos
```

`p` is a public alias of `pg`, matching Bifrost's AUX storage name.
`pg_eos` remains the explicit way to force EOS reconstruction even when AUX
pressure is available. Explicit `_eos` names always force the equilibrium table.

Additional plasma/thermodynamic diagnostics:

```text
pb             magnetic pressure = |B|^2/2 in Bifrost units
pt             total pressure = pg + pb
beta           plasma beta = pg/pb
va             Alfven speed = |B|/sqrt(r)
mach_a         Alfvenic Mach number = |u|/va   (legacy alias: man)

gamma1         first adiabatic exponent from local EOS derivatives
cs             EOS-consistent sound speed sqrt(gamma1*pg_eos/r)
cs_eos         alias of cs
cs_gamma       constant-gamma sound speed sqrt(gamma*pg/r)
mach_s         sonic Mach number = |u|/cs      (legacy alias: mn)
mach_s_gamma   sonic Mach number based on cs_gamma

hpg_hydro      local hydrostatic pressure scale height pg/(r*|grav|)
hpg            legacy alias of hpg_hydro
hpg_grad       signed gradient scale height -pg/(dpg/dz_physical)
               computed as pg/(dpg/dz_native), including with br=True
```

`gamma1` is obtained by differentiating the same bilinear interpolation of the
EOS pressure table used by the rho-e lookup.  `hpg_hydro` uses the run's native
`grav` parameter when present (normally `-2.74` for the solar surface in the
default normalization), with a solar-gravity fallback for older files.

### 500 nm optical depth

`tau500` implements the H-minus bound-free optical-depth approximation used by
the historical Bifrost IDL `br_rhoeetab.pro` routine. It uses the canonical
`tg` and `ne` resolver paths (stored/HION values when available, otherwise EOS)
and the run-composition hydrogen number density. The local H-minus extinction
is fused with a trapezoidal vertical integration in a Numba kernel, so no full
3-D opacity cube is created.

The integration is performed in native Bifrost order from `k=0`, with the IDL
top-boundary value `tau=1e-16`. A requested x/y subdomain is respected before
evaluation, but a z selection is cropped only after the complete native column
has been integrated, because optical depth at a given height depends on every
layer above it. Public physical orientation is applied afterwards as usual.

```python
tau500 = dd.get_var("tau500", snap)
tau500_native = dd.get_var("tau500", snap, br=True)
```

The isolated integration kernel can be timed with:

```bash
python benchmarks/benchmark_tau500.py --n 256 --order F
```

## Units

Supported output systems: `bifrost`, `solar`, `cgs`, `si`.

The registered diagnostic unit families include:

```text
magnetic_gradient  B / length
inverse_time       s^-1 type quantities such as div(u), curl(u)
length             ldiv*, lcurl* (characteristic lengths)
dimensionless          beta, B-hat components
energy_rate_density   erg cm^-3 s^-1 in solar/CGS
specific_energy_rate  erg g^-1 s^-1 in solar/CGS
time                  s in solar/CGS/SI
length                Mm in solar, cm in CGS, m in SI
```

In `solar`, characteristic lengths are in Mm and magnetic gradients in G/Mm.
The four `l*` ratios return `+inf` where their denominator is zero and the
numerator positive, or `NaN` for undefined `0/0`.


## Unit labels and variable introspection

Registered units are available directly from the data object, using the same
unit system as `get_var()`:

```python
dd.get_units("eb")
# "erg cm^-3"

dd.get_units("qjoule", latex=True)
# "erg cm$^{-3}$ s$^{-1}$"

dd.get_units("eb", units="si")
# "J m^-3"

dd.get_unit_type("eb")
# "energy_density"

dd.get_var_name("tauqjoule", latex=True)
# r"$\tau_{Q_J}$"

dd.get_var_name("bx", latex=True)
# r"$B_x$"
```

``get_var_name(var)`` returns the canonical programmatic name; with
``latex=True`` it returns a Matplotlib-mathtext-ready physical label. Aliases
are resolved first, so historical and canonical names share exactly the same
plot label. Physical notation is defined explicitly for every built-in
quantity rather than guessed from arbitrary strings.

Aliases resolve through the same registry, so `dd.get_units("bx")` and
`dd.get_units("b_x")` are identical.  `dd.info(var)` returns concise
user-facing registry metadata: canonical name, aliases, description, stagger
location, unit category, and display units.  Supplying a snapshot adds
`snapshot`, `available`, and `source`.  Sources are native Bifrost file groups
(`snap`, `aux`, `hion`, `helium`), `npy` for a persistent saved array,
`computed` for a resolver recipe, `missing_response_table`, or `unavailable`.
Internal resolver details such as Bifrost storage names are intentionally not
part of the public information dictionary.

The registry is also directly discoverable:

```python
dd.variables()            # all canonical names, alphabetically sorted
dd.variables("joule")     # quick text search
dd.variables("ex")        # exact alias lookup -> ["e_x"]
dd.variables("pressure")  # searches descriptions and unit categories too
```

Search results always use canonical names.  Aliases are searchable but are not
listed as separate variables.

Every package variable must define a human-readable `description` when it is
registered; this keeps the registry itself as living documentation. Session-local
experimental variables registered with `dd.register_variable(...)` may omit it.

### Documentation conventions

The runtime package, response-building tools, and benchmark helpers all carry
function/class docstrings. Public methods use NumPy-style parameter/return
sections where useful. Quantity getters intentionally take their physical
description from the registry, which is the single source of truth exposed by
``dd.info()``; the registration decorator also installs that description as the
getter's runtime docstring. Internal docstrings state the expected native/public
orientation, units, stagger topology, or cache role whenever those conventions
are important to correctness.

## Persistent saved variables

Expensive full 3-D derived quantities can be cached on disk:

```python
dd.savevar("divpoynting", snap)
```

The default layout is deliberately the same simple style used in the legacy
IDL workflow:

```text
run_directory/
    saved/
        divpoynting_0383.npy
        divpoynting_0383.json
```

The `.npy` array is always stored as the full cube in native Bifrost
orientation, Bifrost units, native stagger location, float32, and Fortran
order.  The JSON sidecar records the variable, snapshot, shape, location, unit
type, orientation, dtype, and package version.  Metadata are validated before
a saved array is used.

Saved arrays are opened with `np.load(..., mmap_mode="r")`, so a spatial
selection can read only the requested slab/subvolume instead of loading the
full cube.  Source priority is:

```text
in-memory evaluation cache
    -> native Bifrost snap/aux/hion/helium
    -> saved/*.npy
    -> derived recipe
```

For debugging, the saved layer can be bypassed without deleting anything:

```python
from_saved = dd.get_var("divpoynting", snap)
recomputed = dd.get_var("divpoynting", snap, use_saved=False)
```

`dd.has_savedvar(var, snap)` checks that both `.npy` and `.json` exist and are
consistent.  `savevar(..., overwrite=True)` explicitly regenerates an existing
cache.  A custom directory may be supplied with `BifrostData(...,
saved_dir=...)`; relative paths are interpreted inside the run directory.

## Tests

The suite includes tests for native writes, partial I/O, cache provenance,
strict EOS domains, vector/orientation signs and plotting coordinates. In addition to the previous raw I/O, saved
cache, stagger, EOS, unit, energy, and halo/subdomain regression tests, it now
checks ideal-gas recovery of EOS-derived `gamma1`, EOS and constant-gamma sound
speeds, Alfven speed and Mach numbers, both pressure-scale-height definitions,
length-unit labels, aliases, and the requirement that every official registry
entry carries a description.


### Scalar diagnostic memory traffic

Version 0.1.20 evaluates several large scalar expressions (including ``pb``, ``eb``, ``ek``, ``qpdv``, ``ldivu``, ``va``, ``cs_gamma``, and the pressure scale heights) with NumPy ``out=`` buffers. This preserves the readable NumPy implementation while avoiding unnecessary full-cube temporaries.

## Fast MP4 movies

The new ``dd.movie()`` reuses the scalar ``dd.plot()`` figure and writes frames
straight to FFmpeg using Agg blitting, without intermediate PNG files.

```python
import numpy as np
from matplotlib.colors import LogNorm

path = dd.movie(
    "modb", np.arange(0, 231, 2),
    y=25.5, cmap="turbo", norm=LogNorm(vmin=1, vmax=2e4),
    fps=24, dpi=144,
)
print(path)
# <simulation_dir>/movies/modb_0000_0230_0002_XZ_y=25.50.mp4

# Scan one snapshot through a physical Z interval, one native height in two.
path = dd.movie("tg", 536, scan="z", z=slice(0, 10), every=2, fps=24)
# <simulation_dir>/movies/tg_0536_XY_scan-z=..._every=0002.mp4

# Scan Y while cropping the visible XZ cross-section.
path = dd.movie("modb", 536, scan="y", y=slice(0, 30),
                x=slice(5, 25), z=slice(0, 10), every=4)

# Animated centered magnetic contours, recalculated per frame.
path = dd.movie("tg", [0, 2, 4, 6], y=25.5,
                contours={"var": "bx", "centered": True,
                          "levels": [-100, 100], "colors": "white"})

# Temporal LOS-integrated movie; accumulates native planes (no 3-D cube).
path = dd.movie("r", [0, 2, 4, 6], los="z", fps=24)
```

- Exactly one **scalar physical cut** (``x``, ``y`` or ``z``) is required;
  the remaining axes may be physical ROI slices, e.g. ``x=slice(5, 20)`` and
  ``z=slice(0, 10)`` for a fixed ``y=25.5`` movie.
  A cropped movie appends the selected ranges to its default filename.
- Snapshots must be increasing, integer and **equally spaced**.
- XZ/YZ movies use the same ``uniform_z=True`` remeshing and transpose as
  ``dd.plot()``; XY cuts are not vertically remeshed.
- Color normalization and the colorbar stay fixed. The time title comes from
  each snapshot's ``dd.get_t(..., units="solar")`` metadata.
- For a spatial scan use one integer snapshot and explicitly set the scanned axis
  with ``scan="x"``, ``"y"`` or ``"z"``. ``every`` (default 1) chooses
  every Nth native stagger-aware grid coordinate; scanning Z uses native
  nonuniform heights. Other spatial axes may be cropped with physical slices.
- Contours are available in temporal and spatial-scan movies. They are updated
  as an independent 2-D variable read on every frame; overlays are centered
  by default when the main image uses the center grid. Set ``centered=False``
  to retain native staggered contour coordinates. LOS cannot be combined
  with ``contours`` or ``scan``.
- Temporal LOS movies (``los="x"``, ``"y"``, ``"z"``) accept only ROI slices or
  ``None`` selectors, integrate each snapshot into a 2-D accumulator with the
  same physical quadrature as ``dd.plot()``, and never load a 3-D cube.
- Frame data for ordinary cuts is always read one 2-D plane at a time; no
  full 3-D array is materialized. The initial plot's data is reused for frame zero.
- ``output=None`` chooses a descriptive name inside ``<simulation>/movies``;
  coordinates always use **exactly two decimals** (rounded in Mm).
  A bare filename is also placed there; a relative path including directories
  is resolved from the simulation root; an absolute path is honored.
  ``output_dir`` can override the default directory.
- Existing files are never overwritten unless ``overwrite=True``.
- Optional ``crf=18`` and ``preset="medium"`` control H.264 compression.
- ``progress=True`` uses ``tqdm`` when installed (otherwise runs normally).
- Requires a system ``ffmpeg`` executable with ``libx264`` support.
- The method returns the absolute output ``pathlib.Path``.

Temporal cut movies require exactly one fixed scalar coordinate. For **spatial movies**, supply
one integer snapshot, ``scan`` specifying the moving axis, and optional ROI
slices on any of the other axes:

```python
# Native (possibly nonuniform) height scan, without loading a 3-D cube:
dd.movie("tg", 536, scan="z", z=slice(0, 10), every=2)

# Move a vertical XZ plane along Y; restrict the visible rectangle:
dd.movie("modb", 536, scan="y", y=slice(0, 30),
         x=slice(5, 25), z=slice(0, 10), every=4)
```

``every`` must be a positive integer and applies to native spatial index
steps, not Mm. The first frame is reused directly from ``dd.plot()``; later
frames call ``get_var`` only for the requested two-dimensional cut.
Interactive sweep/slider plotting has been removed.

Filenames always round physical coordinates to 0.01 Mm (including scan limits,
fixed cuts and ROI). Two distinct positions may therefore share a default
filename; overwrite protection avoids accidental replacement. Pass an explicit
``output`` or ``png`` filename to distinguish those cases.


## Resolver memory policy and safe writes (v0.1.36)

``get_var`` uses a strong temporary cache by default, released at the end
of each call. For unusually large derived-variable volumes, request
``low_memory=True`` to use a weak-reference cache: this can substantially
reduce peak memory but may repeat calculations and increase runtime.
``BifrostData(..., low_memory=True)`` enables it by default; the per-call
``get_var(..., low_memory=False)`` overrides that default. An explicit
``with dd.cache():`` block always uses a strong cache.

``write_snap`` and ``write_aux`` raise ``FileExistsError`` for existing files
unless ``overwrite=True`` is passed; no interactive confirmation is used.
When an IDL file explicitly declares ``meshfile``, its absence raises
``FileNotFoundError`` instead of silently synthesizing a uniform mesh.

## v0.1.44: color presets and solar colormaps (no SunPy dependency)

The 169 canonical registry variables have user-reviewed default color maps
and normalization limits in `bifrost_proto/plot_defaults.py`. Values are
specified in `units="solar"` and scaled when you request `cgs`, `si`, or
`bifrost`. Any explicit `cmap=` or `norm=` always takes precedence; aliases
inherit the canonical variable's preset. For instance:

```python
from matplotlib.colors import LogNorm

dd.plot("tg", snap, z=2)              # turbo, LogNorm(1e3, 2e6)
dd.plot("lf_h", snap, x=25.5)        # jet, LogNorm(1e-11, 1)
dd.plot("uperpb", snap, z=2)         # jet, Normalize(0, 50)
dd.plot("tg", snap, z=2, cmap="red_terrain_r")
dd.plot("tg", snap, z=2, norm=LogNorm(5000, 1e6))
```

Custom color maps: `red_terrain`, `green_terrain`, `divergent_terrain`
and their `_r` reversals. Instrument maps: `aia94`, `aia131`, `aia171`,
`aia193`, `aia211`, `aia304`, `aia335`, `eui174`, `xrt`, also with `_r`.
EUI HRI 174 uses the AIA 171 palette; all nine XRT filters use the same
Hinode XRT palette. The instrument RGB arrays use the published SunPy 8.0.0
conventions, but **SunPy and Astropy are not needed** at runtime. The SunPy
license and attribution are included in the package.

Calling `dd.plot` registers the custom names lazily. To use them in arbitrary
Matplotlib calls first, explicitly register once:

```python
from bifrost_proto import register_colormaps
register_colormaps()
plt.imshow(image, cmap="aia193")
```

Instrument emission **cuts** are local `DN cm^-1 s^-1 pix^-1` and default
to `LogNorm(vmin=1e-8, vmax=1e-4)` for all AIA, EUI and XRT channels.
With `los=...`, integrated units are `DN s^-1 pix^-1` and the norm defaults
to `LogNorm(1, 20000)` for AIA/EUI or `LogNorm(0.1, 100)` for XRT. For
non-instrument LOS integrals, the color map is retained but the normalization
is automatically chosen by Matplotlib instead of reusing the cut-plane scale.

`dd.movie()` uses `dd.plot()` for the first frame, so its default color map
and normalization remain fixed for the entire movie, without per-frame
rescaling. As in v0.1.43, `dd.plot()` still requires one physical cut
coordinate `x=`, `y=`, or `z=` unless `los=` is given.
