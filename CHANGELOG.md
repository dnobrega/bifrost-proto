# Changelog

## 0.1.47 — Radiative AUX diagnostics and public-release licensing (2026-11-05)

- Register signed AUX-only `qrad` (full radiative-transfer heating/cooling)
  and `qgenrad` (generalized radiation module heating/cooling), without
  reconstructing missing AUX values.
- Provide `qrador`, `qgenrador` (per-mass rates) and `tauqrad`,
  `tauqgenrad` (`e/abs(q)` timescales), with units, LaTeX symbols and
  matching `qspitz`/`qspitzor`/`tauqspitz` plotting presets.
- Add BSD 3-Clause `LICENSE`, `CITATION.cff`, third-party notices,
  packaging metadata and repository hygiene documentation.
- Explicitly promote the EOS interpolation and `gamma1` kernels to
  float64 intermediates so NumPy 2 pure-Python execution matches Numba JIT.
  Keep float32 outputs and add independent precision regression tests.
- Use PEP 639 license metadata and include `THIRD_PARTY_NOTICES.md`
  in built distributions alongside `LICENSE`.


## 0.1.46 — Direct characteristic lengths and robust MP4 interruption handling (2026-11-05)

- Use `bifrost_proto/version.py` as the single source of the package version, including setuptools metadata. Tests no longer hard-code a previous release number.
- Set `bwr_r` for the nine signed velocity components (`u_*`, `uparb_*`,
  `uperpb_*`), with blue indicating positive signed velocity. Keep `uperpb`
  magnitude's approved `jet` color map.
- **Breaking change:** invert the four previously inverse-length variables
  `ldivb`, `lcurlb`, `ldivu`, `lcurlu`. They now represent characteristic
  lengths (`|B|/|div B|`, `|B|/|curl B|`, `|u|/|div u|`, `|u|/|curl u|`),
  use standard `length` unit conversion (Mm / cm / m) and updated plot labels.
  Zero gradients yield `+inf` with positive fields and `NaN` for `0/0`.
  Plot limits of `LogNorm(1e-4, 100)` are provisional for direct Mm values.
- Preserve completed MP4 encodings when publication is interrupted by a
  `BaseException`, and log (rather than raise) cleanup failures after the
  destination is published. Logging is intentional because a `RuntimeWarning`
  would raise with `-W error` and hide the successful result.

## 0.1.45 — Movie publication and visualization compatibility (2026-11-05)

- Drop the deprecated `N=256` from `ListedColormap`: instrument tables
  already contain exactly 256 entries, with identical output colors.
- Create secure same-directory MP4 temporaries using `O_EXCL` and 0666
  filtered by the process umask (e.g. 0644 under umask 022), instead of
  mkstemp's forced 0600. Applies to both overwritten and new movies.
- Keep the atomic, no-clobber hardlink publication when available. On
  filesystems that reject hardlinks (exFAT/SMB/FUSE), use an exclusive copy
  fallback. The fallback is not atomic for readers; if copy/publication
  fails the encoded temporary is preserved and its path reported.
- Automatically center non-instrumental LOS plots on zero with `CenteredNorm`
  when the *registered preset* is signed and uses a diverging palette.
  Preserve automatic bounds for other LOS quantities and all explicit norms.
- Set `b_h` to `LogNorm(1, 1000)`; tidy Ruff warnings and documentation.

## 0.1.44 — Registered plot presets and colormaps (2026-11-04)

- Applied 163 variable-specific color maps and Matplotlib normalizations from
  the user-reviewed v1.43 spreadsheet. Alias names inherit canonical presets.
  Explicit `cmap` and `norm` always override defaults; unknown variables keep
  Matplotlib behavior. Limits are converted for requested display units.
- Fixed `lf_h` to `LogNorm(1e-11, 1)` and `uperpb` to
  `Normalize(0, 50)` with `jet`.
- Added three HSV-defined terrain colormaps and their `_r` counterparts.
- Added 9 native AIA/EUI/XRT color maps without a runtime SunPy dependency,
  with BSD license attribution. Distinct instrument local and LOS scales;
  non-instrumental LOS keeps automatic normalization.
- `movie()` inherits the selected defaults with constant limits across frames.


## 0.1.43 — Robust movies and fast slab-wise LOS integration

- Accept independent `contours={"units": ...}` without forwarding it to Matplotlib; use the requested unit system for overlay values.
- Use figure-managed colorbars when `layout="constrained"` is active; retain existing axes-divider behavior elsewhere.
- Preflight required `.idl`/`.snap` files for temporal and spatial movies, including expected minimum binary sizes; improve missing-file errors for raw variables.
- Encode to a unique same-directory temporary MP4 and atomically publish only after successful ffmpeg completion, respecting `overwrite=False` even if another process creates the destination meanwhile.
- Integrate `los_stream=True` and temporal LOS movies using transverse 3-D slabs spanning the full LOS, instead of per-LOS-plane reads. Adapt slab size to approximately 64 MiB native input, capped at 64 transverse samples; preserve float64 quadrature and the original non-uniform Z mesh.
- Add regression tests for truncated files, isolated failures, slab integration numerical parity, overlays and constrained colorbars.


## 0.1.42 — LOS movies, centered contours and consistent metadata

- Require both matching `.idl` and `.snap` files for `dd.snaps`; `.aux` is optional, and the legacy unnumbered zero is still recognized. Rescan the simulation directory when queried.
- Round all automatically generated plot/movie filename coordinates and ROI/scan limits to exactly two decimal places (normalize negative zero).
- Automatically center contours when the underlying image is cell-centered to avoid half-cell misalignment; allow `centered=True`/`centered=False` in the contour mapping to explicitly choose centered/native mesh coordinates.
- Support animated `contours={...}` in temporal and spatial-scan movies, removing/redrawing only the contour overlay each frame while reusing image and colorbar.
- Support temporal `dd.movie(..., los="x"|"y"|"z")` with plane-wise float64 accumulation on native coordinates, without 3-D variable reads; expose `dd.plot(..., los_stream=True)` for matching memory-saving static LOS plots.
- Keep `scan` + LOS, and LOS + contours incompatible. Add regression tests for centering, memory-aware LOS parity, all three quadrature axes, and encoded MP4 frames.


## 0.1.41 — Spatial scan movies, IDL snapshots and LOS periodic quadrature

- Extend `dd.movie` with explicit `scan="x"/"y"/"z"` for spatial scans of one snapshot and `every=N` to skip native grid samples. Read only a 2-D cut per frame, respecting staggering and nonuniform native Z. Keep equispaced temporal movies unchanged.
- Reuse the first plotted frame for both temporal and spatial movies, avoiding duplicate variable evaluation. Preserve direct FFmpeg blitting and overwrite protection.
- Add `dd.snaps`: sorted snapshot indices having matching `.idl` files for the selected run root. `.snap`/`.aux` are not required; restart files are excluded; the property rescans on access.
- Use simulation-relative output directories consistently: `<simulation>/plots` for PNG, `<simulation>/movies` for MP4. A PNG path containing a relative directory is resolved relative to the simulation, not the current working directory.
- Correct LOS integration for variables shifted to faces along a complete periodic X/Y axis: include the periodic closing cell through full center-cell widths. Keep trapezoidal integration for cropped face axes, nonperiodic axes and Z.
- Extend and adjust regressions for spatial scan frames, output naming, IDL-only discovery, PNG paths and LOS periodic quadrature.

## 0.1.40 — Automatic run-root detection

- Allow `BifrostData(folder=...)` or `BifrostData(fdir=...)` without a run root;
  discover the unique `snapname` from immediate `*.idl` files.
- If several simulations are detected, display a numbered selection with
  matching `.idl` file counts. Non-interactive callers can supply `file_root`
  or `idlparam` explicitly to skip scanning.
- Keep previous positional `BifrostData(run_root, fdir=...)` behavior intact.
- Add regression tests for directory scanning, selection, and API compatibility.


## 0.1.39 — PNG export and line-of-sight integration

- Add `png=True` to `dd.plot()` to create a named PNG under the current working directory’s `plots/` folder at 300 DPI; allow custom filenames and explicit `overwrite=True`.
- Add `los="x"|"y"|"z"` with optional physical slices, returning the same Matplotlib `(fig, ax, im)` contract.
- Integrate cell-centered quantities using physical cell widths; use trapezoids for face-staggered LOS quantities, including nonuniform Z.
- Accumulate in float64 with NumPy `einsum`, without large weighted temporary cubes; uniformly remesh vertical display planes only after LOS integration.
- Construct colorbar symbols with `\int` and simplify registered unit powers with correct CGS/SI/Bifrost factors, including solar `Mm` powers.
- Reuse the movie filename format via shared lightweight naming helpers; preserve `dd.movie()` unchanged. LOS with `contours` is explicitly unsupported.
- Add regression tests for constant column integrals, staggered face quadrature, nonuniform Z, solar unit conversions, PNG files and collision protection.


## 0.1.38 — Plotting API and physical regions

- Keep one-line scalar `dd.plot()` and add optional `ax=` for multipanel figures.
- Allow physical ROI slices on the two unfixed axes (`x`, `y`, `z`); require exactly one scalar cut coordinate.
- Add optional `contours={"var": ..., **matplotlib_kwargs}` with stagger-aware contour coordinates and uniform Z for vertical cuts.
- Forward `low_memory` to both plotted variables.
- Remove interactive sweep, `PlaneViewer` and slider machinery instead of retaining compatibility shims.
- Extend `dd.movie()` to the same spatial ROI selections; automatic MP4 names identify the ROI without changing legacy full-plane filenames.
- Add regression tests for multipanel plots, ROI selections, contours and cropped MP4 frames.


## 0.1.37

- Raise a descriptive ValueError when centering a registered variable whose mesh location is unknown.
- Fix `movie.py` import ordering for Ruff I001.
- Capture the second expected stale-saved-data warning in its regression test, preserving real user warnings.
- Add regression tests for `center=True` and `centered=True` with missing mesh locations.

## 0.1.36

- Fix nested periodic halo selection for wide X/Y windows without losing unwrapped origins.
- Add optional ``low_memory`` weak cache to ``BifrostData`` and ``get_var``; explicit ``dd.cache()`` stays strong.
- Make ``write_snap`` and ``write_aux`` respect ``overwrite``, with no interactive prompts.
- Raise ``FileNotFoundError`` for IDL-declared missing mesh files.
- Correct README, CHANGELOG and EOS developer documentation.

## 0.1.35 — MP4 movies

- Add ``BifrostData.movie`` for scalar XY/XZ/YZ cuts using a reused figure,
  Matplotlib Agg blitting, and direct FFmpeg H.264 output.
- Reuse registered ``get_var`` units and XZ/YZ uniform-Z remeshing; preserve
  colormap/colorbar normalization and physical snapshot time metadata.
- Auto-name videos and save in the simulation's ``movies/`` directory.
- Reject irregular snapshot sequences and refuse overwrite by default.
- Add synthetic regression tests (including actual FFmpeg frame counts).

## 0.1.34

- Fixed Ruff import ordering, `__all__` sorting/public version export,
  redundant boolean conversions, and an unused local binding.
- `method="nearest"` now accepts a staggered-grid scalar coordinate in the
  edge interval shared with the corresponding center grid. For example,
  requesting `uz` at the lowest center-plane Z selects the nearest Z face,
  rather than rejecting it as outside the face sample range.
- `method="linear"` still rejects coordinates outside the variable's own
  sample coordinates; coordinate slices retain their existing semantics.
- Added regression tests for near-edge nearest/linear behavior in both
  physical and native-Bifrost orientation.

## 0.1.33

- Native snapshot/AUX writes use atomic temporary files to protect
  mmap-backed inputs (interactive confirmation was replaced by explicit
  ``overwrite=True`` in v0.1.36).
- Fixed signed `hpg_grad` using the native downward Z derivative in both
  public orientations and fixed three-scalar coordinate sampling (`float32`).
- Strict EOS table bounds and finite/positive inputs, fused Numba `gamma1`,
  no scaled full-cube rho/ee temporaries, and EOS reuse across snapshots.
- Periodically wrapped X/Y halos load only the small requested windows;
  retained the five-cell halo and exact grouped float32 stagger operations.
- Cache nonuniform Z coefficients per shared mesh and reuse full-grid interior
  rows. Removed unused `build_coefficients`, generic X/Y kernels and unused
  periodic/period arguments from the internal stagger operator dispatcher.
- Persistent NPY/JSON caches validate semantic metadata, recipe/package
  version and native source provenance; ignore descriptions; avoid duplicate
  validation and release one-use expression intermediates.
- Fused common stagger(product) operations into exact-order Numba kernels,
  removing full-cube intermediate products in Poynting and Lorentz recipes.
- Protected shared native resolver arrays from in-place user mutation without
  allocating unconditional copies. Added strict physical-coordinate bounds.
- Plot extents use cell outer boundaries and sweep coordinates come directly
  from the common selection API; `imshow` remains the plotting backend.
- Centralized contiguous flat-view helper; tightened mesh/IDL handling and
  reduced packaging dependencies.


## 0.1.32

- Removed the Z-only parallelization bottleneck from the main algebra and stagger kernels. Three-dimensional strided paths now collapse Y and Z into `prange(ny*nz)` while retaining X as the innermost loop, matching the package's common Fortran-contiguous memory layout.
- Added flat `prange(N)` kernels for contiguous dot/cross algebra, ideal Poynting components and emergence/shear terms, the acute J-B cosine, and energy timescales. Public/internal quantity recipes automatically choose the flat path only when all participating arrays share a compatible contiguous layout; otherwise they fall back to the stride-aware collapsed-row path without input copies.
- Updated the strided two- and three-component magnitude kernels to the same collapsed Y-Z parallel strategy.
- Updated all fixed/generic X/Y and geometry-aware Z stagger kernels from `prange(nz)` to collapsed Y-Z parallelism. Exact legacy interpolation/derivative arithmetic and halo behavior are unchanged.
- Added single-horizontal-plane and strided-view regression coverage plus `benchmarks/benchmark_thin_slab_parallel_v132.py` for local/HPC thread-scaling checks.
- Full regression suite: 170 tests.

## 0.1.31

- Extended the public mesh-coordinate helpers with the same one-axis selection semantics used by `get_var()`: `get_x/get_y/get_z` and `get_xf/get_yf/get_zf` now accept physical coordinate slices/scalars plus `ix/iy/iz` and native `ix_br/iy_br/iz_br` selectors on their corresponding axis.
- Added matching selection support to `get_dx1d/get_dy1d/get_dz1d` and to the scalar mean-spacing helpers `get_dx/get_dy/get_dz`, so selected coordinates and spacing arrays retain compatible sizes.
- Added `uniform_z=True` to selected `get_z/get_zf/get_dz1d/get_dz`, matching the selected endpoints and point count used by `get_var(..., uniform_z=True)`.
- Extended `get_grid(location=...)` with the full `get_var()` spatial selector set and `uniform_z`, including scalar-axis collapse and location-aware center/face coordinates.
- Removed the obsolete public `location=` argument from all spacing helpers. Spacing helpers are now explicitly defined on the cell-center mesh; topological center/face/edge coordinates remain the responsibility of `get_x/get_xf/...` and `get_grid(location=...)`.
- Physical coordinate selectors remain expressed in solar Mm even when coordinate output is requested in CGS, SI, or native Bifrost units, preserving exact selector compatibility with `get_var()`.
- Updated quick-look plotting to obtain uniform-Z coordinates directly from the canonical `get_grid(..., uniform_z=True)` API instead of rebuilding them independently.
- Added mesh-selection regression tests covering physical slices, public/native indices, face coordinates, scalar/linear selection, selected uniform-Z grids, spacing sizes, output-unit conversion, and `get_grid()` parity. Full regression suite: 166 tests.

## 0.1.30

- Added total Poynting-flux diagnostics from the stored electric field and magnetic field: `sp_x/sp_y/sp_z`, horizontal magnitude `sp_h`, and total magnitude `modsp`. The components use the native staggered `E x B` paths and therefore include non-ideal electric-field contributions when they are present in Bifrost AUX data.
- Added ideal-MHD Poynting flux `sp_ideal_x/sp_ideal_y/sp_ideal_z`, `sp_ideal_h`, and `modsp_ideal`, evaluated from `B^2 u - (u.B) B` in native Bifrost orientation before the public physical-frame transform.
- Added the vertical ideal-MHD decomposition `sp_ideal_emerg_z = u_z B_h^2` and `sp_ideal_shear_z = -B_z (u_h.B_h)`. `sp_ideal_z` is constructed as the sum of the two face-centered contributions.
- Added the `energy_flux` unit type with solar/CGS units `erg cm^-2 s^-1`, SI units `W m^-2`, and native normalization `u_e*u_l/u_t`.
- Added Poynting-flux plotting labels, physical-orientation sign tests, total-vs-ideal availability checks, full-domain/subdomain stagger regression tests, and unit/decomposition tests. Full regression suite: 158 tests.

## 0.1.29

- Integrated the quick-look 2-D plotting layer as ``dd.plot(...)`` with scalar physical cuts and interactive ``slice`` sweeps, Matplotlib ``Normalize`` support, configurable aspect/colorbar placement, quantity labels/units, and snapshot-time titles.
- Updated plotting to the canonical v0.1.27+ mesh API. Plot coordinates now come from ``get_grid(location=...)``; the obsolete ``get_x/get_y/get_z(location=...)`` calls were removed. This fixes plotting failures such as ``dd.plot("tg", snap, y=25.5)``.
- Preserved stagger-location-aware coordinates for center, face, and edge quantities and matched uniform-Z plotting coordinates to ``get_var(..., uniform_z=True)``.
- Added Matplotlib as a declared runtime dependency while keeping the plotting module lazily imported by ``BifrostData.plot``.
- Added plotting regression tests for scalar cuts, stagger-aware coordinates, ``Normalize`` passthrough, and interactive sweeps.

## 0.1.28

- Removed the historical compact public getter spellings (`getvar`, `getx`, `gety`, `getz`, `gett`, compact spacing/grid names, and `getvarname`). The public API now exposes only the underscored `get_*` forms, keeping interactive tab completion compact and unambiguous.
- Updated all package tests, examples, benchmarks, README snippets, and developer notes to use the canonical underscored getter API.
- Cleaned Ruff findings in the core runtime modules: sorted imports, removed unused imports, moved `Callable`/`Iterable` to `collections.abc`, and narrowed the IDL-parser fallback exception to `(ValueError, SyntaxError)`.
- Made the fallback LaTeX-label construction compatible with Python 3.10 by moving the backslash-containing replacement out of the f-string expression.
- Applied Ruff's `min(...)` simplification to EOS and instrument-response interpolation index clipping in both contiguous and strided Numba kernels. Physics, interpolation bounds, and output semantics are unchanged; these hot kernels remain covered by the existing numerical regression suite.
- Full regression suite: 148 tests.

## 0.1.27

- Standardized canonical public getter names to Python-style underscores: `get_var`, `get_x/get_y/get_z`, `get_xf/get_yf/get_zf`, `get_t`, `get_grid`, spacing helpers, and `get_var_name`. Historical compact spellings remain silent backward-compatible aliases.
- Added `center=` as an explicit alias of `centered=` for `get_var`; conflicting explicit values are rejected instead of being resolved silently.
- Simplified the axis-coordinate API: `get_x/get_y/get_z` always return cell-center coordinates, while `get_xf/get_yf/get_zf` return the corresponding face coordinates. Full topological locations (`center`, faces, edges) remain available through `get_grid(location=...)`.
- Added the standard `units={solar,bifrost,cgs,si}` length conversion to coordinates and spacing helpers. Solar coordinates are in Mm, CGS in cm, SI in m, and Bifrost returns native run length units.
- Fixed snapshot-zero discovery to prefer the normal numbered `_000` parameter/data filenames. Legacy unnumbered `.idl/.snap/.aux` snapshot-zero runs remain readable as fallbacks; HION/helium zero-snapshot discovery follows the same compatibility principle.
- Added `write_snap()` for the native eight-block MHD snapshot order `r, px, py, pz, e, bx, by, bz`, and `write_aux()` which always requires/writes gas pressure `p` first and temperature `tg` second before ordered optional AUX fields. Writers operate on full-domain arrays already in native Bifrost orientation/units and stream blocks without a 4-D temporary.
- Added `_000`/legacy filename, canonical API, coordinate-unit, `center` alias, native writer, overwrite-safety, and binary round-trip regression tests. Full regression suite: 148 tests.

## 0.1.26

- Audited the built-in quantity registry descriptions for concise, searchable physical terminology. Descriptions no longer duplicate stagger location, storage provenance, historical suffix notes, or centering details already represented by metadata. Magnitude, acceleration, characteristic inverse-length, and internal-energy-timescale families now use consistent keywords.
- Simplified AIA, XRT, and EUI emissivity descriptions to a common response-table wording while retaining the shared `T < 1e4 K` / physical `z < 0 Mm` synthesis mask.
- Cleaned legacy aliases: removed `curlu`, `uxparb/uyparb/uzparb`, `uperb`, and `uperbx/uperby/uzperb`; added the compact horizontal-magnitude aliases `uh -> u_h` and `bh -> b_h`.
- Changed `gradpg_x/y/z` from the mathematical `+grad(pg)` to the physical pressure-gradient force density `-grad(pg)`. The corresponding `gradpgor_*` quantities are therefore the signed pressure-gradient acceleration `-grad(pg)/rho`, directly comparable to `lf_*` / `lfor_*`. Magnitudes are numerically unchanged. Plot labels for signed pressure-gradient components now include the minus sign.
- Clarified energy-source descriptions: Joule/viscous terms are heating, optically thin radiation is cooling, `qpdv` is compression/expansion work, and `qlf` is Lorentz-force work. `tauq*` remain internal-energy timescales `e/abs(q)`.
- Added regression tests for physical/native stagger equivalence (`xup/yup/zup` and `ddzdn`), negative-stride layout handling in magnitude/EOS/instrument-response kernels, alias cleanup, searchable description semantics, and the pressure-force sign convention. Full regression suite: 142 tests.

## 0.1.25

- Added ``dd.get_t(snap, units=...)`` backed by the snapshot IDL ``tsnap`` parameter.
  ``bifrost`` returns native ``tsnap``; ``solar``, ``cgs``, and ``si`` follow the
  package ``time`` conversion and therefore return seconds with the current unit
  conventions. No special minute convention is introduced.
- Reworked the interactive ``dd.stagger.*`` physical/native transforms to use the
  same parallel F-contiguous Y/Z reversal as ``getvar``. Public stagger outputs no
  longer force an F-to-C conversion. Derivative sign changes for physical Y/Z are
  fused into the return transform.
- Made the full-domain requirement of ``dd.stagger.*`` explicit. Cropped arrays are
  rejected with a diagnostic explaining that the public stagger interface lacks
  selection/halo metadata; subdomain stagger operations should go through ``getvar``.
- Added regression tests for snapshot-time units, missing ``tsnap``, F-contiguous
  public stagger outputs, exact equivalence with centered ``getvar`` results, and
  rejection of ambiguous cropped arrays. Full regression suite: 131 tests.

## 0.1.24

- Reworked `_orient_physical()` to materialize the public physical frame directly
  as an F-contiguous array with a parallel Numba copy. The native Y/Z reversal
  and the sign change required for Y/Z vector components are fused into the same
  pass; `fastmath` remains disabled.
- Removed the former `value[:, ::-1, ::-1] -> np.ascontiguousarray(...)` path,
  which forced a multi-GiB F-order Bifrost cube through an expensive F-to-C
  layout conversion. Public `br=False` and native `br=True` full cubes are now
  both naturally F-contiguous.
- Live 720x720x1116 float32 testing on 18 threads measured the orientation copy
  at about 0.0156 s for parallel F output versus about 3.34 s for the previous
  NumPy C-output path; the arrays were bitwise identical. Follow-up checks found
  no penalty for representative Y slices, LOS sums, or element-wise algebra.
- Added orientation regression tests for F-contiguous output, exact scalar/vector
  values, Y/Z component signs, and serial-reference equivalence. Full regression suite: 126 tests.

## 0.1.23

- Parallelized the native `tau500` H-minus optical-depth integration over each
  independent XY plane with Numba `prange`, while keeping the vertical
  trapezoidal recurrence serial in native Bifrost k-order.
- Preserved the existing float32 `chi_prev` work plane and per-cell arithmetic,
  so the parallel kernel remains bitwise identical to the v0.1.22 serial
  implementation.
- Kept `fastmath` disabled and left Numba thread-count control entirely to the
  user/environment (`NUMBA_NUM_THREADS`).
- Live 512x512x384 testing on 18 threads measured the tau500 integration kernel
  at about 0.141 s versus 0.742 s serial (~5.3x speed-up).
- Added a bitwise serial-vs-parallel tau500 regression check; full suite: 117 tests.

## 0.1.22

- Parallelized the large independent-cell Numba kernels with `parallel=True`
  and `prange`, without enabling `fastmath`: 2-D/3-D uniform-Z interpolation,
  2-/3-component vector magnitudes, fused dot/cross algebra, `theta_jb`,
  `tauq*`, rho-e EOS interpolation, and uniform AIA/XRT/EUI response
  interpolation.
- Preserved the memory-layout-aware inner-loop order: C-order uniform-Z cubes
  parallelize over the first axis, while F-order/native 3-D kernels parallelize
  over the outer Z plane or flat physical-memory index as appropriate.
- The parallel kernels retain the same per-cell arithmetic ordering as v0.1.21
  and do not use `fastmath`. Live 512x512x384 testing on 18 Numba threads found
  bitwise-identical outputs and speed-ups from about 2.5x for memory-bound
  vector algebra to about 9-11x for `tauq*`, EOS, instrument responses, and
  uniform-Z interpolation.
- `bifrost-proto` does not set the Numba thread count. Users can control it with
  Numba's normal configuration, e.g. `NUMBA_NUM_THREADS`, which is important
  for avoiding oversubscription in multi-process/HPC workflows.
- Added regression checks that performance-critical dispatchers remain
  parallelized with `fastmath` disabled. Regression suite expanded to 115 tests.

## 0.1.21

- Added `dd.get_varname(var, latex=False)`. The default returns the canonical
  registry name; `latex=True` returns an explicit Matplotlib-mathtext physical
  label (for example `tauqjoule -> $\tau_{Q_J}$`). Aliases resolve before
  formatting, so legacy and canonical names share the same symbol.
- Added explicit plotting notation for every built-in registered quantity,
  including vector components/magnitudes, heating rates and timescales, EOS
  variants, AIA/XRT/EUI emissivities, and topology diagnostics.
- Performed a systematic documentation audit across the runtime package,
  instrument-response build tools, and benchmark helpers. Every Python
  function/class now carries a docstring; public APIs received expanded
  NumPy-style documentation where appropriate.
- Registry descriptions remain the single source of truth for quantity
  physics and are installed as runtime getter docstrings to avoid duplicated
  descriptions drifting apart.
- Added `docs/DEVELOPER_NOTES.md` documenting resolver orientation/units,
  stagger topology, selections/halos, cache ownership, registry semantics, EOS,
  instrument responses, persistent `.npy` quantities, and the performance
  policy.
- Added documentation/plot-label regression tests.

## 0.1.20

- Reduced temporary-array allocation in several large scalar diagnostics using NumPy ``out=`` operations: ``pb``, ``eb``, ``ek``, ``qpdv``, ``ldivu``, ``va``, ``cs_gamma``, ``hpg_hydro``, and ``hpg_grad``.
- Kept simple binary diagnostics unchanged where NumPy already requires only one output array.
- No new Numba kernels were introduced in this pass; the goal is lower memory traffic with transparent NumPy code.
- Added regression tests against the previous algebraic expressions.

## 0.1.19

- Added `dd.get_z(uniform_z=True)`, returning the full-domain uniform Z axis with
  the same endpoints and point count used by `getvar(..., uniform_z=True)`.
- Replaced `_dot3_arrays`' output-plus-scratch NumPy implementation with a
  fused Numba pass that allocates only the result cube.
- Fused the arithmetic part of all three ideal-electric-field components
  (`-(a*b-c*d)`) in Numba, leaving every stagger operation and its ordering
  unchanged.
- Reworked `theta_jb` to compute `|J.B|/(|J||B|)` directly from the already
  centered vector components in one fused Numba pass, avoiding separate
  `modj`, `modb`, dot-product, and denominator cubes. `arccos` remains a NumPy
  ufunc because it benchmarks faster than scalar transcendental evaluation in
  Numba.
- Reworked `uparb` and `qlf` with explicit NumPy `out=` accumulation. This
  removes arithmetic temporaries while preserving the historical stagger
  paths exactly.
- Reworked all `tauq*` energy timescales as a fused Numba pass, preserving
  `+inf` for exactly zero rates and NaN propagation.
- Added algebra/getz regression tests and `benchmark_algebra_v119.py`.
  Regression suite expanded to 105 tests.

## 0.1.18

- Reworked `uniform_z=True` to use fused Numba interpolation instead of two
  full-cube `np.take()` copies plus broadcast arithmetic. C- and F-contiguous
  public arrays use layout-aware loop order; scalar X/Y selections are handled
  by the same path. The new implementation is numerically identical to the
  v0.1.17 vectorized reference for the regression cases.
- Optimized final unit conversion in `getvar()`: unit factors equal to one are
  skipped entirely, and non-native physical-orientation outputs reuse their
  already-detached public buffer for in-place scaling rather than allocating a
  second full cube. Native `br=True` resolver storage is never modified in
  place.
- Simplified `nh` to a single pass, `r * (u_r / grams_per_hydrogen)`, avoiding
  the intermediate full-cube CGS density array.
- Added `benchmark_uniform_z.py` and regression tests for C/F uniform-Z
  interpolation, scalar-axis cases, factor-one unit finalization, and cache
  safety. Regression suite expanded to 100 tests.

## 0.1.17

- Simplified the public `info()` dictionary. Internal resolver details such as
  `requested`, `storage_name`, `derived`, and `component` are no longer exposed.
- Snapshot-aware `info()` now reports only `snapshot`, `available`, and a concise
  `source`: native Bifrost groups (`snap`, `aux`, `hion`, `helium`), `npy` for
  persistent saved arrays, `computed` for resolver recipes,
  `missing_response_table`, or `unavailable`.
- `pg`/`p` information correctly reports `source="aux"` when Bifrost AUX `p` is
  present, while retaining compatibility with raw `pg` datasets.
- Added `dd.variables(query=None)`. With no query it returns all canonical
  variable names in alphabetical order. A query performs a case-insensitive
  search across canonical names, aliases, descriptions, and unit categories;
  exact name/alias matches are returned directly.
- Removed one unused `requested_var` local from `getvar()`.
- Added discoverability/source regression tests. Regression suite expanded to
  96 tests.

## 0.1.16

- Added `tau500`, a center-located dimensionless 500 nm optical-depth
  diagnostic following the H-minus bound-free approximation and trapezoidal
  vertical integration used by the historical Bifrost IDL `br_rhoeetab.pro`
  routine.
- `tau500` uses canonical `tg` and `ne`, so stored/HION thermodynamics are used
  when available and the EOS is only a fallback. The hydrogen number density
  comes from the run composition through the existing `nh` quantity.
- Implemented the optical-depth accumulation with a fused serial Numba kernel.
  It retains only one 2-D plane of the previous local extinction instead of
  materializing a full 3-D opacity cube. Integration is performed in native
  Bifrost k-order and public orientation is applied afterwards.
- Z subdomains are handled non-locally: the requested x/y domain is evaluated
  directly, but the full native z column is integrated before the requested z
  interval is cropped. This preserves the correct optical depth for slices and
  coordinate selections.
- Removed `EOSTable.interpolate_scipy()` from the runtime API. Unusual broadcast
  shapes now use a vectorized NumPy bilinear fallback; the legacy SciPy
  definition lives only in tests/benchmarks as a numerical reference.
- Added regression tests for the IDL H-minus formula, non-uniform z spacing,
  full-column-before-crop behavior, native/public orientation, HION precedence,
  and dimensionless metadata. Regression suite expanded to 92 tests.

## 0.1.15

- Added one common synthetic-emissivity validity mask for every registered AIA,
  Hinode/XRT, and Solar Orbiter/EUI response: local emissivity is set to exactly
  zero for `T < 1e4 K` or physical cell-center height `z < 0 Mm`.
- The instrument mask is applied by a small in-place Numba pass, avoiding 3-D
  boolean-mask temporaries. Heights are derived only for the current native
  `Selection`, so full cubes, native/public subdomains, and later sampling all
  use the same physical rule. Exact threshold values (`T=1e4 K`, `z=0 Mm`)
  remain included.
- Fixed the gas-pressure naming mismatch with Bifrost AUX files. Canonical `pg`
  now uses the stored AUX variable `p` when available; `p` is a public alias of
  `pg`; `pg_eos` continues to force EOS reconstruction. A raw stored `pg` field
  is retained as a compatibility fallback.
- Added regression coverage for AIA/XRT/EUI masking, selected subdomains, public
  versus native orientation, AUX-pressure precedence/aliasing, and forced EOS
  pressure. Regression suite expanded to 87 tests.

## 0.1.14

- Replaced the hot-path `scipy.ndimage.map_coordinates()` rho-e EOS interpolation
  with fused serial Numba bilinear kernels for the canonical Bifrost 3-D arrays.
  The numerical definition remains the same: interpolation on the uniform
  `ln(ee)`/`ln(rho)` table with `mode="nearest"` boundary behavior.
- Added copy-free C- and Fortran-contiguous fast paths plus a stride-aware 3-D
  kernel for non-contiguous views. Resolver/cache inputs are never modified.
- Preserved the pre-v1.14 SciPy implementation as `interpolate_scipy()` for
  validation, benchmarking, and unusual broadcast layouts rather than creating
  hidden contiguous copies.
- `tg`, `ne`, `pg`, and `kr` EOS outputs are all covered by direct regression
  tests against the SciPy reference, including table boundaries, values outside
  the table, C/F layouts, and strided views.
- Native-endian float32 table views are passed directly to Numba; big-endian EOS
  inputs are converted once on table load so the fast path remains usable.
- Added `benchmarks/benchmark_eos_interpolation.py` for isolated SciPy-vs-Numba
  timing. On the local synthetic benchmark, the Numba interpolation was roughly
  2--2.5x faster than the former SciPy path while avoiding its large coordinate
  temporaries.
- Regression suite expanded to 85 tests.

## 0.1.13

- Fixed instrument-response fast-path detection for legacy axes that are intended
  to be uniform but carry small float32 quantization jitter after conversion to
  float64. These tables now use the fused Numba emissivity kernel instead of
  silently falling back to the slower NumPy interpolation path.
- Response-axis origins and spacings are always derived from each table's actual
  `logt`/`logne` arrays. No temperature or density origin is hardcoded. The
  legacy first-interval spacing is retained for interpolation consistency with
  the original IDL recipes, while actual endpoint values define the valid range.
- Added regression tests with deliberately non-round float32 axis origins and
  0.05-dex spacing, plus a guard that genuinely non-uniform float32 axes are not
  misclassified as uniform.
- Added `tools/build_instrument_responses/convert_all.py`, with the historical
  AIA/XRT source-filename mapping, preflight validation before writing, optional
  EUI/HRI 174 conversion via `--nh-over-ne`, output verification, and a safe
  default destination inside the current source tree.
- Expanded the instrument-response documentation and batch-conversion workflow.
  On the user's real AIA 193 cube, correcting the fast-path classification
  reduced the cached response-evaluation step from about 8.25 s to about 0.84 s.

## 0.1.12

- Added a fused Numba fast path for synthetic instrument emissivities
  `ne * nH * G(T, ne)` on the canonical uniform `log10(T)`/`log10(ne)` grids.
  Bilinear interpolation and the final density multiplication are now performed
  in one pass without materializing full-cube log, mask, index, corner-value,
  interpolation, or `G(T,ne)` temporaries.
- C- and Fortran-contiguous response inputs use verified copy-free 1-D memory
  views; arbitrary 3-D strided views use a stride-aware Numba kernel directly.
- Preserved the general NumPy `InstrumentResponseTable.evaluate()` method as a
  reference/fallback for non-uniform response axes and unusual broadcast layouts.
- Instrument recipes such as `aia193`, XRT filters, and `eui174` now call
  `evaluate_emissivity()` directly, while keeping the same public variable names,
  units, fill semantics, and local `DN cm^-1 s^-1 pix^-1` definition.
- Added regression tests for C/F layouts, strided views, input immutability, fill
  behavior, non-uniform fallback, and an explicit guard that the instrument
  recipe uses the fused path for canonical response tables.
- Added `benchmarks/benchmark_instrument_response.py` to compare the v1.11 NumPy
  path against the exact fused v1.12 implementation. Regression suite expanded
  to 79 tests.

## 0.1.11

- Replaced chained `np.hypot(..., out=...)` vector magnitudes with fused serial
  Numba kernels for 2- and 3-component norms. The kernels use one owned output
  buffer, one square root per cell, and never modify resolver/cache inputs.
- Added separate fast paths for C- and Fortran-contiguous arrays using verified
  copy-free 1-D memory views. Bifrost's usual Fortran layout follows this path.
- Added stride-aware 3-D Numba kernels for non-contiguous views. These consume
  NumPy strides directly instead of silently making contiguous copies.
- Added regression tests for float32/float64, C/F layouts, strided views, input
  immutability, and copy-free flattening of contiguous arrays.
- Added exact public subdomain-vs-full-domain crop tests for `modb`, `modj`, and
  `modlf`, covering face-, edge-, and derived-vector magnitude paths.
- Updated `benchmark_magnitude_memory.py` so its `numba` mode benchmarks the
  exact `_norm_arrays()` implementation used by the package.
- Regression suite expanded to 74 tests.

## 0.1.10

- Refactored repeated vector norms into `_register_magnitude()` with explicit
  component tuples and automatic centering from registered stagger locations.
- Magnitudes now use chained `np.hypot(..., out=...)` with an owned output
  buffer, avoiding squared full-cube temporaries and never modifying resolver
  or cache arrays in place.
- Added shared, topology-aware divergence and curl helpers. Divergence accepts
  only x/y/z face vectors; curl maps face vectors to edges with `dn` derivatives
  and edge vectors to faces with `up` derivatives. Invalid topologies fail
  explicitly instead of being silently recentered.
- Added buffer-based three-component dot products for `theta_jb` and `eparb`.
- Renamed horizontal magnitudes `bh` -> `b_h` and `uh` -> `u_h` with no legacy
  aliases, consistent with the `_x/_y/_z/_h` naming convention.
- Removed the `vort` and `vort_x/y/z` aliases. `curlu` remains an alias of
  `modcurlu`.
- Moved the shared stagger-location geometry maps into `geometry.py` so the
  resolver and quantity helpers use one source of truth.
- Added a standalone magnitude memory benchmark and expanded the regression
  suite to 61 tests, including face->edge and edge->face curl paths and
  rejection of divergence on edge vectors.

## 0.1.9

- Added public run-composition properties `elements`, `abundances`,
  `atomic_weights`, `mass_fractions`, and `grams_per_hydrogen`, parsed directly
  from `cel`, `abund`, and `aweight` in `tabparam.in`.
- Added `nh`, the hydrogen-nucleus number density derived from the run mass
  density and composition.
- Added a common `InstrumentResponseTable` runtime with bilinear interpolation
  in `log10(T)` and `log10(ne)` and per-response out-of-grid fill values.
- Registered the seven historical AIA names, nine XRT filter names, and
  `eui174`; every response evaluates the same `ne * nh * G(T,ne)` definition.
- Added `instrument_emissivity` units: `DN cm^-1 s^-1 pix^-1` in solar/CGS
  and `DN m^-1 s^-1 pix^-1` in SI.
- Added packaged resource layout `resources/instrument_responses/<name>/` with
  `.npy` arrays plus provenance-rich `metadata.json`. No legacy IDL/FITS reader
  is required at runtime.
- Added `responses_dir=` for external response-resource roots and
  `response_info(name)` for JSON provenance inspection. `info()` now reports
  response-table availability and distinguishes a missing response table.
- Added reproducible conversion tools for legacy AIA/XRT `.sav` tables and
  Solar Orbiter/EUI HRI 174 FITS responses. EUI conversion explicitly requires
  the historical `proton_dens(T)` helper instead of guessing its CHIANTI/SSW
  normalization.
- Expanded the regression suite to 56 tests, including composition/`nh`, exact
  bilinear interpolation, response metadata, instrument units, and the full
  registered response catalog.

## 0.1.8

- Renamed the perpendicular-velocity magnitude canonically to `uperpb`; the
  provisional beta-stage name `uperp` is intentionally not retained as an
  alias.
- Added `modj` and `theta_jb`; `theta_jb` is the acute J-B angle
  `acos(|J.B|/(|J||B|))`, stored internally in radians and returned in degrees
  in solar units, with clipping and `NaN` where the angle is undefined.
- Added total-electric-field diagnostics `mode`, `eparb`, `eperpb`,
  `eparb_x/y/z`, and `eperpb_x/y/z`. They depend on stored `ex/ey/ez`; the
  non-ideal total field is never reconstructed when AUX values are absent.
- Added gas-pressure-gradient diagnostics `gradpg_x/y/z`, `gradpg_h`,
  `modgradpg`, `gradpgor_x/y/z`, `gradpgor_h`, and `modgradpgor`. The
  mathematical gradient keeps its positive sign.
- Added Lorentz-force diagnostics `lf_x/y/z`, `lf_h`, `modlf`, `lfor_x/y/z`,
  `lfor_h`, and `modlfor`, preserving the legacy Bifrost/IDL stagger path.
- Added `force_density`, `acceleration`, and `angle` unit families; solar
  acceleration is reported in km s^-2 and solar angles in degrees.
- Added an internal halo-safe stagger-expression helper for composite terms
  such as the Lorentz force and edge reconstruction of parallel E.
- Added a regression guard proving that scalar internal energy `e` does not
  collide with vector components `e_x/e_y/e_z`.
- Expanded the regression suite to 52 tests, including analytic pressure
  gradients, Lorentz force, total-E decomposition, J-B angle, unit conversion,
  and exact full-domain/subdomain Lorentz-force equality.

## 0.1.7

- Removed the legacy `get_coord(...)` compatibility helper and its compatibility
  test; `getx()/gety()/getz()` with `location=` and `br=` are now the sole
  coordinate API.
- Added mandatory human-readable `description` metadata for every official
  registered variable; session-local experimental variables may omit it.
- `info()` now reports both the requested/canonical names and the variable
  description. Saved JSON sidecars also record the description for provenance.
- Added `pb`, `pt`, and retained `beta` as `pg/pb`.
- Added cell-centered Alfven speed `va` and Alfvenic Mach number `mach_a` with
  legacy alias `man`.
- Added EOS-derived first adiabatic exponent `gamma1` from local derivatives of
  the rho-e pressure table.
- Added EOS-consistent sound speed `cs` (`cs_eos` alias), constant-run-gamma
  reference `cs_gamma`, and `mach_s` / `mach_s_gamma` (`mn` aliases `mach_s`).
- Added `hpg_hydro` (`hpg` alias), using the run's `grav` parameter, and the
  signed gradient pressure scale height `hpg_grad = -pg/(dpg/dz)`.
- Added a `length` unit family: Mm in solar units, cm in CGS, and m in SI.
- Expanded the regression suite to 41 tests, including analytic EOS, wave-speed,
  Mach-number, pressure-scale-height, description, and unit-label checks.

## 0.1.6

- Added `get_units(var, units=..., latex=...)` for plot-ready unit labels in
  Bifrost, solar, CGS, and SI systems.
- Added `get_unit_type(var)` and `info(var, snap=None)` for registry and source
  introspection.
- Added persistent full-cube derived-variable caching with `savevar()` and
  `has_savedvar()`.
- Saved variables use only `.npy` plus a JSON sidecar in `saved/`; no NPZ or
  IDL saved-file support is included.
- Saved arrays are native Bifrost orientation/units/location, float32 and
  Fortran ordered, and are read with NumPy memmap support.
- Added strict JSON/array metadata validation for saved variables.
- Added `getvar(..., use_saved=False)` for direct saved-vs-recomputed debugging.
- Native Bifrost files retain priority over persistent saved variables, which
  retain priority over derived recipes.
- `savevar(..., overwrite=True)` regenerates caches from native/derived sources
  while bypassing all persistent saved values.
- Expanded the regression suite to 36 tests, including memmap, partial saved
  reads, metadata mismatch detection, explicit overwrite behavior, and unit /
  source introspection.

## 0.1.5

- Added magnetic and kinetic energy densities: `eb = 0.5*modb^2` and
  `ek = 0.5*r*modu^2`.
- Registered `qjoule`, `qvisc`, `qthin`, and `qspitz` as stored-only AUX
  volumetric energy-rate terms.
- Added historical `over r` diagnostics `qjouleor`, `qviscor`, `qthinor`, and
  `qspitzor`, preserving the sign of the source term.
- Added `tauqjoule`, `tauqvisc`, `tauqthin`, and `tauqspitz` as
  `e/abs(q)`, returning `+inf` where the corresponding source is exactly zero.
- Added compression/expansion work `qpdv = -pg*divu`, together with
  `qpdvor` and `tauqpdv`.
- Added Lorentz-force work `qlf = u . (J x B) = J . E_ideal` and `qlfor`; no
  generic Lorentz timescale is defined.
- Generalized `_op_product()` to `_ops_product()` so products on edges can be
  staggered through an arbitrary halo-safe operator chain while preserving the
  historical IDL operation order used by `qlf`.
- Added unit families `energy_rate_density`, `specific_energy_rate`, and
  `time`, with Bifrost/solar/CGS/SI conversions.
- Expanded the regression suite to 32 tests, including analytic checks for
  `eb`, `ek`, `qpdv`, `qlf`, sign-preserving `q*or`, zero-source infinite
  timescales, and exact full-domain/subdomain `qlf` equality.

## 0.1.4

- Added canonical vector-component naming `<quantity>_x/y/z`.
- Added real registry aliases and separate Bifrost `storage_name` metadata.
  Legacy names such as `bx`, `ux`, `jx`, and `ex_ideal` resolve to the same
  canonical cache nodes rather than duplicating calculations.
- Extended `register_vector()` so a vector can be registered from a base name,
  with explicit storage names and legacy aliases.
- Added `_op_product()` for halo-safe expressions such as `xup(u_x*b_x)`.
- Added magnetic quantities: `bh`, `bhat_x/y/z` (`bxob/byob/bzob`), `divb`,
  `curlb_x/y/z`, `modcurlb`, `ldivb`, `lcurlb`, and `beta`.
- Added velocity quantities: `modu`, `uh`, `uparb`, `uparb_x/y/z`,
  `uperpb_x/y/z`, `uperp`, `divu`, `curlu_x/y/z`, `modcurlu`, `lcurlu`, and
  `ldivu`.
- Added `curlu` and `vort` as aliases of `modcurlu`; `vort_x/y/z` alias the
  vorticity components.
- Canonicalized primitive/current/electric vectors as `p_x/y/z`, `b_x/y/z`,
  `u_x/y/z`, `j_x/y/z`, `e_x/y/z`, and `e_ideal_x/y/z` while retaining old
  compact names as aliases.
- Added unit types `magnetic_gradient`, `inverse_time`, `inverse_length`, and
  `dimensionless`.
- Expanded regression suite to 25 tests, including analytic vector-decomposition
  checks and exact full-domain/subdomain `uparb` equality.

## 0.1.3

- Reworked `BifrostUnits` from the fundamental `u_l`, `u_t`, and `u_r`
  normalization.
- Added derived `u_u`, `u_p`, `u_kr`, `u_ee`, `u_e`, `u_B`, `u_i`, and
  `u_el` factors.
- Changed `jx/jy/jz` to `unit_type="current_density"`.
- Registered total `ex/ey/ez` as stored-only edge variables.
- Added derived `ex_ideal/ey_ideal/ez_ideal = -u x B`.
- Added `ne_eos`, `pg_eos`, and `kr_eos`, plus canonical `ne`, `pg`, and `kr`
  fallback behavior matching `tg`.
- EOS `pg` and `kr` are normalized back to Bifrost units internally before
  public output conversion.
- Public physical Y is translated to start at zero while `br=True` remains
  unchanged. Coordinate-based `getvar(..., y=...)` uses the shifted public Y.
- Added SI conversions for current density and electric field.
- Expanded regression suite to 19 tests.

## 0.1.2.1

- Trim fixed-width IDL string padding in parameter files and EOS filenames.

## 0.1.2

- Restore exact legacy grouped float32 arithmetic for X/Y stagger operators.

## 0.1.1

- Add `br=True` raw/native Bifrost orientation to `getvar()` and public stagger.

## 0.1.0

- Initial Bifrost-only prototype.
