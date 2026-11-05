# bifrost-proto developer notes

This file records conventions that are easy to break accidentally when adding
new quantities or changing the resolver.

Canonical public getter names use underscores (`get_var`, `get_x`, `get_xf`, `get_t`, `get_var_name`). Compact spellings without the underscore are intentionally not exposed, so tab completion shows only one public getter API.

## Resolver contract

Derived getters receive arrays in **native Bifrost orientation**, native
stagger location, and Bifrost units unless a recipe explicitly performs a
physical conversion (EOS inputs are the main example). Public transformations
belong at the outer `get_var()` boundary:

1. evaluate the native quantity on the halo-expanded contiguous selection;
2. apply any requested centering through the shared stagger engine;
3. reorient to the public physical convention unless `br=True`; the full
   physical cube is materialized directly as an F-contiguous array, with Y/Z
   reversal and any Y/Z vector-component sign change fused in one parallel pass;
4. apply the user sampling plan;
5. optionally resample Z with `uniform_z=True`;
6. convert to the requested unit system.

A recipe should therefore never inspect `br`, public-axis reversal, or public
unit factors.

## Grid topology and stagger locations

Supported locations are:

- `center`
- `xface`, `yface`, `zface`
- `xedge`, `yedge`, `zedge`

The canonical centering paths live in `geometry.py`. Stagger arithmetic must
remain consistent with the legacy Bifrost/IDL implementation. X/Y use the
fixed periodic sixth-order formulas with the same grouped float32 arithmetic;
Z uses geometry-aware coefficients on the nonuniform grid. Do not replace a
stagger expression with algebraically equivalent code without a golden test.

## Selections and halos

`Selection` is a native-index box. For large periodic X/Y grids its
bounds may temporarily extend below 0 or above n to express a wrapped halo;
the raw reader assembles it from at most two contiguous segments per axis. User-facing scalar,
strided, and linearly interpolated selections are represented by a
`SamplingPlan` and applied only after the native computation. Derived recipes
request the halo they need through `_op()`/`_ops()` rather than constructing
halos themselves.

Quantities involving an integral from a boundary, notably `tau500`, must first
evaluate the full required integration direction and apply the requested crop
only afterwards.

## Cache ownership and in-place operations

Resolver/cache arrays are read-only by convention. In-place NumPy/Numba work is
allowed only on buffers allocated by the current recipe or on the detached
public array created during output finalization. This rule is why `br=True`
unit conversion remains conservative: native output may still share resolver
storage.

`with dd.cache():` intentionally keeps intermediates alive across multiple
public calls. Optimize temporary arrays inside a recipe before changing cache
semantics.

## Variable registry

The registry separates four concepts:

- canonical public name (`b_x`);
- historical alias (`bx`);
- on-disk storage name (`bx`);
- resolver getter, when the quantity can be computed.

Every package variable must have a human-readable `description`. `dd.info()`
exposes user-facing metadata but hides storage implementation details. Descriptions
should state the physical meaning in concise, searchable language and should not
duplicate `location`, storage provenance, or implementation history. Use consistent
keywords for families such as `magnitude`, `acceleration`, `characteristic length`,
`timescale`, `heating`, `cooling`, and `work`. `dd.variables(query)` is the
lightweight discovery interface.

Plot notation is separate from computational metadata. `labels.py` defines the
LaTeX/mathtext symbols returned by `dd.get_var_name(..., latex=True)`. Add a
physical label whenever a new built-in variable is registered; do not infer
scientific notation from arbitrary identifier strings.

## Units

`BifrostUnits` owns numerical conversion factors and unit labels. Internal
recipes should not apply `solar`/CGS/SI scaling. `temperature` and
`number_density` are already stored/interpolated in physical K and cm^-3 and
therefore have unit factor one in Bifrost/CGS-style paths where documented.

## EOS

The rho-e EOS axes are uniform in `ln(rho)` and `ln(e)`. `EOSTable.interpolate`
uses the validated fused bilinear Numba implementation with strict table
bounds: inputs outside the EOS table raise ``ValueError`` instead of
clamping to the nearest boundary. `tg`, `ne`, `pg`, and `kr` use stored/HION values when available and
fall back to EOS; the explicit `*_eos` variables always force EOS evaluation.

## Instrument responses

AIA, XRT, and EUI quantities return local emissivity/count-rate contribution
per unit path length, not LOS-integrated images. Runtime tables are prebuilt
resources; runtime code has no SolarSoft/CHIANTI/Astropy dependency. All
instrument emissivities are set to zero for `T < 1e4 K` or physical `z < 0 Mm`.

## Persistent `.npy` quantities

`savevar()` stores full native-orientation, Bifrost-unit, float32,
Fortran-contiguous arrays plus a JSON sidecar. Raw snapshot/AUX/HION/helium
storage has precedence over `.npy`; `.npy` has precedence over a computed
recipe.

## Performance policy

Optimize memory traffic before arithmetic. Prefer:

- NumPy `out=` for simple element-wise chains;
- fused Numba kernels when they remove several full-cube temporaries or replace
  a general interpolation routine;
- `parallel=True`/`prange` only when output elements (or outer columns/planes)
  are independent and the per-element arithmetic order is unchanged;
- unchanged NumPy when vectorized transcendental operations are already faster.

Do not enable `fastmath` by default: validated parallel kernels preserve the
serial IEEE arithmetic order and already provide the useful speed-up. Never set
the Numba thread count inside the package; users/HPC jobs must remain free to
control `NUMBA_NUM_THREADS` and avoid oversubscription.

Do not optimize stagger geometry by changing the discrete formula. Benchmark
both isolated kernels and end-to-end `get_var()` calls, warming Numba first.

## Public stagger interface versus resolver stagger

`dd.stagger.*` is intentionally a full-domain interactive interface. It accepts
only 3-D arrays whose shape matches the complete run mesh, because a bare ndarray
does not carry the global selection metadata needed to distinguish a cropped X/Y
block from the true periodic domain or to reconstruct the correct nonuniform Z
geometry. Subdomain stagger calculations must go through `get_var`, where the
resolver expands the required five-cell halo per operator, handles true periodic
boundaries, evaluates in native Bifrost geometry, and crops only after the
stagger chain is complete.

Physical/native transforms in `dd.stagger.*` use the same parallel F-contiguous
Y/Z reversal as the public `get_var` path. Do not reintroduce an
`ascontiguousarray` conversion here.

## Strict EOS and persisted provenance (v0.1.33)

EOS tables reject nonfinite, nonpositive and out-of-table rho/specific-energy
inputs; a tabparam/table mismatch is a scientific error, not a signal to clamp.
The EOS object is reused per table across snapshots of one run. The standard
kernels apply conversion-factor logarithms without full-cube CGS temporaries.

Persistent `savevar` JSON records its format/recipe versions and lightweight
source-file fingerprints (sizes and `mtime_ns`). Description/LaTeX changes
are not semantic invalidations. Changed sources or package recipes cause a
warning and transparent recalculation, while malformed cached arrays or
incompatible semantic metadata raise a clear error. These fingerprints are
fast checks, **not cryptographic hashes**: use provenance-controlled input
files for publication-level reproducibility.

The default stagger halo remains five cells because nested operators (e.g.
`xup(ddxdn(...))`) must match full-domain reference results. Horizontal
stagger operators are supported for periodic X/Y only.
