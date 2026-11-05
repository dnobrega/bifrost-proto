"""Core Bifrost reader, resolver, caching, selection, and public API.

The resolver evaluates quantities internally in native Bifrost orientation and
units.  Physical orientation, unit conversion, selections, and optional uniform-Z
resampling are finalized only at the public :meth:`BifrostData.get_var` boundary.
"""

from __future__ import annotations

import glob
import json
import os
import re
import tempfile
import warnings
import weakref
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
from numba import njit, prange

from ._idl_detection import detect_snapname
from .eos import EOSTable, read_idl_ascii
from .geometry import CENTER_OPERATIONS, LOCATION_SHIFTED_AXES
from .labels import latex_varname
from .registry import GLOBAL_REGISTRY, VariableRegistry
from .responses import (
    INSTRUMENT_RESPONSE_NAMES,
    InstrumentResponseTable,
    default_responses_dir,
    response_files,
)
from .selection import AxisPlan, SamplingPlan, Selection
from .stagger import (
    StaggerInterface,
    apply_operator,
    apply_product_operator,
    operator_axis,
)
from .units import BifrostUnits
from .version import __version__

STAGGER_HALO = 5
AMU_CGS = 1.6605402e-24  # legacy value used by the IDL diagnostics


# Fast interpolation onto a uniform public Z grid.  SamplingPlan keeps the
# surviving public axes in x/y/z order, so Z is always the last array axis.
# Layout-specific loop order preserves contiguous writes for both F-order
# public/native arrays and C-order inputs produced by unusual post-processing.
@njit(cache=True)
def _uniform_z_1d(value, lo, hi, w, out):
    """Interpolate a 1-D Z profile onto precomputed uniform-grid indices and weights."""
    for k in range(out.shape[0]):
        wk = w[k]
        out[k] = (1.0 - wk) * value[lo[k]] + wk * value[hi[k]]


@njit(cache=True, parallel=True)
def _uniform_z_2d_c(value, lo, hi, w, out):
    """Interpolate the last axis of a C-oriented 2-D array onto a uniform Z grid."""
    n0, nz = out.shape
    for i in prange(n0):
        for k in range(nz):
            wk = w[k]
            out[i, k] = (1.0 - wk) * value[i, lo[k]] + wk * value[i, hi[k]]


@njit(cache=True, parallel=True)
def _uniform_z_2d_f(value, lo, hi, w, out):
    """Interpolate the last axis of an F-oriented 2-D array onto a uniform Z grid."""
    n0, nz = out.shape
    for k in prange(nz):
        wk = w[k]
        for i in range(n0):
            out[i, k] = (1.0 - wk) * value[i, lo[k]] + wk * value[i, hi[k]]


@njit(cache=True, parallel=True)
def _uniform_z_3d_c(value, lo, hi, w, out):
    """Interpolate a C-oriented 3-D cube onto the requested uniform Z grid."""
    n0, n1, nz = out.shape
    for i in prange(n0):
        for j in range(n1):
            for k in range(nz):
                wk = w[k]
                out[i, j, k] = ((1.0 - wk) * value[i, j, lo[k]]
                                + wk * value[i, j, hi[k]])


@njit(cache=True, parallel=True)
def _uniform_z_3d_f(value, lo, hi, w, out):
    """Interpolate an F-oriented 3-D cube onto the requested uniform Z grid."""
    n0, n1, nz = out.shape
    for k in prange(nz):
        wk = w[k]
        for j in range(n1):
            for i in range(n0):
                out[i, j, k] = ((1.0 - wk) * value[i, j, lo[k]]
                                + wk * value[i, j, hi[k]])


@njit(cache=True, parallel=True)
def _orient_physical_3d_f(value, out):
    """Reverse native Y/Z into an F-contiguous public 3-D output."""
    nx, ny, nz = value.shape
    for k in prange(nz):
        kk = nz - 1 - k
        for j in range(ny):
            jj = ny - 1 - j
            for i in range(nx):
                out[i, j, k] = value[i, jj, kk]


@njit(cache=True, parallel=True)
def _orient_physical_signed_3d_f(value, out):
    """Reverse native Y/Z and negate a Y/Z vector component in one pass."""
    nx, ny, nz = value.shape
    for k in prange(nz):
        kk = nz - 1 - k
        for j in range(ny):
            jj = ny - 1 - j
            for i in range(nx):
                out[i, j, k] = -value[i, jj, kk]


class UnknownVariableError(ValueError):
    """Raised when a requested quantity is neither registered nor stored."""


class StaleSavedVariableError(ValueError):
    """Persistent derived data are outdated and should be recomputed."""




@dataclass
class EvaluationContext:
    """Per-evaluation cache shared by one resolver tree or ``dd.cache()`` block."""

    cache: dict
    use_saved: bool = True


@dataclass(frozen=True)
class RawVariable:
    """On-disk location of one variable inside a Bifrost multi-variable file."""

    filename: str
    index: int
    nvars: int
    transform: str | None = None


@dataclass(frozen=True)
class CompositionMeta:
    """Elemental composition parsed from ``tabparam.in``."""

    elements: tuple[str, ...]
    abundances: np.ndarray
    atomic_weights: np.ndarray
    mass_fractions: np.ndarray
    grams_per_hydrogen: float


@dataclass
class SnapshotMeta:
    """Grid, units, parameters, and raw-variable map for one snapshot."""

    snap: int
    params: dict
    shape: tuple[int, int, int]
    x: np.ndarray
    xdn: np.ndarray
    y: np.ndarray
    ydn: np.ndarray
    z: np.ndarray
    zdn: np.ndarray
    units: BifrostUnits
    raw: dict[str, RawVariable]
    periodic: tuple[bool, bool, bool]
    z_coefficients: dict = field(default_factory=dict)


class BifrostData:
    """Read Bifrost snapshots and evaluate registered analysis quantities.

    Parameters
    ----------
    file_root : str or None, default=None
        Bifrost run root without ``.snap/.aux/.idl`` suffixes. If omitted,
        determine it from ``snapname`` in the directory's ``*.idl`` files.
    fdir : path-like, default="."
        Directory containing the run files.
    folder : path-like or None
        Alias of ``fdir`` for folder-only initialization.
    idlparam : str or None
        Explicit run root (``snapname``); alternative to ``file_root``.
    dtype : str, default="f4"
        Native floating-point dtype used by snapshot files.
    big_endian : bool, default=False
        Interpret binary snapshot files as big-endian when True.
    verbose : bool, default=False
        Enable lightweight diagnostic messages.
    saved_dir : path-like or None
        Directory containing persistent ``savevar`` ``.npy`` caches.
    responses_dir : path-like or None
        Root containing pre-built AIA/XRT/EUI response resources.
    low_memory : bool, default=False
        Use a weak temporary resolver cache to reduce memory at the expense
        of possible repeated computation; ``dd.cache()`` remains strong.

    Notes
    -----
    Internal recipes are evaluated in native Bifrost orientation and units.
    Public orientation, selections, unit conversion, and optional uniform-Z
    resampling are applied only when ``get_var`` returns the final array.
    """

    def __init__(
        self,
        file_root=None,
        *,
        fdir=".",
        folder=None,
        idlparam=None,
        dtype="f4",
        big_endian=False,
        verbose=False,
        saved_dir=None,
        responses_dir=None,
        low_memory=False,
    ):
        """Initialize a run reader without loading snapshot arrays eagerly."""
        if folder is not None:
            if os.fspath(fdir) != "." and os.path.abspath(fdir) != os.path.abspath(folder):
                raise ValueError("folder and fdir must refer to the same directory")
            fdir = folder

        if idlparam is not None:
            if file_root is not None and os.fspath(file_root) != os.fspath(idlparam):
                raise ValueError("Specify only one run root (file_root or idlparam)")
            file_root = idlparam

        self.fdir = os.path.abspath(os.path.expanduser(fdir))
        if file_root is None:
            file_root = detect_snapname(self.fdir)
        self.root_name = file_root
        self.file_root = os.path.join(self.fdir, file_root)
        self.dtype = (">" if big_endian else "<") + dtype
        self.base_dtype = dtype
        self.big_endian = big_endian
        self.verbose = verbose
        self.low_memory = bool(low_memory)
        if saved_dir is None:
            saved_dir = os.path.join(self.fdir, "saved")
        elif not os.path.isabs(saved_dir):
            saved_dir = os.path.join(self.fdir, saved_dir)
        self.saved_dir = os.path.abspath(saved_dir)
        if responses_dir is None:
            responses_dir = default_responses_dir()
        elif not os.path.isabs(os.fspath(responses_dir)):
            responses_dir = os.path.join(self.fdir, os.fspath(responses_dir))
        self.responses_dir = os.path.abspath(os.fspath(responses_dir))

        self.registry = VariableRegistry(GLOBAL_REGISTRY)
        self._meta_cache = {}
        self._eos_cache = {}
        self._saved_validation_cache = {}
        self._composition_cache = {}
        self._response_cache = {}
        self._shared_context = None
        self._grid_snap = None
        self._stagger_interface = None

    @property
    def snaps(self):
        """Sorted snapshots with BOTH matching ``.idl`` and ``.snap`` files.

        Search only this run's root in the simulation directory. No ``.aux``
        is required; restarts are excluded and snapshot zero supports legacy
        unnumbered filenames. Rescan on every access to pick up new output.
        """
        found = set()
        pattern = re.compile(rf"{re.escape(self.root_name)}_(\d+)\.idl\Z")
        for name in os.listdir(self.fdir):
            match = pattern.fullmatch(name)
            if match is None:
                continue
            snap = int(match.group(1))
            # Match exactly the metadata path accepted by get_var(); a
            # root_0002.idl file alone is not a readable root_002.idl.
            if (name == os.path.basename(self._param_filename(snap))
                    and os.path.isfile(os.path.join(self.fdir, name))
                    and os.path.isfile(self._data_filename(snap, "snap"))):
                found.add(snap)
        if (os.path.isfile(os.path.join(self.fdir, self.root_name + ".idl"))
                and os.path.isfile(self._data_filename(0, "snap"))):
            found.add(0)
        return sorted(found)

    # ------------------------------------------------------------------
    # Public registration API
    # ------------------------------------------------------------------
    def register_variable(
        self, name, *, location=None, unit_type=None, component=None,
        storage_name=None, description=None, aliases=(),
    ):
        """Register a session-local scalar quantity and optionally its getter.

        This mirrors the package registry but intentionally permits exploratory
        variables without a description.  Registrations affect only this
        ``BifrostData`` instance.
        """
        # Session-local experimental variables may omit a description.
        self.registry.register(
            name,
            location=location,
            unit_type=unit_type,
            component=component,
            storage_name=storage_name,
            description=description,
            aliases=aliases,
        )

        def decorator(func):
            """Attach a session-local getter to the just-registered variable."""
            self.registry.register(name, getter=func)
            return func

        return decorator

    def register_vector(
        self, names, *, locations, unit_type, storage_names=None, aliases=None,
        description=None, descriptions=None,
    ):
        """Register three session-local vector components with shared metadata."""
        if isinstance(names, str):
            names = tuple(f"{names}_{component}" for component in "xyz")
        else:
            names = tuple(names)
        if storage_names is None:
            storage_names = (None, None, None)
        if aliases is None:
            aliases = ((), (), ())
        if descriptions is None:
            if description is None:
                descriptions = (None, None, None)
            else:
                base = str(description).rstrip(".")
                descriptions = tuple(
                    f"{component.upper()} component of {base[0].lower() + base[1:]}."
                    for component in "xyz"
                )
        for name, location, component, storage_name, component_aliases, component_description in zip(
            names, locations, "xyz", storage_names, aliases, descriptions
        ):
            if isinstance(component_aliases, str):
                component_aliases = (component_aliases,)
            self.registry.register(
                name,
                location=location,
                unit_type=unit_type,
                component=component,
                storage_name=storage_name,
                description=component_description,
                aliases=component_aliases,
            )
        return names

    # ------------------------------------------------------------------
    # Composition and instrument-response metadata
    # ------------------------------------------------------------------
    def _tabparam_filename(self, snap=None):
        """Resolve the run tabparam.in path associated with a snapshot/grid."""
        meta = self._grid_meta() if snap is None else self._meta(int(snap))
        tabparam = str(meta.params.get("tabinputfile", "tabparam.in")).strip().strip("'\"").strip()
        return tabparam if os.path.isabs(tabparam) else os.path.join(self.fdir, tabparam)

    def _composition(self, snap=None):
        """Parse and cache elemental composition metadata from tabparam.in."""
        tabparam = os.path.abspath(self._tabparam_filename(snap))
        cached = getattr(self, "_composition_cache", {}).get(tabparam)
        if cached is not None:
            return cached
        if not os.path.isfile(tabparam):
            raise FileNotFoundError(tabparam)

        params = read_idl_ascii(tabparam)
        missing = [key for key in ("cel", "abund", "aweight") if key not in params]
        if missing:
            raise ValueError(
                f"Composition metadata missing from {tabparam!r}: {', '.join(missing)}"
            )

        elements = tuple(str(params["cel"]).split())
        abundances = np.asarray(str(params["abund"]).split(), dtype=np.float64)
        atomic_weights = np.asarray(str(params["aweight"]).split(), dtype=np.float64)
        if not (len(elements) == abundances.size == atomic_weights.size):
            raise ValueError(
                "cel, abund, and aweight in tabparam.in must contain the same number of entries"
            )
        if not elements:
            raise ValueError("No elements found in tabparam.in composition metadata")

        linear_abundances = 10.0 ** (abundances - 12.0)
        mass_weights = linear_abundances * atomic_weights
        mass_sum = float(np.sum(mass_weights))
        if not np.isfinite(mass_sum) or mass_sum <= 0.0:
            raise ValueError("Invalid abundance/atomic-weight mixture in tabparam.in")
        mass_fractions = mass_weights / mass_sum
        grams_per_hydrogen = float(np.sum(mass_weights) * AMU_CGS)

        composition = CompositionMeta(
            elements=elements,
            abundances=abundances,
            atomic_weights=atomic_weights,
            mass_fractions=mass_fractions,
            grams_per_hydrogen=grams_per_hydrogen,
        )
        if not hasattr(self, "_composition_cache"):
            self._composition_cache = {}
        self._composition_cache[tabparam] = composition
        return composition

    @property
    def elements(self):
        """Element symbols from ``cel`` in the run's ``tabparam.in``."""
        return self._composition().elements

    @property
    def abundances(self):
        """Logarithmic elemental abundances (H=12) from ``tabparam.in``."""
        return self._composition().abundances.copy()

    @property
    def atomic_weights(self):
        """Atomic weights [amu] from ``aweight`` in ``tabparam.in``."""
        return self._composition().atomic_weights.copy()

    @property
    def mass_fractions(self):
        """Elemental mass fractions derived from ``abund`` and ``aweight``."""
        return self._composition().mass_fractions.copy()

    @property
    def grams_per_hydrogen(self):
        """Mass [g] per hydrogen nucleus implied by the run composition."""
        return self._composition().grams_per_hydrogen

    def _instrument_response(self, name):
        """Load and cache one converted instrument-response table."""
        name = str(name)
        if name not in INSTRUMENT_RESPONSE_NAMES:
            raise KeyError(f"Unknown instrument response {name!r}")
        cache = getattr(self, "_response_cache", None)
        if cache is None:
            cache = self._response_cache = {}
        base_dir = getattr(self, "responses_dir", os.fspath(default_responses_dir()))
        key = (os.path.abspath(os.fspath(base_dir)), name)
        if key not in cache:
            cache[key] = InstrumentResponseTable(name, base_dir=base_dir)
        return cache[key]

    def response_info(self, name):
        """Return a copy of the JSON metadata for an instrument response."""
        return dict(self._instrument_response(name).metadata)

    # ------------------------------------------------------------------
    # Explicit multi-call cache scope (optional convenience)
    # ------------------------------------------------------------------
    @contextmanager
    def cache(self, *, use_saved=True):
        """Share one resolver cache across multiple public ``get_var`` calls.

        Arrays are released when the context exits.  This is useful when several
        requested diagnostics reuse expensive intermediate quantities.
        """
        previous = self._shared_context
        self._shared_context = EvaluationContext(cache={}, use_saved=bool(use_saved))
        try:
            yield self
        finally:
            self._shared_context.cache.clear()
            self._shared_context = previous

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------
    def get_var(
        self,
        var,
        snap,
        *,
        x=None,
        y=None,
        z=None,
        ix=None,
        iy=None,
        iz=None,
        ix_br=None,
        iy_br=None,
        iz_br=None,
        method="nearest",
        centered=None,
        center=None,
        uniform_z=False,
        units="solar",
        br=False,
        use_saved=True,
        low_memory=None,
    ):
        """Load or compute one registered quantity for one snapshot.

        Parameters
        ----------
        var : str
            Canonical variable name or alias.
        snap : int
            Snapshot number; ``-1`` addresses the ``.scr`` snapshot.
        x, y, z : scalar, slice, sequence, or None
            Physical-coordinate selections on the public grid.
        ix, iy, iz : index selection or None
            Public-orientation index selections.
        ix_br, iy_br, iz_br : index selection or None
            Native Bifrost index selections.
        method : {"nearest", "linear"}
            Physical-coordinate selection method.
        centered : bool or None
            Center a staggered quantity before applying the final selection.
            ``None`` means not explicitly supplied.
        center : bool or None
            Explicit short alias for ``centered``. Supplying conflicting values
            for ``center`` and ``centered`` raises ``ValueError``.
        uniform_z : bool
            Resample the selected Z extent to the same number of uniformly
            spaced points.
        units : {"solar", "bifrost", "cgs", "si"}
            Output unit system.
        br : bool
            Return native Bifrost orientation/component signs when True.
        use_saved : bool
            Permit persistent ``.npy`` values created by ``savevar``.
        low_memory : bool or None
            If True, use weak references for the temporary resolver cache;
            None inherits the reader setting. An active ``dd.cache()`` block
            always retains strong references to its cached intermediates.

        Returns
        -------
        numpy.ndarray
            Requested scalar/subdomain/cube in the finalized public convention.
        """
        if not isinstance(snap, (int, np.integer)):
            raise TypeError("get_var() intentionally accepts one integer snapshot per call")
        snap = int(snap)

        if centered is None and center is None:
            centered = False
        elif centered is None:
            centered = bool(center)
        elif center is None:
            centered = bool(centered)
        else:
            if bool(centered) != bool(center):
                raise ValueError("center and centered specify conflicting values")
            centered = bool(centered)
        var = self.registry.resolve(var)
        method = self._validate_sampling_method(method)

        spec = self.registry.get(var)
        if spec is None:
            # Unknown AUX variables are intentionally readable in raw Bifrost units.
            if not self._variable_exists(var, snap):
                raise UnknownVariableError(f"Unknown variable {var!r}")
            if units != "bifrost":
                raise ValueError(
                    f"Variable {var!r} exists in a Bifrost file but its units are not registered. "
                    "Use units='bifrost' or register the variable first."
                )
            if centered:
                raise ValueError(
                    f"Cannot center {var!r}: its mesh location is not registered"
                )

        output_location = "center" if centered else (spec.location if spec else None)
        selectors_present = any(v is not None for v in (x, y, z, ix, iy, iz, ix_br, iy_br, iz_br))
        if selectors_present and output_location is None:
            raise ValueError(
                f"Cannot apply coordinate/index selection to {var!r}: mesh location is unknown"
            )

        meta = self._meta(snap)
        plan = self._build_sampling_plan(
            meta,
            location=output_location or "center",
            x=x, y=y, z=z,
            ix=ix, iy=iy, iz=iz,
            ix_br=ix_br, iy_br=iy_br, iz_br=iz_br,
            method=method,
            br=br,
        )

        if self._shared_context is not None:
            # An explicit shared cache always uses strong references,
            # even if use_saved differs between calls.
            if self._shared_context.use_saved == bool(use_saved):
                context = self._shared_context
            else:
                context = EvaluationContext(cache={}, use_saved=bool(use_saved))
        else:
            weak = getattr(self, "low_memory", False) if low_memory is None else bool(low_memory)
            context = EvaluationContext(
                cache=weakref.WeakValueDictionary() if weak else {},
                use_saved=bool(use_saved),
            )
        if centered:
            if spec.location is None:
                raise ValueError(
                    f"Cannot center {var!r}: its mesh location is not registered"
                )
            ops = CENTER_OPERATIONS[spec.location]
            value = self._ops(ops, var, snap, plan.native, context)
        else:
            value = self._getvar(var, snap, plan.native, context)

        if not br:
            value = self._orient_physical(value, spec)
        else:
            # Keep the original Bifrost orientation and component signs.
            # Make the public result independent from memmap/internal cache layout.
            value = np.asfortranarray(value, dtype=np.float32)

        value, axis_names, coords = plan.apply(value)

        if uniform_z and "z" in axis_names and np.ndim(value) > 0:
            value, coords["z"] = self._uniformize_z(value, axis_names, coords["z"])

        if units != "bifrost":
            if spec is None:
                raise ValueError(f"No units are registered for {var!r}")
            factor = meta.units.factor(spec.unit_type, units)
            if factor != 1.0:
                if not br:
                    # _orient_physical() already detached the public array from
                    # the native resolver/cache.  Reuse that buffer instead of
                    # allocating another full cube for the unit conversion.
                    if np.ndim(value) == 0:
                        value = np.float32(value * factor)
                    else:
                        np.multiply(value, factor, out=value)
                else:
                    # Native-orientation output may still share resolver storage;
                    # never modify it in-place.
                    value = value * factor

        if np.ndim(value) == 0:
            return np.float32(value)
        # Native arrays may be shared with a memmap or the resolver cache.
        # Never expose a mutable alias to internal state.
        if br and value.flags.writeable:
            value = value.view()
            value.flags.writeable = False
        return value

    def plot(
        self,
        var,
        snap,
        *,
        x=None,
        y=None,
        z=None,
        ax=None,
        cmap=None,
        norm=None,
        aspect="equal",
        cbar=True,
        cbar_position="top",
        figsize=None,
        title=True,
        minor_ticks=True,
        units="solar",
        contours=None,
        low_memory=None,
        los=None,
        los_stream=False,
        png=False,
        dpi=300,
        overwrite=False,
    ):
        """Plot one physical XY/XZ/YZ cut, optionally cropped to a region.

        Without ``los``, exactly one of ``x``, ``y`` and ``z`` is a scalar
        coordinate in Mm. With ``los="x"|"y"|"z"``, integrate over that
        axis (optionally restricted by a slice); all selectors must then be
        slices or None. The figure is created if
        ``ax=None``, or drawn into the caller's axes otherwise. Contours use
        the same physical selectors and Matplotlib ``contour`` keyword options
        (not combined with LOS integration). ``png=True`` writes a 300-DPI
        image to ``<simulation>/plots``; ``png=path`` selects a filename. Existing files
        are protected unless ``overwrite=True``.

        The implementation is lazily imported to keep non-plotting uses light.
        """
        from .plotting import plot as _plot

        return _plot(
            self,
            var,
            snap,
            x=x,
            y=y,
            z=z,
            ax=ax,
            cmap=cmap,
            norm=norm,
            aspect=aspect,
            cbar=cbar,
            cbar_position=cbar_position,
            figsize=figsize,
            title=title,
            minor_ticks=minor_ticks,
            units=units,
            contours=contours,
            low_memory=low_memory,
            los=los,
            los_stream=los_stream,
            png=png,
            dpi=dpi,
            overwrite=overwrite,
        )

    def movie(self, var, snapshots, **kwargs):
        """Encode a scalar XY/XZ/YZ MP4 with a reused blitting figure.

        See :func:`bifrost_proto.movie.movie` for all options.
        """
        from .movie import movie as _movie

        return _movie(self, var, snapshots, **kwargs)


    # ------------------------------------------------------------------
    # Public mesh API
    # ------------------------------------------------------------------
    @property
    def stagger(self):
        """Stagger operators bound to this run's mesh.

        The public interface expects full-domain arrays in the same physical
        orientation returned by get_var().  Use ``br=True`` in an operator call
        only when the supplied array is in the original Bifrost orientation.
        """
        interface = getattr(self, "_stagger_interface", None)
        if interface is None:
            interface = StaggerInterface(self)
            self._stagger_interface = interface
        return interface

    def get_x(self, *, x=None, ix=None, ix_br=None, method="nearest", units="solar", br=False):
        """Return cell-center X coordinates, optionally applying an X selection.

        Physical ``x=`` selectors are always expressed in solar Mm, matching
        ``get_var()``.  ``ix`` addresses the public output orientation and
        ``ix_br`` addresses native Bifrost indices.
        """
        return self._get_axis_coord(
            "x", location="center", coord_sel=x, index_sel=ix,
            br_index_sel=ix_br, method=method, br=br, units=units,
        )

    def get_xf(self, *, x=None, ix=None, ix_br=None, method="nearest", units="solar", br=False):
        """Return X-face coordinates, optionally applying an X selection."""
        return self._get_axis_coord(
            "x", location="xface", coord_sel=x, index_sel=ix,
            br_index_sel=ix_br, method=method, br=br, units=units,
        )

    def get_y(self, *, y=None, iy=None, iy_br=None, method="nearest", units="solar", br=False):
        """Return cell-center Y coordinates, optionally applying a Y selection.

        Physical ``y=`` selectors are always expressed in solar Mm, matching
        ``get_var()``.  ``iy`` addresses the public output orientation and
        ``iy_br`` addresses native Bifrost indices.
        """
        return self._get_axis_coord(
            "y", location="center", coord_sel=y, index_sel=iy,
            br_index_sel=iy_br, method=method, br=br, units=units,
        )

    def get_yf(self, *, y=None, iy=None, iy_br=None, method="nearest", units="solar", br=False):
        """Return Y-face coordinates, optionally applying a Y selection."""
        return self._get_axis_coord(
            "y", location="yface", coord_sel=y, index_sel=iy,
            br_index_sel=iy_br, method=method, br=br, units=units,
        )

    def get_z(
        self, *, z=None, iz=None, iz_br=None, method="nearest",
        units="solar", br=False, uniform_z=False,
    ):
        """Return cell-center Z coordinates, optionally applying a Z selection.

        Physical ``z=`` selectors are always expressed in solar Mm, matching
        ``get_var()``.  With ``uniform_z=True``, the selected native/public Z
        extent is replaced by the same uniformly spaced coordinate used by
        ``get_var(..., uniform_z=True)``.
        """
        return self._get_axis_coord(
            "z", location="center", coord_sel=z, index_sel=iz,
            br_index_sel=iz_br, method=method, br=br, units=units,
            uniform=bool(uniform_z),
        )

    def get_zf(
        self, *, z=None, iz=None, iz_br=None, method="nearest",
        units="solar", br=False, uniform_z=False,
    ):
        """Return Z-face coordinates, optionally applying a Z selection."""
        return self._get_axis_coord(
            "z", location="zface", coord_sel=z, index_sel=iz,
            br_index_sel=iz_br, method=method, br=br, units=units,
            uniform=bool(uniform_z),
        )

    def get_t(self, snap, *, units="solar"):
        """Return the snapshot time from the IDL ``tsnap`` parameter.

        Parameters
        ----------
        snap : int
            Snapshot number. Use ``-1`` for the restart ``.scr`` snapshot.
        units : {"bifrost", "solar", "cgs", "si"}, default="solar"
            Unit system for the returned time. ``bifrost`` returns ``tsnap``
            unchanged. The other systems follow the registered ``time`` unit
            conversion and therefore return seconds for the current unit
            conventions.

        Returns
        -------
        float
            Snapshot time in the requested unit system.
        """
        meta = self._meta(int(snap))
        if "tsnap" not in meta.params:
            raise KeyError(f"No tsnap entry found for snapshot {snap}")
        tsnap = float(meta.params["tsnap"])
        return tsnap * meta.units.factor("time", units)

    def get_dx1d(self, *, x=None, ix=None, ix_br=None, method="nearest", units="solar", br=False):
        """Return center-grid local X spacing, optionally applying an X selection."""
        return self._get_axis_spacing(
            "x", coord_sel=x, index_sel=ix, br_index_sel=ix_br,
            method=method, br=br, units=units,
        )

    def get_dy1d(self, *, y=None, iy=None, iy_br=None, method="nearest", units="solar", br=False):
        """Return center-grid local Y spacing, optionally applying a Y selection."""
        return self._get_axis_spacing(
            "y", coord_sel=y, index_sel=iy, br_index_sel=iy_br,
            method=method, br=br, units=units,
        )

    def get_dz1d(
        self, *, z=None, iz=None, iz_br=None, method="nearest",
        units="solar", br=False, uniform_z=False,
    ):
        """Return center-grid local Z spacing, optionally applying a Z selection."""
        return self._get_axis_spacing(
            "z", coord_sel=z, index_sel=iz, br_index_sel=iz_br,
            method=method, br=br, units=units, uniform=bool(uniform_z),
        )

    def get_dx(self, *, x=None, ix=None, ix_br=None, method="nearest", units="solar", br=False):
        """Return mean center-grid X spacing over the requested X selection."""
        return float(np.mean(self.get_dx1d(
            x=x, ix=ix, ix_br=ix_br, method=method, units=units, br=br,
        )))

    def get_dy(self, *, y=None, iy=None, iy_br=None, method="nearest", units="solar", br=False):
        """Return mean center-grid Y spacing over the requested Y selection."""
        return float(np.mean(self.get_dy1d(
            y=y, iy=iy, iy_br=iy_br, method=method, units=units, br=br,
        )))

    def get_dz(
        self, *, z=None, iz=None, iz_br=None, method="nearest",
        units="solar", br=False, uniform_z=False,
    ):
        """Return mean center-grid Z spacing over the requested Z selection."""
        return float(np.mean(self.get_dz1d(
            z=z, iz=iz, iz_br=iz_br, method=method, units=units, br=br,
            uniform_z=uniform_z,
        )))

    def get_grid(
        self, *, location="center", x=None, y=None, z=None,
        ix=None, iy=None, iz=None, ix_br=None, iy_br=None, iz_br=None,
        method="nearest", uniform_z=False, units="solar", br=False,
    ):
        """Return selected ``(x, y, z)`` coordinates for one mesh location.

        The selectors have the same meaning as in ``get_var()``.  Scalar
        selections return a scalar coordinate on that axis; slices return 1-D
        coordinate arrays.  ``location`` remains meaningful here because edge
        locations combine center coordinates on one axis with face coordinates
        on the other two.
        """
        method = self._validate_sampling_method(method)
        meta = self._grid_meta()
        plan = self._build_sampling_plan(
            meta,
            location=location,
            x=x, y=y, z=z,
            ix=ix, iy=iy, iz=iz,
            ix_br=ix_br, iy_br=iy_br, iz_br=iz_br,
            method=method,
            br=br,
        )
        factor = (
            meta.units.factor("length", units)
            / meta.units.factor("length", "solar")
        )
        out = []
        for axis, coord, axis_plan in zip("xyz", plan.physical_coords, plan.axes):
            value = self._apply_axis_plan_1d(coord, axis_plan)
            if axis == "z" and uniform_z and np.ndim(value) == 1 and value.size > 1:
                value = np.linspace(value[0], value[-1], value.size, dtype=np.float64)
            out.append(self._scale_coordinate_output(value, factor))
        return tuple(out)

    # ------------------------------------------------------------------
    # Variable metadata API
    # ------------------------------------------------------------------
    def get_location(self, var):
        """Return the registered stagger location of *var*."""
        spec = self.registry.get(var)
        if spec is None or spec.location is None:
            raise UnknownVariableError(f"Mesh location is not registered for {var!r}")
        return spec.location

    def get_unit_type(self, var):
        """Return the registered unit category of *var*."""
        canonical = self.registry.resolve(var)
        spec = self.registry.get(canonical)
        if spec is None or spec.unit_type is None:
            raise UnknownVariableError(f"Unit type is not registered for {var!r}")
        return spec.unit_type

    def get_var_name(self, var, *, latex=False):
        """Return the canonical variable name or its plotting label.

        Parameters
        ----------
        var : str
            Registered variable name or alias. Aliases are resolved before
            formatting, so ``bx`` and ``b_x`` return the same label.
        latex : bool, default=False
            If False, return the canonical programmatic name. If True, return
            a Matplotlib-mathtext-ready physical symbol such as
            ``r"$\tau_{Q_J}$"`` for ``tauqjoule``.

        Returns
        -------
        str
            Canonical name or LaTeX/mathtext plotting label.
        """
        canonical = self.registry.resolve(var)
        if self.registry.get(canonical) is None:
            raise UnknownVariableError(f"Variable {var!r} is not registered")
        return latex_varname(canonical) if latex else canonical

    def get_units(self, var, *, units="solar", latex=False):
        """Return a display-ready unit label for *var*.

        Parameters
        ----------
        var : str
            Variable name or alias.
        units : {"bifrost", "solar", "cgs", "si"}
            Unit system whose label should be returned.
        latex : bool
            If True, format inverse powers for direct Matplotlib use.
        """
        unit_type = self.get_unit_type(var)
        return self._grid_meta().units.label(unit_type, units, latex=latex)

    def variables(self, query=None):
        """Return registered canonical variable names, optionally filtered.

        Parameters
        ----------
        query : str or None
            Case-insensitive substring searched across canonical names,
            aliases, descriptions, and unit categories.  The returned names
            are always canonical registry names.

        Returns
        -------
        list[str]
            Alphabetically sorted canonical variable names.
        """
        names = sorted(self.registry.keys())
        if query is None:
            return names

        query = str(query).strip().lower()
        if not query:
            return names

        exact = []
        searchable = []
        for name in names:
            spec = self.registry[name]
            aliases = [
                alias for alias, target in self.registry.aliases.items()
                if self.registry.resolve(target) == name
            ]
            if query == name.lower() or any(query == alias.lower() for alias in aliases):
                exact.append(name)
            searchable.append((name, spec, aliases))

        # Exact canonical-name/alias lookup should stay sharp even when the
        # same short token occurs incidentally in descriptions (e.g. ``ex``
        # inside "explicitly").
        if exact:
            return exact

        matches = []
        for name, spec, aliases in searchable:
            haystack = [name, *aliases]
            if spec.description:
                haystack.append(spec.description)
            if spec.unit_type:
                haystack.append(spec.unit_type)
            if any(query in str(item).lower() for item in haystack):
                matches.append(name)
        return matches

    def info(self, var, snap=None, *, units="solar", latex=False):
        """Return concise user-facing information for *var*.

        Supplying ``snap`` additionally reports whether the quantity is
        available and where its value comes from.  ``source`` names the
        backing Bifrost file group (``snap``, ``aux``, ``hion``, ``helium``),
        ``npy`` for a persistent saved array, or ``computed`` for a recipe.
        """
        canonical = self.registry.resolve(var)
        spec = self.registry.get(canonical)
        if spec is None:
            raise UnknownVariableError(f"Variable {var!r} is not registered")

        aliases = sorted(
            alias for alias, target in self.registry.aliases.items()
            if self.registry.resolve(target) == canonical
        )
        result = {
            "name": canonical,
            "aliases": aliases,
            "description": spec.description,
            "location": spec.location,
            "unit_type": spec.unit_type,
            "units": (self.get_units(canonical, units=units, latex=latex)
                      if spec.unit_type is not None else None),
        }

        response_complete = None
        if canonical in INSTRUMENT_RESPONSE_NAMES:
            base_dir = getattr(self, "responses_dir", os.fspath(default_responses_dir()))
            files = response_files(base_dir, canonical)
            response_complete = all(
                path.is_file() for path in (files.table, files.logt, files.logne, files.metadata)
            )
            result["response_table_available"] = response_complete
            if response_complete:
                result["response_metadata"] = self.response_info(canonical)

        if snap is not None:
            snap = int(snap)
            storage_name = spec.storage_name or canonical
            raw = self._meta(snap).raw.get(storage_name)
            stored = raw is not None and os.path.isfile(raw.filename)

            # A getter may provide compatibility with a raw field whose name
            # differs from the preferred Bifrost storage name (currently pg).
            if not stored and storage_name != canonical:
                fallback_raw = self._meta(snap).raw.get(canonical)
                if fallback_raw is not None and os.path.isfile(fallback_raw.filename):
                    raw = fallback_raw
                    stored = True

            if canonical in {"qrad", "qgenrad"}:
                stored = stored and self._raw_source(raw.filename) == "aux"
            saved = False if stored or canonical in {"qrad", "qgenrad"} else self.has_savedvar(canonical, snap)
            if stored:
                source = self._raw_source(raw.filename)
            elif saved:
                source = "npy"
            elif canonical in INSTRUMENT_RESPONSE_NAMES and response_complete is False:
                source = "missing_response_table"
            elif canonical in {"tg", "ne"} and self._variable_exists(
                "hion" + canonical, snap
            ):
                source = "hion"
            elif spec.getter is not None:
                source = "computed"
            else:
                source = "unavailable"

            available = stored or saved or spec.getter is not None
            if canonical in INSTRUMENT_RESPONSE_NAMES and response_complete is False and not (stored or saved):
                available = False
            result.update(snapshot=snap, available=available, source=source)

        return result

    # ------------------------------------------------------------------
    # Native Bifrost binary writers
    # ------------------------------------------------------------------
    def _validate_native_write_arrays(self, arrays):
        """Validate full-domain 3-D arrays supplied to a native Bifrost writer."""
        shape = tuple(self._grid_meta().shape)
        for name, value in arrays:
            arr = np.asarray(value)
            if arr.ndim != 3:
                raise ValueError(f"{name} must be a 3-D array; got shape {arr.shape}")
            if tuple(arr.shape) != shape:
                raise ValueError(
                    f"{name} has shape {arr.shape}, expected full Bifrost grid {shape}"
                )
        return shape

    def _write_native_blocks(self, filename, arrays, *, overwrite=False):
        """Write named 3-D arrays sequentially in native Bifrost Fortran layout."""
        arrays = tuple(arrays)
        shape = self._validate_native_write_arrays(arrays)
        filename = os.path.abspath(os.fspath(filename))
        originally_exists = os.path.exists(filename)
        if originally_exists:
            if not overwrite:
                raise FileExistsError(
                    f"Bifrost file {filename!r} already exists; "
                    "pass overwrite=True to replace it"
                )
            warnings.warn(
                f"The existing Bifrost file {filename!r} will be REPLACED. "
                "A new IDL parameter file/run root is recommended.",
                UserWarning, stacklevel=2,
            )
        parent = os.path.dirname(filename)
        os.makedirs(parent, exist_ok=True)
        dtype = np.dtype(self.dtype)
        block_nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        total_nbytes = block_nbytes * len(arrays)
        # Always write to a separate inode: inputs may alias the destination
        # through read-only np.memmap, even with overwrite=True.
        fd, temporary = tempfile.mkstemp(prefix=".bifrost-write-", dir=parent)
        try:
            with os.fdopen(fd, "wb") as fp:
                fp.truncate(total_nbytes)
            for index, (_name, value) in enumerate(arrays):
                mm = np.memmap(
                    temporary, dtype=dtype, mode="r+", order="F",
                    offset=index * block_nbytes, shape=shape,
                )
                mm[...] = np.asarray(value)
                mm.flush()
                del mm
            with open(temporary, "rb") as fp:
                os.fsync(fp.fileno())
            if not originally_exists and os.path.exists(filename):
                raise FileExistsError(
                    f"Destination {filename!r} appeared while writing; "
                    "refusing to overwrite without overwrite=True"
                )
            os.replace(temporary, filename)
            # Existing readers may hold views of the old inode. Clear the
            # resolver cache; external views remain snapshots of the old data.
            if self._shared_context is not None:
                self._shared_context.cache.clear()
            if hasattr(self, "_saved_validation_cache"):
                self._saved_validation_cache.clear()
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return filename

    def write_snap(
        self, filename, r, px, py, pz, e, bx, by, bz, *, overwrite=False
    ):
        """Write a native Bifrost MHD snapshot binary file.

        The eight input arrays must cover the full run grid and already be in
        native Bifrost orientation and Bifrost units. They are written in the
        required order ``r, px, py, pz, e, bx, by, bz`` using the reader's
        configured native floating dtype and endianness. No physical-orientation or unit
        conversion is performed.
        """
        arrays = (
            ("r", r), ("px", px), ("py", py), ("pz", pz),
            ("e", e), ("bx", bx), ("by", by), ("bz", bz),
        )
        return self._write_native_blocks(filename, arrays, overwrite=overwrite)

    def write_aux(self, filename, p, tg, *, extra=None, overwrite=False):
        """Write a native Bifrost AUX binary file with ``p`` and ``tg`` first.

        ``p`` (gas pressure) and ``tg`` (temperature) are mandatory and are
        always written as the first and second blocks. ``extra`` may be an
        ordered iterable of ``(name, array)`` pairs for additional AUX
        quantities. All arrays must cover the full grid and already be in
        native Bifrost orientation and Bifrost units. The matching IDL ``aux``
        list must use the same variable order.
        """
        extras = [] if extra is None else list(extra)
        normalized = []
        seen = {"p", "tg"}
        for item in extras:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("extra must be an ordered iterable of (name, array) pairs")
            name, value = item
            name = str(name)
            if name in seen:
                raise ValueError(f"Duplicate AUX variable name {name!r}")
            seen.add(name)
            normalized.append((name, value))
        arrays = [("p", p), ("tg", tg), *normalized]
        return self._write_native_blocks(filename, arrays, overwrite=overwrite)

    # ------------------------------------------------------------------
    # Persistent derived-variable cache
    # ------------------------------------------------------------------
    def has_savedvar(self, var, snap):
        """Return True if a valid ``saved/<var>_<snap>.npy`` cache exists."""
        canonical = self.registry.resolve(var)
        if self.registry.get(canonical) is None:
            raise UnknownVariableError(f"Variable {var!r} is not registered")
        return self._saved_exists(canonical, int(snap))

    def savevar(self, var, snap, *, overwrite=False):
        """Persist a full 3-D variable in native Bifrost representation.

        Saved arrays are always written in Bifrost orientation, Bifrost units,
        their native stagger location, float32, and Fortran order.  Existing
        saved caches are ignored while regenerating the value.
        """
        snap = int(snap)
        canonical = self.registry.resolve(var)
        spec = self.registry.get(canonical)
        if spec is None:
            raise UnknownVariableError(f"Variable {var!r} is not registered")
        if spec.location is None or spec.unit_type is None:
            raise ValueError(
                f"Cannot save {canonical!r}: location and unit_type must be registered"
            )

        npy_path, json_path = self._saved_paths(canonical, snap)
        exists_any = os.path.exists(npy_path) or os.path.exists(json_path)
        if exists_any and not overwrite:
            raise FileExistsError(
                f"Saved variable {canonical!r} for snapshot {snap} already exists. "
                "Use overwrite=True to replace it."
            )

        value = self.get_var(
            canonical,
            snap,
            units="bifrost",
            br=True,
            use_saved=False,
        )
        value = np.asfortranarray(value, dtype=np.float32)
        meta = self._meta(snap)
        if value.ndim != 3 or value.shape != meta.shape:
            raise ValueError(
                f"savevar() only supports full 3-D variables with shape {meta.shape}; "
                f"got {value.shape}"
            )

        os.makedirs(self.saved_dir, exist_ok=True)
        tmp_npy = npy_path + ".tmp"
        tmp_json = json_path + ".tmp"

        with open(tmp_npy, "wb") as fp:
            np.save(fp, value, allow_pickle=False)

        metadata = {
            "format_version": 2,
            "variable": canonical,
            "snapshot": snap,
            "location": spec.location,
            "unit_type": spec.unit_type,
            "description": spec.description,
            "units": "bifrost",
            "orientation": "bifrost",
            "shape": list(meta.shape),
            "dtype": str(value.dtype),
            "order": "F",
            "package_version": __version__,
            "recipe_version": 1,
            "source_signature": self._source_signature(snap, canonical),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        with open(tmp_json, "w", encoding="utf-8") as fp:
            json.dump(metadata, fp, indent=2, sort_keys=True)
            fp.write("\n")

        os.replace(tmp_npy, npy_path)
        os.replace(tmp_json, json_path)

        if self._shared_context is not None:
            self._shared_context.cache.clear()
        self._saved_validation_cache.clear()
        return npy_path

    @staticmethod
    def _validate_sampling_method(method):
        """Normalize and validate a public coordinate-sampling method."""
        method = str(method).lower()
        if method not in {"nearest", "linear"}:
            raise ValueError("method must be 'nearest' or 'linear'")
        return method

    @staticmethod
    def _apply_axis_plan_1d(values, plan):
        """Apply one :class:`AxisPlan` to a 1-D coordinate-like array."""
        values = np.asarray(values, dtype=np.float64)
        local = values[plan.pstart:plan.pstop]

        if plan.action == "keep":
            return local
        if plan.action == "indices":
            return local[np.asarray(plan.indices, dtype=np.int64)]
        if plan.action == "scalar":
            return np.float64(local[0])
        if plan.action == "linear":
            w = float(plan.weight)
            return np.float64((1.0 - w) * local[0] + w * local[1])
        raise ValueError(f"Unknown AxisPlan action: {plan.action}")

    @staticmethod
    def _scale_coordinate_output(value, factor):
        """Scale one coordinate/spacing result while preserving scalar output."""
        out = np.asarray(value, dtype=np.float64) * factor
        if out.ndim == 0:
            return np.float64(out)
        return out

    def _axis_sampling_plan(
        self, axis, *, location, coord_sel, index_sel, br_index_sel,
        method, br,
    ):
        """Build a normal get_var-style sampling plan for one public mesh axis."""
        kwargs = {
            "x": None, "y": None, "z": None,
            "ix": None, "iy": None, "iz": None,
            "ix_br": None, "iy_br": None, "iz_br": None,
        }
        kwargs[axis] = coord_sel
        kwargs[f"i{axis}"] = index_sel
        kwargs[f"i{axis}_br"] = br_index_sel
        meta = self._grid_meta()
        plan = self._build_sampling_plan(
            meta,
            location=location,
            method=method,
            br=br,
            **kwargs,
        )
        return meta, plan

    def _get_axis_coord(
        self,
        axis,
        *,
        location,
        coord_sel=None,
        index_sel=None,
        br_index_sel=None,
        method="nearest",
        br=False,
        units="solar",
        uniform=False,
    ):
        """Return one selected public/native coordinate axis in length units."""
        self._validate_location(location)
        method = self._validate_sampling_method(method)
        meta, plan = self._axis_sampling_plan(
            axis,
            location=location,
            coord_sel=coord_sel,
            index_sel=index_sel,
            br_index_sel=br_index_sel,
            method=method,
            br=br,
        )
        iaxis = "xyz".index(axis)
        value = self._apply_axis_plan_1d(
            plan.physical_coords[iaxis], plan.axes[iaxis]
        )
        if uniform and np.ndim(value) == 1 and value.size > 1:
            value = np.linspace(value[0], value[-1], value.size, dtype=np.float64)

        # Sampling coordinates are always represented internally here in solar
        # Mm so that x=/y=/z= selectors have exactly the same semantics as
        # get_var(), irrespective of the requested output unit system.
        factor = (
            meta.units.factor("length", units)
            / meta.units.factor("length", "solar")
        )
        return self._scale_coordinate_output(value, factor)

    def _get_axis_spacing(
        self,
        axis,
        *,
        coord_sel=None,
        index_sel=None,
        br_index_sel=None,
        method="nearest",
        br=False,
        units="solar",
        uniform=False,
    ):
        """Return selected local spacing on the cell-center grid."""
        method = self._validate_sampling_method(method)
        meta, plan = self._axis_sampling_plan(
            axis,
            location="center",
            coord_sel=coord_sel,
            index_sel=index_sel,
            br_index_sel=br_index_sel,
            method=method,
            br=br,
        )
        iaxis = "xyz".index(axis)
        coord = np.asarray(plan.physical_coords[iaxis], dtype=np.float64)
        if coord.size < 2:
            spacing_full = np.zeros_like(coord)
        else:
            spacing_full = np.gradient(coord)

        spacing = self._apply_axis_plan_1d(spacing_full, plan.axes[iaxis])
        if uniform:
            selected_coord = self._apply_axis_plan_1d(coord, plan.axes[iaxis])
            if np.ndim(selected_coord) == 1 and selected_coord.size > 1:
                uniform_coord = np.linspace(
                    selected_coord[0], selected_coord[-1], selected_coord.size,
                    dtype=np.float64,
                )
                spacing = np.gradient(uniform_coord)

        factor = (
            meta.units.factor("length", units)
            / meta.units.factor("length", "solar")
        )
        return self._scale_coordinate_output(spacing, factor)

    @staticmethod
    def _validate_location(location):
        """Validate one of the seven supported Bifrost stagger locations."""
        if location not in LOCATION_SHIFTED_AXES:
            allowed = ", ".join(LOCATION_SHIFTED_AXES)
            raise ValueError(f"Unknown location {location!r}. Expected one of: {allowed}")

    # ------------------------------------------------------------------
    # Derived-variable engine
    # ------------------------------------------------------------------
    def _getvar(self, var, snap, selection, context):
        """Resolve one quantity on a contiguous native-Bifrost selection."""
        var = self.registry.resolve(var)
        key = ("var", context.use_saved, var, snap, selection.key)
        cached = context.cache.get(key)
        if cached is not None:
            return cached

        spec = self.registry.get(var)
        storage_name = spec.storage_name if spec and spec.storage_name else var

        # These full/generalized radiative source terms must come from the
        # simulation's AUX output. Never silently read a saved .npy fallback
        # or a same-named quantity from another native storage group.
        if var in {"qrad", "qgenrad"}:
            source = self._meta(snap).raw.get(storage_name)
            if source is None or self._raw_source(source.filename) != "aux":
                raise UnknownVariableError(
                    f"{var!r} is not listed among snapshot {snap}'s AUX fields"
                )
            if not os.path.isfile(source.filename):
                raise FileNotFoundError(
                    f"{var!r} requires the AUX file {source.filename}, which does not exist"
                )

        if self._variable_exists(storage_name, snap):
            value = self._read_variable(storage_name, snap, selection)
        elif context.use_saved and self._saved_exists(var, snap):
            value = self._read_saved_variable(var, snap, selection)
        else:
            if spec is None or spec.getter is None:
                filename = getattr(self._meta(snap).raw.get(storage_name), "filename", None)
                if filename is not None and not os.path.isfile(filename):
                    raise FileNotFoundError(
                        f"{var!r} for snapshot {snap} is stored in {filename}, "
                        "which does not exist"
                    )
                extra = ""
                if spec is not None and spec.storage_name is not None:
                    extra = f" (Bifrost storage name {spec.storage_name!r})"
                raise UnknownVariableError(
                    f"Do not know how to obtain variable {var!r}{extra} for snapshot {snap}"
                )
            value = spec.getter(self, snap, selection, context)

        value = np.asfortranarray(value, dtype=np.float32)
        if isinstance(value, np.ndarray) and value.flags.writeable:
            value.flags.writeable = False
        context.cache[key] = value
        return value

    def _op(self, opname, var, snap, selection, context):
        """Apply one stagger operator to a resolver quantity while managing halo expansion."""
        return self._ops((opname,), var, snap, selection, context)

    def _op_product(self, opname, var1, var2, snap, selection, context):
        """Apply one stagger operator to the pointwise product of two variables."""
        return self._ops_product((opname,), var1, var2, snap, selection, context)

    def _ops_product(self, operations, var1, var2, snap, selection, context):
        """Apply a stagger-operation chain to the pointwise product of two variables.

        The product is formed only after recursively expanding the requested
        domain by the halo required by every operator in the chain.  This
        preserves legacy expressions such as ``yup(zup(E_x * J_x))`` while
        keeping subdomain evaluation identical to cropping a full-domain
        result.
        """
        operations = tuple(operations)
        var1 = self.registry.resolve(var1)
        var2 = self.registry.resolve(var2)
        key = ("ops_product", context.use_saved, operations, var1, var2, snap, selection.key)
        cached = context.cache.get(key)
        if cached is not None:
            return cached

        if not operations:
            a = self._getvar(var1, snap, selection, context)
            b = self._getvar(var2, snap, selection, context)
            value = np.asfortranarray(a * b, dtype=np.float32)
            context.cache[key] = value
            return value

        outer = operations[-1]
        axis = operator_axis(outer)
        meta = self._meta(snap)
        expanded = selection.expand(
            axis,
            STAGGER_HALO,
            meta.shape,
            periodic=meta.periodic[axis],
        )

        center, face = self._local_axis_geometry(meta, axis, expanded)
        coeff = (self._cached_z_coefficients(meta, expanded, outer)
                 if axis == 2 else None)
        if len(operations) == 1:
            # Last/innermost operator: compute product values only at stencil
            # taps. Exact grouped float32 stagger arithmetic, no product cube.
            a = self._getvar(var1, snap, expanded, context)
            b = self._getvar(var2, snap, expanded, context)
            value = apply_product_operator(
                outer, a, b, center, face,
                spacing=self._legacy_axis_spacing(meta, axis), precomputed=coeff,
            )
        else:
            inner = self._ops_product(
                operations[:-1], var1, var2, snap, expanded, context
            )
            value = apply_operator(
                outer, inner, center, face,
                spacing=self._legacy_axis_spacing(meta, axis), precomputed=coeff,
            )
        value = value[selection.relative_to(expanded, meta.shape)]
        value = np.asfortranarray(value, dtype=np.float32)
        if isinstance(value, np.ndarray) and value.flags.writeable:
            value.flags.writeable = False
        context.cache[key] = value
        return value

    def _ops_expression(self, operations, tag, snap, selection, context, evaluator):
        """Apply a stagger-operation chain to a lazily evaluated expression.

        ``evaluator(selection)`` must return the expression on exactly the
        requested native Bifrost selection.  The helper recursively expands
        that selection by the standard stagger halo before evaluating the
        expression, applies each operator, and crops back to the requested
        domain.  This keeps composite stagger expressions identical between
        full-domain and subdomain evaluation.
        """
        operations = tuple(operations)
        key = (
            "ops_expression", context.use_saved, tag, operations, snap, selection.key
        )
        cached = context.cache.get(key)
        if cached is not None:
            return cached

        if not operations:
            value = np.asfortranarray(evaluator(selection), dtype=np.float32)
            context.cache[key] = value
            return value

        outer = operations[-1]
        axis = operator_axis(outer)
        meta = self._meta(snap)
        expanded = selection.expand(
            axis,
            STAGGER_HALO,
            meta.shape,
            periodic=meta.periodic[axis],
        )

        inner = self._ops_expression(
            operations[:-1], tag, snap, expanded, context, evaluator
        )
        if self._shared_context is None:
            # This intermediate is consumed by exactly one outer operator.
            inner_key = ("ops_expression", context.use_saved, tag,
                         operations[:-1], snap, expanded.key)
            context.cache.pop(inner_key, None)
        center, face = self._local_axis_geometry(meta, axis, expanded)
        value = apply_operator(
            outer,
            inner,
            center,
            face,
            spacing=self._legacy_axis_spacing(meta, axis),
            precomputed=(self._cached_z_coefficients(meta, expanded, outer)
                         if axis == 2 else None),
        )
        value = value[selection.relative_to(expanded, meta.shape)]
        value = np.asfortranarray(value, dtype=np.float32)
        if isinstance(value, np.ndarray) and value.flags.writeable:
            value.flags.writeable = False
        context.cache[key] = value
        return value

    def _ops(self, operations, var, snap, selection, context):
        """Apply a sequence of stagger operators while preserving resolver/cache semantics."""
        operations = tuple(operations)
        if not operations:
            return self._getvar(var, snap, selection, context)

        key = ("ops", context.use_saved, operations, var, snap, selection.key)
        cached = context.cache.get(key)
        if cached is not None:
            return cached

        outer = operations[-1]
        axis = operator_axis(outer)
        meta = self._meta(snap)
        expanded = selection.expand(
            axis,
            STAGGER_HALO,
            meta.shape,
            periodic=meta.periodic[axis],
        )

        inner = self._ops(operations[:-1], var, snap, expanded, context)
        center, face = self._local_axis_geometry(meta, axis, expanded)
        value = apply_operator(
            outer,
            inner,
            center,
            face,
            spacing=self._legacy_axis_spacing(meta, axis),
            precomputed=(self._cached_z_coefficients(meta, expanded, outer)
                         if axis == 2 else None),
        )
        value = value[selection.relative_to(expanded, meta.shape)]
        value = np.asfortranarray(value, dtype=np.float32)
        if isinstance(value, np.ndarray) and value.flags.writeable:
            value.flags.writeable = False
        context.cache[key] = value
        return value

    # ------------------------------------------------------------------
    # Snapshot metadata / raw I/O
    # ------------------------------------------------------------------
    def _grid_meta(self):
        """Return metadata for a representative snapshot of this run.

        Bifrost's mesh belongs to the run, not to a particular analysis call.
        Prefer snapshot 0, then the restart snapshot, then the first numbered
        snapshot found on disk.  The chosen snapshot is cached.
        """
        grid_snap = getattr(self, "_grid_snap", None)
        if grid_snap is not None:
            return self._meta(grid_snap)

        # If some snapshot metadata has already been loaded, its mesh is the
        # natural representative mesh for this run.
        if getattr(self, "_meta_cache", None):
            grid_snap = next(iter(self._meta_cache))
            self._grid_snap = grid_snap
            return self._meta(grid_snap)

        zero_param = self._param_filename(0)
        if os.path.isfile(zero_param):
            snap = 0
        elif os.path.isfile(self.file_root + ".idl.scr"):
            snap = -1
        else:
            pattern = self.file_root + "_*.idl"
            numbered = []
            rx = re.compile(re.escape(self.file_root) + r"_(\d+)\.idl$")
            for filename in glob.glob(pattern):
                match = rx.match(filename)
                if match:
                    numbered.append(int(match.group(1)))
            if not numbered:
                raise FileNotFoundError(
                    f"Could not find a Bifrost .idl file for {self.file_root!r}"
                )
            snap = min(numbered)

        self._grid_snap = snap
        return self._meta(snap)

    def _meta(self, snap):
        """Read and cache all metadata required to access one snapshot."""
        if snap in self._meta_cache:
            return self._meta_cache[snap]

        paramfile = self._param_filename(snap)
        if not os.path.isfile(paramfile):
            raise FileNotFoundError(paramfile)
        params = read_idl_ascii(paramfile)

        nx = int(params["mx"])
        ny = int(params["my"])
        nz = int(params["mz"])
        shape = (nx, ny, nz)

        units = BifrostUnits(
            u_l=float(params.get("u_l", 1.0e8)),
            u_t=float(params.get("u_t", 1.0e2)),
            u_r=float(params.get("u_r", 1.0e-7)),
        )

        x, xdn, y, ydn, z, zdn = self._read_mesh(params, shape)
        periodic = tuple(
            self._as_bool(params.get(f"periodic_{axis}", default))
            for axis, default in zip("xyz", (True, True, False))
        )
        raw = self._build_raw_map(snap, params)

        meta = SnapshotMeta(
            snap=snap,
            params=params,
            shape=shape,
            x=x, xdn=xdn,
            y=y, ydn=ydn,
            z=z, zdn=zdn,
            units=units,
            raw=raw,
            periodic=periodic,
        )
        for previous in self._meta_cache.values():
            if np.array_equal(meta.z, previous.z) and np.array_equal(meta.zdn, previous.zdn):
                meta.z_coefficients = previous.z_coefficients
                break
        self._meta_cache[snap] = meta
        return meta

    def _param_filename(self, snap):
        """Return the parameter filename, preferring numbered snapshot names.

        Non-negative snapshots use ``_<NNN>.idl`` uniformly, including
        snapshot zero. For compatibility with older runs, snapshot zero falls
        back to the historical unnumbered ``.idl`` filename when ``_000.idl``
        is absent.
        """
        snap = int(snap)
        if snap < 0:
            return self.file_root + ".idl.scr"
        numbered = self.file_root + f"_{snap:03d}.idl"
        if snap == 0 and not os.path.isfile(numbered):
            legacy = self.file_root + ".idl"
            if os.path.isfile(legacy):
                return legacy
        return numbered

    @staticmethod
    def _as_bool(value):
        """Interpret IDL-style logical values as Python booleans."""
        if isinstance(value, str):
            value = value.strip().upper()
            if value in {"T", "TRUE", ".TRUE."}:
                return True
            if value in {"F", "FALSE", ".FALSE."}:
                return False
        return bool(value)

    def _read_mesh(self, params, shape):
        """Read center/face mesh coordinates or construct a uniform fallback mesh."""
        meshfile = str(params.get("meshfile", "")).strip().strip("'\"").strip()
        if "meshfile" in params and not meshfile:
            raise ValueError("IDL declares an empty meshfile; a valid mesh path is required")
        if meshfile:
            meshfile = meshfile if os.path.isabs(meshfile) else os.path.join(self.fdir, meshfile)

        if meshfile and not os.path.isfile(meshfile):
            raise FileNotFoundError(
                f"Mesh file declared by IDL does not exist: {meshfile!r}"
            )

        if meshfile:
            out = []
            with open(meshfile) as fp:
                for n_expected in shape:
                    n = int(fp.readline().strip())
                    if n != n_expected:
                        raise ValueError(
                            f"Mesh dimension {n} does not match snapshot dimension {n_expected}"
                        )
                    center = np.fromstring(fp.readline(), sep=" ", dtype=np.float32)
                    face = np.fromstring(fp.readline(), sep=" ", dtype=np.float32)
                    if center.size != n or face.size != n:
                        raise ValueError(
                            f"Mesh file {meshfile!r} has incomplete coordinates: "
                            f"expected {n}, got center={center.size}, face={face.size}"
                        )
                    fp.readline()  # inverse-spacing data not needed by stagger_new
                    fp.readline()
                    out.extend((center, face))
            return tuple(out)

        result = []
        for axis, n in zip("xyz", shape):
            d = float(params.get("d" + axis, 1.0))
            if d == 0.0:
                d = 1.0
            if d < 0.0:
                d = -d / n
            center = np.arange(n, dtype=np.float32) * np.float32(d)
            face = center - np.float32(0.5 * d)
            result.extend((center, face))
        return tuple(result)

    def _build_raw_map(self, snap, params):
        """Map stored variable names to their binary file, index, and transform."""
        do_mhd = self._as_bool(params.get("do_mhd", True))
        snapvars = ["r", "px", "py", "pz", "e"]
        if do_mhd:
            snapvars += ["bx", "by", "bz"]

        aux = str(params.get("aux", "")).split()
        # v0 intentionally skips Bifrost's special 2-D AUX storage.
        aux = [v for v in aux if v not in {"ixy1"}]  # known special 2-D AUX block

        hion = []
        if self._as_bool(params.get("do_hion", False)):
            hion = ["hionne", "hiontg", "n1", "n2", "n3", "n4", "n5", "n6", "nh2"]

        helium = []
        if self._as_bool(params.get("do_helium", False)):
            helium = ["nhe1", "nhe2", "nhe3"]

        groups = [
            (snapvars, self._data_filename(snap, "snap"), None),
            (aux, self._data_filename(snap, "aux"), None),
            (hion, self._data_filename(snap, "hion"), None),
            (helium, self._data_filename(snap, "helium"), "exp"),
        ]

        raw = {}
        for variables, filename, transform in groups:
            if not variables:
                continue
            for idx, name in enumerate(variables):
                raw[name] = RawVariable(filename, idx, len(variables), transform)
        return raw

    def _data_filename(self, snap, group):
        """Construct a Bifrost binary filename for one storage group.

        Snapshot zero follows the same numbered ``_000`` convention as other
        non-negative snapshots, with legacy unnumbered names accepted only as
        fallbacks when the numbered file is absent.
        """
        root = self.file_root
        snap = int(snap)
        if group in {"snap", "aux"}:
            suffix = f".{group}"
            if snap < 0:
                return root + suffix + ".scr"
            numbered = root + f"_{snap:03d}" + suffix
            if snap == 0 and not os.path.isfile(numbered):
                legacy = root + suffix
                if os.path.isfile(legacy):
                    return legacy
            return numbered

        if group == "hion":
            if snap < 0:
                return root + ".hion.snap.scr"
            numbered = root + f".hion_{snap:03d}.snap"
            if snap == 0 and not os.path.isfile(numbered):
                legacy = root + ".hion.snap"
                if os.path.isfile(legacy):
                    return legacy
            return numbered

        if group == "helium":
            if snap < 0:
                return root + ".helium.snap.scr"
            # Preserve the historical non-padded helium convention for
            # existing non-zero snapshots while also accepting padded names.
            padded = root + f".helium_{snap:03d}.snap"
            historical = root + f".helium_{snap}.snap"
            if os.path.isfile(historical):
                return historical
            if os.path.isfile(padded):
                return padded
            if snap == 0:
                legacy = root + ".helium.snap"
                if os.path.isfile(legacy):
                    return legacy
            return padded

        raise ValueError(group)

    @staticmethod
    def _raw_source(filename):
        """Classify a raw binary filename as snap, aux, hion, or helium."""
        name = os.path.basename(filename)
        if ".hion" in name:
            return "hion"
        if ".helium" in name:
            return "helium"
        if name.endswith(".aux") or ".aux." in name or re.search(r"_\d+\.aux$", name):
            return "aux"
        return "snap"

    @staticmethod
    def _snap_label(snap):
        """Return the persistent-cache filename label for a snapshot number."""
        return "scr" if int(snap) < 0 else f"{int(snap):04d}"

    def _saved_paths(self, var, snap):
        """Return .npy and JSON paths for one persistent saved quantity."""
        canonical = self.registry.resolve(var)
        label = self._snap_label(snap)
        base = os.path.join(self.saved_dir, f"{canonical}_{label}")
        return base + ".npy", base + ".json"

    def _source_signature(self, snap, var=None):
        """Fast provenance fingerprint for native files supplying a snapshot.

        No full-file hashes are calculated during routine access; file size
        and nanosecond modification timestamps detect ordinary replacements.
        """
        meta = self._meta(int(snap))
        paths = {os.path.abspath(self._param_filename(snap))}
        paths.update(os.path.abspath(raw.filename) for raw in meta.raw.values()
                     if os.path.isfile(raw.filename))
        meshfile = str(meta.params.get("meshfile", "")).strip().strip("'\"")
        if meshfile:
            paths.add(os.path.abspath(meshfile if os.path.isabs(meshfile)
                                      else os.path.join(self.fdir, meshfile)))
        tabparam = str(meta.params.get("tabinputfile", "tabparam.in")).strip().strip("'\"")
        tabpath = os.path.abspath(tabparam if os.path.isabs(tabparam)
                                   else os.path.join(self.fdir, tabparam))
        if os.path.isfile(tabpath):
            paths.add(tabpath)
            tabinfo = read_idl_ascii(tabpath)
            table = str(tabinfo.get("eostablefile", "")).strip().strip("'\"")
            if table:
                paths.add(os.path.abspath(table if os.path.isabs(table)
                                          else os.path.join(self.fdir, table)))
        if var in INSTRUMENT_RESPONSE_NAMES:
            response = response_files(self.responses_dir, var)
            paths.update(os.fspath(p) for p in (
                response.table, response.logt, response.logne, response.metadata
            ))
        result = {}
        for path in sorted(paths):
            if os.path.isfile(path):
                stat = os.stat(path)
                result[path] = [int(stat.st_size), int(stat.st_mtime_ns)]
        return result

    def _saved_exists(self, var, snap):
        """Check and validate whether a persistent saved quantity is available."""
        # Lightweight in-memory test readers used by the unit suite do not
        # define a persistent run directory and therefore simply have no
        # disk-backed saved cache.
        if not hasattr(self, "saved_dir"):
            return False

        npy_path, json_path = self._saved_paths(var, snap)
        has_npy = os.path.isfile(npy_path)
        has_json = os.path.isfile(json_path)
        if not has_npy and not has_json:
            return False
        if has_npy != has_json:
            raise ValueError(
                f"Incomplete saved cache for {var!r}, snapshot {snap}: "
                f"expected both {os.path.basename(npy_path)!r} and "
                f"{os.path.basename(json_path)!r}"
            )
        source = self._source_signature(snap, var)
        st_json, st_npy = os.stat(json_path), os.stat(npy_path)
        key = (self.registry.resolve(var), snap, st_json.st_mtime_ns,
               st_json.st_size, st_npy.st_mtime_ns, st_npy.st_size)
        if self._saved_validation_cache.get(key) == source:
            return True
        try:
            self._validate_saved(var, snap, npy_path, json_path,
                                 source_signature=source)
        except StaleSavedVariableError as exc:
            warnings.warn(str(exc) + "; recalculating", UserWarning, stacklevel=2)
            return False
        self._saved_validation_cache[key] = source
        return True

    def _validate_saved(self, var, snap, npy_path, json_path, *, source_signature=None):
        """Validate saved-array metadata and storage layout against the current run."""
        canonical = self.registry.resolve(var)
        spec = self.registry.get(canonical)
        if spec is None:
            raise UnknownVariableError(f"Variable {var!r} is not registered")

        with open(json_path, encoding="utf-8") as fp:
            metadata = json.load(fp)

        meta = self._meta(int(snap))
        expected = {
            "format_version": 2,
            "variable": canonical,
            "snapshot": int(snap),
            "location": spec.location,
            "unit_type": spec.unit_type,
            "units": "bifrost",
            "orientation": "bifrost",
            "shape": list(meta.shape),
            "dtype": "float32",
            "order": "F",
        }
        if metadata.get("format_version") != 2:
            raise StaleSavedVariableError(
                f"Old persistent cache format for {canonical!r}, snapshot {snap}"
            )
        if metadata.get("package_version") != __version__ or metadata.get("recipe_version") != 1:
            raise StaleSavedVariableError(
                f"Outdated calculation recipe for {canonical!r}, snapshot {snap}"
            )
        if source_signature is None:
            source_signature = self._source_signature(snap, canonical)
        if metadata.get("source_signature") != source_signature:
            raise StaleSavedVariableError(
                f"Source files changed for {canonical!r}, snapshot {snap}"
            )
        mismatches = []
        for key, expected_value in expected.items():
            if metadata.get(key) != expected_value:
                mismatches.append(
                    f"{key}: expected {expected_value!r}, saved {metadata.get(key)!r}"
                )
        if mismatches:
            raise ValueError(
                "Saved variable metadata mismatch for "
                f"{canonical!r}, snapshot {snap}:\n  " + "\n  ".join(mismatches)
            )

        mm = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        array_mismatches = []
        if list(mm.shape) != list(meta.shape):
            array_mismatches.append(f"array shape: expected {meta.shape!r}, saved {mm.shape!r}")
        if mm.dtype != np.dtype(np.float32):
            array_mismatches.append(f"array dtype: expected float32, saved {mm.dtype}")
        if not mm.flags.f_contiguous:
            array_mismatches.append("array order: expected Fortran-contiguous")
        if array_mismatches:
            raise ValueError(
                "Saved variable array mismatch for "
                f"{canonical!r}, snapshot {snap}:\n  " + "\n  ".join(array_mismatches)
            )
        return metadata

    @staticmethod
    def _slice_native_region(array, selection, shape):
        """Read contiguous or periodically wrapped native windows.

        Wrapped axes are represented by unwrapped slices, not huge full-domain
        reads. A wrapped selection requires at most two contiguous reads per
        axis; native memmap views remain zero-copy for ordinary windows.
        """
        result = array
        for axis, slc in enumerate(selection.normalized(shape).slices):
            n = shape[axis]
            lo, hi = slc.start, slc.stop
            if lo >= 0 and hi <= n:
                slices = [slice(None)] * result.ndim
                slices[axis] = slice(lo, hi)
                result = result[tuple(slices)]
                continue
            if hi - lo > n or lo < -n or hi > 2 * n:
                raise ValueError("Wrapped sampling exceeds one periodic grid")
            offset = lo % n
            pieces = []
            current = offset
            left = hi - lo
            while left:
                end = min(n, current + left)
                slices = [slice(None)] * result.ndim
                slices[axis] = slice(current, end)
                pieces.append(result[tuple(slices)])
                left -= end - current
                current = 0
            result = np.concatenate(pieces, axis=axis)
        return result

    def _read_saved_variable(self, var, snap, selection):
        """Load one validated persistent .npy quantity, preferably as a memmap."""
        npy_path, _ = self._saved_paths(var, snap)
        # _getvar only reaches this method after _saved_exists validated it.
        mm = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        value = np.asarray(self._slice_native_region(mm, selection, self._meta(snap).shape), dtype=np.float32)
        return np.asfortranarray(value)

    def _variable_exists(self, var, snap):
        """Return whether a raw Bifrost variable is available for a snapshot."""
        raw = self._meta(snap).raw.get(var)
        return raw is not None and os.path.isfile(raw.filename)

    def _read_variable(self, var, snap, selection):
        """Read one stored Bifrost variable on a native contiguous selection."""
        meta = self._meta(snap)
        raw = meta.raw[var]
        nx, ny, nz = meta.shape
        dsize = np.dtype(self.dtype).itemsize
        offset = np.int64(nx) * np.int64(ny) * np.int64(nz) * np.int64(raw.index) * np.int64(dsize)
        mm = np.memmap(
            raw.filename,
            dtype=self.dtype,
            mode="r",
            order="F",
            offset=int(offset),
            shape=meta.shape,
        )
        value = np.asarray(self._slice_native_region(mm, selection, self._meta(snap).shape), dtype=np.float32)
        if raw.transform == "exp":
            value = np.exp(value)
        return np.asfortranarray(value)

    # ------------------------------------------------------------------
    # EOS
    # ------------------------------------------------------------------
    def _eos(self, snap):
        """Load and cache the Bifrost rho-e EOS object for a snapshot."""
        meta = self._meta(snap)
        tabparam = str(meta.params.get("tabinputfile", "tabparam.in")).strip().strip("'\"").strip()
        tabparam = os.path.abspath(tabparam if os.path.isabs(tabparam) else os.path.join(self.fdir, tabparam))
        if tabparam in self._eos_cache:
            return self._eos_cache[tabparam]
        eos = EOSTable(
            tabparam,
            fdir=self.fdir,
            dtype=self.base_dtype,
            big_endian=self.big_endian,
        )
        self._eos_cache[tabparam] = eos
        return eos

    # ------------------------------------------------------------------
    # Geometry / sampling
    # ------------------------------------------------------------------
    def _coords_for_location(self, meta, location, *, physical):
        """Return native center/face coordinates for a registered mesh location."""
        self._validate_location(location)
        shifted = LOCATION_SHIFTED_AXES[location]
        coords = []

        # Public physical Y is translated so that the centered Y grid starts
        # at zero.  The same translation is applied to all stagger locations,
        # preserving their relative half-cell geometry.  Native ``br=True``
        # coordinates remain completely untouched.
        y_center_phys = -np.asarray(meta.y, dtype=np.float64)[::-1] * meta.units.length_to_Mm
        y_origin = y_center_phys[0]

        for axis, center, face in zip("xyz", (meta.x, meta.y, meta.z), (meta.xdn, meta.ydn, meta.zdn)):
            arr = face if axis in shifted else center
            arr = np.asarray(arr, dtype=np.float64)
            if physical:
                arr = arr * meta.units.length_to_Mm
                if axis in {"y", "z"}:
                    arr = -arr[::-1]
                if axis == "y":
                    arr = arr - y_origin
            coords.append(arr)
        return tuple(coords)

    def _build_sampling_plan(
        self, meta, *, location, x, y, z, ix, iy, iz,
        ix_br, iy_br, iz_br, method, br=False,
    ):
        """Build a selection in native Bifrost index space.

        ``br=False`` keeps the normal public convention: x/y/z and ix/iy/iz
        refer to the physically reoriented output, while ix_br/iy_br/iz_br
        refer explicitly to native Bifrost indices.

        ``br=True`` changes the output frame itself to native Bifrost
        orientation.  In that case x/y/z and ix/iy/iz naturally refer to
        that native output frame; the *_br index names remain accepted as
        explicit aliases.
        """
        if br:
            native_coords = self._coords_for_location(meta, location, physical=False)
            output_coords = tuple(
                np.asarray(c, dtype=np.float64) * meta.units.length_to_Mm
                for c in native_coords
            )
        else:
            output_coords = self._coords_for_location(meta, location, physical=True)

        # An axis's staggered samples can be offset half a cell from the
        # cell-centered extent.  Allow nearest-neighbor selection at the
        # neighboring center/face boundary without admitting arbitrary
        # out-of-domain coordinates or extrapolating with method="linear".
        center_coords = self._coords_for_location(meta, "center", physical=not br)
        if br:
            center_coords = tuple(
                np.asarray(c, dtype=np.float64) * meta.units.length_to_Mm
                for c in center_coords
            )

        shape = meta.shape
        axis_plans = []
        native_slices = []

        for axis, coord, center_coord, n, csel, isel, brsel in zip(
            "xyz", output_coords, center_coords, shape,
            (x, y, z), (ix, iy, iz), (ix_br, iy_br, iz_br),
        ):
            supplied = sum(v is not None for v in (csel, isel, brsel))
            if supplied > 1:
                raise ValueError(f"Specify only one of {axis}, i{axis}, or i{axis}_br")

            # In native-output mode, *_br is simply an explicit alias for
            # the ordinary output index selector.
            if br and brsel is not None:
                isel_eff, brsel_eff = brsel, None
            else:
                isel_eff, brsel_eff = isel, brsel

            plan = self._axis_plan(
                axis, coord, n, csel, isel_eff, brsel_eff, method,
                center_bounds=(float(center_coord[0]), float(center_coord[-1])),
            )
            axis_plans.append(plan)

            if br:
                # Output and internal index orientation are the same.
                start, stop = plan.pstart, plan.pstop
            else:
                # Convert physical output interval to native Bifrost interval.
                if axis == "x":
                    start, stop = plan.pstart, plan.pstop
                else:
                    start, stop = n - plan.pstop, n - plan.pstart
            native_slices.append(slice(start, stop, 1))

        return SamplingPlan(
            Selection(*native_slices),
            tuple(axis_plans),
            output_coords,
        )

    def _axis_plan(self, axis, coord, n, csel, isel, brsel, method, *, center_bounds=None):
        """Translate one physical/index selector into a native contiguous axis plan."""
        if brsel is not None:
            if not isinstance(brsel, (int, np.integer)):
                raise NotImplementedError(f"i{axis}_br slices are deferred; integer indices work in v0")
            i_br = int(brsel)
            if i_br < 0:
                i_br += n
            if not 0 <= i_br < n:
                raise IndexError(i_br)
            ip = i_br if axis == "x" else n - 1 - i_br
            return AxisPlan(ip, ip + 1, action="scalar")

        if isel is not None:
            if isinstance(isel, (int, np.integer)):
                ip = int(isel)
                if ip < 0:
                    ip += n
                if not 0 <= ip < n:
                    raise IndexError(ip)
                return AxisPlan(ip, ip + 1, action="scalar")

            if not isinstance(isel, slice):
                raise TypeError(f"i{axis} must be int or slice")
            start, stop, step = isel.indices(n)
            if step <= 0:
                raise ValueError("Negative/zero index strides are not supported in v0")
            idx = tuple(range(start, stop, step))
            if not idx:
                return AxisPlan(0, 0, action="keep")
            pstart, pstop = idx[0], idx[-1] + 1
            rel = tuple(i - pstart for i in idx)
            if rel == tuple(range(pstop - pstart)):
                return AxisPlan(pstart, pstop, action="keep")
            return AxisPlan(pstart, pstop, action="indices", indices=rel)

        if csel is not None:
            if isinstance(csel, slice):
                if csel.step is not None:
                    raise ValueError(
                        f"Physical-coordinate slice {axis}=slice(...) does not use step in v0; "
                        f"use i{axis}=slice(..., step) for decimation"
                    )
                lo = coord[0] if csel.start is None else float(csel.start)
                hi = coord[-1] if csel.stop is None else float(csel.stop)
                if hi < lo:
                    lo, hi = hi, lo
                pstart = int(np.searchsorted(coord, lo, side="left"))
                pstop = int(np.searchsorted(coord, hi, side="right"))
                if pstart == pstop:
                    raise ValueError(
                        f"Physical {axis} slice {csel!r} selects no grid points "
                        f"within [{coord[0]}, {coord[-1]}]"
                    )
                return AxisPlan(pstart, pstop, action="keep")

            target = float(csel)
            outside_samples = target < coord[0] or target > coord[-1]
            # Only nearest can clamp, and only within the union of the
            # variable's staggered positions and the corresponding centers.
            if center_bounds is None:
                allowed_lo, allowed_hi = float(coord[0]), float(coord[-1])
            else:
                allowed_lo = min(float(coord[0]), center_bounds[0])
                allowed_hi = max(float(coord[-1]), center_bounds[1])
            if (not np.isfinite(target)
                    or (outside_samples and method != "nearest")
                    or target < allowed_lo or target > allowed_hi):
                raise ValueError(
                    f"Physical {axis}={target} outside grid [{coord[0]}, {coord[-1]}] "
                    f"(nearest permitted within [{allowed_lo}, {allowed_hi}])"
                )
            if method == "nearest":
                ip = int(np.argmin(np.abs(coord - target)))
                return AxisPlan(ip, ip + 1, action="scalar", target=target)

            # linear
            if target <= coord[0]:
                return AxisPlan(0, 1, action="scalar", target=target)
            if target >= coord[-1]:
                return AxisPlan(n - 1, n, action="scalar", target=target)
            hi = int(np.searchsorted(coord, target, side="right"))
            lo = hi - 1
            if coord[lo] == target:
                return AxisPlan(lo, lo + 1, action="scalar", target=target)
            w = (target - coord[lo]) / (coord[hi] - coord[lo])
            return AxisPlan(lo, hi + 1, action="linear", weight=float(w), target=target)

        return AxisPlan(0, n, action="keep")

    def _reverse_yz_f(self, value, *, negate=False):
        """Reverse Y/Z into a new F-contiguous cube, optionally negating values.

        The Y/Z reversal is an involution and is therefore used both for the
        native->physical public transform and by the public stagger interface
        when converting user arrays back to native Bifrost orientation.
        """
        value = np.asarray(value)
        if value.ndim != 3:
            raise ValueError("Y/Z orientation transforms expect a 3-D array")
        out = np.empty(value.shape, dtype=value.dtype, order="F")
        if negate:
            _orient_physical_signed_3d_f(value, out)
        else:
            _orient_physical_3d_f(value, out)
        return out

    def _orient_physical(self, value, spec):
        """Convert a native 3-D result to F-contiguous public orientation.

        Native Bifrost Y/Z are reversed in the public frame.  Y and Z vector
        components also change sign.  Both operations are fused into one
        parallel copy so large cubes are not first exposed as negative-stride
        views and then expensively transposed into C order.
        """
        negate = spec is not None and spec.component in {"y", "z"}
        return self._reverse_yz_f(value, negate=negate)


    @staticmethod
    def _legacy_axis_spacing(meta, axis):
        """Return the scalar spacing used by legacy X/Y stagger derivatives.

        Prefer the scalar dx/dy stored in the IDL parameter file, matching the
        legacy BifrostData object used by the validated stagger_new_v2 code.
        Fall back to the native coordinate difference when unavailable.
        """
        key = ("dx", "dy", "dz")[axis]
        value = meta.params.get(key, None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None

        coord = (meta.x, meta.y, meta.z)[axis]
        if value is None or value == 0.0 or value < 0.0:
            if len(coord) < 2:
                return np.float32(1.0)
            value = float(np.float32(coord[1] - coord[0]))
        return np.float32(value)

    def _local_axis_geometry(self, meta, axis, selection):
        """Extract local center/face geometry needed by one stagger operation."""
        s = selection.normalized(meta.shape).slices[axis]
        center_all = (meta.x, meta.y, meta.z)[axis]
        face_all = (meta.xdn, meta.ydn, meta.zdn)[axis]
        if s.start < 0 or s.stop > meta.shape[axis]:
            if axis == 2:
                raise ValueError("Z is nonperiodic and cannot wrap")
            idx = np.arange(s.start, s.stop) % meta.shape[axis]
            center = center_all[idx]
            face = face_all[idx]
        else:
            center = center_all[s]
            face = face_all[s]

        return center, face

    @staticmethod
    def _cached_z_coefficients(meta, selection, operator):
        """Memoize local Z stencils per fixed mesh, reusing exact global rows.

        Artificial-block boundary rows are computed once with legacy local
        geometry; valid interior rows are copied from global coefficients,
        with identical floating-point values and shifted indices.
        """
        from .stagger import build_nonuniform_coefficients

        s = selection.normalized(meta.shape).iz
        lo, hi = s.start, s.stop
        direction = operator[-2:]
        derivative = operator.startswith("dd")
        global_key = (0, meta.shape[2], direction, derivative)
        if global_key not in meta.z_coefficients:
            meta.z_coefficients[global_key] = build_nonuniform_coefficients(
                meta.z, meta.zdn, direction, derivative
            )
        if (lo, hi) == (0, meta.shape[2]):
            return meta.z_coefficients[global_key]
        key = (lo, hi, direction, derivative)
        if key not in meta.z_coefficients:
            cg, ig = meta.z_coefficients[global_key]
            coeff = cg[lo:hi].copy()
            index = ig[lo:hi].copy() - lo
            unsafe = np.any((index < 0) | (index >= hi - lo), axis=1)
            if np.any(unsafe):
                local_c, local_i = build_nonuniform_coefficients(
                    meta.z[lo:hi], meta.zdn[lo:hi], direction, derivative
                )
                coeff[unsafe] = local_c[unsafe]
                index[unsafe] = local_i[unsafe]
            meta.z_coefficients[key] = coeff, index
        return meta.z_coefficients[key]

    @staticmethod
    def _uniformize_z(value, axis_names, z):
        """Interpolate the selected public array onto a uniform Z grid.

        The output keeps the same number of Z points as the selected native
        domain.  Interpolation is fused in Numba so only the output array is
        allocated; the older vectorized path required two full ``np.take``
        copies plus broadcast temporaries.
        """
        z = np.asarray(z, dtype=np.float64)
        if z.size <= 1:
            return value, z
        znew = np.linspace(z[0], z[-1], z.size)
        if np.allclose(z, znew):
            return value, znew

        axis = axis_names.index("z")
        hi = np.searchsorted(z, znew, side="right")
        hi = np.clip(hi, 1, z.size - 1).astype(np.int64, copy=False)
        lo = (hi - 1).astype(np.int64, copy=False)
        w = ((znew - z[lo]) / (z[hi] - z[lo])).astype(np.float64, copy=False)

        if axis != value.ndim - 1:
            raise RuntimeError("Internal sampling invariant violated: Z must be the last surviving axis")

        out = np.empty_like(value)
        c_layout = bool(value.flags.c_contiguous and not value.flags.f_contiguous)
        if value.ndim == 1:
            _uniform_z_1d(value, lo, hi, w, out)
        elif value.ndim == 2:
            (_uniform_z_2d_c if c_layout else _uniform_z_2d_f)(value, lo, hi, w, out)
        elif value.ndim == 3:
            (_uniform_z_3d_c if c_layout else _uniform_z_3d_f)(value, lo, hi, w, out)
        else:
            raise ValueError("uniform_z supports public arrays with 1-3 dimensions")
        return out, znew
