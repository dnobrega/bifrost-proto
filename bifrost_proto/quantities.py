"""Registry of stored and derived Bifrost analysis quantities.

Quantity recipes operate on native Bifrost arrays through the shared resolver.
Stagger operations are delegated to :mod:`bifrost_proto.stagger`; algebraic
helpers may use NumPy ``out=`` or small fused Numba kernels to avoid temporary
3-D arrays without changing the underlying discrete geometry.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .array_utils import _flat_memory_view
from .geometry import CENTER_OPERATIONS, CURL_OUTPUT_LOCATIONS, FACE_VECTOR_LOCATIONS
from .registry import GLOBAL_REGISTRY, register_variable, register_vector
from .selection import Selection


# =============================================================================
# Generic vector helpers
# =============================================================================
def _component_locations(registry, components):
    """Return the canonical x/y/z stagger locations for a registered vector."""
    locations = []
    for varname in components:
        spec = registry.get(varname)
        if spec is None or spec.location is None:
            raise ValueError(f"Variable {varname!r} has no registered stagger location")
        locations.append(spec.location)
    return tuple(locations)


def _to_center(dd, varname, snap, selection, context):
    """Move one registered variable to cell centers using its native location."""
    spec = dd.registry.get(varname)
    if spec is None or spec.location is None:
        raise ValueError(f"Variable {varname!r} has no registered stagger location")
    try:
        operations = CENTER_OPERATIONS[spec.location]
    except KeyError as exc:
        raise ValueError(
            f"Variable {varname!r} has unsupported location {spec.location!r}"
        ) from exc
    return dd._ops(operations, varname, snap, selection, context)


@njit(cache=True, parallel=True)
def _norm2_flat(x, y, out):
    """Fused 2-component Euclidean norm on contiguous 1-D views."""
    for i in prange(out.size):
        xv = x[i]
        yv = y[i]
        out[i] = np.sqrt(xv * xv + yv * yv)


@njit(cache=True, parallel=True)
def _norm3_flat(x, y, z, out):
    """Fused 3-component Euclidean norm on contiguous 1-D views."""
    for i in prange(out.size):
        xv = x[i]
        yv = y[i]
        zv = z[i]
        out[i] = np.sqrt(xv * xv + yv * yv + zv * zv)


@njit(cache=True, parallel=True)
def _norm2_strided(x, y, out):
    """Fused norm for arbitrary 3-D strided views without copying inputs."""
    nx, ny, nz = x.shape
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            xv = x[i, j, k]
            yv = y[i, j, k]
            out[i, j, k] = np.sqrt(xv * xv + yv * yv)


@njit(cache=True, parallel=True)
def _norm3_strided(x, y, z, out):
    """Fused norm for arbitrary 3-D strided views without copying inputs."""
    nx, ny, nz = x.shape
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            xv = x[i, j, k]
            yv = y[i, j, k]
            zv = z[i, j, k]
            out[i, j, k] = np.sqrt(xv * xv + yv * yv + zv * zv)


@njit(cache=True, parallel=True)
def _dot3_flat(ax, ay, az, bx, by, bz, out):
    """Fused three-component dot product on matching contiguous 1-D views."""
    for p in prange(out.size):
        out[p] = ax[p] * bx[p] + ay[p] * by[p] + az[p] * bz[p]


@njit(cache=True, parallel=True)
def _dot3_native(ax, ay, az, bx, by, bz, out):
    """Fused dot product on arbitrary 3-D views, parallel over Y-Z rows."""
    nx, ny, nz = out.shape
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            out[i, j, k] = (
                ax[i, j, k] * bx[i, j, k]
                + ay[i, j, k] * by[i, j, k]
                + az[i, j, k] * bz[i, j, k]
            )


@njit(cache=True, parallel=True)
def _negative_cross_term_flat(a, b, c, d, out):
    """Evaluate ``-(a*b-c*d)`` on matching contiguous 1-D views."""
    for p in prange(out.size):
        out[p] = -(a[p] * b[p] - c[p] * d[p])


@njit(cache=True, parallel=True)
def _negative_cross_term_native(a, b, c, d, out):
    """Evaluate ``-(a*b-c*d)`` on arbitrary 3-D views."""
    nx, ny, nz = out.shape
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            out[i, j, k] = -(
                a[i, j, k] * b[i, j, k]
                - c[i, j, k] * d[i, j, k]
            )


@njit(cache=True, parallel=True)
def _ideal_poynting_component_flat(ux, uy, uz, bx, by, bz, component, out):
    """Compute one centered ideal-MHD Poynting component on flat views."""
    for p in prange(out.size):
        u0 = ux[p]
        u1 = uy[p]
        u2 = uz[p]
        b0 = bx[p]
        b1 = by[p]
        b2 = bz[p]
        if component == 0:
            out[p] = u0 * (b1 * b1 + b2 * b2) - b0 * (u1 * b1 + u2 * b2)
        elif component == 1:
            out[p] = u1 * (b0 * b0 + b2 * b2) - b1 * (u0 * b0 + u2 * b2)
        else:
            out[p] = u2 * (b0 * b0 + b1 * b1) - b2 * (u0 * b0 + u1 * b1)


@njit(cache=True, parallel=True)
def _ideal_poynting_component_native(ux, uy, uz, bx, by, bz, component, out):
    """Compute one centered ideal-MHD Poynting component on arbitrary 3-D views."""
    nx, ny, nz = out.shape
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            u0 = ux[i, j, k]
            u1 = uy[i, j, k]
            u2 = uz[i, j, k]
            b0 = bx[i, j, k]
            b1 = by[i, j, k]
            b2 = bz[i, j, k]
            if component == 0:
                out[i, j, k] = (
                    u0 * (b1 * b1 + b2 * b2)
                    - b0 * (u1 * b1 + u2 * b2)
                )
            elif component == 1:
                out[i, j, k] = (
                    u1 * (b0 * b0 + b2 * b2)
                    - b1 * (u0 * b0 + u2 * b2)
                )
            else:
                out[i, j, k] = (
                    u2 * (b0 * b0 + b1 * b1)
                    - b2 * (u0 * b0 + u1 * b1)
                )


@njit(cache=True, parallel=True)
def _ideal_poynting_emerg_z_flat(uz, bx, by, out):
    """Compute ``uz*(bx^2+by^2)`` on matching contiguous flat views."""
    for p in prange(out.size):
        b0 = bx[p]
        b1 = by[p]
        out[p] = uz[p] * (b0 * b0 + b1 * b1)


@njit(cache=True, parallel=True)
def _ideal_poynting_emerg_z_native(uz, bx, by, out):
    """Compute ``uz*(bx^2+by^2)`` on arbitrary 3-D views."""
    nx, ny, nz = out.shape
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            b0 = bx[i, j, k]
            b1 = by[i, j, k]
            out[i, j, k] = uz[i, j, k] * (b0 * b0 + b1 * b1)


@njit(cache=True, parallel=True)
def _ideal_poynting_shear_z_flat(ux, uy, bx, by, bz, out):
    """Compute ``-bz*(ux*bx+uy*by)`` on matching contiguous flat views."""
    for p in prange(out.size):
        out[p] = -bz[p] * (ux[p] * bx[p] + uy[p] * by[p])


@njit(cache=True, parallel=True)
def _ideal_poynting_shear_z_native(ux, uy, bx, by, bz, out):
    """Compute ``-bz*(ux*bx+uy*by)`` on arbitrary 3-D views."""
    nx, ny, nz = out.shape
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            out[i, j, k] = -bz[i, j, k] * (
                ux[i, j, k] * bx[i, j, k]
                + uy[i, j, k] * by[i, j, k]
            )


@njit(cache=True, parallel=True)
def _acute_cosine_flat(jx, jy, jz, bx, by, bz, out):
    """Compute ``|J.B|/(|J||B|)`` on matching contiguous flat views."""
    for p in prange(out.size):
        j0 = jx[p]
        j1 = jy[p]
        j2 = jz[p]
        b0 = bx[p]
        b1 = by[p]
        b2 = bz[p]
        dot = j0 * b0 + j1 * b1 + j2 * b2
        modj = np.sqrt(j0 * j0 + j1 * j1 + j2 * j2)
        modb = np.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
        denominator = modj * modb
        if denominator == 0.0:
            out[p] = np.nan
        else:
            cosine = abs(dot) / denominator
            if cosine < 0.0:
                cosine = 0.0
            elif cosine > 1.0:
                cosine = 1.0
            out[p] = cosine


@njit(cache=True, parallel=True)
def _acute_cosine_native(jx, jy, jz, bx, by, bz, out):
    """Compute ``|J.B|/(|J||B|)`` on arbitrary 3-D views."""
    nx, ny, nz = out.shape
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            j0 = jx[i, j, k]
            j1 = jy[i, j, k]
            j2 = jz[i, j, k]
            b0 = bx[i, j, k]
            b1 = by[i, j, k]
            b2 = bz[i, j, k]
            dot = j0 * b0 + j1 * b1 + j2 * b2
            modj = np.sqrt(j0 * j0 + j1 * j1 + j2 * j2)
            modb = np.sqrt(b0 * b0 + b1 * b1 + b2 * b2)
            denominator = modj * modb
            if denominator == 0.0:
                out[i, j, k] = np.nan
            else:
                cosine = abs(dot) / denominator
                if cosine < 0.0:
                    cosine = 0.0
                elif cosine > 1.0:
                    cosine = 1.0
                out[i, j, k] = cosine


@njit(cache=True, parallel=True)
def _timescale_flat(e, q, out):
    """Compute ``e/|q|`` on matching contiguous flat views."""
    for p in prange(out.size):
        qv = q[p]
        if qv == 0.0:
            out[p] = np.inf
        else:
            out[p] = e[p] / abs(qv)


@njit(cache=True, parallel=True)
def _timescale_native(e, q, out):
    """Compute ``e/|q|`` on arbitrary 3-D views."""
    nx, ny, nz = out.shape
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            qv = q[i, j, k]
            if qv == 0.0:
                out[i, j, k] = np.inf
            else:
                out[i, j, k] = e[i, j, k] / abs(qv)



def _flat_compatible(*arrays):
    """Return True when all arrays share one contiguous memory layout."""
    all_c = all(a.flags.c_contiguous for a in arrays)
    all_f = all(a.flags.f_contiguous for a in arrays)
    return all_c or all_f


def _norm_arrays(parts):
    """Euclidean norm with one owned output buffer and fused Numba kernels.

    C- and F-contiguous arrays use a fast flat kernel. Arbitrary 3-D strided
    views use a stride-aware kernel directly, so no contiguous input copies are
    introduced. Arrays supplied by the resolver are never modified in-place.
    """
    parts = tuple(parts)
    if not parts:
        raise ValueError("At least one array is required")
    if len(parts) not in (1, 2, 3):
        raise ValueError("Vector magnitude supports one, two, or three components")

    shape = parts[0].shape
    if any(part.shape != shape for part in parts[1:]):
        raise ValueError("All magnitude components must have the same shape")

    out = np.empty_like(parts[0])
    if len(parts) == 1:
        np.abs(parts[0], out=out)
        return out

    all_c = all(part.flags.c_contiguous for part in parts) and out.flags.c_contiguous
    all_f = all(part.flags.f_contiguous for part in parts) and out.flags.f_contiguous

    if all_c or all_f:
        flat_parts = tuple(_flat_memory_view(part) for part in parts)
        flat_out = _flat_memory_view(out)
        if len(parts) == 2:
            _norm2_flat(flat_parts[0], flat_parts[1], flat_out)
        else:
            _norm3_flat(flat_parts[0], flat_parts[1], flat_parts[2], flat_out)
        return out

    if parts[0].ndim != 3 or any(part.ndim != 3 for part in parts):
        raise ValueError(
            "Non-contiguous magnitude inputs are supported only for 3-D arrays"
        )

    if len(parts) == 2:
        _norm2_strided(parts[0], parts[1], out)
    else:
        _norm3_strided(parts[0], parts[1], parts[2], out)
    return out


def _vector_magnitude(dd, snap, selection, context, components):
    """Center vector components and return their fused Euclidean magnitude."""
    parts = tuple(
        _to_center(dd, varname, snap, selection, context)
        for varname in components
    )
    return _norm_arrays(parts)


def _register_magnitude(name, components, unit_type, description, aliases=()):
    """Register a scalar magnitude recipe for explicit vector components."""
    components = tuple(components)
    # Validate at import time so an invalid official definition fails loudly.
    for varname, location in zip(
        components, _component_locations(GLOBAL_REGISTRY, components)
    ):
        if location not in CENTER_OPERATIONS:
            raise ValueError(
                f"Cannot center {varname!r} from unsupported location {location!r}"
            )

    @register_variable(
        name,
        location="center",
        unit_type=unit_type,
        aliases=aliases,
        description=description,
    )
    def getter(dd, snap, selection, context, _components=components):
        """Resolver getter generated by a quantity-registration helper."""
        return _vector_magnitude(dd, snap, selection, context, _components)

    return getter


def _vector_divergence(dd, snap, selection, context, components):
    """Evaluate the divergence of a face-centered vector at cell centers."""
    components = tuple(components)
    topology = _component_locations(dd.registry, components)
    if topology != FACE_VECTOR_LOCATIONS:
        raise ValueError(
            "Divergence helper requires components on xface/yface/zface; "
            f"got {topology!r}"
        )

    vx, vy, vz = components
    dx = dd._op("ddxup", vx, snap, selection, context)
    dy = dd._op("ddyup", vy, snap, selection, context)
    dz = dd._op("ddzup", vz, snap, selection, context)
    out = np.empty_like(dx)
    np.add(dx, dy, out=out)
    np.add(out, dz, out=out)
    return out


def _register_divergence(name, components, unit_type, description, aliases=()):
    """Register a divergence recipe after validating face-vector topology."""
    components = tuple(components)
    topology = _component_locations(GLOBAL_REGISTRY, components)
    if topology != FACE_VECTOR_LOCATIONS:
        raise ValueError(
            f"Cannot register {name!r}: divergence requires an xface/yface/zface "
            f"vector, got {topology!r}"
        )

    @register_variable(
        name,
        location="center",
        unit_type=unit_type,
        aliases=aliases,
        description=description,
    )
    def getter(dd, snap, selection, context, _components=components):
        """Resolver getter generated by a quantity-registration helper."""
        return _vector_divergence(dd, snap, selection, context, _components)

    return getter


def _curl_component(dd, snap, selection, context, components, component):
    """Evaluate one topology-aware component of a registered vector curl."""
    components = tuple(components)
    topology = _component_locations(dd.registry, components)
    if topology not in CURL_OUTPUT_LOCATIONS:
        raise ValueError(
            "Curl helper requires either xface/yface/zface or "
            f"xedge/yedge/zedge components; got {topology!r}"
        )

    vx, vy, vz = components
    if topology == FACE_VECTOR_LOCATIONS:
        terms = {
            "x": (("ddydn", vz), ("ddzdn", vy)),
            "y": (("ddzdn", vx), ("ddxdn", vz)),
            "z": (("ddxdn", vy), ("ddydn", vx)),
        }
    else:
        terms = {
            "x": (("ddyup", vz), ("ddzup", vy)),
            "y": (("ddzup", vx), ("ddxup", vz)),
            "z": (("ddxup", vy), ("ddyup", vx)),
        }
    try:
        (op_a, var_a), (op_b, var_b) = terms[component]
    except KeyError as exc:
        raise ValueError(f"Unknown curl component {component!r}") from exc

    a = dd._op(op_a, var_a, snap, selection, context)
    b = dd._op(op_b, var_b, snap, selection, context)
    out = np.empty_like(a)
    np.subtract(a, b, out=out)
    return out


def _register_curl(basename, components, unit_type, description):
    """Register the three curl components for a face- or edge-centered vector."""
    components = tuple(components)
    topology = _component_locations(GLOBAL_REGISTRY, components)
    try:
        output_locations = CURL_OUTPUT_LOCATIONS[topology]
    except KeyError as exc:
        raise ValueError(
            f"Cannot register {basename!r}: curl requires a face or edge vector, "
            f"got {topology!r}"
        ) from exc

    for component, location in zip("xyz", output_locations):
        name = f"{basename}_{component}"
        component_description = (
            f"{component.upper()} component of {description.rstrip('.')}."
        )

        def getter(
            dd, snap, selection, context,
            _components=components, _component=component,
        ):
            """Resolver getter generated by a quantity-registration helper."""
            return _curl_component(
                dd, snap, selection, context, _components, _component
            )

        register_variable(
            name,
            location=location,
            unit_type=unit_type,
            component=component,
            description=component_description,
        )(getter)


def _dot3_arrays(ax, ay, az, bx, by, bz):
    """Three-component dot product with one owned output buffer."""
    out = np.empty_like(ax)
    arrays = (ax, ay, az, bx, by, bz, out)
    if _flat_compatible(*arrays):
        flat = tuple(_flat_memory_view(a) for a in arrays)
        _dot3_flat(*flat)
    else:
        _dot3_native(ax, ay, az, bx, by, bz, out)
    return out


def _negative_cross_term_arrays(a, b, c, d):
    """Return ``-(a*b-c*d)`` without allocating arithmetic temporaries."""
    out = np.empty_like(a)
    arrays = (a, b, c, d, out)
    if _flat_compatible(*arrays):
        flat = tuple(_flat_memory_view(v) for v in arrays)
        _negative_cross_term_flat(*flat)
    else:
        _negative_cross_term_native(a, b, c, d, out)
    return out


def _ideal_poynting_component_arrays(ux, uy, uz, bx, by, bz, component):
    """Return one centered ideal-MHD Poynting-flux component."""
    out = np.empty_like(ux)
    arrays = (ux, uy, uz, bx, by, bz, out)
    if _flat_compatible(*arrays):
        flat = tuple(_flat_memory_view(v) for v in arrays)
        _ideal_poynting_component_flat(*flat[:-1], component, flat[-1])
    else:
        _ideal_poynting_component_native(ux, uy, uz, bx, by, bz, component, out)
    return out


def _ideal_poynting_emerg_z_arrays(uz, bx, by):
    """Return the centered vertical emergence contribution to ideal Poynting flux."""
    out = np.empty_like(uz)
    arrays = (uz, bx, by, out)
    if _flat_compatible(*arrays):
        flat = tuple(_flat_memory_view(v) for v in arrays)
        _ideal_poynting_emerg_z_flat(*flat)
    else:
        _ideal_poynting_emerg_z_native(uz, bx, by, out)
    return out


def _ideal_poynting_shear_z_arrays(ux, uy, bx, by, bz):
    """Return the centered vertical shear contribution to ideal Poynting flux."""
    out = np.empty_like(ux)
    arrays = (ux, uy, bx, by, bz, out)
    if _flat_compatible(*arrays):
        flat = tuple(_flat_memory_view(v) for v in arrays)
        _ideal_poynting_shear_z_flat(*flat)
    else:
        _ideal_poynting_shear_z_native(ux, uy, bx, by, bz, out)
    return out


# =============================================================================
# Raw/native metadata
# =============================================================================
register_variable(
    "r",
    location="center",
    unit_type="density",
    description="Mass density.",
)
register_variable(
    "e",
    location="center",
    unit_type="energy_density",
    description="Internal energy per unit volume.",
)

# Canonical vector naming is <quantity>_x/y/z. Bifrost's compact on-disk
# names remain fully supported as aliases.
register_vector(
    "p",
    locations=("xface", "yface", "zface"),
    unit_type="momentum_density",
    description="Momentum density",
    storage_names=("px", "py", "pz"),
    aliases=(("px",), ("py",), ("pz",)),
)
register_vector(
    "b",
    locations=("xface", "yface", "zface"),
    unit_type="magnetic_field",
    description="Magnetic field",
    storage_names=("bx", "by", "bz"),
    aliases=(("bx",), ("by",), ("bz",)),
)

# The total electric field is stored-only. A snapshot does not contain enough
# information to reconstruct Bifrost's non-ideal/hyperdiffusive contribution.
register_vector(
    "e",
    locations=("xedge", "yedge", "zedge"),
    unit_type="electric_field",
    description="Total electric field, including non-ideal terms",
    storage_names=("ex", "ey", "ez"),
    aliases=(("ex",), ("ey",), ("ez",)),
)

_STORED_Q_DESCRIPTIONS = {
    "qjoule": "Heating due to Joule dissipation.",
    "qvisc": "Heating due to viscous dissipation.",
    "qthin": "Cooling due to optically thin radiation.",
    "qspitz": "Heating/cooling due to Spitzer thermal conduction.",
    "qrad": "Signed radiative heating/cooling from the full radiative-transfer module.",
    "qgenrad": "Signed heating/cooling from Bifrost's generalized radiation module.",
}
for _name, _description in _STORED_Q_DESCRIPTIONS.items():
    register_variable(
        _name,
        location="center",
        unit_type="energy_rate_density",
        description=_description,
    )

# HION/helium metadata. These populations/electron density are stored in
# physical number-density units, while hiontg is stored in K.
register_variable(
    "hiontg",
    location="center",
    unit_type="temperature",
    description="HION non-equilibrium hydrogen temperature.",
)
register_variable(
    "hionne",
    location="center",
    unit_type="number_density",
    description="HION non-equilibrium electron number density.",
)
_HION_DESCRIPTIONS = {
    "n1": "Non-equilibrium hydrogen population n1.",
    "n2": "Non-equilibrium hydrogen population n2.",
    "n3": "Non-equilibrium hydrogen population n3.",
    "n4": "Non-equilibrium hydrogen population n4.",
    "n5": "Non-equilibrium hydrogen population n5.",
    "n6": "Non-equilibrium hydrogen population n6.",
    "nh2": "Non-equilibrium molecular-hydrogen number density.",
    "nhe1": "Non-equilibrium neutral-helium population.",
    "nhe2": "Non-equilibrium singly ionized helium population.",
    "nhe3": "Non-equilibrium doubly ionized helium population.",
}
for _name, _description in _HION_DESCRIPTIONS.items():
    register_variable(
        _name,
        location="center",
        unit_type="number_density",
        description=_description,
    )


# =============================================================================
# Hydrodynamic primitives and velocity
# =============================================================================
@register_variable(
    "ee",
    location="center",
    unit_type="specific_energy",
    description="Specific internal energy, e/r.",
)
def get_ee(dd, snap, selection, context):
    """Resolver getter for the registered ``ee`` quantity; physical meaning and units are defined by its registry metadata."""
    e = dd._getvar("e", snap, selection, context)
    r = dd._getvar("r", snap, selection, context)
    return e / r


@register_variable(
    "u_x",
    location="xface",
    unit_type="velocity",
    component="x",
    aliases=("ux",),
    description="X component of the plasma velocity.",
)
def get_u_x(dd, snap, selection, context):
    """Resolver getter for the registered ``u_x`` quantity; physical meaning and units are defined by its registry metadata."""
    p = dd._getvar("p_x", snap, selection, context)
    rho = dd._op("xdn", "r", snap, selection, context)
    return p / rho


@register_variable(
    "u_y",
    location="yface",
    unit_type="velocity",
    component="y",
    aliases=("uy",),
    description="Y component of the plasma velocity.",
)
def get_u_y(dd, snap, selection, context):
    """Resolver getter for the registered ``u_y`` quantity; physical meaning and units are defined by its registry metadata."""
    p = dd._getvar("p_y", snap, selection, context)
    rho = dd._op("ydn", "r", snap, selection, context)
    return p / rho


@register_variable(
    "u_z",
    location="zface",
    unit_type="velocity",
    component="z",
    aliases=("uz",),
    description="Z component of the plasma velocity.",
)
def get_u_z(dd, snap, selection, context):
    """Resolver getter for the registered ``u_z`` quantity; physical meaning and units are defined by its registry metadata."""
    p = dd._getvar("p_z", snap, selection, context)
    rho = dd._op("zdn", "r", snap, selection, context)
    return p / rho


_register_magnitude(
    "modu",
    ("u_x", "u_y", "u_z"),
    "velocity",
    "Magnitude of the velocity, |u|.",
)

_register_magnitude(
    "u_h",
    ("u_x", "u_y"),
    "velocity",
    "Horizontal magnitude of the velocity, sqrt(u_x^2 + u_y^2).",
    aliases=("uh",),
)


# =============================================================================
# Magnetic field
# =============================================================================
_register_magnitude(
    "modb",
    ("b_x", "b_y", "b_z"),
    "magnetic_field",
    "Magnitude of the magnetic field, |B|.",
)

_register_magnitude(
    "b_h",
    ("b_x", "b_y"),
    "magnetic_field",
    "Horizontal magnitude of the magnetic field, sqrt(Bx^2 + By^2).",
    aliases=("bh",),
)

@register_variable(
    "bhat_x",
    location="xface",
    unit_type="dimensionless",
    component="x",
    aliases=("bxob",),
    description="X component of the magnetic-field unit vector, Bx/xdn(|B|).",
)
def get_bhat_x(dd, snap, selection, context):
    """Resolver getter for the registered ``bhat_x`` quantity; physical meaning and units are defined by its registry metadata."""
    bx = dd._getvar("b_x", snap, selection, context)
    b = dd._op("xdn", "modb", snap, selection, context)
    return bx / b


@register_variable(
    "bhat_y",
    location="yface",
    unit_type="dimensionless",
    component="y",
    aliases=("byob",),
    description="Y component of the magnetic-field unit vector, By/ydn(|B|).",
)
def get_bhat_y(dd, snap, selection, context):
    """Resolver getter for the registered ``bhat_y`` quantity; physical meaning and units are defined by its registry metadata."""
    by = dd._getvar("b_y", snap, selection, context)
    b = dd._op("ydn", "modb", snap, selection, context)
    return by / b


@register_variable(
    "bhat_z",
    location="zface",
    unit_type="dimensionless",
    component="z",
    aliases=("bzob",),
    description="Z component of the magnetic-field unit vector, Bz/zdn(|B|).",
)
def get_bhat_z(dd, snap, selection, context):
    """Resolver getter for the registered ``bhat_z`` quantity; physical meaning and units are defined by its registry metadata."""
    bz = dd._getvar("b_z", snap, selection, context)
    b = dd._op("zdn", "modb", snap, selection, context)
    return bz / b


_register_divergence(
    "divb",
    ("b_x", "b_y", "b_z"),
    "magnetic_gradient",
    "Magnetic-field divergence, div(B).",
)

_register_curl(
    "curlb",
    ("b_x", "b_y", "b_z"),
    "magnetic_gradient",
    "curl(B)",
)

_register_magnitude(
    "modcurlb",
    ("curlb_x", "curlb_y", "curlb_z"),
    "magnetic_gradient",
    "Magnitude of the magnetic-field curl.",
)

@register_variable(
    "ldivb",
    location="center",
    unit_type="length",
    description="Characteristic magnetic divergence length, |B|/|div(B)|.",
)
def get_ldivb(dd, snap, selection, context):
    """Resolver getter for the registered ``ldivb`` quantity; physical meaning and units are defined by its registry metadata."""
    divb = dd._getvar("divb", snap, selection, context)
    b = dd._getvar("modb", snap, selection, context)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(b, np.abs(divb))


@register_variable(
    "lcurlb",
    location="center",
    unit_type="length",
    description="Characteristic magnetic curl length, |B|/|curl(B)|.",
)
def get_lcurlb(dd, snap, selection, context):
    """Resolver getter for the registered ``lcurlb`` quantity; physical meaning and units are defined by its registry metadata."""
    curlb = dd._getvar("modcurlb", snap, selection, context)
    b = dd._getvar("modb", snap, selection, context)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(b, curlb)


# Current density. Numerically the native components are curl(B); the unit
# conversion differs and includes c/(4*pi), hence a separate vector quantity.
@register_variable(
    "j_x",
    location="xedge",
    unit_type="current_density",
    component="x",
    aliases=("jx",),
    description="X component of the electric current density.",
)
def get_j_x(dd, snap, selection, context):
    """Resolver getter for the registered ``j_x`` quantity; physical meaning and units are defined by its registry metadata."""
    return dd._getvar("curlb_x", snap, selection, context)


@register_variable(
    "j_y",
    location="yedge",
    unit_type="current_density",
    component="y",
    aliases=("jy",),
    description="Y component of the electric current density.",
)
def get_j_y(dd, snap, selection, context):
    """Resolver getter for the registered ``j_y`` quantity; physical meaning and units are defined by its registry metadata."""
    return dd._getvar("curlb_y", snap, selection, context)


@register_variable(
    "j_z",
    location="zedge",
    unit_type="current_density",
    component="z",
    aliases=("jz",),
    description="Z component of the electric current density.",
)
def get_j_z(dd, snap, selection, context):
    """Resolver getter for the registered ``j_z`` quantity; physical meaning and units are defined by its registry metadata."""
    return dd._getvar("curlb_z", snap, selection, context)


_register_magnitude(
    "modj",
    ("j_x", "j_y", "j_z"),
    "current_density",
    "Magnitude of the electric current density.",
)

@register_variable(
    "theta_jb",
    location="center",
    unit_type="angle",
    description="Acute angle between J and B, arccos(|J dot B|/(|J||B|)); undefined points are NaN.",
)
def get_theta_jb(dd, snap, selection, context):
    """Resolver getter for the registered ``theta_jb`` quantity; physical meaning and units are defined by its registry metadata."""
    bx = dd._op("xup", "b_x", snap, selection, context)
    by = dd._op("yup", "b_y", snap, selection, context)
    bz = dd._op("zup", "b_z", snap, selection, context)
    jx = dd._ops(("yup", "zup"), "j_x", snap, selection, context)
    jy = dd._ops(("zup", "xup"), "j_y", snap, selection, context)
    jz = dd._ops(("xup", "yup"), "j_z", snap, selection, context)

    # Fuse dot product, both magnitudes, division, absolute value, clipping,
    # and the undefined-point test in Numba.  Keep arccos as NumPy's ufunc: it
    # is substantially faster than evaluating the transcendental in a scalar
    # Numba loop while still requiring only the final output cube.
    cosine = np.empty_like(bx)
    arrays = (jx, jy, jz, bx, by, bz, cosine)
    if _flat_compatible(*arrays):
        flat = tuple(_flat_memory_view(v) for v in arrays)
        _acute_cosine_flat(*flat)
    else:
        _acute_cosine_native(jx, jy, jz, bx, by, bz, cosine)
    np.arccos(cosine, out=cosine)
    return cosine


# =============================================================================
# Velocity decomposition relative to B
# =============================================================================
@register_variable(
    "uparb",
    location="center",
    unit_type="velocity",
    description="Signed velocity component parallel to the magnetic field, u dot B / |B|.",
)
def get_uparb(dd, snap, selection, context):
    """Resolver getter for the registered ``uparb`` quantity; physical meaning and units are defined by its registry metadata."""
    # Preserve the legacy IDL operation order exactly:
    #   [xup(ux*bx) + yup(uy*by) + zup(uz*bz)] / |B|
    ubx = dd._op_product("xup", "u_x", "b_x", snap, selection, context)
    uby = dd._op_product("yup", "u_y", "b_y", snap, selection, context)
    ubz = dd._op_product("zup", "u_z", "b_z", snap, selection, context)
    b = dd._getvar("modb", snap, selection, context)
    out = np.empty_like(ubx)
    np.add(ubx, uby, out=out)
    np.add(out, ubz, out=out)
    np.divide(out, b, out=out)
    return out


@register_variable(
    "uparb_x",
    location="xface",
    unit_type="velocity",
    component="x",
    description="X component of the velocity vector parallel to B.",
)
def get_uparb_x(dd, snap, selection, context):
    """Resolver getter for the registered ``uparb_x`` quantity; physical meaning and units are defined by its registry metadata."""
    upar = dd._op("xdn", "uparb", snap, selection, context)
    bhat = dd._getvar("bhat_x", snap, selection, context)
    return upar * bhat


@register_variable(
    "uparb_y",
    location="yface",
    unit_type="velocity",
    component="y",
    description="Y component of the velocity vector parallel to B.",
)
def get_uparb_y(dd, snap, selection, context):
    """Resolver getter for the registered ``uparb_y`` quantity; physical meaning and units are defined by its registry metadata."""
    upar = dd._op("ydn", "uparb", snap, selection, context)
    bhat = dd._getvar("bhat_y", snap, selection, context)
    return upar * bhat


@register_variable(
    "uparb_z",
    location="zface",
    unit_type="velocity",
    component="z",
    description="Z component of the velocity vector parallel to B.",
)
def get_uparb_z(dd, snap, selection, context):
    """Resolver getter for the registered ``uparb_z`` quantity; physical meaning and units are defined by its registry metadata."""
    upar = dd._op("zdn", "uparb", snap, selection, context)
    bhat = dd._getvar("bhat_z", snap, selection, context)
    return upar * bhat


@register_variable(
    "uperpb_x",
    location="xface",
    unit_type="velocity",
    component="x",
    description="X component of the velocity perpendicular to B.",
)
def get_uperpb_x(dd, snap, selection, context):
    """Resolver getter for the registered ``uperpb_x`` quantity; physical meaning and units are defined by its registry metadata."""
    return dd._getvar("u_x", snap, selection, context) - dd._getvar(
        "uparb_x", snap, selection, context
    )


@register_variable(
    "uperpb_y",
    location="yface",
    unit_type="velocity",
    component="y",
    description="Y component of the velocity perpendicular to B.",
)
def get_uperpb_y(dd, snap, selection, context):
    """Resolver getter for the registered ``uperpb_y`` quantity; physical meaning and units are defined by its registry metadata."""
    return dd._getvar("u_y", snap, selection, context) - dd._getvar(
        "uparb_y", snap, selection, context
    )


@register_variable(
    "uperpb_z",
    location="zface",
    unit_type="velocity",
    component="z",
    description="Z component of the velocity perpendicular to B.",
)
def get_uperpb_z(dd, snap, selection, context):
    """Resolver getter for the registered ``uperpb_z`` quantity; physical meaning and units are defined by its registry metadata."""
    return dd._getvar("u_z", snap, selection, context) - dd._getvar(
        "uparb_z", snap, selection, context
    )


_register_magnitude(
    "uperpb",
    ("uperpb_x", "uperpb_y", "uperpb_z"),
    "velocity",
    "Magnitude of the velocity perpendicular to the magnetic field, |u_perp,B|.",
)


# =============================================================================
# Velocity derivatives
# =============================================================================
_register_divergence(
    "divu",
    ("u_x", "u_y", "u_z"),
    "inverse_time",
    "Velocity divergence, div(u).",
)

_register_curl(
    "curlu",
    ("u_x", "u_y", "u_z"),
    "inverse_time",
    "curl(u)",
)

_register_magnitude(
    "modcurlu",
    ("curlu_x", "curlu_y", "curlu_z"),
    "inverse_time",
    "Magnitude of the velocity curl.",
)

@register_variable(
    "lcurlu",
    location="center",
    unit_type="length",
    description="Characteristic velocity curl length, |u|/|curl(u)|.",
)
def get_lcurlu(dd, snap, selection, context):
    """Resolver getter for the registered ``lcurlu`` quantity; physical meaning and units are defined by its registry metadata."""
    curlu = dd._getvar("modcurlu", snap, selection, context)
    u = dd._getvar("modu", snap, selection, context)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.divide(u, curlu)


@register_variable(
    "ldivu",
    location="center",
    unit_type="length",
    description="Characteristic velocity divergence length, |u|/|div(u)|.",
)
def get_ldivu(dd, snap, selection, context):
    """Resolver getter for the registered ``ldivu`` quantity; physical meaning and units are defined by its registry metadata."""
    divu = dd._getvar("divu", snap, selection, context)
    u = dd._getvar("modu", snap, selection, context)
    out = np.empty_like(divu)
    np.abs(divu, out=out)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(u, out, out=out)
    return out


# =============================================================================
# Ideal electric field
# =============================================================================
@register_variable(
    "e_ideal_x",
    location="xedge",
    unit_type="electric_field",
    component="x",
    aliases=("ex_ideal",),
    description="X component of the ideal electric field, -u cross B.",
)
def get_e_ideal_x(dd, snap, selection, context):
    """Resolver getter for the registered ``e_ideal_x`` quantity; physical meaning and units are defined by its registry metadata."""
    uy = dd._op("zdn", "u_y", snap, selection, context)
    bz = dd._op("ydn", "b_z", snap, selection, context)
    uz = dd._op("ydn", "u_z", snap, selection, context)
    by = dd._op("zdn", "b_y", snap, selection, context)
    return _negative_cross_term_arrays(uy, bz, uz, by)


@register_variable(
    "e_ideal_y",
    location="yedge",
    unit_type="electric_field",
    component="y",
    aliases=("ey_ideal",),
    description="Y component of the ideal electric field, -u cross B.",
)
def get_e_ideal_y(dd, snap, selection, context):
    """Resolver getter for the registered ``e_ideal_y`` quantity; physical meaning and units are defined by its registry metadata."""
    uz = dd._op("xdn", "u_z", snap, selection, context)
    bx = dd._op("zdn", "b_x", snap, selection, context)
    ux = dd._op("zdn", "u_x", snap, selection, context)
    bz = dd._op("xdn", "b_z", snap, selection, context)
    return _negative_cross_term_arrays(uz, bx, ux, bz)


@register_variable(
    "e_ideal_z",
    location="zedge",
    unit_type="electric_field",
    component="z",
    aliases=("ez_ideal",),
    description="Z component of the ideal electric field, -u cross B.",
)
def get_e_ideal_z(dd, snap, selection, context):
    """Resolver getter for the registered ``e_ideal_z`` quantity; physical meaning and units are defined by its registry metadata."""
    ux = dd._op("ydn", "u_x", snap, selection, context)
    by = dd._op("xdn", "b_y", snap, selection, context)
    uy = dd._op("xdn", "u_y", snap, selection, context)
    bx = dd._op("ydn", "b_x", snap, selection, context)
    return _negative_cross_term_arrays(ux, by, uy, bx)


# =============================================================================
# Poynting flux
# =============================================================================
def _total_poynting_term(
    dd, snap, selection, context, tag, outer_op, evar, bvar, bdown,
):
    """Evaluate one stagger-aware product entering ``E cross B``."""
    def edge_product(sel):
        """Multiply one electric-field edge component by co-located B."""
        electric = dd._getvar(evar, snap, sel, context)
        magnetic = dd._op(bdown, bvar, snap, sel, context)
        out = np.empty_like(electric)
        np.multiply(electric, magnetic, out=out)
        return out

    return dd._ops_expression(
        (outer_op,), ("total_poynting_term", tag), snap, selection, context, edge_product
    )


@register_variable(
    "sp_x",
    location="xface",
    unit_type="energy_flux",
    component="x",
    description="X component of the total Poynting flux.",
)
def get_sp_x(dd, snap, selection, context):
    """Return the native X component of ``E cross B`` on x-faces."""
    eybz = _total_poynting_term(
        dd, snap, selection, context, "x_eybz", "zup", "e_y", "b_z", "xdn"
    )
    ezby = _total_poynting_term(
        dd, snap, selection, context, "x_ezby", "yup", "e_z", "b_y", "xdn"
    )
    out = np.empty_like(eybz)
    np.subtract(eybz, ezby, out=out)
    return out


@register_variable(
    "sp_y",
    location="yface",
    unit_type="energy_flux",
    component="y",
    description="Y component of the total Poynting flux.",
)
def get_sp_y(dd, snap, selection, context):
    """Return the native Y component of ``E cross B`` on y-faces."""
    ezbx = _total_poynting_term(
        dd, snap, selection, context, "y_ezbx", "xup", "e_z", "b_x", "ydn"
    )
    exbz = _total_poynting_term(
        dd, snap, selection, context, "y_exbz", "zup", "e_x", "b_z", "ydn"
    )
    out = np.empty_like(ezbx)
    np.subtract(ezbx, exbz, out=out)
    return out


@register_variable(
    "sp_z",
    location="zface",
    unit_type="energy_flux",
    component="z",
    description="Z component of the total Poynting flux.",
)
def get_sp_z(dd, snap, selection, context):
    """Return the native Z component of ``E cross B`` on z-faces."""
    exby = _total_poynting_term(
        dd, snap, selection, context, "z_exby", "yup", "e_x", "b_y", "zdn"
    )
    eybx = _total_poynting_term(
        dd, snap, selection, context, "z_eybx", "xup", "e_y", "b_x", "zdn"
    )
    out = np.empty_like(exby)
    np.subtract(exby, eybx, out=out)
    return out


_register_magnitude(
    "sp_h",
    ("sp_x", "sp_y"),
    "energy_flux",
    "Horizontal magnitude of the total Poynting flux.",
)

_register_magnitude(
    "modsp",
    ("sp_x", "sp_y", "sp_z"),
    "energy_flux",
    "Magnitude of the total Poynting flux.",
)


def _centered_velocity_and_field(dd, snap, selection, context):
    """Return all velocity and magnetic components at cell centers."""
    ux = dd._op("xup", "u_x", snap, selection, context)
    uy = dd._op("yup", "u_y", snap, selection, context)
    uz = dd._op("zup", "u_z", snap, selection, context)
    bx = dd._op("xup", "b_x", snap, selection, context)
    by = dd._op("yup", "b_y", snap, selection, context)
    bz = dd._op("zup", "b_z", snap, selection, context)
    return ux, uy, uz, bx, by, bz


def _ideal_poynting_face_component(dd, snap, selection, context, axis):
    """Construct one ideal-MHD Poynting component at centers, then stagger to a face."""
    component_index = {"x": 0, "y": 1}[axis]

    def centered_expression(sel):
        """Evaluate ``B^2 u - (u.B) B`` on cell centers."""
        ux, uy, uz, bx, by, bz = _centered_velocity_and_field(
            dd, snap, sel, context
        )
        return _ideal_poynting_component_arrays(
            ux, uy, uz, bx, by, bz, component_index
        )

    return dd._ops_expression(
        (f"{axis}dn",),
        ("ideal_poynting_component", axis),
        snap,
        selection,
        context,
        centered_expression,
    )


@register_variable(
    "sp_ideal_x",
    location="xface",
    unit_type="energy_flux",
    component="x",
    description="X component of the ideal-MHD Poynting flux.",
)
def get_sp_ideal_x(dd, snap, selection, context):
    """Return the ideal-MHD Poynting-flux X component."""
    return _ideal_poynting_face_component(dd, snap, selection, context, "x")


@register_variable(
    "sp_ideal_y",
    location="yface",
    unit_type="energy_flux",
    component="y",
    description="Y component of the ideal-MHD Poynting flux.",
)
def get_sp_ideal_y(dd, snap, selection, context):
    """Return the ideal-MHD Poynting-flux Y component."""
    return _ideal_poynting_face_component(dd, snap, selection, context, "y")


@register_variable(
    "sp_ideal_emerg_z",
    location="zface",
    unit_type="energy_flux",
    component="z",
    description="Emergence contribution to the vertical ideal-MHD Poynting flux.",
)
def get_sp_ideal_emerg_z(dd, snap, selection, context):
    """Return ``zdn[uz*(bx^2+by^2)]`` using native Bifrost staggering."""
    def centered_expression(sel):
        """Evaluate the vertical emergence term on cell centers."""
        uz = dd._op("zup", "u_z", snap, sel, context)
        bx = dd._op("xup", "b_x", snap, sel, context)
        by = dd._op("yup", "b_y", snap, sel, context)
        return _ideal_poynting_emerg_z_arrays(uz, bx, by)

    return dd._ops_expression(
        ("zdn",),
        ("ideal_poynting_emerg_z",),
        snap,
        selection,
        context,
        centered_expression,
    )


@register_variable(
    "sp_ideal_shear_z",
    location="zface",
    unit_type="energy_flux",
    component="z",
    description="Shear contribution to the vertical ideal-MHD Poynting flux.",
)
def get_sp_ideal_shear_z(dd, snap, selection, context):
    """Return ``zdn[-bz*(ux*bx+uy*by)]`` using native Bifrost staggering."""
    def centered_expression(sel):
        """Evaluate the vertical shear term on cell centers."""
        ux = dd._op("xup", "u_x", snap, sel, context)
        uy = dd._op("yup", "u_y", snap, sel, context)
        bx = dd._op("xup", "b_x", snap, sel, context)
        by = dd._op("yup", "b_y", snap, sel, context)
        bz = dd._op("zup", "b_z", snap, sel, context)
        return _ideal_poynting_shear_z_arrays(ux, uy, bx, by, bz)

    return dd._ops_expression(
        ("zdn",),
        ("ideal_poynting_shear_z",),
        snap,
        selection,
        context,
        centered_expression,
    )


@register_variable(
    "sp_ideal_z",
    location="zface",
    unit_type="energy_flux",
    component="z",
    description="Z component of the ideal-MHD Poynting flux.",
)
def get_sp_ideal_z(dd, snap, selection, context):
    """Return the vertical ideal Poynting flux as emergence plus shear terms."""
    emerg = dd._getvar("sp_ideal_emerg_z", snap, selection, context)
    shear = dd._getvar("sp_ideal_shear_z", snap, selection, context)
    out = np.empty_like(emerg)
    np.add(emerg, shear, out=out)
    return out


_register_magnitude(
    "sp_ideal_h",
    ("sp_ideal_x", "sp_ideal_y"),
    "energy_flux",
    "Horizontal magnitude of the ideal-MHD Poynting flux.",
)

_register_magnitude(
    "modsp_ideal",
    ("sp_ideal_x", "sp_ideal_y", "sp_ideal_z"),
    "energy_flux",
    "Magnitude of the ideal-MHD Poynting flux.",
)


# =============================================================================
# Total electric-field decomposition relative to B
# =============================================================================
_register_magnitude(
    "mode",
    ("e_x", "e_y", "e_z"),
    "electric_field",
    "Magnitude of the total electric field.",
)

@register_variable(
    "eparb",
    location="center",
    unit_type="electric_field",
    description="Signed total electric-field component parallel to B, E dot B / |B|.",
)
def get_eparb(dd, snap, selection, context):
    """Resolver getter for the registered ``eparb`` quantity; physical meaning and units are defined by its registry metadata."""
    ex = dd._ops(("yup", "zup"), "e_x", snap, selection, context)
    ey = dd._ops(("zup", "xup"), "e_y", snap, selection, context)
    ez = dd._ops(("xup", "yup"), "e_z", snap, selection, context)
    bx = dd._op("xup", "b_x", snap, selection, context)
    by = dd._op("yup", "b_y", snap, selection, context)
    bz = dd._op("zup", "b_z", snap, selection, context)
    modb = dd._getvar("modb", snap, selection, context)
    dot = _dot3_arrays(ex, ey, ez, bx, by, bz)
    np.divide(dot, modb, out=dot)
    return dot


def _eparb_vector_component(dd, snap, selection, context, axis):
    """Return one staggered component of the electric field parallel to B."""
    # E_parallel vector is constructed at cell centers first, then moved to
    # the native electric-field edge.  The down-operator order is the reverse
    # of the corresponding edge-to-center path.
    down_ops = {
        "x": ("zdn", "ydn"),
        "y": ("xdn", "zdn"),
        "z": ("ydn", "xdn"),
    }[axis]

    def centered_component(sel):
        """Center one electric-field component while preserving its resolver cache path."""
        epar = dd._getvar("eparb", snap, sel, context)
        bcomp = dd._op(f"{axis}up", f"b_{axis}", snap, sel, context)
        modb = dd._getvar("modb", snap, sel, context)
        return epar * (bcomp / modb)

    return dd._ops_expression(
        down_ops,
        ("eparb_component", axis),
        snap,
        selection,
        context,
        centered_component,
    )


@register_variable(
    "eparb_x",
    location="xedge",
    unit_type="electric_field",
    component="x",
    description="X component of the total electric field parallel to B.",
)
def get_eparb_x(dd, snap, selection, context):
    """Resolver getter for the registered ``eparb_x`` quantity; physical meaning and units are defined by its registry metadata."""
    return _eparb_vector_component(dd, snap, selection, context, "x")


@register_variable(
    "eparb_y",
    location="yedge",
    unit_type="electric_field",
    component="y",
    description="Y component of the total electric field parallel to B.",
)
def get_eparb_y(dd, snap, selection, context):
    """Resolver getter for the registered ``eparb_y`` quantity; physical meaning and units are defined by its registry metadata."""
    return _eparb_vector_component(dd, snap, selection, context, "y")


@register_variable(
    "eparb_z",
    location="zedge",
    unit_type="electric_field",
    component="z",
    description="Z component of the total electric field parallel to B.",
)
def get_eparb_z(dd, snap, selection, context):
    """Resolver getter for the registered ``eparb_z`` quantity; physical meaning and units are defined by its registry metadata."""
    return _eparb_vector_component(dd, snap, selection, context, "z")


@register_variable(
    "eperpb_x",
    location="xedge",
    unit_type="electric_field",
    component="x",
    description="X component of the total electric field perpendicular to B.",
)
def get_eperpb_x(dd, snap, selection, context):
    """Resolver getter for the registered ``eperpb_x`` quantity; physical meaning and units are defined by its registry metadata."""
    return dd._getvar("e_x", snap, selection, context) - dd._getvar(
        "eparb_x", snap, selection, context
    )


@register_variable(
    "eperpb_y",
    location="yedge",
    unit_type="electric_field",
    component="y",
    description="Y component of the total electric field perpendicular to B.",
)
def get_eperpb_y(dd, snap, selection, context):
    """Resolver getter for the registered ``eperpb_y`` quantity; physical meaning and units are defined by its registry metadata."""
    return dd._getvar("e_y", snap, selection, context) - dd._getvar(
        "eparb_y", snap, selection, context
    )


@register_variable(
    "eperpb_z",
    location="zedge",
    unit_type="electric_field",
    component="z",
    description="Z component of the total electric field perpendicular to B.",
)
def get_eperpb_z(dd, snap, selection, context):
    """Resolver getter for the registered ``eperpb_z`` quantity; physical meaning and units are defined by its registry metadata."""
    return dd._getvar("e_z", snap, selection, context) - dd._getvar(
        "eparb_z", snap, selection, context
    )


_register_magnitude(
    "eperpb",
    ("eperpb_x", "eperpb_y", "eperpb_z"),
    "electric_field",
    "Magnitude of the total electric field perpendicular to B, |E_perp,B|.",
)


# =============================================================================
# Thermodynamics / plasma diagnostics
# =============================================================================
@register_variable(
    "pb",
    location="center",
    unit_type="pressure",
    description="Magnetic pressure, B^2/(8 pi).",
)
def get_pb(dd, snap, selection, context):
    """Resolver getter for the registered ``pb`` quantity; physical meaning and units are defined by its registry metadata."""
    b = dd._getvar("modb", snap, selection, context)
    out = np.empty_like(b)
    np.multiply(b, np.float32(0.5), out=out)
    np.multiply(out, b, out=out)
    return out


@register_variable(
    "pt",
    location="center",
    unit_type="pressure",
    description="Total gas plus magnetic pressure, pg + pb.",
)
def get_pt(dd, snap, selection, context):
    """Resolver getter for the registered ``pt`` quantity; physical meaning and units are defined by its registry metadata."""
    pg = dd._getvar("pg", snap, selection, context)
    pb = dd._getvar("pb", snap, selection, context)
    return pg + pb


@register_variable(
    "beta",
    location="center",
    unit_type="dimensionless",
    description="Plasma beta, pg/pb.",
)
def get_beta(dd, snap, selection, context):
    """Resolver getter for the registered ``beta`` quantity; physical meaning and units are defined by its registry metadata."""
    pg = dd._getvar("pg", snap, selection, context)
    pb = dd._getvar("pb", snap, selection, context)
    return pg / pb


@register_variable(
    "va",
    location="center",
    unit_type="velocity",
    description="Alfven speed, |B|/sqrt(r) in Bifrost units.",
)
def get_va(dd, snap, selection, context):
    """Resolver getter for the registered ``va`` quantity; physical meaning and units are defined by its registry metadata."""
    b = dd._getvar("modb", snap, selection, context)
    r = dd._getvar("r", snap, selection, context)
    out = np.empty_like(r)
    np.sqrt(r, out=out)
    np.divide(b, out, out=out)
    return out


@register_variable(
    "mach_a",
    location="center",
    unit_type="dimensionless",
    aliases=("man",),
    description="Alfvenic Mach number, |u|/va.",
)
def get_mach_a(dd, snap, selection, context):
    """Resolver getter for the registered ``mach_a`` quantity; physical meaning and units are defined by its registry metadata."""
    u = dd._getvar("modu", snap, selection, context)
    va = dd._getvar("va", snap, selection, context)
    return u / va


def _run_gamma(dd, snap):
    """Return the run adiabatic gamma parameter, defaulting to 5/3."""
    return np.float32(dd._meta(snap).params.get("gamma", 5.0 / 3.0))


@register_variable(
    "cs_gamma",
    location="center",
    unit_type="velocity",
    description="Sound speed using the run's constant gamma (default 5/3): sqrt(gamma*pg/r).",
)
def get_cs_gamma(dd, snap, selection, context):
    """Resolver getter for the registered ``cs_gamma`` quantity; physical meaning and units are defined by its registry metadata."""
    pg = dd._getvar("pg", snap, selection, context)
    r = dd._getvar("r", snap, selection, context)
    out = np.empty_like(pg)
    np.multiply(pg, _run_gamma(dd, snap), out=out)
    np.divide(out, r, out=out)
    np.sqrt(out, out=out)
    return out


@register_variable(
    "mach_s_gamma",
    location="center",
    unit_type="dimensionless",
    description="Sonic Mach number computed with cs_gamma, |u|/cs_gamma.",
)
def get_mach_s_gamma(dd, snap, selection, context):
    """Resolver getter for the registered ``mach_s_gamma`` quantity; physical meaning and units are defined by its registry metadata."""
    u = dd._getvar("modu", snap, selection, context)
    cs = dd._getvar("cs_gamma", snap, selection, context)
    return u / cs


def _gravity_native(dd, snap):
    """Return the native signed vertical gravity in Bifrost acceleration units."""
    meta = dd._meta(snap)
    if "grav" in meta.params:
        return np.float32(abs(float(meta.params["grav"])))
    # Solar-surface fallback in case an old parameter file omitted grav.
    g_sun_cgs = 2.74e4
    acceleration_unit = meta.units.u_l / (meta.units.u_t * meta.units.u_t)
    return np.float32(g_sun_cgs / acceleration_unit)


@register_variable(
    "hpg_hydro",
    location="center",
    unit_type="length",
    aliases=("hpg",),
    description="Local hydrostatic gas-pressure scale height, pg/(r*|g|), using the run's grav parameter.",
)
def get_hpg_hydro(dd, snap, selection, context):
    """Resolver getter for the registered ``hpg_hydro`` quantity; physical meaning and units are defined by its registry metadata."""
    pg = dd._getvar("pg", snap, selection, context)
    r = dd._getvar("r", snap, selection, context)
    out = np.empty_like(pg)
    np.multiply(r, _gravity_native(dd, snap), out=out)
    np.divide(pg, out, out=out)
    return out


@register_variable(
    "hpg_grad",
    location="center",
    unit_type="length",
    description="Signed gas-pressure scale height -pg/(dpg/dz_physical), evaluated as pg/(dpg/dz_native).",
)
def get_hpg_grad(dd, snap, selection, context):
    """Resolver getter for the registered ``hpg_grad`` quantity; physical meaning and units are defined by its registry metadata."""
    pg = dd._getvar("pg", snap, selection, context)
    dpdz = dd._ops(("ddzdn", "zup"), "pg", snap, selection, context)
    out = np.full_like(pg, np.inf, dtype=np.float32)
    nonzero = dpdz != 0.0
    np.divide(pg, dpdz, out=out, where=nonzero)
    # Native Bifrost z grows downward; physical height is -z_native.
    return out


# =============================================================================
# Gas-pressure gradient and Lorentz force
# =============================================================================
def _pressure_gradient_force_component(dd, snap, selection, context, operator):
    """Return one component of the pressure-gradient force density, ``-grad(pg)``."""
    grad = dd._op(operator, "pg", snap, selection, context)
    out = np.empty_like(grad)
    np.negative(grad, out=out)
    return out


@register_variable(
    "gradpg_x",
    location="xface",
    unit_type="force_density",
    component="x",
    description="X component of the pressure-gradient force density, -dpg/dx.",
)
def get_gradpg_x(dd, snap, selection, context):
    """Resolver getter for the registered ``gradpg_x`` quantity; physical meaning and units are defined by its registry metadata."""
    return _pressure_gradient_force_component(dd, snap, selection, context, "ddxdn")


@register_variable(
    "gradpg_y",
    location="yface",
    unit_type="force_density",
    component="y",
    description="Y component of the pressure-gradient force density, -dpg/dy.",
)
def get_gradpg_y(dd, snap, selection, context):
    """Resolver getter for the registered ``gradpg_y`` quantity; physical meaning and units are defined by its registry metadata."""
    return _pressure_gradient_force_component(dd, snap, selection, context, "ddydn")


@register_variable(
    "gradpg_z",
    location="zface",
    unit_type="force_density",
    component="z",
    description="Z component of the pressure-gradient force density, -dpg/dz.",
)
def get_gradpg_z(dd, snap, selection, context):
    """Resolver getter for the registered ``gradpg_z`` quantity; physical meaning and units are defined by its registry metadata."""
    return _pressure_gradient_force_component(dd, snap, selection, context, "ddzdn")


_register_magnitude(
    "gradpg_h",
    ('gradpg_x', 'gradpg_y'),
    "force_density",
    'Horizontal magnitude of the pressure-gradient force density.',
)

_register_magnitude(
    "modgradpg",
    ('gradpg_x', 'gradpg_y', 'gradpg_z'),
    "force_density",
    'Magnitude of the pressure-gradient force density.',
)

@register_variable(
    "gradpgor_x",
    location="xface",
    unit_type="acceleration",
    component="x",
    description="X component of the acceleration due to the pressure-gradient force, -(1/rho)dpg/dx.",
)
def get_gradpgor_x(dd, snap, selection, context):
    """Resolver getter for the registered ``gradpgor_x`` quantity; physical meaning and units are defined by its registry metadata."""
    grad = dd._getvar("gradpg_x", snap, selection, context)
    rho = dd._op("xdn", "r", snap, selection, context)
    return grad / rho


@register_variable(
    "gradpgor_y",
    location="yface",
    unit_type="acceleration",
    component="y",
    description="Y component of the acceleration due to the pressure-gradient force, -(1/rho)dpg/dy.",
)
def get_gradpgor_y(dd, snap, selection, context):
    """Resolver getter for the registered ``gradpgor_y`` quantity; physical meaning and units are defined by its registry metadata."""
    grad = dd._getvar("gradpg_y", snap, selection, context)
    rho = dd._op("ydn", "r", snap, selection, context)
    return grad / rho


@register_variable(
    "gradpgor_z",
    location="zface",
    unit_type="acceleration",
    component="z",
    description="Z component of the acceleration due to the pressure-gradient force, -(1/rho)dpg/dz.",
)
def get_gradpgor_z(dd, snap, selection, context):
    """Resolver getter for the registered ``gradpgor_z`` quantity; physical meaning and units are defined by its registry metadata."""
    grad = dd._getvar("gradpg_z", snap, selection, context)
    rho = dd._op("zdn", "r", snap, selection, context)
    return grad / rho


_register_magnitude(
    "gradpgor_h",
    ('gradpgor_x', 'gradpgor_y'),
    "acceleration",
    'Horizontal magnitude of the acceleration due to the pressure-gradient force.',
)

_register_magnitude(
    "modgradpgor",
    ('gradpgor_x', 'gradpgor_y', 'gradpgor_z'),
    "acceleration",
    'Magnitude of the acceleration due to the pressure-gradient force.',
)

def _lf_term(dd, snap, selection, context, tag, outer_op, jvar, bvar, bdown):
    """Evaluate one Lorentz-force component with the exact legacy stagger product path."""
    def edge_product(sel):
        """Evaluate one stagger-aware product entering a Lorentz-force component."""
        j = dd._getvar(jvar, snap, sel, context)
        b = dd._op(bdown, bvar, snap, sel, context)
        return j * b

    return dd._ops_expression(
        (outer_op,), ("lf_term", tag), snap, selection, context, edge_product
    )


@register_variable(
    "lf_x",
    location="xface",
    unit_type="force_density",
    component="x",
    description="X component of the Lorentz force density, J cross B.",
)
def get_lf_x(dd, snap, selection, context):
    """Resolver getter for the registered ``lf_x`` quantity; physical meaning and units are defined by its registry metadata."""
    return (
        _lf_term(dd, snap, selection, context, "x_jybz", "zup", "j_y", "b_z", "xdn")
        - _lf_term(dd, snap, selection, context, "x_jzby", "yup", "j_z", "b_y", "xdn")
    )


@register_variable(
    "lf_y",
    location="yface",
    unit_type="force_density",
    component="y",
    description="Y component of the Lorentz force density, J cross B.",
)
def get_lf_y(dd, snap, selection, context):
    """Resolver getter for the registered ``lf_y`` quantity; physical meaning and units are defined by its registry metadata."""
    return (
        _lf_term(dd, snap, selection, context, "y_jzbx", "xup", "j_z", "b_x", "ydn")
        - _lf_term(dd, snap, selection, context, "y_jxbz", "zup", "j_x", "b_z", "ydn")
    )


@register_variable(
    "lf_z",
    location="zface",
    unit_type="force_density",
    component="z",
    description="Z component of the Lorentz force density, J cross B.",
)
def get_lf_z(dd, snap, selection, context):
    """Resolver getter for the registered ``lf_z`` quantity; physical meaning and units are defined by its registry metadata."""
    return (
        _lf_term(dd, snap, selection, context, "z_jxby", "yup", "j_x", "b_y", "zdn")
        - _lf_term(dd, snap, selection, context, "z_jybx", "xup", "j_y", "b_x", "zdn")
    )


_register_magnitude(
    "lf_h",
    ('lf_x', 'lf_y'),
    "force_density",
    'Horizontal magnitude of the Lorentz force density, J cross B.',
)

_register_magnitude(
    "modlf",
    ('lf_x', 'lf_y', 'lf_z'),
    "force_density",
    'Magnitude of the Lorentz force density, J cross B.',
)

@register_variable(
    "lfor_x",
    location="xface",
    unit_type="acceleration",
    component="x",
    description="X component of the Lorentz-force acceleration.",
)
def get_lfor_x(dd, snap, selection, context):
    """Resolver getter for the registered ``lfor_x`` quantity; physical meaning and units are defined by its registry metadata."""
    lf = dd._getvar("lf_x", snap, selection, context)
    rho = dd._op("xdn", "r", snap, selection, context)
    return lf / rho


@register_variable(
    "lfor_y",
    location="yface",
    unit_type="acceleration",
    component="y",
    description="Y component of the Lorentz-force acceleration.",
)
def get_lfor_y(dd, snap, selection, context):
    """Resolver getter for the registered ``lfor_y`` quantity; physical meaning and units are defined by its registry metadata."""
    lf = dd._getvar("lf_y", snap, selection, context)
    rho = dd._op("ydn", "r", snap, selection, context)
    return lf / rho


@register_variable(
    "lfor_z",
    location="zface",
    unit_type="acceleration",
    component="z",
    description="Z component of the Lorentz-force acceleration.",
)
def get_lfor_z(dd, snap, selection, context):
    """Resolver getter for the registered ``lfor_z`` quantity; physical meaning and units are defined by its registry metadata."""
    lf = dd._getvar("lf_z", snap, selection, context)
    rho = dd._op("zdn", "r", snap, selection, context)
    return lf / rho


_register_magnitude(
    "lfor_h",
    ('lfor_x', 'lfor_y'),
    "acceleration",
    'Horizontal magnitude of the Lorentz-force acceleration.',
)

_register_magnitude(
    "modlfor",
    ('lfor_x', 'lfor_y', 'lfor_z'),
    "acceleration",
    'Magnitude of the Lorentz-force acceleration.',
)

# =============================================================================
# Energy reservoirs and source / conversion terms
# =============================================================================
@register_variable(
    "eb",
    location="center",
    unit_type="energy_density",
    description="Magnetic energy per unit volume.",
)
def get_eb(dd, snap, selection, context):
    """Resolver getter for the registered ``eb`` quantity; physical meaning and units are defined by its registry metadata."""
    b = dd._getvar("modb", snap, selection, context)
    out = np.empty_like(b)
    np.multiply(b, np.float32(0.5), out=out)
    np.multiply(out, b, out=out)
    return out


@register_variable(
    "ek",
    location="center",
    unit_type="energy_density",
    description="Kinetic energy per unit volume, r*|u|^2/2.",
)
def get_ek(dd, snap, selection, context):
    """Resolver getter for the registered ``ek`` quantity; physical meaning and units are defined by its registry metadata."""
    r = dd._getvar("r", snap, selection, context)
    u = dd._getvar("modu", snap, selection, context)
    out = np.empty_like(r)
    np.multiply(r, np.float32(0.5), out=out)
    np.multiply(out, u, out=out)
    np.multiply(out, u, out=out)
    return out


def _timescale_from_energy(e, q):
    """Return e/|q| with +inf where q is exactly zero, in one pass."""
    out = np.empty_like(e)
    arrays = (e, q, out)
    if _flat_compatible(*arrays):
        flat = tuple(_flat_memory_view(v) for v in arrays)
        _timescale_flat(*flat)
    else:
        _timescale_native(e, q, out)
    return out


def _register_q_diagnostics(name, specific_description, *, tau_description=None):
    """Register a per-mass energy-rate diagnostic and optional energy timescale."""

    @register_variable(
        f"{name}or",
        location="center",
        unit_type="specific_energy_rate",
        description=specific_description,
    )
    def get_q_over_r(dd, snap, selection, context, _name=name):
        """Resolver getter generated for a volumetric rate divided by mass density."""
        q = dd._getvar(_name, snap, selection, context)
        r = dd._getvar("r", snap, selection, context)
        return q / r

    if tau_description is not None:
        @register_variable(
            f"tau{name}",
            location="center",
            unit_type="time",
            description=tau_description,
        )
        def get_tau_q(dd, snap, selection, context, _name=name):
            """Resolver getter generated for an internal-energy timescale."""
            e = dd._getvar("e", snap, selection, context)
            q = dd._getvar(_name, snap, selection, context)
            return _timescale_from_energy(e, q)


_Q_DIAGNOSTICS = {
    "qjoule": (
        "Joule heating per unit mass.",
        "Internal-energy timescale for Joule heating, e/|qjoule|.",
    ),
    "qvisc": (
        "Viscous heating per unit mass.",
        "Internal-energy timescale for viscous heating, e/|qvisc|.",
    ),
    "qthin": (
        "Optically thin radiative cooling per unit mass.",
        "Internal-energy timescale for optically thin radiative cooling, e/|qthin|.",
    ),
    "qspitz": (
        "Spitzer thermal-conduction heating/cooling per unit mass.",
        "Internal-energy timescale for Spitzer thermal conduction, e/|qspitz|.",
    ),
    "qrad": (
        "Full radiative-transfer heating/cooling per unit mass.",
        "Internal-energy timescale for full radiative transfer, e/|qrad|.",
    ),
    "qgenrad": (
        "Generalized-radiation heating/cooling per unit mass.",
        "Internal-energy timescale for generalized radiation, e/|qgenrad|.",
    ),
}
for _qname, (_specific_description, _tau_description) in _Q_DIAGNOSTICS.items():
    _register_q_diagnostics(
        _qname,
        _specific_description,
        tau_description=_tau_description,
    )


@register_variable(
    "qpdv",
    location="center",
    unit_type="energy_rate_density",
    description="Compression/expansion work, -pg*div(u).",
)
def get_qpdv(dd, snap, selection, context):
    """Resolver getter for the registered ``qpdv`` quantity; physical meaning and units are defined by its registry metadata."""
    pg = dd._getvar("pg", snap, selection, context)
    divu = dd._getvar("divu", snap, selection, context)
    out = np.empty_like(pg)
    np.negative(pg, out=out)
    np.multiply(out, divu, out=out)
    return out


_register_q_diagnostics(
    "qpdv",
    "Compression/expansion work per unit mass.",
    tau_description="Internal-energy timescale for compression/expansion work, e/|qpdv|.",
)


@register_variable(
    "qlf",
    location="center",
    unit_type="energy_rate_density",
    description="Work by the Lorentz force, u dot (J cross B) = J dot E_ideal.",
)
def get_qlf(dd, snap, selection, context):
    """Resolver getter for the registered ``qlf`` quantity; physical meaning and units are defined by its registry metadata."""
    qx = dd._ops_product(
        ("zup", "yup"), "e_ideal_x", "j_x", snap, selection, context
    )
    qy = dd._ops_product(
        ("xup", "zup"), "e_ideal_y", "j_y", snap, selection, context
    )
    qz = dd._ops_product(
        ("yup", "xup"), "e_ideal_z", "j_z", snap, selection, context
    )
    out = np.empty_like(qx)
    np.add(qx, qy, out=out)
    np.add(out, qz, out=out)
    return out


_register_q_diagnostics(
    "qlf",
    "Lorentz-force work per unit mass.",
)


# =============================================================================
# EOS helpers
# =============================================================================
def _eos_input(dd, snap, selection, context):
    """Return native rho/ee and units without allocating scaled 3-D arrays."""
    return (dd._getvar("r", snap, selection, context),
            dd._getvar("ee", snap, selection, context),
            dd._meta(snap).units)


def _eos_evaluate(dd, snap, rho, ee, units, *, out=None, gamma=False):
    """Pass unit factors to the tabulated EOS kernels (no scaled cubes)."""
    from .eos import EOSTable
    eos = dd._eos(snap)
    if isinstance(eos, EOSTable):
        kwargs = {"rho_scale": units.u_r, "ee_scale": units.u_ee}
        return eos.gamma1(rho, ee, **kwargs) if gamma else eos.interpolate(rho, ee, out=out, **kwargs)
    # Experimental/custom EOS objects retain their original CGS interface.
    return (eos.gamma1(rho * units.u_r, ee * units.u_ee) if gamma
            else eos.interpolate(rho * units.u_r, ee * units.u_ee, out=out))


@register_variable(
    "tg_eos",
    location="center",
    unit_type="temperature",
    description="Temperature from the Bifrost rho-e EOS table.",
)
def get_tg_eos(dd, snap, selection, context):
    """Resolver getter for the registered ``tg_eos`` quantity; physical meaning and units are defined by its registry metadata."""
    rho_cgs, ee_cgs, units = _eos_input(dd, snap, selection, context)
    return _eos_evaluate(dd, snap, rho_cgs, ee_cgs, units, out="tg")


@register_variable(
    "ne_eos",
    location="center",
    unit_type="number_density",
    description="Electron number density from the Bifrost rho-e EOS table.",
)
def get_ne_eos(dd, snap, selection, context):
    """Resolver getter for the registered ``ne_eos`` quantity; physical meaning and units are defined by its registry metadata."""
    rho_cgs, ee_cgs, units = _eos_input(dd, snap, selection, context)
    return _eos_evaluate(dd, snap, rho_cgs, ee_cgs, units, out="ne")


@register_variable(
    "pg_eos",
    location="center",
    unit_type="pressure",
    description="Gas pressure from the Bifrost rho-e EOS table.",
)
def get_pg_eos(dd, snap, selection, context):
    """Resolver getter for the registered ``pg_eos`` quantity; physical meaning and units are defined by its registry metadata."""
    rho_cgs, ee_cgs, units = _eos_input(dd, snap, selection, context)
    pg_cgs = _eos_evaluate(dd, snap, rho_cgs, ee_cgs, units, out="pg")
    return pg_cgs / units.u_p


@register_variable(
    "kr_eos",
    location="center",
    unit_type="opacity",
    description="Rosseland mean opacity from the Bifrost rho-e EOS table.",
)
def get_kr_eos(dd, snap, selection, context):
    """Resolver getter for the registered ``kr_eos`` quantity; physical meaning and units are defined by its registry metadata."""
    rho_cgs, ee_cgs, units = _eos_input(dd, snap, selection, context)
    kr_cgs = _eos_evaluate(dd, snap, rho_cgs, ee_cgs, units, out="kr")
    return kr_cgs / units.u_kr


@register_variable(
    "gamma1",
    location="center",
    unit_type="dimensionless",
    description="First adiabatic exponent Gamma_1 derived from local derivatives of the Bifrost rho-e EOS table.",
)
def get_gamma1(dd, snap, selection, context):
    """Resolver getter for the registered ``gamma1`` quantity; physical meaning and units are defined by its registry metadata."""
    rho_cgs, ee_cgs, units = _eos_input(dd, snap, selection, context)
    return _eos_evaluate(dd, snap, rho_cgs, ee_cgs, units, gamma=True)


@register_variable(
    "cs",
    location="center",
    unit_type="velocity",
    aliases=("cs_eos",),
    description="EOS-consistent adiabatic sound speed, sqrt(Gamma_1*pg_eos/r).",
)
def get_cs(dd, snap, selection, context):
    """Resolver getter for the registered ``cs`` quantity; physical meaning and units are defined by its registry metadata."""
    gamma1 = dd._getvar("gamma1", snap, selection, context)
    pg = dd._getvar("pg_eos", snap, selection, context)
    r = dd._getvar("r", snap, selection, context)
    return np.sqrt(gamma1 * pg / r)


@register_variable(
    "mach_s",
    location="center",
    unit_type="dimensionless",
    aliases=("mn",),
    description="Sonic Mach number based on the EOS-consistent sound speed, |u|/cs.",
)
def get_mach_s(dd, snap, selection, context):
    """Resolver getter for the registered ``mach_s`` quantity; physical meaning and units are defined by its registry metadata."""
    u = dd._getvar("modu", snap, selection, context)
    cs = dd._getvar("cs", snap, selection, context)
    return u / cs


@register_variable(
    "tg",
    location="center",
    unit_type="temperature",
    description="Temperature.",
)
def get_tg(dd, snap, selection, context):
    """Resolver getter for the registered ``tg`` quantity; physical meaning and units are defined by its registry metadata."""
    if dd._variable_exists("hiontg", snap):
        return dd._getvar("hiontg", snap, selection, context)
    return get_tg_eos(dd, snap, selection, context)


@register_variable(
    "ne",
    location="center",
    unit_type="number_density",
    description="Electron number density.",
)
def get_ne(dd, snap, selection, context):
    """Resolver getter for the registered ``ne`` quantity; physical meaning and units are defined by its registry metadata."""
    if dd._variable_exists("hionne", snap):
        return dd._getvar("hionne", snap, selection, context)
    return get_ne_eos(dd, snap, selection, context)


@register_variable(
    "pg",
    location="center",
    unit_type="pressure",
    storage_name="p",
    aliases=("p",),
    description="Gas pressure.",
)
def get_pg(dd, snap, selection, context):
    """Resolver getter for the registered ``pg`` quantity; physical meaning and units are defined by its registry metadata."""
    # ``p`` is Bifrost's standard AUX storage name for gas pressure.  The
    # resolver checks that storage name before calling this getter.  Keep a
    # compatibility fallback for datasets that actually store a raw ``pg``
    # field, then fall back to the EOS.  ``pg_eos`` always forces the table.
    if dd._variable_exists("pg", snap):
        return dd._read_variable("pg", snap, selection)
    return get_pg_eos(dd, snap, selection, context)


@register_variable(
    "kr",
    location="center",
    unit_type="opacity",
    description="Rosseland mean opacity.",
)
def get_kr(dd, snap, selection, context):
    """Resolver getter for the registered ``kr`` quantity; physical meaning and units are defined by its registry metadata."""
    return get_kr_eos(dd, snap, selection, context)


# =============================================================================
# Composition and synthetic instrument responses
# =============================================================================
@register_variable(
    "nh",
    location="center",
    unit_type="number_density",
    description=(
        "Hydrogen number density derived from mass density and the run elemental "
        "abundances/atomic weights."
    ),
)
def get_nh(dd, snap, selection, context):
    """Resolver getter for the registered ``nh`` quantity; physical meaning and units are defined by its registry metadata."""
    rho = dd._getvar("r", snap, selection, context)
    grams_per_h = dd._composition(snap).grams_per_hydrogen
    # Collapse rho -> CGS and mass-per-hydrogen conversion into one pass over
    # the cube instead of materializing an intermediate rho_cgs array.
    factor = dd._meta(snap).units.u_r / grams_per_h
    return rho * factor


# H-minus bound-free optical-depth approximation used by the historical
# Bifrost IDL br_rhoeetab.pro tau keyword. The local extinction is evaluated
# on the fly and never materialized as a full 3-D auxiliary cube.
_TAU500_HMINUS_SAHA = 1.03526e-16
_TAU500_HMINUS_SIGMA_CM2 = 2.9256e-17
_TAU500_HMINUS_BINDING_EV = 0.754
_TAU500_EV_TO_ERG = 1.60217733e-12
_TAU500_K_B_CGS = 1.38e-16
_TAU500_TOP = 1.0e-16


@njit(cache=True, parallel=True)
def _integrate_tau500_native(ne, nh, temperature, z_cm, out):
    """Integrate tau500 in native Bifrost k-order with one 2-D work plane.

    This follows the H- bound-free approximation and trapezoidal vertical
    integration used by ``br_rhoeetab.pro``.  ``ne`` and ``nh`` are number
    densities [cm^-3], ``temperature`` is [K], and ``z_cm`` is the native
    center coordinate [cm].
    """
    nx, ny, nz = temperature.shape
    if nz == 0:
        return

    chi_prev = np.empty((nx, ny), dtype=np.float32)
    exponent_constant = (
        _TAU500_HMINUS_BINDING_EV * _TAU500_EV_TO_ERG / _TAU500_K_B_CGS
    )

    # First native plane is the top boundary in the historical Bifrost
    # convention.  Keep the tiny non-zero boundary value used by the IDL
    # reference rather than forcing exact zero.
    for j in prange(ny):
        for i in range(nx):
            t = temperature[i, j, 0]
            chi = (
                _TAU500_HMINUS_SAHA
                * ne[i, j, 0]
                * _TAU500_HMINUS_SIGMA_CM2
                / (t ** 1.5)
                * np.exp(exponent_constant / t)
                * nh[i, j, 0]
            )
            chi_prev[i, j] = chi
            out[i, j, 0] = _TAU500_TOP

    for k in range(1, nz):
        dz_cm = abs(z_cm[k] - z_cm[k - 1])
        for j in prange(ny):
            for i in range(nx):
                t = temperature[i, j, k]
                chi = (
                    _TAU500_HMINUS_SAHA
                    * ne[i, j, k]
                    * _TAU500_HMINUS_SIGMA_CM2
                    / (t ** 1.5)
                    * np.exp(exponent_constant / t)
                    * nh[i, j, k]
                )
                out[i, j, k] = (
                    out[i, j, k - 1]
                    + 0.5 * (chi_prev[i, j] + chi) * dz_cm
                )
                chi_prev[i, j] = chi


@register_variable(
    "tau500",
    location="center",
    unit_type="dimensionless",
    description=(
        "Approximate 500 nm continuum optical depth from H-minus bound-free "
        "opacity, integrated vertically from the native Bifrost top boundary."
    ),
)
def get_tau500(dd, snap, selection, context):
    """Resolver getter for the registered ``tau500`` quantity; physical meaning and units are defined by its registry metadata."""
    meta = dd._meta(snap)
    requested = selection.normalized(meta.shape)

    # Optical depth at a requested z depends on every layer above it. Keep the
    # requested x/y domain, but evaluate the full native z column before
    # cropping back to the user's selection.
    full_z = Selection(requested.ix, requested.iy, slice(0, meta.shape[2], 1))

    ne = dd._getvar("ne", snap, full_z, context)
    tg = dd._getvar("tg", snap, full_z, context)
    nh = dd._getvar("nh", snap, full_z, context)

    z_cm = np.asarray(meta.z, dtype=np.float64) * meta.units.u_l
    tau = np.empty_like(tg, dtype=np.float32, order="K")
    _integrate_tau500_native(ne, nh, tg, z_cm, tau)

    return tau[requested.relative_to(full_z, meta.shape)]


# Common validity mask for optically thin coronal synthetic observables.
# These thresholds intentionally reproduce the long-standing Bifrost/IDL
# synthesis convention used for AIA, XRT, and EUI: cool chromospheric plasma
# and all plasma below the z=0 surface are excluded from the local emissivity.
_INSTRUMENT_TMIN_K = 1.0e4
_INSTRUMENT_ZMIN_MM = 0.0


@njit(cache=True)
def _apply_instrument_mask(emissivity, temperature, height_mm):
    """Apply the common AIA/XRT/EUI mask in-place without 3-D temporaries."""
    nx, ny, nz = emissivity.shape
    for k in range(nz):
        if height_mm[k] < _INSTRUMENT_ZMIN_MM:
            for j in range(ny):
                for i in range(nx):
                    emissivity[i, j, k] = 0.0
            continue

        for j in range(ny):
            for i in range(nx):
                if temperature[i, j, k] < _INSTRUMENT_TMIN_K:
                    emissivity[i, j, k] = 0.0


def _instrument_height_native_mm(dd, snap, selection):
    """Physical cell-center height [Mm] in the current native selection order."""
    meta = dd._meta(snap)
    iz = selection.normalized(meta.shape).iz

    # Public physical Z is -native_z[::-1].  Recipes are evaluated before the
    # public reorientation, so in native array order the physical height is
    # simply -native_z.  Slice only the requested internal domain so full
    # cubes, subdomains, and scalar/linear public selections share one path.
    return np.asarray(
        -np.asarray(meta.z[iz], dtype=np.float64) * meta.units.length_to_Mm,
        dtype=np.float32,
    )


def _get_instrument_emissivity(dd, name, snap, selection, context):
    """Evaluate one AIA/XRT/EUI local emissivity on the native selection."""
    ne = dd._getvar("ne", snap, selection, context)
    tg = dd._getvar("tg", snap, selection, context)
    nh = dd._getvar("nh", snap, selection, context)
    response = dd._instrument_response(name)
    emissivity = response.evaluate_emissivity(tg, ne, nh)
    height_mm = _instrument_height_native_mm(dd, snap, selection)
    _apply_instrument_mask(emissivity, tg, height_mm)
    return emissivity


_INSTRUMENT_RESPONSES = {
    "aia94": "SDO/AIA 94 A emissivity using a tabulated density- and temperature-dependent response table.",
    "aia131": "SDO/AIA 131 A emissivity using a tabulated density- and temperature-dependent response table.",
    "aia171": "SDO/AIA 171 A emissivity using a tabulated density- and temperature-dependent response table.",
    "aia193": "SDO/AIA 193 A emissivity using a tabulated density- and temperature-dependent response table.",
    "aia211": "SDO/AIA 211 A emissivity using a tabulated density- and temperature-dependent response table.",
    "aia304": "SDO/AIA 304 A emissivity using a tabulated density- and temperature-dependent response table.",
    "aia335": "SDO/AIA 335 A emissivity using a tabulated density- and temperature-dependent response table.",
    "xrt_almed": "Hinode/XRT Al-med emissivity using a tabulated density- and temperature-dependent response table.",
    "xrt_almesh": "Hinode/XRT Al-mesh emissivity using a tabulated density- and temperature-dependent response table.",
    "xrt_alpoly": "Hinode/XRT Al-poly emissivity using a tabulated density- and temperature-dependent response table.",
    "xrt_althick": "Hinode/XRT Al-thick emissivity using a tabulated density- and temperature-dependent response table.",
    "xrt_bemed": "Hinode/XRT Be-med emissivity using a tabulated density- and temperature-dependent response table.",
    "xrt_bethick": "Hinode/XRT Be-thick emissivity using a tabulated density- and temperature-dependent response table.",
    "xrt_bethin": "Hinode/XRT Be-thin emissivity using a tabulated density- and temperature-dependent response table.",
    "xrt_cpoly": "Hinode/XRT C-poly emissivity using a tabulated density- and temperature-dependent response table.",
    "xrt_tipoly": "Hinode/XRT Ti-poly emissivity using a tabulated density- and temperature-dependent response table.",
    "eui174": "Solar Orbiter/EUI HRI 174 emissivity using a tabulated density- and temperature-dependent response table.",
}


def _register_instrument_response(name, description):
    """Register one synthetic instrument-response quantity."""
    description = (
        description
        + " It is set to zero for T < 1e4 K or physical height z < 0 Mm."
    )
    @register_variable(
        name,
        location="center",
        unit_type="instrument_emissivity",
        description=description,
    )
    def getter(dd, snap, selection, context, _name=name):
        """Resolver getter generated by a quantity-registration helper."""
        return _get_instrument_emissivity(dd, _name, snap, selection, context)

    return getter


for _response_name, _response_description in _INSTRUMENT_RESPONSES.items():
    _register_instrument_response(_response_name, _response_description)
