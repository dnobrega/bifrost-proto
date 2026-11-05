"""Exact Bifrost stagger operators.

X/Y use the legacy fixed-coefficient periodic operators evaluated in the same
float32 grouping order as the IDL/C implementation.  Z uses geometry-aware
coefficients generated with the legacy calc_stagger_inv algorithm, including
its non-periodic boundary orders.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

# =============================================================================
# Geometry-aware coefficient construction (used for Z and as a fallback)
# =============================================================================

@njit(cache=True)
def _calc_stagger_inv(x, n, deriv_order, offset):
    """Return one legacy interpolation/derivative coefficient vector."""
    c = np.zeros(6, dtype=np.float32)
    b = np.zeros(6, dtype=np.float32)
    r = np.zeros(6, dtype=np.float32)
    c[n] = -x[0]
    for i in range(1, n + 1):
        for j in range(n - i, n):
            c[j] -= x[i] * c[j + 1]
        c[n] -= x[i]
    for i in range(n + 1):
        t = np.float32(1.0)
        b[n] = np.float32(1.0)
        for j in range(n, 0, -1):
            b[j - 1] = c[j] + x[i] * b[j]
            t = x[i] * t + b[j - 1]
        r[i + offset] = b[deriv_order] / t
    return r


def _boundary_orders(direction, derivative):
    """Return legacy one-sided stencil orders near lower/upper Z boundaries."""
    if direction == "up":
        lower = (3, 4, 5)
        upper = (4, 3, 2) if derivative else (4, 3, 1)
    else:
        lower = (2, 3, 4) if derivative else (1, 3, 4)
        upper = (5, 4, 3)
    return lower, upper


def build_nonuniform_coefficients(center, face, direction, derivative=False):
    """Build six-point geometry-aware coefficients for a nonuniform axis."""
    center = np.asarray(center, dtype=np.float32)
    face = np.asarray(face, dtype=np.float32)
    if center.ndim != 1 or face.ndim != 1 or center.shape != face.shape:
        raise ValueError("center and face must be matching 1-D arrays")
    if center.size < 6:
        raise ValueError("At least 6 grid points are required")
    if direction not in ("up", "dn"):
        raise ValueError("direction must be 'up' or 'dn'")

    n = center.size
    deriv_order = 1 if derivative else 0
    if direction == "up":
        source, target, shift = face, center, 2
    else:
        source, target, shift = center, face, 3

    coeff = np.zeros((n, 6), dtype=np.float32)
    index = np.zeros((n, 6), dtype=np.int64)
    lower_orders, upper_orders = _boundary_orders(direction, derivative)

    for k in range(n):
        xx = np.zeros(6, dtype=np.float32)
        if k < 3:
            order = lower_orders[k]
            for l in range(order + 1):
                xx[l] = source[l] - target[k]
            coeff[k] = _calc_stagger_inv(xx, order, deriv_order, 0)
            index[k] = np.arange(6)
        elif k > n - 4:
            p = k - (n - 3)
            order = upper_orders[p]
            offset = 5 - order
            source_start = n - order - 1
            for l in range(order + 1):
                xx[l] = source[source_start + l] - target[k]
            coeff[k] = _calc_stagger_inv(xx, order, deriv_order, offset)
            index[k] = np.arange(n - 6, n)
        else:
            m = k - shift
            for l in range(6):
                index[k, l] = m + l
                xx[l] = source[m + l] - target[k]
            coeff[k] = _calc_stagger_inv(xx, 5, deriv_order, 0)
    return coeff, index


# =============================================================================
# Legacy fixed periodic X/Y setup
# =============================================================================

def _periodic_indices(n, direction):
    """Return six periodic stencil indices for every point on a uniform axis."""
    shift = 2 if direction == "up" else 3
    return (np.arange(n, dtype=np.int64)[:, None] - shift
            + np.arange(6, dtype=np.int64)[None, :]) % n


def _legacy_interp_constants():
    """Return the exact float32 legacy sixth-order interpolation constants."""
    return (
        np.float32(150.0 / 256.0),
        np.float32(-25.0 / 256.0),
        np.float32(3.0 / 256.0),
    )


def _legacy_deriv_constants(spacing):
    """Return the exact float32 legacy derivative constants for *spacing*."""
    h = np.float32(spacing)
    if h == 0:
        raise ValueError("grid spacing cannot be zero")
    c0 = 9.0 / 1920.0
    b0 = (-1.0 - 120.0 * c0) / 24.0
    a0 = 1.0 - 3.0 * b0 - 5.0 * c0
    return (
        np.float32(a0 / float(h)),
        np.float32(b0 / float(h)),
        np.float32(c0 / float(h)),
    )


def _infer_uniform_spacing(coord):
    """Infer one uniform spacing from the first two coordinates."""
    coord = np.asarray(coord, dtype=np.float32)
    if coord.size < 2:
        return np.float32(1.0)
    return np.float32(coord[1] - coord[0])


# =============================================================================
# Exact-order kernels
# =============================================================================

@njit(parallel=True, cache=True)
def _apply_fixed_x_interp(f, index, a, b, c):
    """Apply exact grouped legacy interpolation along X."""
    nx, ny, nz = f.shape
    out = np.empty_like(f)
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            out[i, j, k] = (
                a * (f[index[i, 3], j, k] + f[index[i, 2], j, k])
                + b * (f[index[i, 4], j, k] + f[index[i, 1], j, k])
                + c * (f[index[i, 5], j, k] + f[index[i, 0], j, k])
            )
    return out


@njit(parallel=True, cache=True)
def _apply_fixed_y_interp(f, index, a, b, c):
    """Apply exact grouped legacy interpolation along Y."""
    nx, ny, nz = f.shape
    out = np.empty_like(f)
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            out[i, j, k] = (
                a * (f[i, index[j, 3], k] + f[i, index[j, 2], k])
                + b * (f[i, index[j, 4], k] + f[i, index[j, 1], k])
                + c * (f[i, index[j, 5], k] + f[i, index[j, 0], k])
            )
    return out


@njit(parallel=True, cache=True)
def _apply_fixed_x_deriv(f, index, a, b, c):
    """Apply exact grouped legacy first derivative along X."""
    nx, ny, nz = f.shape
    out = np.empty_like(f)
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            out[i, j, k] = (
                a * (f[index[i, 3], j, k] - f[index[i, 2], j, k])
                + b * (f[index[i, 4], j, k] - f[index[i, 1], j, k])
                + c * (f[index[i, 5], j, k] - f[index[i, 0], j, k])
            )
    return out


@njit(parallel=True, cache=True)
def _apply_fixed_y_deriv(f, index, a, b, c):
    """Apply exact grouped legacy first derivative along Y."""
    nx, ny, nz = f.shape
    out = np.empty_like(f)
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            out[i, j, k] = (
                a * (f[i, index[j, 3], k] - f[i, index[j, 2], k])
                + b * (f[i, index[j, 4], k] - f[i, index[j, 1], k])
                + c * (f[i, index[j, 5], k] - f[i, index[j, 0], k])
            )
    return out


@njit(parallel=True, cache=True)
def _apply_z(f, coeff, index):
    """Apply precomputed geometry-aware coefficients along Z."""
    nx, ny, nz = f.shape
    out = np.empty_like(f)
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            value = np.float32(0.0)
            for l in range(6):
                value += coeff[k, l] * f[i, j, index[k, l]]
            out[i, j, k] = value
    return out


@njit(parallel=True, cache=True)
def _apply_product_xy(u, v, index, a, b, c, axis, derivative):
    """Fuse one staggered product and one X/Y operator without a product cube."""
    nx, ny, nz = u.shape
    out = np.empty_like(u)
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            q = i if axis == 0 else j
            t0, t1, t2 = index[q, 0], index[q, 1], index[q, 2]
            t3, t4, t5 = index[q, 3], index[q, 4], index[q, 5]
            if axis == 0:
                v0 = u[t0, j, k] * v[t0, j, k]
                v1 = u[t1, j, k] * v[t1, j, k]
                v2 = u[t2, j, k] * v[t2, j, k]
                v3 = u[t3, j, k] * v[t3, j, k]
                v4 = u[t4, j, k] * v[t4, j, k]
                v5 = u[t5, j, k] * v[t5, j, k]
            else:
                v0 = u[i, t0, k] * v[i, t0, k]
                v1 = u[i, t1, k] * v[i, t1, k]
                v2 = u[i, t2, k] * v[i, t2, k]
                v3 = u[i, t3, k] * v[i, t3, k]
                v4 = u[i, t4, k] * v[i, t4, k]
                v5 = u[i, t5, k] * v[i, t5, k]
            if derivative:
                out[i, j, k] = (a * (v3 - v2) + b * (v4 - v1)
                                + c * (v5 - v0))
            else:
                out[i, j, k] = (a * (v3 + v2) + b * (v4 + v1)
                                + c * (v5 + v0))
    return out


@njit(parallel=True, cache=True)
def _apply_product_z(u, v, coeff, index):
    """Fuse a pointwise product with the legacy six-tap nonuniform Z operator."""
    nx, ny, nz = u.shape
    out = np.empty_like(u)
    for jk in prange(ny * nz):
        k = jk // ny
        j = jk - k * ny
        for i in range(nx):
            value = np.float32(0.0)
            for l in range(6):
                q = index[k, l]
                value += coeff[k, l] * (u[i, j, q] * v[i, j, q])
            out[i, j, k] = value
    return out


def apply_product_operator(name, u, v, center, face, *, spacing=None, precomputed=None):
    """Evaluate one stagger operator on u*v without materializing the product."""
    u, v = np.asarray(u), np.asarray(v)
    if u.shape != v.shape or u.ndim != 3:
        raise ValueError("Staggered products require same-shaped 3-D arrays")
    axis, derivative = operator_axis(name), name.startswith("dd")
    direction = name[-2:]
    if axis in (0, 1):
        if spacing is None:
            spacing = _infer_uniform_spacing(center)
        a, b, c = (_legacy_deriv_constants(spacing) if derivative
                   else _legacy_interp_constants())
        return _apply_product_xy(u, v, _periodic_indices(u.shape[axis], direction),
                                 a, b, c, axis, derivative)
    coeff, index = (precomputed if precomputed is not None else
                    build_nonuniform_coefficients(center, face, direction, derivative))
    return _apply_product_z(u, v, coeff, index)


def operator_axis(name):
    """Return the numerical axis acted on by a stagger operator name."""
    if name.startswith("dd"):
        return "xyz".index(name[2])
    return "xyz".index(name[0])


def apply_operator(name, f, center, face, *, spacing=None, precomputed=None):
    """Apply one operator on a local block.

    For periodic X/Y this uses the exact grouped IDL/C arithmetic.  A local
    halo block may wrap at its artificial edge; callers crop at least five
    cells, so those artificial-edge values never enter the returned domain.
    When a requested periodic selection reaches the true global boundary the
    resolver supplies the full axis, making the wrap physically correct.
    """
    derivative = name.startswith("dd")
    axis = operator_axis(name)
    direction = name[-2:]
    f = np.asarray(f, dtype=np.float32, order="F")
    if f.ndim != 3:
        raise ValueError("stagger expects a 3-D array")

    if axis in (0, 1):
        # Global and halo-local X/Y both use periodic fixed-coefficient stencils.
        # Artificial local-edge outputs are cropped by the five-cell halo.
        index = _periodic_indices(f.shape[axis], direction)
        if derivative:
            if spacing is None:
                spacing = _infer_uniform_spacing(center)
            a, b, c = _legacy_deriv_constants(spacing)
            return (_apply_fixed_x_deriv if axis == 0 else _apply_fixed_y_deriv)(f, index, a, b, c)
        a, b, c = _legacy_interp_constants()
        return (_apply_fixed_x_interp if axis == 0 else _apply_fixed_y_interp)(f, index, a, b, c)

    coeff, index = (precomputed if precomputed is not None
                    else build_nonuniform_coefficients(center, face, direction, derivative))
    return _apply_z(f, coeff, index)


class StaggerGrid:
    """Precomputed exact operators for one full Bifrost mesh."""

    def __init__(self, x, xdn, y, ydn, z, zdn, dx=None, dy=None):
        """Precompute full-grid periodic X/Y and nonuniform Z stencils."""
        self.x = np.asarray(x, dtype=np.float32)
        self.x_face = np.asarray(xdn, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.y_face = np.asarray(ydn, dtype=np.float32)
        self.z = np.asarray(z, dtype=np.float32)
        self.z_face = np.asarray(zdn, dtype=np.float32)
        self.shape = (self.x.size, self.y.size, self.z.size)
        self.dx = np.float32(_infer_uniform_spacing(self.x) if dx is None else dx)
        self.dy = np.float32(_infer_uniform_spacing(self.y) if dy is None else dy)

        self._xup_i = _periodic_indices(self.x.size, "up")
        self._xdn_i = _periodic_indices(self.x.size, "dn")
        self._yup_i = _periodic_indices(self.y.size, "up")
        self._ydn_i = _periodic_indices(self.y.size, "dn")
        self._interp_abc = _legacy_interp_constants()
        self._ddx_abc = _legacy_deriv_constants(self.dx)
        self._ddy_abc = _legacy_deriv_constants(self.dy)

        self._zops = {}
        for direction in ("up", "dn"):
            self._zops[direction] = build_nonuniform_coefficients(self.z, self.z_face, direction, False)
            self._zops["dd" + direction] = build_nonuniform_coefficients(self.z, self.z_face, direction, True)

    def _check(self, f):
        """Validate full-domain float32 input required for exact operators."""
        f = np.asarray(f)
        if f.ndim != 3:
            raise ValueError(f"stagger expects a 3-D array, got {f.ndim}-D")
        if f.shape != self.shape:
            raise ValueError(f"array shape {f.shape} does not match stagger mesh {self.shape}")
        if f.dtype != np.float32:
            raise TypeError(f"stagger expects float32 arrays for IDL/C reproducibility; got {f.dtype}")
        return f

    def xup(self, f):
        """Interpolate x-face values up to centers along X."""
        f = self._check(f); return _apply_fixed_x_interp(f, self._xup_i, *self._interp_abc)
    def xdn(self, f):
        """Interpolate centered values down to x-faces along X."""
        f = self._check(f); return _apply_fixed_x_interp(f, self._xdn_i, *self._interp_abc)
    def yup(self, f):
        """Interpolate y-face values up to centers along Y."""
        f = self._check(f); return _apply_fixed_y_interp(f, self._yup_i, *self._interp_abc)
    def ydn(self, f):
        """Interpolate centered values down to y-faces along Y."""
        f = self._check(f); return _apply_fixed_y_interp(f, self._ydn_i, *self._interp_abc)
    def zup(self, f):
        """Interpolate z-face values up to centers along Z."""
        f = self._check(f); coeff, index = self._zops["up"]; return _apply_z(f, coeff, index)
    def zdn(self, f):
        """Interpolate centered values down to z-faces along Z."""
        f = self._check(f); coeff, index = self._zops["dn"]; return _apply_z(f, coeff, index)
    def ddxup(self, f):
        """Differentiate x-face values up to centers along X."""
        f = self._check(f); return _apply_fixed_x_deriv(f, self._xup_i, *self._ddx_abc)
    def ddxdn(self, f):
        """Differentiate centered values down to x-faces along X."""
        f = self._check(f); return _apply_fixed_x_deriv(f, self._xdn_i, *self._ddx_abc)
    def ddyup(self, f):
        """Differentiate y-face values up to centers along Y."""
        f = self._check(f); return _apply_fixed_y_deriv(f, self._yup_i, *self._ddy_abc)
    def ddydn(self, f):
        """Differentiate centered values down to y-faces along Y."""
        f = self._check(f); return _apply_fixed_y_deriv(f, self._ydn_i, *self._ddy_abc)
    def ddzup(self, f):
        """Differentiate z-face values up to centers along Z."""
        f = self._check(f); coeff, index = self._zops["ddup"]; return _apply_z(f, coeff, index)
    def ddzdn(self, f):
        """Differentiate centered values down to z-faces along Z."""
        f = self._check(f); coeff, index = self._zops["dddn"]; return _apply_z(f, coeff, index)


class StaggerInterface:
    """Public IDL-like stagger interface bound to one BifrostData run."""

    def __init__(self, data):
        """Bind public stagger operations to the mesh of *data*."""
        self._data = data
        meta = data._grid_meta()
        dx = data._legacy_axis_spacing(meta, 0)
        dy = data._legacy_axis_spacing(meta, 1)
        self.grid = StaggerGrid(meta.x, meta.xdn, meta.y, meta.ydn, meta.z, meta.zdn, dx=dx, dy=dy)

    def _apply(self, name, f, *, br=False):
        """Apply one full-domain stagger operator in native or physical orientation.

        This interactive interface intentionally accepts only arrays covering
        the complete run mesh.  It has no selection metadata or halo context, so
        a cropped ndarray cannot safely represent periodic X/Y boundaries or the
        global nonuniform Z geometry.  For subdomains use ``get_var(...)`` and let
        the resolver expand/crop the required stagger halo.
        """
        arr = np.asarray(f)
        if arr.ndim != 3:
            raise ValueError(
                "dd.stagger.* expects a full-domain 3-D array; for subdomains "
                "use get_var(..., centered=True) or a registered derived quantity"
            )
        if tuple(arr.shape) != tuple(self.grid.shape):
            raise ValueError(
                f"Array shape {arr.shape} does not match the full run mesh {self.grid.shape}; "
                "dd.stagger.* cannot infer the global geometry of a cropped subdomain"
            )
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)

        if br:
            native = np.asfortranarray(arr)
        else:
            # The public physical frame differs from native Bifrost by a Y/Z
            # reversal.  Use the same parallel F-contiguous transform as
            # get_var() instead of materializing a negative-stride view.
            native = self._data._reverse_yz_f(arr)

        out = getattr(self.grid, name)(native)
        if br:
            return np.asfortranarray(out)

        axis = operator_axis(name)
        negate = name.startswith("dd") and axis in (1, 2)
        return self._data._reverse_yz_f(out, negate=negate)

    def xup(self, f, *, br=False):
        """Interpolate x-face values up to cell centers along X."""
        return self._apply("xup", f, br=br)
    def xdn(self, f, *, br=False):
        """Interpolate cell-centered values down to x-faces along X."""
        return self._apply("xdn", f, br=br)
    def yup(self, f, *, br=False):
        """Interpolate y-face values up to cell centers along Y."""
        return self._apply("yup", f, br=br)
    def ydn(self, f, *, br=False):
        """Interpolate cell-centered values down to y-faces along Y."""
        return self._apply("ydn", f, br=br)
    def zup(self, f, *, br=False):
        """Interpolate z-face values up to cell centers along Z."""
        return self._apply("zup", f, br=br)
    def zdn(self, f, *, br=False):
        """Interpolate cell-centered values down to z-faces along Z."""
        return self._apply("zdn", f, br=br)
    def ddxup(self, f, *, br=False):
        """Differentiate x-face values up to cell centers along X."""
        return self._apply("ddxup", f, br=br)
    def ddxdn(self, f, *, br=False):
        """Differentiate cell-centered values down to x-faces along X."""
        return self._apply("ddxdn", f, br=br)
    def ddyup(self, f, *, br=False):
        """Differentiate y-face values up to cell centers along Y."""
        return self._apply("ddyup", f, br=br)
    def ddydn(self, f, *, br=False):
        """Differentiate cell-centered values down to y-faces along Y."""
        return self._apply("ddydn", f, br=br)
    def ddzup(self, f, *, br=False):
        """Differentiate z-face values up to cell centers along Z."""
        return self._apply("ddzup", f, br=br)
    def ddzdn(self, f, *, br=False):
        """Differentiate cell-centered values down to z-faces along Z."""
        return self._apply("ddzdn", f, br=br)
