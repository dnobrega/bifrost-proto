"""Bifrost rho-e equation-of-state table loading and interpolation.

Canonical EOS tables are uniform in ``ln(rho)`` and ``ln(e)`` and are sampled
with fused bilinear Numba kernels.  Public quantities decide separately whether
to use stored/HION thermodynamics or explicit EOS fallbacks.
"""

from __future__ import annotations

import ast
import os

import numpy as np
from numba import njit, prange

from .array_utils import _flat_memory_view


def read_idl_ascii(filename):
    """Read the simple IDL-style key=value files used by Bifrost."""

    params = {}
    with open(filename) as fp:
        for line in fp:
            line, _, _ = line.partition(";")
            key, sep, value = line.partition("=")
            if not sep:
                continue
            key = key.strip().lower()
            value = value.strip()
            if not key or not value:
                continue
            if value.lower() in {".false.", ".true."}:
                params[key] = value.lower() == ".true."
                continue
            try:
                parsed = ast.literal_eval(value)
                # IDL character parameters are often fixed-width strings, so
                # leading/trailing blanks may live *inside* the quotes. They
                # are padding, not part of filenames/variable names.
                params[key] = parsed.strip() if isinstance(parsed, str) else parsed
            except (ValueError, SyntaxError):
                try:
                    params[key] = float(value.replace("D", "e").replace("d", "e"))
                except ValueError:
                    params[key] = value.strip()
    return params


@njit(cache=True, parallel=True)
def _eos_interp_flat(
    rho,
    ee,
    table,
    lnei0,
    inv_dlnei,
    nei,
    lnrho0,
    inv_dlnrho,
    nrho,
    exponentiate,
    log_ee_scale,
    log_rho_scale,
    out,
):
    """Bilinear rho-e EOS interpolation for contiguous 1-D memory views."""
    for p in prange(out.size):
        # Keep the same coordinate definition as the historical NumPy/SciPy
        # path. The table axes are uniformly spaced in ln(ee) and ln(rho).
        # Promote explicitly: with NUMBA_DISABLE_JIT and NumPy >= 2 (NEP 50),
        # float32 scalars would otherwise keep the whole interpolant in float32.
        x = (np.log(np.float64(ee[p])) + log_ee_scale - lnei0) * inv_dlnei
        y = (np.log(np.float64(rho[p])) + log_rho_scale - lnrho0) * inv_dlnrho

        # scipy.ndimage.map_coordinates(..., order=1, mode="nearest") is
        # equivalent to clipping each coordinate to the valid table domain
        # before evaluating the local linear interpolant.
        if x < 0.0:
            x = 0.0
        elif x > nei - 1.0:
            x = nei - 1.0

        if y < 0.0:
            y = 0.0
        elif y > nrho - 1.0:
            y = nrho - 1.0

        i0 = int(np.floor(x))
        j0 = int(np.floor(y))
        i0 = min(i0, nei - 2)
        j0 = min(j0, nrho - 2)

        fx = x - i0
        fy = y - j0

        q00 = np.float64(table[i0, j0])
        q10 = np.float64(table[i0 + 1, j0])
        q01 = np.float64(table[i0, j0 + 1])
        q11 = np.float64(table[i0 + 1, j0 + 1])

        value = (
            (1.0 - fx) * (1.0 - fy) * q00
            + fx * (1.0 - fy) * q10
            + (1.0 - fx) * fy * q01
            + fx * fy * q11
        )

        if exponentiate:
            value = np.exp(value)

        out[p] = value


@njit(cache=True, parallel=True)
def _eos_interp_strided_3d(
    rho,
    ee,
    table,
    lnei0,
    inv_dlnei,
    nei,
    lnrho0,
    inv_dlnrho,
    nrho,
    exponentiate,
    log_ee_scale,
    log_rho_scale,
    out,
):
    """Stride-aware 3-D EOS interpolation without contiguizing the inputs."""
    nx, ny, nz = rho.shape

    # Bifrost arrays are normally Fortran ordered, so keep x as the innermost
    # index. NumPy strides are nevertheless honored for arbitrary 3-D views.
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                x = (np.log(np.float64(ee[i, j, k])) + log_ee_scale - lnei0) * inv_dlnei
                y = (
                    np.log(np.float64(rho[i, j, k])) + log_rho_scale - lnrho0
                ) * inv_dlnrho

                if x < 0.0:
                    x = 0.0
                elif x > nei - 1.0:
                    x = nei - 1.0

                if y < 0.0:
                    y = 0.0
                elif y > nrho - 1.0:
                    y = nrho - 1.0

                i0 = int(np.floor(x))
                j0 = int(np.floor(y))
                i0 = min(i0, nei - 2)
                j0 = min(j0, nrho - 2)

                fx = x - i0
                fy = y - j0

                q00 = np.float64(table[i0, j0])
                q10 = np.float64(table[i0 + 1, j0])
                q01 = np.float64(table[i0, j0 + 1])
                q11 = np.float64(table[i0 + 1, j0 + 1])

                value = (
                    (1.0 - fx) * (1.0 - fy) * q00
                    + fx * (1.0 - fy) * q10
                    + (1.0 - fx) * fy * q01
                    + fx * fy * q11
                )

                if exponentiate:
                    value = np.exp(value)

                out[i, j, k] = value


def _same_contiguous_layout(*arrays):
    """Return True when all arrays are contiguous in a compatible layout."""
    all_c = all(a.flags.c_contiguous for a in arrays)
    all_f = all(a.flags.f_contiguous for a in arrays)
    return all_c or all_f


def _eos_interp_general(
    rho,
    ee,
    table,
    lnei0,
    inv_dlnei,
    lnrho0,
    inv_dlnrho,
    exponentiate,
):
    """Vectorized fallback for unusual broadcast/lower-dimensional inputs.

    Normal Bifrost cubes use the fused Numba kernels above. This path keeps
    ``EOSTable.interpolate`` general without exposing or maintaining a second
    SciPy interpolation method in the runtime API.
    """
    x = (np.log(ee) - lnei0) * inv_dlnei
    y = (np.log(rho) - lnrho0) * inv_dlnrho

    ne, nrho = table.shape
    x = np.clip(x, 0.0, ne - 1.0)
    y = np.clip(y, 0.0, nrho - 1.0)

    i0 = np.minimum(np.floor(x).astype(np.int64), ne - 2)
    j0 = np.minimum(np.floor(y).astype(np.int64), nrho - 2)
    fx = x - i0
    fy = y - j0

    q00 = table[i0, j0]
    q10 = table[i0 + 1, j0]
    q01 = table[i0, j0 + 1]
    q11 = table[i0 + 1, j0 + 1]

    value = (
        (1.0 - fx) * (1.0 - fy) * q00
        + fx * (1.0 - fy) * q10
        + (1.0 - fx) * fy * q01
        + fx * fy * q11
    )
    if exponentiate:
        value = np.exp(value)
    return np.asarray(value, dtype=np.float32)


@njit(cache=True)
def _gamma1_cell(
    rho,
    ee,
    table,
    lnei0,
    inv_dle,
    lnrho0,
    inv_dlr,
    log_ee_scale,
    log_rho_scale,
    product_scale,
):
    """Evaluate Gamma_1 from one bilinear log-pressure interpolant."""
    rho = np.float64(rho)  # float64 arithmetic with and without JIT
    ee = np.float64(ee)
    x = (np.log(ee) + log_ee_scale - lnei0) * inv_dle
    y = (np.log(rho) + log_rho_scale - lnrho0) * inv_dlr
    x = max(0.0, min(x, table.shape[0] - 1.0))
    y = max(0.0, min(y, table.shape[1] - 1.0))
    i = min(int(np.floor(x)), table.shape[0] - 2)
    j = min(int(np.floor(y)), table.shape[1] - 2)
    fx, fy = x - i, y - j
    q00 = np.float64(table[i, j])
    q10 = np.float64(table[i + 1, j])
    q01 = np.float64(table[i, j + 1])
    q11 = np.float64(table[i + 1, j + 1])
    lnpg = (
        (1.0 - fx) * (1.0 - fy) * q00
        + fx * (1.0 - fy) * q10
        + (1.0 - fx) * fy * q01
        + fx * fy * q11
    )
    chi_e = ((1.0 - fy) * (q10 - q00) + fy * (q11 - q01)) * inv_dle
    chi_r = ((1.0 - fx) * (q01 - q00) + fx * (q11 - q10)) * inv_dlr
    return chi_r + chi_e * np.exp(lnpg) / (rho * ee * product_scale)


@njit(cache=True, parallel=True)
def _gamma1_flat(
    rho,
    ee,
    table,
    lnei0,
    inv_dle,
    lnrho0,
    inv_dlr,
    log_ee_scale,
    log_rho_scale,
    product_scale,
    out,
):
    """Evaluate Gamma_1 for contiguous C/F arrays with one output buffer."""
    for p in prange(out.size):
        out[p] = _gamma1_cell(
            rho[p],
            ee[p],
            table,
            lnei0,
            inv_dle,
            lnrho0,
            inv_dlr,
            log_ee_scale,
            log_rho_scale,
            product_scale,
        )


@njit(cache=True, parallel=True)
def _gamma1_strided(
    rho,
    ee,
    table,
    lnei0,
    inv_dle,
    lnrho0,
    inv_dlr,
    log_ee_scale,
    log_rho_scale,
    product_scale,
    out,
):
    """Evaluate Gamma_1 on noncontiguous 3-D views without input copies."""
    nx, ny, nz = out.shape
    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                out[i, j, k] = _gamma1_cell(
                    rho[i, j, k],
                    ee[i, j, k],
                    table,
                    lnei0,
                    inv_dle,
                    lnrho0,
                    inv_dlr,
                    log_ee_scale,
                    log_rho_scale,
                    product_scale,
                )


class EOSTable:
    """Minimal rho-e Bifrost EOS reader/interpolator."""

    def __init__(self, tabparam, *, fdir=".", dtype="f4", big_endian=False):
        """Load EOS axis metadata and memory-map the four Bifrost tables."""
        self.fdir = os.path.abspath(fdir)
        self.params = read_idl_ascii(tabparam)
        p = self.params

        self.lnrho = np.linspace(
            np.log(p["rhomin"]), np.log(p["rhomax"]), int(p["nrhobin"])
        )
        self.lnei = np.linspace(
            np.log(p["eimin"]), np.log(p["eimax"]), int(p["neibin"])
        )
        self.dlnrho = self.lnrho[1] - self.lnrho[0]
        self.dlnei = self.lnei[1] - self.lnei[0]

        tablefile = str(p["eostablefile"]).strip().strip("'\"").strip()
        tablefile = (
            tablefile
            if os.path.isabs(tablefile)
            else os.path.join(self.fdir, tablefile)
        )
        dt = (">" if big_endian else "<") + dtype
        table = np.memmap(
            tablefile,
            mode="r",
            shape=(int(p["neibin"]), int(p["nrhobin"]), 4),
            dtype=dt,
            order="F",
        )

        # np.asarray(..., dtype=np.float32) remains a zero-copy view for the
        # usual native-endian f4 tables. For big-endian input it performs a
        # one-time conversion to a Numba-compatible native-endian array.
        self.lnpg = np.asarray(table[:, :, 0], dtype=np.float32)
        self.tg = np.asarray(table[:, :, 1], dtype=np.float32)
        self.lnne = np.asarray(table[:, :, 2], dtype=np.float32)
        self.lnkr = np.asarray(table[:, :, 3], dtype=np.float32)

    def _validate_inputs(self, rho, ee, *, rho_scale=1.0, ee_scale=1.0):
        """Reject invalid thermodynamics before unchecked Numba table indexing.

        Ranges use a small tolerance to accommodate float32 rounding of
        otherwise valid table-boundary values; genuine outliers still fail.
        """
        ranges = (
            ("rho", rho, self.lnrho, rho_scale),
            ("specific internal energy", ee, self.lnei, ee_scale),
        )
        for name, values, logaxis, scale in ranges:
            low, high = float(np.exp(logaxis[0])), float(np.exp(logaxis[-1]))
            # A reduction does not materialize any full-cube temporary apart
            # from NumPy's finite/range masks, and is executed outside Numba.
            minimum, maximum = (
                float(np.min(values)) * scale,
                float(np.max(values)) * scale,
            )
            if not (np.isfinite(minimum) and np.isfinite(maximum)):
                raise ValueError(f"EOS {name} contains NaN or infinite values")
            if minimum < low * (1 - 1e-6) or maximum > high * (1 + 1e-6):
                raise ValueError(
                    f"EOS {name} is outside its table range [{low:.8g}, {high:.8g}] "
                    f"(data min={minimum:.8g}, max={maximum:.8g}); "
                    "select the EOS table matching this simulation"
                )
            if minimum <= 0.0:
                raise ValueError(f"EOS {name} must be positive; minimum={minimum}")

    def _table(self, out):
        """Return the stored 2-D EOS table associated with an output name."""
        try:
            return {
                "tg": self.tg,
                "pg": self.lnpg,
                "ne": self.lnne,
                "kr": self.lnkr,
            }[out]
        except KeyError as exc:
            raise ValueError(f"Unknown EOS output {out!r}") from exc

    def interpolate(self, rho_cgs, ee_cgs, out="tg", *, rho_scale=1.0, ee_scale=1.0):
        """Interpolate a Bifrost rho-e EOS quantity with a fused Numba kernel.

        The numerical definition matches the historical SciPy implementation:
        bilinear interpolation on the uniform ``ln(ee)``/``ln(rho)`` grid with
        nearest-boundary behavior outside the table. ``pg``, ``ne`` and ``kr``
        are stored logarithmically and exponentiated after interpolation;
        ``tg`` is stored linearly.
        """
        table = self._table(out)
        rho, ee = np.broadcast_arrays(np.asarray(rho_cgs), np.asarray(ee_cgs))
        self._validate_inputs(rho, ee, rho_scale=rho_scale, ee_scale=ee_scale)

        # The normal Bifrost path is a same-shaped 3-D float array. Preserve
        # the input memory order for the output and never modify the inputs.
        order = "F" if rho.flags.f_contiguous and not rho.flags.c_contiguous else "C"
        result = np.empty(rho.shape, dtype=np.float32, order=order)

        exponentiate = out != "tg"
        args = (
            table,
            float(self.lnei[0]),
            float(1.0 / self.dlnei),
            int(table.shape[0]),
            float(self.lnrho[0]),
            float(1.0 / self.dlnrho),
            int(table.shape[1]),
            exponentiate,
            float(np.log(ee_scale)),
            float(np.log(rho_scale)),
        )

        if _same_contiguous_layout(rho, ee, result):
            _eos_interp_flat(
                _flat_memory_view(rho),
                _flat_memory_view(ee),
                *args,
                _flat_memory_view(result),
            )
            return result

        if rho.ndim == 3 and ee.ndim == 3:
            _eos_interp_strided_3d(rho, ee, *args, result)
            return result

        # Broadcasting or unusual lower-dimensional strided arrays are not a
        # hot path in Bifrost analysis. Keep complete API compatibility with a
        # vectorized NumPy fallback rather than retaining a second SciPy
        # interpolation implementation in the runtime API.
        return _eos_interp_general(
            rho * rho_scale,
            ee * ee_scale,
            table,
            float(self.lnei[0]),
            float(1.0 / self.dlnei),
            float(self.lnrho[0]),
            float(1.0 / self.dlnrho),
            exponentiate,
        )

    def pressure_log_derivatives(self, rho_cgs, ee_cgs):
        """Return local logarithmic pressure derivatives of the EOS.

        The pressure table is linearly interpolated in ``ln(ee)`` and
        ``ln(rho)``. This routine differentiates that same bilinear
        interpolant analytically, avoiding a separate finite-difference grid.

        Returns
        -------
        chi_rho, chi_ee : ndarray
            ``d ln p / d ln rho |_ee`` and ``d ln p / d ln ee |_rho``.
        """
        rho_cgs, ee_cgs = np.broadcast_arrays(rho_cgs, ee_cgs)
        self._validate_inputs(rho_cgs, ee_cgs)
        x = (np.log(ee_cgs) - self.lnei[0]) / self.dlnei
        y = (np.log(rho_cgs) - self.lnrho[0]) / self.dlnrho

        ne = self.lnpg.shape[0]
        nr = self.lnpg.shape[1]
        x = np.clip(x, 0.0, ne - 1.0)
        y = np.clip(y, 0.0, nr - 1.0)

        i0 = np.minimum(np.floor(x).astype(np.int64), ne - 2)
        j0 = np.minimum(np.floor(y).astype(np.int64), nr - 2)
        fx = x - i0
        fy = y - j0

        q00 = np.asarray(self.lnpg[i0, j0], dtype=np.float64)
        q10 = np.asarray(self.lnpg[i0 + 1, j0], dtype=np.float64)
        q01 = np.asarray(self.lnpg[i0, j0 + 1], dtype=np.float64)
        q11 = np.asarray(self.lnpg[i0 + 1, j0 + 1], dtype=np.float64)

        dlnp_dx = (1.0 - fy) * (q10 - q00) + fy * (q11 - q01)
        dlnp_dy = (1.0 - fx) * (q01 - q00) + fx * (q11 - q10)

        chi_ee = dlnp_dx / self.dlnei
        chi_rho = dlnp_dy / self.dlnrho
        return (
            chi_rho.astype(np.float32, copy=False),
            chi_ee.astype(np.float32, copy=False),
        )

    def gamma1(self, rho_cgs, ee_cgs, *, rho_scale=1.0, ee_scale=1.0):
        """Gamma_1 with a fused Numba kernel and one float32 output cube.

        Avoids the dozens of float64-sized NumPy temporaries formerly produced
        by pressure_log_derivatives() + interpolate() + vectorized algebra.
        """
        rho, ee = np.broadcast_arrays(np.asarray(rho_cgs), np.asarray(ee_cgs))
        self._validate_inputs(rho, ee, rho_scale=rho_scale, ee_scale=ee_scale)
        order = "F" if rho.flags.f_contiguous and not rho.flags.c_contiguous else "C"
        out = np.empty(rho.shape, dtype=np.float32, order=order)
        args = (
            self.lnpg,
            float(self.lnei[0]),
            float(1.0 / self.dlnei),
            float(self.lnrho[0]),
            float(1.0 / self.dlnrho),
            float(np.log(ee_scale)),
            float(np.log(rho_scale)),
            float(ee_scale * rho_scale),
        )
        if _same_contiguous_layout(rho, ee, out):
            _gamma1_flat(
                _flat_memory_view(rho),
                _flat_memory_view(ee),
                *args,
                _flat_memory_view(out),
            )
        elif rho.ndim == 3:
            _gamma1_strided(rho, ee, *args, out)
        else:
            # Small general/broadcasted inputs; retain a simple fallback.
            pg = self.interpolate(
                rho, ee, out="pg", rho_scale=rho_scale, ee_scale=ee_scale
            )
            chi_r, chi_e = self.pressure_log_derivatives(rho * rho_scale, ee * ee_scale)
            out[...] = chi_r + chi_e * pg / (rho * ee * rho_scale * ee_scale)
        return out
