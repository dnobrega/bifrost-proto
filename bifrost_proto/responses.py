"""Runtime interpolation of synthetic AIA, XRT, and EUI response tables.

Response resources are pre-built NumPy arrays.  Runtime evaluation therefore
depends only on NumPy/Numba and preserves the source-table density and
temperature dependence without requiring SolarSoft, CHIANTI, or Astropy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit, prange

from .array_utils import _flat_memory_view


def _uniform_axis_info(axis):
    """Return ``(is_uniform, step)`` for a monotonic response axis.

    Legacy IDL response axes are often stored as float32 values even when the
    intended grid is exactly uniform.  Consecutive differences therefore vary
    at the ~float32-ULP level after conversion to float64.  Detect uniformity
    using a tolerance tied to the represented axis scale, while deriving the
    interpolation step from the actual first interval to preserve the legacy
    IDL indexing convention.  No response origin or spacing is hardcoded.
    """
    axis = np.asarray(axis, dtype=np.float64)
    if axis.ndim != 1 or axis.size < 2:
        raise ValueError("Response axis must be one-dimensional with at least two points")

    delta = np.diff(axis)
    if not np.all(delta > 0.0):
        return False, np.nan

    representative = float(np.median(delta))
    scale = max(1.0, float(np.max(np.abs(axis))))
    atol = 8.0 * np.finfo(np.float32).eps * scale
    uniform = bool(np.allclose(delta, representative, rtol=5e-5, atol=atol))

    # Match the historical IDL recipes, which use axis[1]-axis[0] as dlog.
    return uniform, float(delta[0])


@njit(cache=True, parallel=True)
def _uniform_response_emissivity_flat(
    temperature,
    electron_density,
    hydrogen_density,
    table,
    logt0,
    logt_max,
    inv_dlogt,
    nlogt,
    logne0,
    logne_max,
    inv_dlogne,
    nlogne,
    fill_value,
    out,
):
    """Fused bilinear response interpolation and ne*nH multiplication."""
    for i in prange(out.size):
        t = temperature[i]
        ne = electron_density[i]
        nh = hydrogen_density[i]

        if (
            not np.isfinite(t)
            or not np.isfinite(ne)
            or t <= 0.0
            or ne <= 0.0
        ):
            out[i] = ne * nh * fill_value
            continue

        lt = np.log10(t)
        ln = np.log10(ne)

        if (
            lt < logt0
            or lt > logt_max
            or ln < logne0
            or ln > logne_max
        ):
            out[i] = ne * nh * fill_value
            continue

        xt = (lt - logt0) * inv_dlogt
        xn = (ln - logne0) * inv_dlogne

        it = int(np.floor(xt))
        ine = int(np.floor(xn))

        it = min(it, nlogt - 2)
        ine = min(ine, nlogne - 2)

        ft = xt - it
        fn = xn - ine

        q00 = table[ine, it]
        q10 = table[ine + 1, it]
        q01 = table[ine, it + 1]
        q11 = table[ine + 1, it + 1]

        g = (
            (1.0 - fn) * (1.0 - ft) * q00
            + fn * (1.0 - ft) * q10
            + (1.0 - fn) * ft * q01
            + fn * ft * q11
        )

        # The legacy NumPy path casts the interpolated response to float32
        # before multiplying by ne*nH. Keep the same rounding point here.
        g32 = np.float32(g)
        out[i] = ne * nh * g32


@njit(cache=True, parallel=True)
def _uniform_response_emissivity_strided(
    temperature,
    electron_density,
    hydrogen_density,
    table,
    logt0,
    logt_max,
    inv_dlogt,
    nlogt,
    logne0,
    logne_max,
    inv_dlogne,
    nlogne,
    fill_value,
    out,
):
    """Stride-aware 3-D fused response kernel without input copies."""
    nx, ny, nz = temperature.shape

    for k in prange(nz):
        for j in range(ny):
            for i in range(nx):
                t = temperature[i, j, k]
                ne = electron_density[i, j, k]
                nh = hydrogen_density[i, j, k]

                if (
                    not np.isfinite(t)
                    or not np.isfinite(ne)
                    or t <= 0.0
                    or ne <= 0.0
                ):
                    out[i, j, k] = ne * nh * fill_value
                    continue

                lt = np.log10(t)
                ln = np.log10(ne)

                if (
                    lt < logt0
                    or lt > logt_max
                    or ln < logne0
                    or ln > logne_max
                ):
                    out[i, j, k] = ne * nh * fill_value
                    continue

                xt = (lt - logt0) * inv_dlogt
                xn = (ln - logne0) * inv_dlogne

                it = int(np.floor(xt))
                ine = int(np.floor(xn))

                it = min(it, nlogt - 2)
                ine = min(ine, nlogne - 2)

                ft = xt - it
                fn = xn - ine

                q00 = table[ine, it]
                q10 = table[ine + 1, it]
                q01 = table[ine, it + 1]
                q11 = table[ine + 1, it + 1]

                g = (
                    (1.0 - fn) * (1.0 - ft) * q00
                    + fn * (1.0 - ft) * q10
                    + (1.0 - fn) * ft * q01
                    + fn * ft * q11
                )
                out[i, j, k] = ne * nh * np.float32(g)



INSTRUMENT_RESPONSE_NAMES = (
    "aia94",
    "aia131",
    "aia171",
    "aia193",
    "aia211",
    "aia304",
    "aia335",
    "xrt_almed",
    "xrt_almesh",
    "xrt_alpoly",
    "xrt_althick",
    "xrt_bemed",
    "xrt_bethick",
    "xrt_bethin",
    "xrt_cpoly",
    "xrt_tipoly",
    "eui174",
)


def default_responses_dir() -> Path:
    """Return the instrument-response resource directory shipped with the package."""
    return Path(__file__).resolve().parent / "resources" / "instrument_responses"


@dataclass(frozen=True)
class ResponseFiles:
    """Filesystem paths comprising one converted instrument response."""

    directory: Path
    table: Path
    logt: Path
    logne: Path
    metadata: Path


def response_files(base_dir, name: str) -> ResponseFiles:
    """Return canonical resource paths for response *name* under *base_dir*."""
    base = Path(base_dir)
    directory = base / name
    return ResponseFiles(
        directory=directory,
        table=directory / "table.npy",
        logt=directory / "logt.npy",
        logne=directory / "logne.npy",
        metadata=directory / "metadata.json",
    )


class InstrumentResponseTable:
    """Density- and temperature-dependent synthetic instrument response.

    The canonical on-disk representation is

    ``table[ine, it] = G(log10(ne/cm^-3), log10(T/K))``

    and recipes multiply the interpolated response by ``ne * nH``.  The
    resulting local synthetic count-rate contribution therefore has units
    ``DN cm^-1 s^-1 pix^-1``.
    """

    def __init__(self, name: str, *, base_dir=None):
        """Load response axes, table, and provenance metadata from disk."""
        self.name = str(name)
        if base_dir is None:
            base_dir = default_responses_dir()
        self.files = response_files(base_dir, self.name)

        missing = [
            path.name
            for path in (
                self.files.table,
                self.files.logt,
                self.files.logne,
                self.files.metadata,
            )
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Instrument response {self.name!r} is incomplete in "
                f"{str(self.files.directory)!r}; missing: {', '.join(missing)}. "
                "Build/copy the response tables with tools/build_instrument_responses/."
            )

        with self.files.metadata.open() as fp:
            self.metadata = json.load(fp)

        meta_name = self.metadata.get("name")
        if meta_name is not None and meta_name != self.name:
            raise ValueError(
                f"Response metadata name {meta_name!r} does not match requested {self.name!r}"
            )

        self.logt = np.asarray(np.load(self.files.logt, mmap_mode="r", allow_pickle=False), dtype=np.float64)
        self.logne = np.asarray(np.load(self.files.logne, mmap_mode="r", allow_pickle=False), dtype=np.float64)
        self.table = np.load(self.files.table, mmap_mode="r", allow_pickle=False)

        if self.logt.ndim != 1 or self.logne.ndim != 1:
            raise ValueError("Response logt.npy and logne.npy must both be one-dimensional")
        if self.logt.size < 2 or self.logne.size < 2:
            raise ValueError("Response axes must each contain at least two points")
        if not np.all(np.diff(self.logt) > 0.0):
            raise ValueError("Response log10(T) axis must be strictly increasing")
        if not np.all(np.diff(self.logne) > 0.0):
            raise ValueError("Response log10(ne) axis must be strictly increasing")
        expected = (self.logne.size, self.logt.size)
        if self.table.shape != expected:
            raise ValueError(
                f"Response table shape {self.table.shape!r} does not match canonical "
                f"(n_logne, n_logt)={expected!r}"
            )

        self.fill_value = float(self.metadata.get("fill_value", 0.0))
        self._dt = np.diff(self.logt)
        self._dne = np.diff(self.logne)
        self._uniform_t, self._dlogt = _uniform_axis_info(self.logt)
        self._uniform_ne, self._dlogne = _uniform_axis_info(self.logne)

    def evaluate_emissivity(self, temperature, electron_density, hydrogen_density):
        """Return ``ne * nH * G(T, ne)`` using a fused Numba kernel.

        This is the fast path used by the public synthetic-instrument recipes.
        The response axes must be uniform in log10(T) and log10(ne), which is
        the canonical format produced by ``tools/build_instrument_responses``.
        C- and F-contiguous arrays are evaluated through copy-free 1-D views;
        arbitrary 3-D strided views use a stride-aware kernel directly.

        For non-uniform response axes or unsupported array layouts, the method
        falls back to the general ``evaluate`` implementation to preserve the
        full public semantics. Resolver/cache inputs are never modified in-place.
        """
        temperature, electron_density, hydrogen_density = np.broadcast_arrays(
            temperature, electron_density, hydrogen_density
        )

        if not (self._uniform_t and self._uniform_ne):
            return (
                electron_density
                * hydrogen_density
                * self.evaluate(temperature, electron_density)
            )

        out = np.empty_like(temperature, dtype=np.float32, order="K", subok=False)

        all_c = (
            temperature.flags.c_contiguous
            and electron_density.flags.c_contiguous
            and hydrogen_density.flags.c_contiguous
            and out.flags.c_contiguous
        )
        all_f = (
            temperature.flags.f_contiguous
            and electron_density.flags.f_contiguous
            and hydrogen_density.flags.f_contiguous
            and out.flags.f_contiguous
        )

        args = (
            self.table,
            float(self.logt[0]),
            float(self.logt[-1]),
            1.0 / float(self._dlogt),
            int(self.logt.size),
            float(self.logne[0]),
            float(self.logne[-1]),
            1.0 / float(self._dlogne),
            int(self.logne.size),
            float(self.fill_value),
        )

        if all_c or all_f:
            _uniform_response_emissivity_flat(
                _flat_memory_view(temperature),
                _flat_memory_view(electron_density),
                _flat_memory_view(hydrogen_density),
                *args,
                _flat_memory_view(out),
            )
            return out

        if temperature.ndim == 3:
            _uniform_response_emissivity_strided(
                temperature,
                electron_density,
                hydrogen_density,
                *args,
                out,
            )
            return out

        # Arbitrary broadcasted non-contiguous shapes are uncommon in the
        # Bifrost runtime; keep the fully general NumPy implementation here.
        return (
            electron_density
            * hydrogen_density
            * self.evaluate(temperature, electron_density)
        ).astype(np.float32, copy=False)

    def evaluate(self, temperature, electron_density):
        """Bilinearly interpolate ``G(T, ne)`` on arbitrary array shapes.

        Parameters are physical ``T`` [K] and ``ne`` [cm^-3].  Values that
        are non-positive, non-finite, or outside the response grid receive the
        response-specific ``fill_value`` from ``metadata.json``.
        """
        temperature, electron_density = np.broadcast_arrays(temperature, electron_density)
        out = np.full(temperature.shape, self.fill_value, dtype=np.float64)

        valid_input = (
            np.isfinite(temperature)
            & np.isfinite(electron_density)
            & (temperature > 0.0)
            & (electron_density > 0.0)
        )
        if not np.any(valid_input):
            return out.astype(np.float32)

        lt = np.empty(temperature.shape, dtype=np.float64)
        ln = np.empty(electron_density.shape, dtype=np.float64)
        lt.fill(np.nan)
        ln.fill(np.nan)
        lt[valid_input] = np.log10(temperature[valid_input])
        ln[valid_input] = np.log10(electron_density[valid_input])

        valid = (
            valid_input
            & (lt >= self.logt[0])
            & (lt <= self.logt[-1])
            & (ln >= self.logne[0])
            & (ln <= self.logne[-1])
        )
        if not np.any(valid):
            return out.astype(np.float32)

        ltv = lt[valid]
        lnv = ln[valid]

        if self._uniform_t:
            xt = (ltv - self.logt[0]) / self._dlogt
            it = np.floor(xt).astype(np.int64)
            it = np.minimum(it, self.logt.size - 2)
            ft = xt - it
        else:
            it = np.searchsorted(self.logt, ltv, side="right") - 1
            it = np.clip(it, 0, self.logt.size - 2)
            ft = (ltv - self.logt[it]) / (self.logt[it + 1] - self.logt[it])

        if self._uniform_ne:
            xn = (lnv - self.logne[0]) / self._dlogne
            ine = np.floor(xn).astype(np.int64)
            ine = np.minimum(ine, self.logne.size - 2)
            fn = xn - ine
        else:
            ine = np.searchsorted(self.logne, lnv, side="right") - 1
            ine = np.clip(ine, 0, self.logne.size - 2)
            fn = (lnv - self.logne[ine]) / (self.logne[ine + 1] - self.logne[ine])

        table = self.table
        q00 = np.asarray(table[ine, it], dtype=np.float64)
        q10 = np.asarray(table[ine + 1, it], dtype=np.float64)
        q01 = np.asarray(table[ine, it + 1], dtype=np.float64)
        q11 = np.asarray(table[ine + 1, it + 1], dtype=np.float64)

        interp = (
            (1.0 - fn) * (1.0 - ft) * q00
            + fn * (1.0 - ft) * q10
            + (1.0 - fn) * ft * q01
            + fn * ft * q11
        )
        out[valid] = interp
        return out.astype(np.float32, copy=False)
