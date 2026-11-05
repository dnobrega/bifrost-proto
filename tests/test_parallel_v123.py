"""Regression checks for the v0.1.23 tau500 parallelization."""

import numpy as np
from numba import njit

import bifrost_proto.quantities as quantities


@njit(cache=True)
def _tau500_serial_reference(ne, nh, temperature, z_cm, out):
    """Serial v0.1.22 tau500 kernel retained only as a test reference."""
    nx, ny, nz = temperature.shape
    if nz == 0:
        return

    chi_prev = np.empty((nx, ny), dtype=np.float32)
    exponent_constant = (
        quantities._TAU500_HMINUS_BINDING_EV
        * quantities._TAU500_EV_TO_ERG
        / quantities._TAU500_K_B_CGS
    )

    for j in range(ny):
        for i in range(nx):
            t = temperature[i, j, 0]
            chi = (
                quantities._TAU500_HMINUS_SAHA
                * ne[i, j, 0]
                * quantities._TAU500_HMINUS_SIGMA_CM2
                / (t ** 1.5)
                * np.exp(exponent_constant / t)
                * nh[i, j, 0]
            )
            chi_prev[i, j] = chi
            out[i, j, 0] = quantities._TAU500_TOP

    for k in range(1, nz):
        dz_cm = abs(z_cm[k] - z_cm[k - 1])
        for j in range(ny):
            for i in range(nx):
                t = temperature[i, j, k]
                chi = (
                    quantities._TAU500_HMINUS_SAHA
                    * ne[i, j, k]
                    * quantities._TAU500_HMINUS_SIGMA_CM2
                    / (t ** 1.5)
                    * np.exp(exponent_constant / t)
                    * nh[i, j, k]
                )
                out[i, j, k] = (
                    out[i, j, k - 1]
                    + 0.5 * (chi_prev[i, j] + chi) * dz_cm
                )
                chi_prev[i, j] = chi


def test_tau500_dispatcher_is_parallel_without_fastmath():
    opts = quantities._integrate_tau500_native.targetoptions
    assert opts.get("parallel") is True
    assert not opts.get("fastmath", False)


def test_parallel_tau500_is_bitwise_equal_to_v122_serial_kernel():
    rng = np.random.default_rng(123)
    shape = (19, 13, 11)

    ne = np.array(10.0 ** rng.uniform(10.0, 13.0, size=shape), dtype=np.float32, order="F")
    nh = np.array(10.0 ** rng.uniform(14.0, 17.0, size=shape), dtype=np.float32, order="F")
    temperature = np.array(rng.uniform(4000.0, 15000.0, size=shape), dtype=np.float32, order="F")

    # Deliberately non-uniform native Bifrost center grid in centimeters.
    dz = rng.uniform(4.0e6, 1.6e7, size=shape[2] - 1)
    z_cm = np.concatenate(([0.0], np.cumsum(dz))).astype(np.float64)

    expected = np.empty(shape, dtype=np.float32, order="F")
    got = np.empty(shape, dtype=np.float32, order="F")

    _tau500_serial_reference(ne, nh, temperature, z_cm, expected)
    quantities._integrate_tau500_native(ne, nh, temperature, z_cm, got)

    assert np.array_equal(got, expected, equal_nan=True)
