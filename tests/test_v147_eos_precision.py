"""EOS kernels give the same float64-accurate result with and without JIT.

With NumPy >= 2 (NEP 50), a float32 scalar combined with a Python float stays
float32. Without explicit promotion, the pure-Python execution of the Numba
kernels (NUMBA_DISABLE_JIT=1, debuggers, coverage) interpolated in float32
and differed from the compiled kernels by up to ~4e-6 for exp() outputs.
"""

import numpy as np
import pytest
from test_eos_numba_v114 import make_eos, physical_inputs

from bifrost_proto import eos as eos_module
from bifrost_proto.eos import _flat_memory_view

OUTPUTS = ("tg", "ne", "pg", "kr")


def _python(kernel):
    """Pure-Python body of a Numba dispatcher (the function itself without JIT)."""
    return getattr(kernel, "py_func", kernel)


def _float64_reference(eos, rho, ee, name):
    table = np.asarray(eos._table(name), dtype=np.float64)
    x = (np.log(ee.astype(np.float64)) - eos.lnei[0]) / eos.dlnei
    y = (np.log(rho.astype(np.float64)) - eos.lnrho[0]) / eos.dlnrho
    i0 = np.minimum(np.floor(x).astype(np.int64), table.shape[0] - 2)
    j0 = np.minimum(np.floor(y).astype(np.int64), table.shape[1] - 2)
    fx, fy = x - i0, y - j0
    value = ((1 - fx) * (1 - fy) * table[i0, j0] + fx * (1 - fy) * table[i0 + 1, j0]
             + (1 - fx) * fy * table[i0, j0 + 1] + fx * fy * table[i0 + 1, j0 + 1])
    return value if name == "tg" else np.exp(value)


def _args(eos, name):
    table = eos._table(name)
    return (table, float(eos.lnei[0]), float(1.0 / eos.dlnei), int(table.shape[0]),
            float(eos.lnrho[0]), float(1.0 / eos.dlnrho), int(table.shape[1]),
            name != "tg", 0.0, 0.0)


@pytest.mark.parametrize("name", OUTPUTS)
def test_compiled_and_python_flat_kernel_agree(name):
    eos = make_eos()
    rho, ee = physical_inputs(eos, (13, 11, 7), np.random.default_rng(5), "F")
    compiled = eos.interpolate(rho, ee, out=name)
    python = np.empty_like(compiled)
    _python(eos_module._eos_interp_flat)(
        _flat_memory_view(rho), _flat_memory_view(ee), *_args(eos, name),
        _flat_memory_view(python),
    )
    np.testing.assert_allclose(python, compiled, rtol=1.2e-7, atol=0.0)


@pytest.mark.parametrize("name", OUTPUTS)
def test_compiled_and_python_strided_kernel_agree(name):
    eos = make_eos()
    rho0, ee0 = physical_inputs(eos, (14, 12, 10), np.random.default_rng(6), "F")
    rho, ee = rho0[::2, 1:, ::-1], ee0[::2, 1:, ::-1]
    compiled = eos.interpolate(rho, ee, out=name)
    python = np.empty(rho.shape, dtype=np.float32)
    _python(eos_module._eos_interp_strided_3d)(rho, ee, *_args(eos, name), python)
    np.testing.assert_allclose(python, compiled, rtol=1.2e-7, atol=0.0)


@pytest.mark.parametrize("name", OUTPUTS)
def test_interpolation_is_float64_accurate(name):
    eos = make_eos()
    rho, ee = physical_inputs(eos, (19, 15, 11), np.random.default_rng(12345), "C")
    got = eos.interpolate(rho, ee, out=name).astype(np.float64)
    expected = _float64_reference(eos, rho, ee, name)
    # Only the final float32 rounding remains (half an ulp, ~6e-8).
    np.testing.assert_allclose(got, expected, rtol=1.2e-7, atol=0.0)


def _float64_gamma1_reference(eos, rho, ee):
    """Independent NumPy formulation of Gamma_1 from the bilinear ln(p) table."""
    rho64 = np.asarray(rho, dtype=np.float64)
    ee64 = np.asarray(ee, dtype=np.float64)
    table = np.asarray(eos.lnpg, dtype=np.float64)
    x = np.clip((np.log(ee64) - eos.lnei[0]) / eos.dlnei, 0, table.shape[0] - 1)
    y = np.clip((np.log(rho64) - eos.lnrho[0]) / eos.dlnrho, 0, table.shape[1] - 1)
    i = np.minimum(np.floor(x).astype(np.int64), table.shape[0] - 2)
    j = np.minimum(np.floor(y).astype(np.int64), table.shape[1] - 2)
    fx, fy = x - i, y - j
    p00 = table[i, j]
    p10 = table[i + 1, j]
    p01 = table[i, j + 1]
    p11 = table[i + 1, j + 1]
    lnp = ((1 - fx) * (1 - fy) * p00 + fx * (1 - fy) * p10
           + (1 - fx) * fy * p01 + fx * fy * p11)
    chi_ee = ((1 - fy) * (p10 - p00) + fy * (p11 - p01)) / eos.dlnei
    chi_rho = ((1 - fx) * (p01 - p00) + fx * (p11 - p10)) / eos.dlnrho
    return chi_rho + chi_ee * np.exp(lnp) / (rho64 * ee64)


@pytest.mark.parametrize("layout", ("C", "F", "strided"))
def test_gamma1_matches_python_body_and_float64_reference(layout):
    eos = make_eos()
    order = "F" if layout == "strided" else layout
    rho, ee = physical_inputs(eos, (17, 13, 11), np.random.default_rng(141), order)
    if layout == "strided":
        rho, ee = rho[::2, 1:, ::-1], ee[::2, 1:, ::-1]
    expected = _float64_gamma1_reference(eos, rho, ee)
    compiled = eos.gamma1(rho, ee)
    assert compiled.dtype == np.float32

    args = (eos.lnpg, float(eos.lnei[0]), float(1 / eos.dlnei),
            float(eos.lnrho[0]), float(1 / eos.dlnrho), 0.0, 0.0, 1.0)
    pyout = np.empty_like(compiled)
    if layout == "strided":
        _python(eos_module._gamma1_strided)(rho, ee, *args, pyout)
    else:
        _python(eos_module._gamma1_flat)(
            _flat_memory_view(rho), _flat_memory_view(ee), *args,
            _flat_memory_view(pyout),
        )
    np.testing.assert_allclose(pyout, compiled, rtol=1.2e-7, atol=1e-8)
    np.testing.assert_allclose(compiled, expected, rtol=1.2e-7, atol=1e-8)
