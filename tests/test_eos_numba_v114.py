import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from bifrost_proto.eos import EOSTable


def scipy_reference(eos, rho_cgs, ee_cgs, out="tg"):
    """Test-only reference preserving the pre-v1.14 SciPy definition."""
    table = eos._table(out)
    x = (np.log(ee_cgs) - eos.lnei[0]) / eos.dlnei
    y = (np.log(rho_cgs) - eos.lnrho[0]) / eos.dlnrho
    value = map_coordinates(table, [x, y], order=1, mode="nearest")
    if out != "tg":
        value = np.exp(value)
    return value.astype(np.float32, copy=False)


def make_eos(nei=41, nrho=37):
    eos = EOSTable.__new__(EOSTable)
    eos.lnei = np.linspace(np.log(2.0e11), np.log(5.0e15), nei)
    eos.lnrho = np.linspace(np.log(3.0e-18), np.log(2.0e-5), nrho)
    eos.dlnei = eos.lnei[1] - eos.lnei[0]
    eos.dlnrho = eos.lnrho[1] - eos.lnrho[0]

    x = np.arange(nei, dtype=np.float64)[:, None]
    y = np.arange(nrho, dtype=np.float64)[None, :]

    # Use nontrivial but smooth synthetic tables. The logarithmic outputs are
    # kept in a numerically safe range for exponentiation.
    eos.tg = np.asfortranarray(
        4200.0 + 19.0 * x + 13.0 * y + 0.07 * x * y, dtype=np.float32
    )
    eos.lnpg = np.asfortranarray(
        8.0 + 0.017 * x + 0.011 * y + 2.0e-4 * x * y, dtype=np.float32
    )
    eos.lnne = np.asfortranarray(
        24.0 + 0.013 * x + 0.009 * y + 1.0e-4 * x * y, dtype=np.float32
    )
    eos.lnkr = np.asfortranarray(
        -7.0 + 0.009 * x + 0.006 * y + 7.0e-5 * x * y, dtype=np.float32
    )
    return eos


def physical_inputs(eos, shape, rng, order):
    # Strict EOS: every input must fall within its own table domain.
    xe = rng.uniform(0.2, len(eos.lnei) - 1.2, size=shape)
    yr = rng.uniform(0.2, len(eos.lnrho) - 1.2, size=shape)
    ee = np.array(np.exp(eos.lnei[0] + xe * eos.dlnei), dtype=np.float32, order=order)
    rho = np.array(
        np.exp(eos.lnrho[0] + yr * eos.dlnrho), dtype=np.float32, order=order
    )
    return rho, ee


def test_numba_eos_matches_scipy_reference_for_all_outputs_c_and_f():
    eos = make_eos()
    rng = np.random.default_rng(12345)

    for order in ("C", "F"):
        rho, ee = physical_inputs(eos, (19, 15, 11), rng, order)
        rho_before = rho.copy()
        ee_before = ee.copy()

        for name in ("tg", "ne", "pg", "kr"):
            expected = scipy_reference(eos, rho, ee, out=name)
            got = eos.interpolate(rho, ee, out=name)
            assert got.dtype == np.float32
            np.testing.assert_allclose(got, expected, rtol=3e-6, atol=0.0)

        np.testing.assert_array_equal(rho, rho_before)
        np.testing.assert_array_equal(ee, ee_before)


def test_numba_eos_strided_3d_view_matches_scipy_without_input_copies():
    eos = make_eos()
    rng = np.random.default_rng(7)
    rho0, ee0 = physical_inputs(eos, (34, 30, 26), rng, "F")
    rho = rho0[1:-1:2, 2:-2:2, 1:-1:2]
    ee = ee0[1:-1:2, 2:-2:2, 1:-1:2]

    assert not rho.flags.c_contiguous and not rho.flags.f_contiguous
    assert not ee.flags.c_contiguous and not ee.flags.f_contiguous

    rho_before = rho.copy()
    ee_before = ee.copy()

    for name in ("tg", "ne", "pg", "kr"):
        expected = scipy_reference(eos, rho, ee, out=name)
        got = eos.interpolate(rho, ee, out=name)
        np.testing.assert_allclose(got, expected, rtol=3e-6, atol=0.0)

    np.testing.assert_array_equal(rho, rho_before)
    np.testing.assert_array_equal(ee, ee_before)


def test_numba_eos_rejects_outside_table_instead_of_clamping():
    eos = make_eos(nei=8, nrho=7)
    x = np.array([-5.0, 0.0, 0.3, 6.7, 7.0, 12.0], dtype=np.float64)
    y = np.array([10.0, -3.0, 0.6, 5.2, 6.0, -8.0], dtype=np.float64)
    ee = np.exp(eos.lnei[0] + x * eos.dlnei).astype(np.float32)
    rho = np.exp(eos.lnrho[0] + y * eos.dlnrho).astype(np.float32)
    with pytest.raises(ValueError, match="outside its table range"):
        eos.interpolate(rho, ee, out="pg")


def test_unusual_broadcast_shape_preserves_general_reference_semantics():
    eos = make_eos()
    rho = np.asarray([[1.0e-15], [2.0e-14]], dtype=np.float32)
    ee = np.asarray([[5.0e12, 1.0e13, 2.0e13]], dtype=np.float32)

    # The broadcasted arrays are zero-stride/non-contiguous and deliberately
    # exercise the compatibility fallback rather than a hidden contiguization.
    rho_b, ee_b = np.broadcast_arrays(rho, ee)
    expected = scipy_reference(eos, rho_b, ee_b, out="tg")
    got = eos.interpolate(rho, ee, out="tg")
    np.testing.assert_allclose(got, expected, rtol=3e-6, atol=0.0)


def test_numba_eos_negative_stride_3d_view_matches_scipy():
    """EOS interpolation must support full-size reversed views without copies."""
    eos = make_eos()
    rng = np.random.default_rng(20260924)
    rho0, ee0 = physical_inputs(eos, (19, 15, 13), rng, "F")
    rho = rho0[:, :, ::-1]
    ee = ee0[:, :, ::-1]

    assert not rho.flags.c_contiguous and not rho.flags.f_contiguous
    assert not ee.flags.c_contiguous and not ee.flags.f_contiguous
    assert rho.strides[2] < 0 and ee.strides[2] < 0

    for name in ("tg", "ne", "pg", "kr"):
        expected = scipy_reference(eos, rho, ee, out=name)
        got = eos.interpolate(rho, ee, out=name)
        np.testing.assert_allclose(got, expected, rtol=3e-6, atol=0.0)
