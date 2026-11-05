import numpy as np
import pytest

from bifrost_proto.core import UnknownVariableError
from test_engine import ArrayData


def interior(a):
    return a[5:-5, 5:-5, 5:-5]


def make_constant_energy_data(n=24, include_q=True):
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    shape = (n, n, n)
    r = np.full(shape, 2.0, dtype=np.float32)
    arrays = {
        "r": r,
        "e": np.full(shape, 10.0, dtype=np.float32),
        "px": r * 1.0,
        "py": r * 2.0,
        "pz": r * 3.0,
        "bx": np.full(shape, 4.0, dtype=np.float32),
        "by": np.full(shape, 5.0, dtype=np.float32),
        "bz": np.full(shape, 6.0, dtype=np.float32),
        "pg": np.full(shape, 7.0, dtype=np.float32),
    }
    if include_q:
        arrays.update({
            "qjoule": np.full(shape, 4.0, dtype=np.float32),
            "qvisc": np.full(shape, 2.0, dtype=np.float32),
            "qthin": np.full(shape, -5.0, dtype=np.float32),
            "qspitz": np.zeros(shape, dtype=np.float32),
        })
    return ArrayData(arrays, x, y, z)


def test_energy_reservoirs_eb_and_ek():
    dd = make_constant_energy_data()
    eb = dd.get_var("eb", 0, units="bifrost", br=True)
    ek = dd.get_var("ek", 0, units="bifrost", br=True)

    # B=(4,5,6) -> |B|^2=77; r=2, u=(1,2,3) -> r|u|^2/2=14.
    assert np.allclose(interior(eb), 38.5)
    assert np.allclose(interior(ek), 14.0)


def test_stored_q_terms_over_r_and_timescales_preserve_sign():
    dd = make_constant_energy_data()

    expected = {
        "qjoule": (4.0, 2.0, 2.5),
        "qvisc": (2.0, 1.0, 5.0),
        "qthin": (-5.0, -2.5, 2.0),
    }
    for name, (q, qor, tau) in expected.items():
        assert np.all(interior(dd.get_var(name, 0, units="bifrost", br=True)) == q)
        assert np.allclose(interior(dd.get_var(name + "or", 0, units="bifrost", br=True)), qor)
        assert np.allclose(interior(dd.get_var("tau" + name, 0, units="bifrost", br=True)), tau)

    # q=0 means no finite characteristic timescale.
    tau = dd.get_var("tauqspitz", 0, units="bifrost", br=True)
    assert np.all(np.isinf(tau))


def test_q_terms_are_stored_only():
    dd = make_constant_energy_data(include_q=False)
    with pytest.raises(UnknownVariableError):
        dd.get_var("qjoule", 0, units="bifrost", br=True)
    with pytest.raises(UnknownVariableError):
        dd.get_var("qjouleor", 0, units="bifrost", br=True)


def test_qpdv_compression_expansion_family():
    n = 24
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    shape = (n, n, n)
    r = np.full(shape, 2.0, dtype=np.float32)
    xface = x - np.float32(0.5)
    ux_face = np.broadcast_to(xface[:, None, None], shape).astype(np.float32).copy()

    arrays = {
        "r": r,
        "e": np.full(shape, 10.0, dtype=np.float32),
        "px": r * ux_face,
        "py": np.zeros(shape, dtype=np.float32),
        "pz": np.zeros(shape, dtype=np.float32),
        "bx": np.zeros(shape, dtype=np.float32),
        "by": np.zeros(shape, dtype=np.float32),
        "bz": np.ones(shape, dtype=np.float32),
        "pg": np.full(shape, 7.0, dtype=np.float32),
    }
    dd = ArrayData(arrays, x, y, z)

    divu = dd.get_var("divu", 0, units="bifrost", br=True)
    qpdv = dd.get_var("qpdv", 0, units="bifrost", br=True)
    qpdvor = dd.get_var("qpdvor", 0, units="bifrost", br=True)
    tau = dd.get_var("tauqpdv", 0, units="bifrost", br=True)

    assert np.allclose(interior(divu), 1.0, atol=2e-5)
    assert np.allclose(interior(qpdv), -7.0, atol=2e-4)
    assert np.allclose(interior(qpdvor), -3.5, atol=2e-4)
    assert np.allclose(interior(tau), 10.0 / 7.0, atol=2e-4)


def make_lorentz_data(n=28):
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    shape = (n, n, n)
    yy = np.broadcast_to(y[None, :, None], shape).astype(np.float32).copy()
    r = np.ones(shape, dtype=np.float32)
    arrays = {
        "r": r,
        "e": np.ones(shape, dtype=np.float32),
        "px": np.zeros(shape, dtype=np.float32),
        "py": np.full(shape, 2.0, dtype=np.float32),
        "pz": np.zeros(shape, dtype=np.float32),
        "bx": np.zeros(shape, dtype=np.float32),
        "by": np.zeros(shape, dtype=np.float32),
        "bz": yy,
        "pg": np.ones(shape, dtype=np.float32),
    }
    return ArrayData(arrays, x, y, z), yy


def test_qlf_and_qlfor_against_simple_analytic_field():
    dd, yy = make_lorentz_data()
    qlf = dd.get_var("qlf", 0, units="bifrost", br=True)
    qlfor = dd.get_var("qlfor", 0, units="bifrost", br=True)

    # B=(0,0,y), J=curl(B)=(1,0,0), u=(0,2,0):
    # u . (J x B) = -2 y.
    expected = -2.0 * yy
    assert np.allclose(interior(qlf), interior(expected), atol=5e-4)
    assert np.allclose(interior(qlfor), interior(expected), atol=5e-4)


def test_qlf_subdomain_matches_full_domain_exactly():
    dd, _ = make_lorentz_data(n=28)
    full = dd.get_var("qlf", 0, units="bifrost", br=True)
    sub = dd.get_var(
        "qlf", 0,
        ix=slice(8, 18), iy=slice(9, 19), iz=slice(7, 17),
        units="bifrost", br=True,
    )
    assert np.array_equal(sub, full[8:18, 9:19, 7:17])
