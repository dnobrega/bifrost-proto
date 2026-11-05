import numpy as np

from bifrost_proto.eos import EOSTable
from bifrost_proto.registry import GLOBAL_REGISTRY
from test_engine import ArrayData, make_data
from test_quantities_v14 import make_constant_data, interior


class IdealGasEOS:
    def __init__(self, gamma=5.0 / 3.0):
        self.gamma = np.float32(gamma)

    def interpolate(self, rho_cgs, ee_cgs, out="pg"):
        if out != "pg":
            raise ValueError(out)
        return np.asarray((self.gamma - 1.0) * rho_cgs * ee_cgs, dtype=np.float32)

    def gamma1(self, rho_cgs, ee_cgs):
        shape = np.broadcast(rho_cgs, ee_cgs).shape
        return np.full(shape, self.gamma, dtype=np.float32)


def test_eos_gamma1_recovers_ideal_gas_exactly_from_log_pressure_table():
    gamma = 5.0 / 3.0
    eos = EOSTable.__new__(EOSTable)
    eos.lnrho = np.linspace(np.log(1e-10), np.log(1e-6), 12)
    eos.lnei = np.linspace(np.log(1e10), np.log(1e13), 11)
    eos.dlnrho = eos.lnrho[1] - eos.lnrho[0]
    eos.dlnei = eos.lnei[1] - eos.lnei[0]

    ee, rho = np.meshgrid(np.exp(eos.lnei), np.exp(eos.lnrho), indexing="ij")
    eos.lnpg = np.log((gamma - 1.0) * rho * ee).astype(np.float32)
    eos.tg = np.zeros_like(eos.lnpg)
    eos.lnne = np.zeros_like(eos.lnpg)
    eos.lnkr = np.zeros_like(eos.lnpg)

    rho_q = np.array([2e-9, 7e-8], dtype=np.float64)
    ee_q = np.array([3e11, 2e12], dtype=np.float64)
    got = eos.gamma1(rho_q, ee_q)
    assert np.allclose(got, gamma, rtol=2e-5, atol=2e-5)


def test_pressure_alfven_and_mach_block_on_constant_fields():
    dd = make_constant_data()
    b2 = 77.0
    modu = np.sqrt(14.0)
    pb = 0.5 * b2
    va = np.sqrt(b2 / 2.0)

    assert np.allclose(interior(dd.get_var("pb", 0, units="bifrost", br=True)), pb)
    assert np.allclose(interior(dd.get_var("pt", 0, units="bifrost", br=True)), 7.0 + pb)
    assert np.allclose(interior(dd.get_var("beta", 0, units="bifrost", br=True)), 7.0 / pb)
    assert np.allclose(interior(dd.get_var("va", 0, units="bifrost", br=True)), va)
    assert np.allclose(interior(dd.get_var("mach_a", 0, units="bifrost", br=True)), modu / va)
    assert np.array_equal(
        dd.get_var("man", 0, units="bifrost", br=True),
        dd.get_var("mach_a", 0, units="bifrost", br=True),
    )


def test_eos_sound_speed_and_constant_gamma_reference():
    dd = make_data()
    dd._eos = lambda snap: IdealGasEOS()

    gamma = 5.0 / 3.0
    # r=2, e=6 -> ee=3; ideal-gas pg=(gamma-1)*r*ee=4 in native units.
    expected_cs = np.sqrt(gamma * 4.0 / 2.0)
    cs = dd.get_var("cs", 0, units="bifrost", br=True)
    cs_alias = dd.get_var("cs_eos", 0, units="bifrost", br=True)
    cs_gamma = dd.get_var("cs_gamma", 0, units="bifrost", br=True)

    assert np.allclose(cs, expected_cs)
    assert np.array_equal(cs, cs_alias)
    assert np.allclose(cs_gamma, expected_cs)

    modu = np.sqrt(4.0**2 + 5.0**2 + 6.0**2)
    mach = dd.get_var("mach_s", 0, units="bifrost", br=True)
    assert np.allclose(mach, modu / expected_cs, atol=2e-5)
    assert np.array_equal(mach, dd.get_var("mn", 0, units="bifrost", br=True))
    assert np.allclose(
        dd.get_var("mach_s_gamma", 0, units="bifrost", br=True),
        modu / expected_cs,
        atol=2e-5,
    )


def test_pressure_scale_heights_and_length_units():
    dd = make_constant_data()
    # No grav parameter in ArrayData -> solar fallback corresponds to 2.74 in
    # the default Bifrost normalization.
    expected_hydro = 7.0 / (2.0 * 2.74)
    h = dd.get_var("hpg_hydro", 0, units="bifrost", br=True)
    assert np.allclose(interior(h), expected_hydro)
    assert np.array_equal(h, dd.get_var("hpg", 0, units="bifrost", br=True))
    assert dd.get_units("hpg_hydro") == "Mm"
    assert dd.get_units("hpg_hydro", units="cgs") == "cm"
    assert dd.get_units("hpg_hydro", units="si") == "m"

    n = 24
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    shape = (n, n, n)
    r = np.full(shape, 2.0, dtype=np.float32)
    zz = np.broadcast_to(z[None, None, :], shape)
    pg = (100.0 + 2.0 * zz).astype(np.float32)
    arrays = {
        "r": r,
        "e": np.full(shape, 6.0, dtype=np.float32),
        "px": r,
        "py": r,
        "pz": r,
        "bx": np.ones(shape, dtype=np.float32),
        "by": np.ones(shape, dtype=np.float32),
        "bz": np.ones(shape, dtype=np.float32),
        "pg": pg,
    }
    dd2 = ArrayData(arrays, x, y, z)
    hgrad = dd2.get_var("hpg_grad", 0, units="bifrost", br=True)
    assert np.allclose(interior(hgrad), interior(pg) / 2.0, atol=2e-4)


def test_official_registry_is_self_documenting_and_info_reports_description():
    missing = [name for name, spec in GLOBAL_REGISTRY.items() if not spec.description]
    assert missing == []

    dd = make_data()
    info = dd.info("mn")
    assert "mn" in info["aliases"]
    assert info["name"] == "mach_s"
    assert "Sonic Mach number" in info["description"]
