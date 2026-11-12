import numpy as np
from test_engine import make_data


class DummyEOS:
    def interpolate(self, rho_cgs, ee_cgs, out="tg"):
        shape = np.broadcast(rho_cgs, ee_cgs).shape
        values = {
            "tg": 6500.0,
            "ne": 2.5e12,
            "pg": 3.2e5,
            "kr": 0.45,
        }
        return np.full(shape, values[out], dtype=np.float32)


def test_eos_tg_ne_pg_kr_and_native_internal_normalization():
    dd = make_data()
    dummy = DummyEOS()
    dd._eos = lambda snap: dummy
    u = dd._meta(0).units

    tg_native = dd.get_var("tg_eos", 0, units="bifrost", br=True)
    ne_native = dd.get_var("ne_eos", 0, units="bifrost", br=True)
    pg_native = dd.get_var("pg_eos", 0, units="bifrost", br=True)
    kr_native = dd.get_var("kr_eos", 0, units="bifrost", br=True)

    assert np.allclose(tg_native, 6500.0)
    assert np.allclose(ne_native, 2.5e12)
    assert np.allclose(pg_native, 3.2e5 / u.u_p)
    assert np.allclose(kr_native, 0.45 / u.u_kr)

    # Solar/CGS output recovers the physical values from the table.
    assert np.allclose(dd.get_var("pg_eos", 0, units="cgs", br=True), 3.2e5)
    assert np.allclose(dd.get_var("kr_eos", 0, units="cgs", br=True), 0.45)
    assert np.allclose(dd.get_var("ne_eos", 0, units="cgs", br=True), 2.5e12)


def test_stored_and_hion_thermodynamics_take_precedence_over_eos():
    dd = make_data()
    dd._eos = lambda snap: DummyEOS()
    shape = dd.arrays["r"].shape

    # HION is preferred over EOS when canonical tg/ne are not stored.
    dd.arrays["hiontg"] = np.full(shape, 7100.0, dtype=np.float32)
    dd.arrays["hionne"] = np.full(shape, 8.0e11, dtype=np.float32)
    assert np.all(dd.get_var("tg", 0, units="bifrost", br=True) == 7100.0)
    assert np.all(dd.get_var("ne", 0, units="bifrost", br=True) == 8.0e11)

    # Stored canonical AUX variables win over both HION and EOS.
    dd.arrays["tg"] = np.full(shape, 7200.0, dtype=np.float32)
    dd.arrays["ne"] = np.full(shape, 9.0e11, dtype=np.float32)
    dd.arrays["p"] = np.full(shape, 4.0, dtype=np.float32)
    dd.arrays["kr"] = np.full(shape, 5.0, dtype=np.float32)

    assert np.all(dd.get_var("tg", 0, units="bifrost", br=True) == 7200.0)
    assert np.all(dd.get_var("ne", 0, units="bifrost", br=True) == 9.0e11)
    assert np.all(dd.get_var("pg", 0, units="bifrost", br=True) == 4.0)
    assert np.all(dd.get_var("p", 0, units="bifrost", br=True) == 4.0)
    assert np.all(dd.get_var("kr", 0, units="bifrost", br=True) == 5.0)

    # Explicit EOS names still force the table even when AUX p exists.
    assert np.allclose(
        dd.get_var("pg_eos", 0, units="bifrost", br=True), 3.2e5 / dd._meta(0).units.u_p
    )


def test_pg_prefers_aux_p_and_keeps_raw_pg_compatibility():
    dd = make_data()
    dd._eos = lambda snap: DummyEOS()
    shape = dd.arrays["r"].shape

    # Standard Bifrost AUX spelling.
    dd.arrays["p"] = np.full(shape, 7.0, dtype=np.float32)
    assert np.all(dd.get_var("pg", 0, units="bifrost", br=True) == 7.0)
    assert np.all(dd.get_var("p", 0, units="bifrost", br=True) == 7.0)

    # Compatibility with datasets that expose a raw pg field instead.
    del dd.arrays["p"]
    dd.arrays["pg"] = np.full(shape, 8.0, dtype=np.float32)
    assert np.all(dd.get_var("pg", 0, units="bifrost", br=True) == 8.0)

    del dd.arrays["pg"]
    expected = 3.2e5 / dd._meta(0).units.u_p
    assert np.allclose(dd.get_var("pg", 0, units="bifrost", br=True), expected)
