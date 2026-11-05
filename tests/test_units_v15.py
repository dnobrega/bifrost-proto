import numpy as np

from bifrost_proto.units import BifrostUnits


def test_energy_rate_and_time_units():
    u = BifrostUnits()

    assert np.isclose(u.u_q, u.u_e / u.u_t)
    assert np.isclose(u.u_qor, u.u_ee / u.u_t)

    assert np.isclose(u.factor("energy_rate_density", "cgs"), u.u_q)
    assert np.isclose(u.factor("specific_energy_rate", "cgs"), u.u_qor)
    assert np.isclose(u.factor("time", "cgs"), u.u_t)

    assert np.isclose(u.factor("energy_rate_density", "solar"), u.u_q)
    assert np.isclose(u.factor("specific_energy_rate", "solar"), u.u_qor)
    assert np.isclose(u.factor("time", "solar"), u.u_t)

    assert np.isclose(u.factor("energy_rate_density", "si"), u.u_q * 0.1)
    assert np.isclose(u.factor("specific_energy_rate", "si"), u.u_qor * 1.0e-4)
    assert np.isclose(u.factor("time", "si"), u.u_t)
