import math

import numpy as np

from bifrost_proto.units import C_LIGHT, BifrostUnits


def test_bifrost_units_are_derived_from_fundamental_units():
    u = BifrostUnits(u_l=1.0e8, u_t=1.0e2, u_r=1.0e-7)

    assert np.isclose(u.u_u, 1.0e6)
    assert np.isclose(u.u_p, u.u_r * u.u_u**2)
    assert np.isclose(u.u_kr, 1.0 / (u.u_r * u.u_l))
    assert np.isclose(u.u_ee, u.u_u**2)
    assert np.isclose(u.u_e, u.u_r * u.u_ee)
    assert np.isclose(u.u_B, u.u_u * math.sqrt(4.0 * math.pi * u.u_r))
    assert np.isclose(u.u_i, u.u_B / u.u_l * C_LIGHT / (4.0 * math.pi))
    assert np.isclose(u.u_el, u.u_B * u.u_l / u.u_t / C_LIGHT)

    assert np.isclose(u.factor("pressure", "cgs"), u.u_p)
    assert np.isclose(u.factor("opacity", "cgs"), u.u_kr)
    assert np.isclose(u.factor("current_density", "cgs"), u.u_i)
    assert np.isclose(u.factor("electric_field", "cgs"), u.u_el)
    assert np.isclose(u.factor("velocity", "solar"), 10.0)


def test_current_and_electric_si_factors_match_gaussian_cgs_conversion():
    u = BifrostUnits()
    assert np.isclose(
        u.factor("current_density", "si"),
        u.u_i * 3.33564095198152e-6,
    )
    assert np.isclose(
        u.factor("electric_field", "si"),
        u.u_el * 29979.2458,
    )
