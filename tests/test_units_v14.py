import numpy as np

from bifrost_proto.units import BifrostUnits


def test_new_derivative_unit_types():
    u = BifrostUnits()

    assert np.isclose(u.factor("dimensionless", "solar"), 1.0)
    assert np.isclose(u.factor("inverse_time", "solar"), 1.0 / u.u_t)
    assert np.isclose(u.factor("inverse_length", "cgs"), 1.0 / u.u_l)
    assert np.isclose(u.factor("inverse_length", "solar"), 1.0 / u.length_to_Mm)
    assert np.isclose(
        u.factor("magnetic_gradient", "solar"),
        u.u_B / u.length_to_Mm,
    )
