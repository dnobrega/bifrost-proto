import numpy as np

from bifrost_proto.units import BifrostUnits


def test_force_acceleration_and_angle_unit_factors():
    u = BifrostUnits()

    assert np.isclose(u.factor("force_density", "cgs"), u.u_p / u.u_l)
    assert np.isclose(
        u.factor("acceleration", "cgs"),
        u.u_l / (u.u_t * u.u_t),
    )
    assert np.isclose(
        u.factor("acceleration", "solar"),
        u.u_l / (u.u_t * u.u_t) / 1.0e5,
    )
    assert np.isclose(u.factor("angle", "solar"), 180.0 / np.pi)
    assert np.isclose(u.factor("angle", "cgs"), 1.0)
    assert np.isclose(u.factor("angle", "si"), 1.0)


def test_force_acceleration_and_angle_labels():
    u = BifrostUnits()
    assert u.label("force_density", "solar") == "dyn cm^-3"
    assert u.label("acceleration", "solar") == "km s^-2"
    assert u.label("angle", "solar") == "deg"
    assert u.label("angle", "cgs") == "rad"
    assert u.label("angle", "si") == "rad"
