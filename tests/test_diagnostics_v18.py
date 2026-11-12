import numpy as np
import pytest

from bifrost_proto.core import UnknownVariableError
from test_engine import ArrayData, make_data


def interior(a):
    return a[5:-5, 5:-5, 5:-5]


def constant_data(n=24, *, b=(1.0, 0.0, 0.0), efield=None, j=None, rho=2.0):
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    shape = (n, n, n)
    r = np.full(shape, rho, dtype=np.float32)
    arrays = {
        "r": r,
        "e": np.full(shape, 6.0, dtype=np.float32),
        "px": r,
        "py": 2.0 * r,
        "pz": 3.0 * r,
        "bx": np.full(shape, b[0], dtype=np.float32),
        "by": np.full(shape, b[1], dtype=np.float32),
        "bz": np.full(shape, b[2], dtype=np.float32),
        "pg": np.full(shape, 7.0, dtype=np.float32),
    }
    if efield is not None:
        arrays.update(
            ex=np.full(shape, efield[0], dtype=np.float32),
            ey=np.full(shape, efield[1], dtype=np.float32),
            ez=np.full(shape, efield[2], dtype=np.float32),
        )
    if j is not None:
        arrays.update(
            j_x=np.full(shape, j[0], dtype=np.float32),
            j_y=np.full(shape, j[1], dtype=np.float32),
            j_z=np.full(shape, j[2], dtype=np.float32),
        )
    return ArrayData(arrays, x, y, z)


def test_energy_e_and_electric_vector_registry_do_not_collide():
    dd = constant_data(efield=(2.0, 3.0, 4.0))
    einfo = dd.info("e")
    exinfo = dd.info("e_x")
    legacy = dd.info("ex")

    assert einfo["name"] == "e"
    assert einfo["unit_type"] == "energy_density"
    assert einfo["location"] == "center"
    assert exinfo["name"] == "e_x"
    assert exinfo["unit_type"] == "electric_field"
    assert legacy["name"] == "e_x"
    assert legacy["aliases"] == exinfo["aliases"]


def test_velocity_perpendicular_canonical_name_is_uperpb_only():
    dd = constant_data(b=(4.0, 5.0, 6.0))
    value = dd.get_var("uperpb", 0, units="bifrost", br=True)
    assert value.shape == dd.arrays["r"].shape
    with pytest.raises(UnknownVariableError):
        dd.info("uperp")


def test_total_electric_field_parallel_perpendicular_decomposition():
    dd = constant_data(b=(1.0, 0.0, 0.0), efield=(2.0, 3.0, 4.0))

    assert np.allclose(
        interior(dd.get_var("mode", 0, units="bifrost", br=True)), np.sqrt(29.0)
    )
    assert np.allclose(interior(dd.get_var("eparb", 0, units="bifrost", br=True)), 2.0)

    expected_par = (2.0, 0.0, 0.0)
    expected_perp = (0.0, 3.0, 4.0)
    for axis, expected in zip("xyz", expected_par):
        got = dd.get_var(f"eparb_{axis}", 0, units="bifrost", br=True)
        assert np.allclose(interior(got), expected, atol=3e-5)
    for axis, expected in zip("xyz", expected_perp):
        got = dd.get_var(f"eperpb_{axis}", 0, units="bifrost", br=True)
        assert np.allclose(interior(got), expected, atol=3e-5)

    assert np.allclose(
        interior(dd.get_var("eperpb", 0, units="bifrost", br=True)), 5.0, atol=3e-5
    )


def test_theta_jb_is_acute_angle_and_uses_degrees_in_solar_units():
    invsqrt2 = np.float32(1.0 / np.sqrt(2.0))
    dd = constant_data(b=(1.0, 0.0, 0.0), j=(invsqrt2, invsqrt2, 0.0))

    theta_rad = dd.get_var("theta_jb", 0, units="bifrost", br=True)
    theta_deg = dd.get_var("theta_jb", 0, units="solar", br=True)
    assert np.allclose(interior(theta_rad), np.pi / 4.0, atol=2e-5)
    assert np.allclose(interior(theta_deg), 45.0, atol=2e-4)
    assert dd.get_units("theta_jb") == "deg"
    assert dd.get_units("theta_jb", units="cgs") == "rad"

    # Anti-parallel is also zero degrees because theta_jb uses |J dot B|.
    dd.arrays["j_x"][:] = -1.0
    dd.arrays["j_y"][:] = 0.0
    theta_anti = dd.get_var("theta_jb", 0, units="solar", br=True)
    assert np.allclose(interior(theta_anti), 0.0, atol=2e-4)


def test_theta_jb_is_nan_when_angle_is_undefined():
    dd = constant_data(b=(1.0, 0.0, 0.0), j=(0.0, 0.0, 0.0))
    theta = dd.get_var("theta_jb", 0, units="solar", br=True)
    assert np.all(np.isnan(interior(theta)))


def test_pressure_gradient_and_over_r_on_linear_pressure():
    n = 24
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    shape = (n, n, n)
    r = np.full(shape, 2.0, dtype=np.float32)
    arrays = {
        "r": r,
        "e": np.ones(shape, dtype=np.float32),
        "px": np.zeros(shape, dtype=np.float32),
        "py": np.zeros(shape, dtype=np.float32),
        "pz": np.zeros(shape, dtype=np.float32),
        "bx": np.ones(shape, dtype=np.float32),
        "by": np.zeros(shape, dtype=np.float32),
        "bz": np.zeros(shape, dtype=np.float32),
        "pg": (2.0 * xx + 3.0 * yy + 4.0 * zz + 20.0).astype(np.float32),
    }
    dd = ArrayData(arrays, x, y, z)

    # gradpg_* is the physical pressure-gradient force density, -grad(pg).
    for axis, expected in zip("xyz", (-2.0, -3.0, -4.0)):
        got = dd.get_var(f"gradpg_{axis}", 0, units="bifrost", br=True)
        assert np.allclose(interior(got), expected, atol=3e-5)
        got_or = dd.get_var(f"gradpgor_{axis}", 0, units="bifrost", br=True)
        assert np.allclose(interior(got_or), expected / 2.0, atol=3e-5)

    # In the public physical frame z is reversed relative to native Bifrost,
    # so a pressure decreasing upward gives positive (upward) acceleration.
    grad_z_phys = dd.get_var("gradpg_z", 0, units="bifrost", br=False)
    acc_z_phys = dd.get_var("gradpgor_z", 0, units="bifrost", br=False)
    assert np.allclose(interior(grad_z_phys), 4.0, atol=3e-5)
    assert np.allclose(interior(acc_z_phys), 2.0, atol=3e-5)

    assert np.allclose(
        interior(dd.get_var("gradpg_h", 0, units="bifrost", br=True)),
        np.sqrt(13.0),
        atol=5e-5,
    )
    assert np.allclose(
        interior(dd.get_var("modgradpg", 0, units="bifrost", br=True)),
        np.sqrt(29.0),
        atol=5e-5,
    )
    assert np.allclose(
        interior(dd.get_var("gradpgor_h", 0, units="bifrost", br=True)),
        np.sqrt(13.0) / 2.0,
        atol=5e-5,
    )
    assert np.allclose(
        interior(dd.get_var("modgradpgor", 0, units="bifrost", br=True)),
        np.sqrt(29.0) / 2.0,
        atol=5e-5,
    )


def test_lorentz_force_legacy_stagger_path_on_constant_j_and_b():
    dd = constant_data(b=(4.0, 5.0, 6.0), j=(1.0, 2.0, 3.0), rho=2.0)
    expected = (-3.0, 6.0, -3.0)

    for axis, target in zip("xyz", expected):
        lf = dd.get_var(f"lf_{axis}", 0, units="bifrost", br=True)
        lfor = dd.get_var(f"lfor_{axis}", 0, units="bifrost", br=True)
        assert np.allclose(interior(lf), target, atol=4e-5)
        assert np.allclose(interior(lfor), target / 2.0, atol=4e-5)

    assert np.allclose(
        interior(dd.get_var("lf_h", 0, units="bifrost", br=True)),
        np.sqrt(45.0),
        atol=6e-5,
    )
    assert np.allclose(
        interior(dd.get_var("modlf", 0, units="bifrost", br=True)),
        np.sqrt(54.0),
        atol=6e-5,
    )
    assert np.allclose(
        interior(dd.get_var("lfor_h", 0, units="bifrost", br=True)),
        np.sqrt(45.0) / 2.0,
        atol=6e-5,
    )
    assert np.allclose(
        interior(dd.get_var("modlfor", 0, units="bifrost", br=True)),
        np.sqrt(54.0) / 2.0,
        atol=6e-5,
    )


def test_lorentz_subdomain_matches_full_domain_exactly():
    dd = make_data(n=28)
    rng = np.random.default_rng(1234)
    for name in ("j_x", "j_y", "j_z"):
        dd.arrays[name] = rng.normal(size=(28, 28, 28)).astype(np.float32)

    full = dd.get_var("lf_x", 0, units="bifrost", br=True)
    sub = dd.get_var(
        "lf_x",
        0,
        ix=slice(8, 18),
        iy=slice(9, 19),
        iz=slice(7, 17),
        units="bifrost",
        br=True,
    )
    assert np.array_equal(sub, full[8:18, 9:19, 7:17])


def test_new_force_and_acceleration_units():
    dd = constant_data()
    assert dd.get_units("lf_x") == "dyn cm^-3"
    assert dd.get_units("lfor_x") == "km s^-2"
    assert dd.get_units("lfor_x", units="cgs") == "cm s^-2"
    assert dd.get_units("lfor_x", units="si") == "m s^-2"
