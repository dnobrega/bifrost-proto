import numpy as np
import pytest

from bifrost_proto.core import UnknownVariableError
from bifrost_proto.units import BifrostUnits
from test_engine import ArrayData


def _interior(a):
    return a[6:-6, 6:-6, 6:-6]


def make_poynting_data(n=28, *, include_total_e=True):
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    shape = (n, n, n)
    r = np.full(shape, 2.0, dtype=np.float32)
    arrays = {
        "r": r,
        "e": np.ones(shape, dtype=np.float32),
        "px": r * np.float32(1.0),
        "py": r * np.float32(2.0),
        "pz": r * np.float32(3.0),
        "bx": np.full(shape, 4.0, dtype=np.float32),
        "by": np.full(shape, 5.0, dtype=np.float32),
        "bz": np.full(shape, 6.0, dtype=np.float32),
    }
    if include_total_e:
        arrays.update(
            {
                "ex": np.full(shape, 10.0, dtype=np.float32),
                "ey": np.full(shape, 20.0, dtype=np.float32),
                "ez": np.full(shape, 30.0, dtype=np.float32),
            }
        )
    return ArrayData(arrays, x, y, z)


def test_total_poynting_flux_from_stored_e_cross_b():
    dd = make_poynting_data()

    sx = dd.get_var("sp_x", 0, units="bifrost", br=True)
    sy = dd.get_var("sp_y", 0, units="bifrost", br=True)
    sz = dd.get_var("sp_z", 0, units="bifrost", br=True)

    # E=(10,20,30), B=(4,5,6): E x B = (-30, 60, -30).
    assert np.allclose(_interior(sx), -30.0, atol=2e-5)
    assert np.allclose(_interior(sy), 60.0, atol=2e-5)
    assert np.allclose(_interior(sz), -30.0, atol=2e-5)

    sh = dd.get_var("sp_h", 0, units="bifrost", br=True)
    mod = dd.get_var("modsp", 0, units="bifrost", br=True)
    assert np.allclose(_interior(sh), np.hypot(-30.0, 60.0), atol=2e-4)
    assert np.allclose(_interior(mod), np.sqrt(30.0**2 + 60.0**2 + 30.0**2), atol=2e-4)


def test_total_poynting_flux_requires_stored_total_electric_field():
    dd = make_poynting_data(include_total_e=False)
    with pytest.raises(UnknownVariableError):
        dd.get_var("sp_z", 0, units="bifrost", br=True)


def test_ideal_poynting_flux_and_vertical_decomposition():
    dd = make_poynting_data(include_total_e=False)

    sx = dd.get_var("sp_ideal_x", 0, units="bifrost", br=True)
    sy = dd.get_var("sp_ideal_y", 0, units="bifrost", br=True)
    sz = dd.get_var("sp_ideal_z", 0, units="bifrost", br=True)
    emerg = dd.get_var("sp_ideal_emerg_z", 0, units="bifrost", br=True)
    shear = dd.get_var("sp_ideal_shear_z", 0, units="bifrost", br=True)

    # u=(1,2,3), B=(4,5,6):
    # B^2 u - (u.B)B = (-51, -6, 39).
    assert np.allclose(_interior(sx), -51.0, atol=2e-4)
    assert np.allclose(_interior(sy), -6.0, atol=2e-4)
    assert np.allclose(_interior(sz), 39.0, atol=2e-4)

    # Sz,emerg = uz*(bx^2+by^2) = 123
    # Sz,shear = -bz*(ux*bx+uy*by) = -84
    assert np.allclose(_interior(emerg), 123.0, atol=2e-4)
    assert np.allclose(_interior(shear), -84.0, atol=2e-4)
    assert np.array_equal(sz, emerg + shear)

    sh = dd.get_var("sp_ideal_h", 0, units="bifrost", br=True)
    mod = dd.get_var("modsp_ideal", 0, units="bifrost", br=True)
    assert np.allclose(_interior(sh), np.hypot(-51.0, -6.0), atol=2e-4)
    assert np.allclose(_interior(mod), np.sqrt(51.0**2 + 6.0**2 + 39.0**2), atol=2e-4)


def test_poynting_vector_signs_follow_public_physical_orientation():
    dd = make_poynting_data()

    # Public physical orientation reverses native Y/Z and changes the sign of
    # Y/Z vector components. Constant fields make the spatial reversal inert.
    sx = dd.get_var("sp_x", 0, units="bifrost")
    sy = dd.get_var("sp_y", 0, units="bifrost")
    sz = dd.get_var("sp_z", 0, units="bifrost")
    assert np.allclose(_interior(sx), -30.0, atol=2e-5)
    assert np.allclose(_interior(sy), -60.0, atol=2e-5)
    assert np.allclose(_interior(sz), 30.0, atol=2e-5)

    emerg = dd.get_var("sp_ideal_emerg_z", 0, units="bifrost")
    shear = dd.get_var("sp_ideal_shear_z", 0, units="bifrost")
    assert np.allclose(_interior(emerg), -123.0, atol=2e-4)
    assert np.allclose(_interior(shear), 84.0, atol=2e-4)


def test_poynting_subdomain_matches_full_domain():
    dd = make_poynting_data()
    selection = dict(ix=slice(8, 18), iy=slice(9, 19), iz=slice(7, 17))

    for name in ("sp_z", "sp_ideal_z", "sp_ideal_emerg_z", "sp_ideal_shear_z"):
        full = dd.get_var(name, 0, units="bifrost", br=True)
        sub = dd.get_var(name, 0, units="bifrost", br=True, **selection)
        assert np.array_equal(sub, full[8:18, 9:19, 7:17]), name


def test_energy_flux_units_and_labels():
    u = BifrostUnits()
    assert np.isclose(u.u_flux, u.u_e * u.u_l / u.u_t)
    assert np.isclose(u.factor("energy_flux", "cgs"), u.u_flux)
    assert np.isclose(u.factor("energy_flux", "solar"), u.u_flux)
    assert np.isclose(u.factor("energy_flux", "si"), u.u_flux * 1.0e-3)
    assert u.label("energy_flux", "solar") == "erg cm^-2 s^-1"
    assert u.label("energy_flux", "si") == "W m^-2"

    dd = make_poynting_data()
    assert dd.get_unit_type("sp_z") == "energy_flux"
    assert dd.get_units("sp_z") == "erg cm^-2 s^-1"
    assert dd.get_units("sp_z", units="si") == "W m^-2"


def test_poynting_latex_labels():
    dd = make_poynting_data()
    assert dd.get_var_name("sp_x", latex=True) == r"$S_x$"
    assert dd.get_var_name("sp_ideal_z", latex=True) == r"$S^{\mathrm{ideal}}_z$"
    assert dd.get_var_name("sp_h", latex=True) == r"$S_h$"
    assert dd.get_var_name("modsp", latex=True) == r"$|\mathbf{S}|$"
    assert dd.get_var_name("modsp_ideal", latex=True) == r"$|\mathbf{S}^{\mathrm{ideal}}|$"
    assert dd.get_var_name("sp_ideal_emerg_z", latex=True) == r"$S^{\mathrm{ideal}}_{z,\mathrm{emerg}}$"
    assert dd.get_var_name("sp_ideal_shear_z", latex=True) == r"$S^{\mathrm{ideal}}_{z,\mathrm{shear}}$"
