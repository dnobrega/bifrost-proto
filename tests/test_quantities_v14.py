import numpy as np
import pytest

from bifrost_proto.core import UnknownVariableError
from test_engine import ArrayData, make_data


def make_constant_data(n=24):
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    shape = (n, n, n)
    r = np.full(shape, 2.0, dtype=np.float32)
    arrays = {
        "r": r,
        "e": np.full(shape, 6.0, dtype=np.float32),
        "px": r * 1.0,
        "py": r * 2.0,
        "pz": r * 3.0,
        "bx": np.full(shape, 4.0, dtype=np.float32),
        "by": np.full(shape, 5.0, dtype=np.float32),
        "bz": np.full(shape, 6.0, dtype=np.float32),
        "pg": np.full(shape, 7.0, dtype=np.float32),
    }
    return ArrayData(arrays, x, y, z)


def interior(a):
    return a[5:-5, 5:-5, 5:-5]


def test_canonical_vector_names_and_legacy_aliases_share_cache():
    dd = make_data()

    with dd.cache():
        canonical = dd.get_var("b_x", 0, units="bifrost", br=True)
        legacy = dd.get_var("bx", 0, units="bifrost", br=True)
        assert np.array_equal(canonical, legacy)
        assert dd.read_counts["bx"] == 1

    assert dd.get_location("b_x") == "xface"
    assert dd.get_location("bx") == "xface"


def test_magnetic_block_on_constant_fields():
    dd = make_constant_data()
    b = np.sqrt(np.float32(77.0))

    assert np.allclose(interior(dd.get_var("modb", 0, units="bifrost", br=True)), b)
    assert np.allclose(
        interior(dd.get_var("b_h", 0, units="bifrost", br=True)), np.sqrt(41.0)
    )
    assert np.allclose(
        interior(dd.get_var("bhat_x", 0, units="bifrost", br=True)), 4.0 / b
    )
    assert np.allclose(
        interior(dd.get_var("bxob", 0, units="bifrost", br=True)), 4.0 / b
    )
    assert np.allclose(
        interior(dd.get_var("bhat_y", 0, units="bifrost", br=True)), 5.0 / b
    )
    assert np.allclose(
        interior(dd.get_var("bhat_z", 0, units="bifrost", br=True)), 6.0 / b
    )

    assert np.allclose(
        interior(dd.get_var("divb", 0, units="bifrost", br=True)), 0.0, atol=2e-6
    )
    assert np.allclose(
        interior(dd.get_var("modcurlb", 0, units="bifrost", br=True)), 0.0, atol=2e-6
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = dd.get_var("modb", 0, units="bifrost", br=True) / np.abs(
            dd.get_var("divb", 0, units="bifrost", br=True)
        )
    np.testing.assert_allclose(
        interior(dd.get_var("ldivb", 0, units="bifrost", br=True)),
        interior(expected),
        rtol=1e-6,
        equal_nan=True,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = dd.get_var("modb", 0, units="bifrost", br=True) / np.abs(
            dd.get_var("modcurlb", 0, units="bifrost", br=True)
        )
    np.testing.assert_allclose(
        interior(dd.get_var("lcurlb", 0, units="bifrost", br=True)),
        interior(expected),
        rtol=1e-6,
        equal_nan=True,
    )
    assert np.allclose(
        interior(dd.get_var("beta", 0, units="bifrost", br=True)), 14.0 / 77.0
    )


def test_velocity_parallel_perpendicular_block_on_constant_fields():
    dd = make_constant_data()

    modu = np.sqrt(14.0)
    uh = np.sqrt(5.0)
    modb = np.sqrt(77.0)
    uparb = 32.0 / modb

    assert np.allclose(interior(dd.get_var("modu", 0, units="bifrost", br=True)), modu)
    assert np.allclose(interior(dd.get_var("u_h", 0, units="bifrost", br=True)), uh)
    assert np.allclose(
        interior(dd.get_var("uparb", 0, units="bifrost", br=True)), uparb
    )

    expected_par = np.array([128.0 / 77.0, 160.0 / 77.0, 192.0 / 77.0])
    for axis, expected in zip("xyz", expected_par):
        got = dd.get_var(f"uparb_{axis}", 0, units="bifrost", br=True)
        assert np.allclose(interior(got), expected)

    expected_perp = np.array([1.0, 2.0, 3.0]) - expected_par
    for axis, expected in zip("xyz", expected_perp):
        got = dd.get_var(f"uperpb_{axis}", 0, units="bifrost", br=True)
        assert np.allclose(interior(got), expected)

    uperpb = np.sqrt(np.sum(expected_perp * expected_perp))
    assert np.allclose(
        interior(dd.get_var("uperpb", 0, units="bifrost", br=True)), uperpb
    )


def test_velocity_derivative_block_and_clean_aliases():
    dd = make_constant_data()

    divu = dd.get_var("divu", 0, units="bifrost", br=True)
    modcurlu = dd.get_var("modcurlu", 0, units="bifrost", br=True)

    assert np.allclose(interior(divu), 0.0, atol=2e-6)
    assert np.allclose(interior(modcurlu), 0.0, atol=2e-6)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = dd.get_var("modu", 0, units="bifrost", br=True) / np.abs(
            dd.get_var("modcurlu", 0, units="bifrost", br=True)
        )
    np.testing.assert_allclose(
        interior(dd.get_var("lcurlu", 0, units="bifrost", br=True)),
        interior(expected),
        rtol=1e-6,
        equal_nan=True,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = dd.get_var("modu", 0, units="bifrost", br=True) / np.abs(
            dd.get_var("divu", 0, units="bifrost", br=True)
        )
    np.testing.assert_allclose(
        interior(dd.get_var("ldivu", 0, units="bifrost", br=True)),
        interior(expected),
        rtol=1e-6,
        equal_nan=True,
    )

    assert dd.info("uh")["name"] == "u_h"
    assert dd.info("bh")["name"] == "b_h"

    for removed in (
        "curlu",
        "vort",
        "vort_x",
        "vort_y",
        "vort_z",
        "uperb",
        "uperbx",
        "uperby",
        "uzperb",
        "uxparb",
        "uyparb",
        "uzparb",
    ):
        with pytest.raises(UnknownVariableError):
            dd.info(removed)


def test_uparb_subdomain_matches_full_domain_exactly():
    dd = make_data(n=28)
    full = dd.get_var("uparb", 0, units="bifrost", br=True)
    sub = dd.get_var(
        "uparb",
        0,
        ix=slice(8, 18),
        iy=slice(9, 19),
        iz=slice(7, 17),
        units="bifrost",
        br=True,
    )
    assert np.array_equal(sub, full[8:18, 9:19, 7:17])
