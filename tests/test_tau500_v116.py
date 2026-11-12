import numpy as np
from test_engine import ArrayData

HMINUS_SAHA = 1.03526e-16
HMINUS_SIGMA_CM2 = 2.9256e-17
HMINUS_BINDING_EV = 0.754
EV_TO_ERG = 1.60217733e-12
K_B_CGS = 1.38e-16
TAU_TOP = 1.0e-16


def make_tau_data(n=9):
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    # Deliberately non-uniform native z to exercise the trapezoidal geometry.
    z = np.asarray(
        [0.0, 0.12, 0.29, 0.53, 0.86, 1.31, 1.90, 2.67, 3.61], dtype=np.float32
    )[:n]
    shape = (n, n, n)
    ii, jj, kk = np.meshgrid(
        np.arange(n, dtype=np.float32),
        np.arange(n, dtype=np.float32),
        np.arange(n, dtype=np.float32),
        indexing="ij",
    )

    arrays = {
        "r": np.full(shape, 2.0, dtype=np.float32, order="F"),
        "e": np.full(shape, 6.0, dtype=np.float32, order="F"),
        "px": np.zeros(shape, dtype=np.float32, order="F"),
        "py": np.zeros(shape, dtype=np.float32, order="F"),
        "pz": np.zeros(shape, dtype=np.float32, order="F"),
        "bx": np.zeros(shape, dtype=np.float32, order="F"),
        "by": np.zeros(shape, dtype=np.float32, order="F"),
        "bz": np.zeros(shape, dtype=np.float32, order="F"),
        "ne": np.asfortranarray(
            2.0e12 * (1.0 + 0.015 * ii + 0.02 * kk), dtype=np.float32
        ),
        "tg": np.asfortranarray(5200.0 + 45.0 * jj + 120.0 * kk, dtype=np.float32),
        "nh": np.asfortranarray(
            8.0e16 * (1.0 + 0.01 * ii + 0.03 * kk), dtype=np.float32
        ),
    }
    return ArrayData(arrays, x, y, z)


def tau500_reference(ne, nh, tg, z_cm):
    exponent_constant = HMINUS_BINDING_EV * EV_TO_ERG / K_B_CGS
    chi = (
        HMINUS_SAHA
        * ne.astype(np.float64)
        * HMINUS_SIGMA_CM2
        / tg.astype(np.float64) ** 1.5
        * np.exp(exponent_constant / tg.astype(np.float64))
        * nh.astype(np.float64)
    )

    tau = np.empty(tg.shape, dtype=np.float32, order="F")
    tau[:, :, 0] = TAU_TOP
    for k in range(1, tg.shape[2]):
        dz = abs(float(z_cm[k]) - float(z_cm[k - 1]))
        tau[:, :, k] = (
            tau[:, :, k - 1].astype(np.float64)
            + 0.5 * (chi[:, :, k - 1] + chi[:, :, k]) * dz
        ).astype(np.float32)
    return tau


def test_tau500_matches_idl_hminus_trapezoid_reference():
    dd = make_tau_data()
    ne = dd.arrays["ne"]
    nh = dd.arrays["nh"]
    tg = dd.arrays["tg"]
    z_cm = dd._meta(0).z.astype(np.float64) * dd._meta(0).units.u_l

    expected = tau500_reference(ne, nh, tg, z_cm)
    got = dd.get_var("tau500", 0, units="bifrost", br=True)

    assert got.dtype == np.float32
    np.testing.assert_allclose(got, expected, rtol=3e-6, atol=0.0)
    assert np.all(got[:, :, 0] == np.float32(TAU_TOP))
    assert np.all(np.diff(got, axis=2) >= 0.0)


def test_tau500_z_subdomain_is_cropped_only_after_full_column_integration():
    dd = make_tau_data()
    full = dd.get_var("tau500", 0, units="bifrost", br=True)

    sub = dd.get_var(
        "tau500",
        0,
        ix_br=slice(2, 7),
        iy_br=slice(1, 6),
        iz_br=slice(3, 8),
        units="bifrost",
        br=True,
    )
    np.testing.assert_array_equal(sub, full[2:7, 1:6, 3:8])

    # In particular, the first requested z plane must not be reset to tau_top.
    assert np.all(sub[:, :, 0] > TAU_TOP)


def test_tau500_physical_orientation_is_only_applied_after_native_integration():
    dd = make_tau_data()
    native = dd.get_var("tau500", 0, units="bifrost", br=True)
    physical = dd.get_var("tau500", 0, units="bifrost", br=False)
    np.testing.assert_array_equal(physical, native[:, ::-1, ::-1])

    # Public-index subdomains must be identical to cropping the already
    # oriented full result, including a z interval that does not start at the
    # native top boundary.
    sub = dd.get_var(
        "tau500",
        0,
        ix=slice(1, 7),
        iy=slice(2, 8),
        iz=slice(2, 7),
        units="bifrost",
        br=False,
    )
    np.testing.assert_array_equal(sub, physical[1:7, 2:8, 2:7])


def test_tau500_uses_canonical_hion_temperature_and_electron_density():
    dd = make_tau_data()
    shape = dd.arrays["r"].shape

    # Remove stored canonical fields so tg/ne resolve through HION. If tau500
    # incorrectly forces the EOS this test fails immediately.
    del dd.arrays["tg"]
    del dd.arrays["ne"]
    dd.arrays["hiontg"] = np.full(shape, 6100.0, dtype=np.float32, order="F")
    dd.arrays["hionne"] = np.full(shape, 3.0e12, dtype=np.float32, order="F")

    def no_eos(_snap):
        raise AssertionError(
            "tau500 must not force EOS thermodynamics when HION is available"
        )

    dd._eos = no_eos

    z_cm = dd._meta(0).z.astype(np.float64) * dd._meta(0).units.u_l
    expected = tau500_reference(
        dd.arrays["hionne"], dd.arrays["nh"], dd.arrays["hiontg"], z_cm
    )
    got = dd.get_var("tau500", 0, units="bifrost", br=True)
    np.testing.assert_allclose(got, expected, rtol=3e-6, atol=0.0)


def test_tau500_is_registered_as_dimensionless():
    dd = make_tau_data()
    info = dd.info("tau500")
    assert info["location"] == "center"
    assert info["unit_type"] == "dimensionless"
    assert "H-minus" in info["description"]
    assert dd.get_units("tau500") == ""
