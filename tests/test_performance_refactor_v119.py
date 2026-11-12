import numpy as np
from test_engine import make_data

from bifrost_proto import quantities


def test_getz_uniform_z_matches_full_domain_uniformization_grid():
    dd = make_data(n=17)
    # make_data uses a uniform mesh, so perturb the representative Z grid to
    # exercise the new coordinate-only path without involving field data.
    meta = dd._meta(0)
    z_native = np.asarray(meta.z, dtype=np.float64)
    z_native[:] = (
        z_native[0]
        + np.sign(z_native[-1] - z_native[0])
        * np.linspace(0.0, abs(z_native[-1] - z_native[0]), z_native.size) ** 1.2
    )

    for br in (False, True):
        z = dd.get_z(br=br)
        zuni = dd.get_z(br=br, uniform_z=True)
        expected = np.linspace(z[0], z[-1], z.size, dtype=np.float64)
        assert np.array_equal(zuni, expected)
        assert zuni.size == z.size
        assert zuni[0] == z[0]
        assert zuni[-1] == z[-1]


def test_dot3_numba_matches_explicit_float32_reference():
    rng = np.random.default_rng(123)
    shape = (13, 11, 9)
    arrays = [
        np.array(rng.normal(size=shape), dtype=np.float32, order="F") for _ in range(6)
    ]
    ax, ay, az, bx, by, bz = arrays

    got = quantities._dot3_arrays(ax, ay, az, bx, by, bz)
    expected = np.empty_like(ax)
    np.multiply(ax, bx, out=expected)
    scratch = np.empty_like(ax)
    np.multiply(ay, by, out=scratch)
    np.add(expected, scratch, out=expected)
    np.multiply(az, bz, out=scratch)
    np.add(expected, scratch, out=expected)

    np.testing.assert_allclose(got, expected, rtol=2e-7, atol=2e-7)


def test_negative_cross_term_numba_matches_numpy_reference():
    rng = np.random.default_rng(456)
    shape = (12, 10, 8)
    arrays = [
        np.array(rng.normal(size=shape), dtype=np.float32, order="F") for _ in range(4)
    ]
    a, b, c, d = arrays
    got = quantities._negative_cross_term_arrays(a, b, c, d)
    expected = -(a * b - c * d)
    np.testing.assert_allclose(got, expected, rtol=2e-7, atol=2e-7)


def test_timescale_numba_preserves_zero_inf_sign_and_nan_semantics():
    e = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32).reshape(1, 1, 4)
    q = np.array([2.0, -4.0, 0.0, np.nan], dtype=np.float32).reshape(1, 1, 4)
    got = quantities._timescale_from_energy(e, q)
    assert got[0, 0, 0] == np.float32(0.5)
    assert got[0, 0, 1] == np.float32(0.5)
    assert np.isinf(got[0, 0, 2]) and got[0, 0, 2] > 0
    assert np.isnan(got[0, 0, 3])


def test_acute_cosine_kernel_matches_vectorized_reference():
    rng = np.random.default_rng(789)
    shape = (11, 9, 7)
    vals = [
        np.array(rng.normal(size=shape), dtype=np.float32, order="F") for _ in range(6)
    ]
    jx, jy, jz, bx, by, bz = vals
    # One undefined point exercises the explicit NaN path.
    jx[0, 0, 0] = jy[0, 0, 0] = jz[0, 0, 0] = 0.0

    got = np.empty_like(jx)
    quantities._acute_cosine_native(jx, jy, jz, bx, by, bz, got)

    dot = jx * bx + jy * by + jz * bz
    modj = np.sqrt(jx * jx + jy * jy + jz * jz)
    modb = np.sqrt(bx * bx + by * by + bz * bz)
    with np.errstate(invalid="ignore", divide="ignore"):
        expected = np.abs(dot) / (modj * modb)
    expected = np.clip(expected, 0.0, 1.0).astype(np.float32)
    expected[modj * modb == 0.0] = np.nan

    np.testing.assert_allclose(got, expected, rtol=4e-7, atol=4e-7, equal_nan=True)
