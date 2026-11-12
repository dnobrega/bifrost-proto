import numpy as np
from test_engine import make_data

import bifrost_proto.core as core_module
from bifrost_proto.core import BifrostData


def _uniform_z_reference(value, axis_names, z):
    """v1.17 vectorized implementation kept only as a regression reference."""
    z = np.asarray(z, dtype=np.float64)
    znew = np.linspace(z[0], z[-1], z.size)
    axis = axis_names.index("z")
    hi = np.searchsorted(z, znew, side="right")
    hi = np.clip(hi, 1, z.size - 1)
    lo = hi - 1
    w = (znew - z[lo]) / (z[hi] - z[lo])
    v0 = np.take(value, lo, axis=axis)
    v1 = np.take(value, hi, axis=axis)
    shape = [1] * value.ndim
    shape[axis] = znew.size
    out = (1.0 - w.reshape(shape)) * v0 + w.reshape(shape) * v1
    return out.astype(value.dtype, copy=False), znew


def test_uniform_z_numba_matches_vectorized_reference_c_and_f():
    rng = np.random.default_rng(1234)
    z = np.linspace(-2.0, 5.0, 17, dtype=np.float64)
    z = z[0] + (z - z[0]) ** 1.15

    for order in ("C", "F"):
        value = np.array(
            rng.standard_normal((11, 13, 17)), dtype=np.float32, order=order
        )
        expected, z_expected = _uniform_z_reference(value, ("x", "y", "z"), z)
        got, z_got = BifrostData._uniformize_z(value, ("x", "y", "z"), z)

        assert got.dtype == value.dtype
        assert np.array_equal(got, expected)
        assert np.array_equal(z_got, z_expected)


def test_uniform_z_numba_matches_reference_after_scalar_axes():
    rng = np.random.default_rng(5678)
    z = np.linspace(0.0, 3.0, 19, dtype=np.float64) ** 1.25

    cases = [
        (rng.standard_normal((12, 19)).astype(np.float32), ("x", "z")),
        (rng.standard_normal(19).astype(np.float32), ("z",)),
    ]
    for value, axis_names in cases:
        expected, z_expected = _uniform_z_reference(value, axis_names, z)
        got, z_got = BifrostData._uniformize_z(value, axis_names, z)
        assert np.array_equal(got, expected)
        assert np.array_equal(z_got, z_expected)


def test_getvar_skips_unit_multiply_when_factor_is_one(monkeypatch):
    dd = make_data(n=8)
    dd.arrays["tg"] = np.full((8, 8, 8), 6500.0, dtype=np.float32)

    def fail_multiply(*args, **kwargs):
        raise AssertionError("np.multiply must not be called for a unit factor of one")

    monkeypatch.setattr(core_module.np, "multiply", fail_multiply)
    got = dd.get_var("tg", 0, units="solar")
    assert np.all(got == 6500.0)


def test_inplace_public_unit_conversion_does_not_modify_resolver_cache():
    dd = make_data(n=8)
    with dd.cache():
        solar = dd.get_var("r", 0, units="solar")
        native = dd.get_var("r", 0, units="bifrost", br=True)

    assert np.allclose(solar, 2.0 * dd._meta(0).units.u_r)
    assert np.all(native == 2.0)
