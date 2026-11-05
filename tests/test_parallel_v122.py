"""Regression checks for the v0.1.22 Numba parallelization pass."""

import numpy as np

import bifrost_proto.core as core
import bifrost_proto.eos as eos
import bifrost_proto.quantities as quantities
import bifrost_proto.responses as responses


def _assert_parallel_no_fastmath(dispatcher):
    opts = dispatcher.targetoptions
    assert opts.get("parallel") is True
    assert not opts.get("fastmath", False)


def test_performance_critical_numba_dispatchers_are_parallel_without_fastmath():
    dispatchers = (
        core._uniform_z_2d_c,
        core._uniform_z_2d_f,
        core._uniform_z_3d_c,
        core._uniform_z_3d_f,
        quantities._norm2_flat,
        quantities._norm3_flat,
        quantities._norm2_strided,
        quantities._norm3_strided,
        quantities._dot3_native,
        quantities._negative_cross_term_native,
        quantities._acute_cosine_native,
        quantities._timescale_native,
        eos._eos_interp_flat,
        eos._eos_interp_strided_3d,
        responses._uniform_response_emissivity_flat,
        responses._uniform_response_emissivity_strided,
    )
    for dispatcher in dispatchers:
        _assert_parallel_no_fastmath(dispatcher)


def test_uniform_z_parallel_kernels_are_bitwise_equal_to_direct_reference():
    rng = np.random.default_rng(122)
    z = np.linspace(-2.0, 5.0, 23, dtype=np.float64)
    z = z[0] + (z - z[0]) ** 1.13
    zu = np.linspace(z[0], z[-1], z.size)
    hi = np.clip(np.searchsorted(z, zu, side="right"), 1, z.size - 1).astype(np.int64)
    lo = (hi - 1).astype(np.int64)
    w = (zu - z[lo]) / (z[hi] - z[lo])

    for order, kernel in (("C", core._uniform_z_3d_c), ("F", core._uniform_z_3d_f)):
        value = np.array(rng.normal(size=(17, 13, 23)), dtype=np.float32, order=order)
        got = np.empty_like(value)
        kernel(value, lo, hi, w, got)

        expected = np.empty_like(value)
        for k in range(value.shape[2]):
            wk = w[k]
            expected[:, :, k] = (
                (1.0 - wk) * value[:, :, lo[k]] + wk * value[:, :, hi[k]]
            ).astype(np.float32)
        assert np.array_equal(got, expected)


def test_parallel_vector_algebra_is_bitwise_stable():
    rng = np.random.default_rng(123)
    shape = (15, 11, 9)
    arrays = [np.array(rng.normal(size=shape), dtype=np.float32, order="F") for _ in range(6)]
    ax, ay, az, bx, by, bz = arrays

    got_dot = np.empty_like(ax)
    quantities._dot3_native(ax, ay, az, bx, by, bz, got_dot)
    ref_dot = np.empty_like(ax)
    for k in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                ref_dot[i, j, k] = (
                    ax[i, j, k] * bx[i, j, k]
                    + ay[i, j, k] * by[i, j, k]
                    + az[i, j, k] * bz[i, j, k]
                )
    assert np.array_equal(got_dot, ref_dot)

    got_cross = np.empty_like(ax)
    quantities._negative_cross_term_native(ax, by, ay, bx, got_cross)
    ref_cross = np.empty_like(ax)
    for k in range(shape[2]):
        for j in range(shape[1]):
            for i in range(shape[0]):
                ref_cross[i, j, k] = -(
                    ax[i, j, k] * by[i, j, k]
                    - ay[i, j, k] * bx[i, j, k]
                )
    assert np.array_equal(got_cross, ref_cross)


def test_parallel_timescale_preserves_exact_special_value_semantics():
    e = np.arange(1, 1 + 4 * 3 * 5, dtype=np.float32).reshape(4, 3, 5, order="F")
    q = np.linspace(-3.0, 4.0, e.size, dtype=np.float32).reshape(e.shape, order="F")
    q[0, 0, 0] = 0.0
    q[1, 1, 1] = np.nan
    got = np.empty_like(e)
    quantities._timescale_native(e, q, got)

    ref = np.empty_like(e)
    for k in range(e.shape[2]):
        for j in range(e.shape[1]):
            for i in range(e.shape[0]):
                qv = q[i, j, k]
                ref[i, j, k] = np.inf if qv == 0.0 else e[i, j, k] / abs(qv)
    assert np.array_equal(got, ref, equal_nan=True)
