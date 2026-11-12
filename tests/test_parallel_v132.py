"""Regression tests for the v0.1.32 thin-slab parallelization pass."""

import numpy as np

from bifrost_proto import quantities, stagger


def _farray(rng, shape):
    return np.array(rng.normal(size=shape), dtype=np.float32, order="F")


def test_flat_and_row_parallel_dot_are_bitwise_equal_on_horizontal_plane():
    rng = np.random.default_rng(132)
    shape = (37, 29, 1)
    ax, ay, az, bx, by, bz = [_farray(rng, shape) for _ in range(6)]

    flat_out = quantities._dot3_arrays(ax, ay, az, bx, by, bz)
    row_out = np.empty_like(ax)
    quantities._dot3_native(ax, ay, az, bx, by, bz, row_out)

    assert np.array_equal(flat_out, row_out)


def test_pointwise_strided_single_plane_paths_preserve_values_without_input_copies():
    rng = np.random.default_rng(133)
    shape = (42, 34, 3)
    values = [_farray(rng, shape) for _ in range(6)]
    views = [v[1::2, ::2, 1:2] for v in values]
    ax, ay, az, bx, by, bz = views

    assert all(v.shape[2] == 1 for v in views)
    assert all(not (v.flags.c_contiguous or v.flags.f_contiguous) for v in views)

    dot = quantities._dot3_arrays(ax, ay, az, bx, by, bz)
    expected_dot = ax * bx + ay * by + az * bz
    np.testing.assert_allclose(dot, expected_dot, rtol=2e-7, atol=2e-7)

    norm = quantities._norm_arrays((ax, ay, az))
    expected_norm = np.sqrt(ax * ax + ay * ay + az * az)
    np.testing.assert_allclose(norm, expected_norm, rtol=2e-7, atol=2e-7)

    spz = quantities._ideal_poynting_component_arrays(ax, ay, az, bx, by, bz, 2)
    expected_spz = az * (bx * bx + by * by) - bz * (ax * bx + ay * by)
    np.testing.assert_allclose(spz, expected_spz, rtol=3e-7, atol=3e-7)

    q = bx.copy(order="K")
    q[0, 0, 0] = 0.0
    tau = quantities._timescale_from_energy(ax, q)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_tau = ax / np.abs(q)
    expected_tau[q == 0.0] = np.inf
    np.testing.assert_allclose(tau, expected_tau, rtol=2e-7, atol=2e-7)


def test_new_flat_dispatchers_are_parallel_without_fastmath():
    dispatchers = (
        quantities._dot3_flat,
        quantities._negative_cross_term_flat,
        quantities._ideal_poynting_component_flat,
        quantities._ideal_poynting_emerg_z_flat,
        quantities._ideal_poynting_shear_z_flat,
        quantities._acute_cosine_flat,
        quantities._timescale_flat,
    )
    for dispatcher in dispatchers:
        opts = dispatcher.targetoptions
        assert opts.get("parallel") is True
        assert not opts.get("fastmath", False)


def test_xy_stagger_operators_accept_one_plane_after_collapsing_yz_parallelism():
    rng = np.random.default_rng(134)
    f = _farray(rng, (31, 27, 1))
    x = np.arange(f.shape[0], dtype=np.float32)
    y = np.arange(f.shape[1], dtype=np.float32)
    xf = x + np.float32(0.5)
    yf = y + np.float32(0.5)

    # The point of this regression is the nz=1 execution path. Existing exact
    # stagger tests cover the legacy arithmetic itself.
    xup = stagger.apply_operator("xup", f, x, xf, spacing=1.0)
    yup = stagger.apply_operator("yup", f, y, yf, spacing=1.0)

    assert xup.shape == f.shape
    assert yup.shape == f.shape
    assert np.isfinite(xup).all()
    assert np.isfinite(yup).all()
