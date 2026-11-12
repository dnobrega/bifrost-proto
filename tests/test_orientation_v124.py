"""Regression tests for the parallel F-order public-orientation path."""

from __future__ import annotations

import numpy as np
import pytest
from test_engine import make_data


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("signed", [False, True])
def test_private_orientation_matches_numpy_reference(order, signed):
    """Parallel orientation must preserve the historical public values exactly."""
    dd = make_data(n=12)
    base = np.arange(12 * 11 * 10, dtype=np.float32).reshape((12, 11, 10), order=order)
    spec = dd.registry.get("by") if signed else dd.registry.get("bx")

    got = dd._orient_physical(base, spec)
    expected = base[:, ::-1, ::-1]
    if signed:
        expected = -expected

    assert got.flags.f_contiguous
    assert not got.flags.c_contiguous
    np.testing.assert_array_equal(got, expected)


def test_private_orientation_accepts_strided_native_input():
    """The direct indexed kernel must also handle unusual strided resolver arrays."""
    dd = make_data(n=12)
    base = np.asfortranarray(
        np.arange(24 * 22 * 20, dtype=np.float32).reshape((24, 22, 20), order="F")
    )
    view = base[::2, ::2, ::2]
    assert not view.flags.c_contiguous and not view.flags.f_contiguous

    got = dd._orient_physical(view, dd.registry.get("bz"))
    expected = -view[:, ::-1, ::-1]

    assert got.flags.f_contiguous
    np.testing.assert_array_equal(got, expected)


def test_public_full_cube_is_f_contiguous_and_preserves_component_signs():
    """Default public cubes should be F-order with the established Y/Z signs."""
    dd = make_data(n=24)

    bx_native = dd.get_var("bx", 0, units="bifrost", br=True)
    by_native = dd.get_var("by", 0, units="bifrost", br=True)
    bz_native = dd.get_var("bz", 0, units="bifrost", br=True)

    bx = dd.get_var("bx", 0, units="bifrost")
    by = dd.get_var("by", 0, units="bifrost")
    bz = dd.get_var("bz", 0, units="bifrost")

    for value in (bx, by, bz):
        assert value.flags.f_contiguous
        assert not value.flags.c_contiguous

    np.testing.assert_array_equal(bx, bx_native[:, ::-1, ::-1])
    np.testing.assert_array_equal(by, -by_native[:, ::-1, ::-1])
    np.testing.assert_array_equal(bz, -bz_native[:, ::-1, ::-1])


def test_physical_unit_conversion_keeps_f_order_output():
    """In-place public unit scaling must not destroy the new F-contiguous layout."""
    dd = make_data(n=24)
    got = dd.get_var("bx", 0, units="solar")
    assert got.flags.f_contiguous
    assert not got.flags.c_contiguous


def test_uniform_z_keeps_f_order_for_full_public_cube():
    """Full-cube uniform-Z resampling should stay on the optimized F-order path."""
    dd = make_data(n=24)
    got = dd.get_var("bx", 0, units="bifrost", uniform_z=True)
    assert got.flags.f_contiguous
    assert not got.flags.c_contiguous


def test_orientation_dispatchers_are_parallel_without_fastmath():
    """The public-orientation kernels must stay parallel and IEEE-conservative."""
    from bifrost_proto import core

    for dispatcher in (core._orient_physical_3d_f, core._orient_physical_signed_3d_f):
        opts = dispatcher.targetoptions
        assert opts.get("parallel") is True
        assert not opts.get("fastmath", False)
