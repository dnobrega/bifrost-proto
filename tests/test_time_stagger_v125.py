"""Regression tests for snapshot time and the v0.1.25 public stagger contract."""

from __future__ import annotations

import numpy as np
import pytest

from test_engine import make_data


def test_get_t_uses_standard_time_unit_conversion():
    dd = make_data(n=12)
    dd._meta_cache[0].params["tsnap"] = 1.5

    assert dd.get_t(0, units="bifrost") == pytest.approx(1.5)
    assert dd.get_t(0) == pytest.approx(150.0)
    assert dd.get_t(0, units="solar") == pytest.approx(150.0)
    assert dd.get_t(0, units="cgs") == pytest.approx(150.0)
    assert dd.get_t(0, units="si") == pytest.approx(150.0)


def test_get_t_raises_when_tsnap_is_missing():
    dd = make_data(n=12)
    with pytest.raises(KeyError, match="tsnap"):
        dd.get_t(0)


def test_public_stagger_physical_output_is_f_contiguous():
    dd = make_data(n=24)
    raw = dd.get_var("bz", 0, units="bifrost")
    got = dd.stagger.zup(raw)
    expected = dd.get_var("bz", 0, centered=True, units="bifrost")

    assert got.flags.f_contiguous
    assert not got.flags.c_contiguous
    np.testing.assert_array_equal(got, expected)


def test_public_stagger_native_output_is_f_contiguous():
    dd = make_data(n=24)
    raw = dd.get_var("bx", 0, units="bifrost", br=True)
    got = dd.stagger.xup(raw, br=True)
    expected = dd.get_var("bx", 0, centered=True, units="bifrost", br=True)

    assert got.flags.f_contiguous
    assert not got.flags.c_contiguous
    np.testing.assert_array_equal(got, expected)


def test_public_stagger_rejects_cropped_subdomain():
    dd = make_data(n=24)
    cropped = dd.get_var("bx", 0, ix=slice(6, 18), units="bifrost")

    with pytest.raises(ValueError, match="full run mesh|cropped subdomain"):
        dd.stagger.xup(cropped)


@pytest.mark.parametrize("op", ["xup", "yup", "zup"])
def test_public_stagger_physical_native_interpolation_equivalence(op):
    """Public physical stagger must be native Bifrost stagger plus reorientation."""
    dd = make_data(n=24)
    rng = np.random.default_rng(20260924)
    native = np.asfortranarray(rng.normal(size=dd._grid_meta().shape).astype(np.float32))
    physical = dd._reverse_yz_f(native)

    got = getattr(dd.stagger, op)(physical, br=False)
    native_result = getattr(dd.stagger, op)(native, br=True)
    expected = dd._reverse_yz_f(native_result)

    assert got.flags.f_contiguous
    np.testing.assert_array_equal(got, expected)


def test_public_stagger_physical_native_ddzdn_equivalence():
    """Physical Z derivatives must include the sign from the reversed Z axis."""
    dd = make_data(n=24)
    rng = np.random.default_rng(20260925)
    native = np.asfortranarray(rng.normal(size=dd._grid_meta().shape).astype(np.float32))
    physical = dd._reverse_yz_f(native)

    got = dd.stagger.ddzdn(physical, br=False)
    native_result = dd.stagger.ddzdn(native, br=True)
    expected = dd._reverse_yz_f(native_result, negate=True)

    assert got.flags.f_contiguous
    np.testing.assert_array_equal(got, expected)
