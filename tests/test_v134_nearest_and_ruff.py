"""Regression tests for staggered-grid nearest boundaries and public exports."""

from __future__ import annotations

import numpy as np
import pytest

import bifrost_proto
from bifrost_proto.version import __version__ as source_version
from test_engine import make_data


@pytest.mark.parametrize("br", [False, True])
@pytest.mark.parametrize(
    ("axis", "var", "coord_getter"),
    [
        ("x", "bx", "get_xf"),
        ("y", "by", "get_yf"),
        ("z", "uz", "get_zf"),
    ],
)
def test_nearest_uses_nearest_staggered_sample_at_center_boundary(br, axis, var, coord_getter):
    dd = make_data(n=24)
    centers = getattr(dd, f"get_{axis}")(br=br)
    faces = getattr(dd, coord_getter)(br=br)
    edge = next(i for i in (0, -1) if centers[i] < faces[0] or centers[i] > faces[-1])
    target = float(centers[edge])
    nearest_i = int(np.argmin(np.abs(faces - target)))

    actual = dd.get_var(var, 0, br=br, units="bifrost", method="nearest", **{axis: target})
    expected = dd.get_var(var, 0, br=br, units="bifrost", **{f"i{axis}": nearest_i})
    np.testing.assert_array_equal(actual, expected)

    # The coordinate-query helpers follow exactly the same sampling policy.
    assert getattr(dd, coord_getter)(br=br, **{axis: target}) == faces[nearest_i]

    # Linear should never extrapolate beyond actual sampled coordinate range.
    with pytest.raises(ValueError, match="outside grid"):
        dd.get_var(var, 0, br=br, method="linear", **{axis: target})

    # A coordinate beyond both center and face ranges should still be rejected.
    invalid = min(centers[0], faces[0]) - 2.0
    with pytest.raises(ValueError, match="outside grid"):
        dd.get_var(var, 0, br=br, method="nearest", **{axis: invalid})


def test_public_version_is_exported_and_registry_is_populated():
    # Verify the public export without hard-coding a release number.
    assert bifrost_proto.__version__ == source_version
    assert "__version__" in bifrost_proto.__all__
    dd = make_data(n=12)
    assert dd.registry.get("u_z") is not None
