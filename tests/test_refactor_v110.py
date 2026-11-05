import numpy as np
import pytest

from bifrost_proto.core import EvaluationContext, UnknownVariableError
from bifrost_proto.quantities import (
    _curl_component,
    _norm_arrays,
    _vector_divergence,
)
from bifrost_proto.selection import Selection
from test_engine import ArrayData


def _full(n):
    return Selection(slice(0, n), slice(0, n), slice(0, n))


def _interior(a):
    return a[5:-5, 5:-5, 5:-5]


def test_norm_helper_matches_direct_norm_and_does_not_modify_inputs():
    rng = np.random.default_rng(123)
    parts = [rng.normal(size=(12, 11, 10)).astype(np.float32) for _ in range(3)]
    originals = [p.copy() for p in parts]

    got = _norm_arrays(parts)
    expected = np.sqrt(parts[0] * parts[0] + parts[1] * parts[1] + parts[2] * parts[2])

    assert got.dtype == np.float32
    assert np.allclose(got, expected, rtol=2e-6, atol=2e-6)
    for part, original in zip(parts, originals):
        assert np.array_equal(part, original)


def test_public_horizontal_names_are_b_h_and_u_h_only():
    n = 24
    shape = (n, n, n)
    coords = np.arange(n, dtype=np.float32)
    rho = np.ones(shape, dtype=np.float32)
    dd = ArrayData(
        {
            "r": rho,
            "e": rho,
            "px": rho,
            "py": 2 * rho,
            "pz": 3 * rho,
            "bx": 4 * rho,
            "by": 5 * rho,
            "bz": 6 * rho,
            "pg": rho,
        },
        coords, coords, coords,
    )

    assert np.allclose(_interior(dd.get_var("b_h", 0, units="bifrost", br=True)), np.hypot(4.0, 5.0))
    assert np.allclose(_interior(dd.get_var("u_h", 0, units="bifrost", br=True)), np.hypot(1.0, 2.0))
    assert dd.info("bh")["name"] == "b_h"
    assert dd.info("uh")["name"] == "u_h"


def test_edge_vector_curl_uses_up_derivatives_and_returns_face_values():
    n = 24
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    one = np.ones((n, n, n), dtype=np.float32)
    dd = ArrayData(
        {
            "r": one,
            "e": one,
            "px": one,
            "py": one,
            "pz": one,
            "bx": one,
            "by": one,
            "bz": one,
            # Edge vector E = (z, x, y) has curl(E) = (1, 1, 1).
            "ex": zz.astype(np.float32),
            "ey": xx.astype(np.float32),
            "ez": yy.astype(np.float32),
            "pg": one,
        },
        x, y, z,
    )
    context = EvaluationContext(cache={})
    selection = _full(n)

    for axis in "xyz":
        got = _curl_component(
            dd, 0, selection, context, ("e_x", "e_y", "e_z"), axis
        )
        assert np.allclose(_interior(got), 1.0, atol=3e-5)


def test_divergence_helper_rejects_edge_vectors():
    n = 16
    coords = np.arange(n, dtype=np.float32)
    one = np.ones((n, n, n), dtype=np.float32)
    dd = ArrayData(
        {
            "r": one,
            "e": one,
            "px": one,
            "py": one,
            "pz": one,
            "bx": one,
            "by": one,
            "bz": one,
            "ex": one,
            "ey": one,
            "ez": one,
            "pg": one,
        },
        coords, coords, coords,
    )
    with pytest.raises(ValueError, match="xface/yface/zface"):
        _vector_divergence(
            dd,
            0,
            _full(n),
            EvaluationContext(cache={}),
            ("e_x", "e_y", "e_z"),
        )


def test_face_vector_curl_keeps_dn_path_and_edge_locations():
    n = 24
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    one = np.ones((n, n, n), dtype=np.float32)
    dd = ArrayData(
        {
            "r": one,
            "e": one,
            "px": one,
            "py": one,
            "pz": one,
            # Face vector B = (z, x, y) has curl(B) = (1, 1, 1).
            "bx": zz.astype(np.float32),
            "by": xx.astype(np.float32),
            "bz": yy.astype(np.float32),
            "pg": one,
        },
        x, y, z,
    )

    for axis, location in zip("xyz", ("xedge", "yedge", "zedge")):
        got = dd.get_var(f"curlb_{axis}", 0, units="bifrost", br=True)
        assert np.allclose(_interior(got), 1.0, atol=3e-5)
        assert dd.get_location(f"curlb_{axis}") == location
