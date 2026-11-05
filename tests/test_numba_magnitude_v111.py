import numpy as np
import pytest

import bifrost_proto.quantities as q
from test_engine import ArrayData


def _expected(*parts):
    total = np.zeros_like(parts[0])
    for part in parts:
        total = total + part * part
    return np.sqrt(total)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ncomp", [2, 3])
def test_norm_arrays_contiguous_matches_numpy_without_modifying_inputs(order, dtype, ncomp):
    rng = np.random.default_rng(20260918)
    parts = tuple(
        np.array(rng.normal(size=(17, 15, 13)), dtype=dtype, order=order)
        for _ in range(ncomp)
    )
    originals = tuple(part.copy(order="K") for part in parts)

    got = q._norm_arrays(parts)
    expected = _expected(*parts)

    assert got.dtype == dtype
    assert got.shape == parts[0].shape
    if order == "C":
        assert got.flags.c_contiguous
    else:
        assert got.flags.f_contiguous
    np.testing.assert_allclose(
        got,
        expected,
        rtol=2e-6 if dtype == np.float32 else 2e-14,
        atol=2e-6 if dtype == np.float32 else 2e-14,
    )
    for part, original in zip(parts, originals):
        np.testing.assert_array_equal(part, original)


def test_flat_memory_view_is_copy_free_for_c_and_f_arrays():
    for order in ("C", "F"):
        arr = np.arange(8 * 7 * 6, dtype=np.float32).reshape((8, 7, 6), order=order)
        flat = q._flat_memory_view(arr)
        assert flat.ndim == 1
        assert np.shares_memory(flat, arr)
        assert flat.size == arr.size


def test_norm_arrays_strided_view_uses_stride_kernel_without_flattening(monkeypatch):
    rng = np.random.default_rng(7)
    bases = tuple(
        np.asfortranarray(rng.normal(size=(30, 28, 26)).astype(np.float32))
        for _ in range(3)
    )
    parts = tuple(base[1:28:2, 2:26:3, 3:24:2] for base in bases)

    assert all(not p.flags.c_contiguous and not p.flags.f_contiguous for p in parts)
    assert all(np.shares_memory(p, base) for p, base in zip(parts, bases))
    originals = tuple(p.copy() for p in parts)

    # If the strided branch accidentally tries to flatten/contiguize the input,
    # fail loudly.  The intended path passes the strided views straight to Numba.
    monkeypatch.setattr(
        q,
        "_flat_memory_view",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("strided path must not call _flat_memory_view")
        ),
    )

    got = q._norm_arrays(parts)
    expected = _expected(*parts)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)
    for part, original in zip(parts, originals):
        np.testing.assert_array_equal(part, original)


def _diagnostic_data(n=36):
    rng = np.random.default_rng(991)
    coords = np.arange(n, dtype=np.float32)
    shape = (n, n, n)
    one = np.ones(shape, dtype=np.float32)
    # Smooth-ish nontrivial fields.  Fortran order mirrors Bifrost I/O.
    arrays = {
        "r": np.asfortranarray(1.0 + 0.1 * rng.random(shape, dtype=np.float32)),
        "e": np.asfortranarray(2.0 + 0.1 * rng.random(shape, dtype=np.float32)),
        "px": np.asfortranarray(0.2 * rng.normal(size=shape).astype(np.float32)),
        "py": np.asfortranarray(0.2 * rng.normal(size=shape).astype(np.float32)),
        "pz": np.asfortranarray(0.2 * rng.normal(size=shape).astype(np.float32)),
        "bx": np.asfortranarray(rng.normal(size=shape).astype(np.float32)),
        "by": np.asfortranarray(rng.normal(size=shape).astype(np.float32)),
        "bz": np.asfortranarray(rng.normal(size=shape).astype(np.float32)),
        "pg": one,
    }
    return ArrayData(arrays, coords, coords, coords)


@pytest.mark.parametrize("var", ["modb", "modj", "modlf"])
def test_public_subdomain_matches_crop_of_full_domain(var):
    dd = _diagnostic_data()
    slx = slice(11, 23)
    sly = slice(10, 24)
    slz = slice(9, 22)

    full = dd.get_var(var, 0, units="bifrost", br=True)
    sub = dd.get_var(
        var,
        0,
        ix_br=slx,
        iy_br=sly,
        iz_br=slz,
        units="bifrost",
        br=True,
    )

    expected = full[slx, sly, slz]
    # The halo engine should reproduce the same arithmetic on the requested
    # cells; require exact equality, not merely numerical closeness.
    np.testing.assert_array_equal(sub, expected)


def test_norm_arrays_negative_stride_view_uses_strided_kernel(monkeypatch):
    """Negative-stride views must never be routed through the flat fast path."""
    rng = np.random.default_rng(20260924)
    bases = tuple(
        np.asfortranarray(rng.normal(size=(18, 16, 14)).astype(np.float32))
        for _ in range(3)
    )
    parts = tuple(base[:, :, ::-1] for base in bases)

    assert all(not p.flags.c_contiguous and not p.flags.f_contiguous for p in parts)
    assert all(p.strides[2] < 0 for p in parts)

    monkeypatch.setattr(
        q,
        "_flat_memory_view",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("negative-stride path must not call _flat_memory_view")
        ),
    )

    got = q._norm_arrays(parts)
    expected = _expected(*parts)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)
