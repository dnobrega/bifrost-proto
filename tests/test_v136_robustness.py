"""Regression tests for v0.1.36 memory, periodic halos and fail-closed I/O."""

from __future__ import annotations

import numpy as np
import pytest

from bifrost_proto import BifrostData
from bifrost_proto.selection import Selection
from test_v133_safety_and_wrapped import _binary_run


def test_periodic_expand_preserves_unwrapped_child_for_wide_window():
    shape = (72, 72, 16)
    child = Selection(slice(-5, 65), slice(0, 72), slice(0, 16))
    parent = child.expand(0, 5, shape, periodic=True)
    assert parent.ix.stop - parent.ix.start == 72
    assert parent.ix.start <= child.ix.start
    assert parent.ix.stop >= child.ix.stop
    rel = child.relative_to(parent, shape)
    assert rel[0].stop - rel[0].start == 70
    assert rel[0].start >= 0
    assert parent.expand(0, 5, shape, periodic=True).ix == parent.ix


@pytest.mark.parametrize("n", [72, 96])
@pytest.mark.parametrize("variable", ["modcurlb", "curlb_z", "qlf"])
@pytest.mark.parametrize(
    "axis,low,high", [("x", 0, 60), ("x", 12, 70), ("y", 0, 60), ("y", 12, 70)]
)
def test_wide_periodic_selection_is_bitwise_full_crop(
    tmp_path, n, variable, axis, low, high
):
    dd = _binary_run(tmp_path, n=n)
    full = dd.get_var(variable, 0, br=True, units="bifrost")
    selection = {"i" + axis: slice(low, high)}
    actual = dd.get_var(variable, 0, br=True, units="bifrost", **selection)
    expected = full[low:high, :, :] if axis == "x" else full[:, low:high, :]
    np.testing.assert_array_equal(actual, expected)


def test_weak_cache_matches_strong_and_default_overrides(tmp_path):
    dd = _binary_run(tmp_path, n=24)
    assert dd.low_memory is False
    baseline = dd.get_var("uperpb", 0, ix=slice(3, 20), units="bifrost")
    weak = dd.get_var("uperpb", 0, ix=slice(3, 20), units="bifrost", low_memory=True)
    np.testing.assert_array_equal(weak, baseline)

    dd_weak = BifrostData("example", fdir=tmp_path, low_memory=True)
    assert dd_weak.low_memory is True
    inherited = dd_weak.get_var("uperpb", 0, ix=slice(3, 20), units="bifrost")
    strong_override = dd_weak.get_var(
        "uperpb", 0, ix=slice(3, 20), units="bifrost", low_memory=False
    )
    np.testing.assert_array_equal(inherited, strong_override)
    np.testing.assert_array_equal(inherited, baseline)

    with dd_weak.cache():
        assert type(dd_weak._shared_context.cache) is dict
        # Strong cache always stays strong despite the per-call flag.
        dd_weak.get_var("modcurlb", 0, low_memory=True, units="bifrost")
        assert len(dd_weak._shared_context.cache) > 0
        dd_weak.get_var("modcurlb", 0, low_memory=False, units="bifrost")


def test_weak_context_only_retains_live_numpy_objects():
    from weakref import WeakValueDictionary

    cache = WeakValueDictionary()
    arr = np.ones(4, dtype=np.float32)
    cache["x"] = arr
    assert cache.get("x") is arr
    del arr
    assert cache.get("x") is None


def test_explicit_missing_meshfile_is_error_not_uniform_fallback(tmp_path):
    dd = _binary_run(tmp_path, n=8)
    path = tmp_path / "mesh.mesh"
    path.unlink()
    # Metadata must not have been read yet.
    with pytest.raises(FileNotFoundError, match="mesh.mesh"):
        dd.get_var("r", 0)


def test_missing_mesh_key_retains_legacy_uniform_fallback(tmp_path):
    dd = _binary_run(tmp_path, n=8)
    path = tmp_path / "example_000.idl"
    path.write_text(path.read_text().replace("meshfile='mesh.mesh'\n", ""))
    (tmp_path / "mesh.mesh").unlink()
    assert dd.get_var("r", 0).shape == (8, 8, 8)


def test_write_aux_never_prompts_and_requires_overwrite(tmp_path, monkeypatch):
    dd = _binary_run(tmp_path, n=8)
    destination = tmp_path / "custom.aux"
    p = np.ones((8, 8, 8), dtype=np.float32)
    tg = p * 2
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("unexpected input"))
    dd.write_aux(destination, p, tg)
    with pytest.raises(FileExistsError):
        dd.write_aux(destination, p, tg)
    with pytest.warns(UserWarning, match="REPLACED"):
        dd.write_aux(destination, p * 3, tg, overwrite=True)
    np.testing.assert_array_equal(
        np.memmap(destination, dtype="<f4", shape=(8, 8, 8), order="F"), p * 3
    )


@pytest.mark.parametrize("periodic_axis", [0, 1])
def test_512_periodic_axis_wide_window_matches_full(tmp_path, periodic_axis):
    """Exercise the original 512-grid bug without allocating a full 512^3 run."""
    from test_api_io_v127 import _write_mesh

    shape = (512, 16, 16) if periodic_axis == 0 else (16, 512, 16)
    coords = tuple(np.arange(n, dtype=np.float32) for n in shape)
    _write_mesh(tmp_path / "mesh.mesh", *coords)
    nx, ny, nz = shape
    (tmp_path / "example_000.idl").write_text(
        f"mx={nx}\nmy={ny}\nmz={nz}\n"
        "dx=1\ndy=1\ndz=1\n"
        "do_mhd=.true.\ndo_hion=.false.\ndo_helium=.false.\n"
        "aux=''\nmeshfile='mesh.mesh'\nu_l=1e8\nu_t=1e2\nu_r=1e-7\n"
    )
    a = np.memmap(
        tmp_path / "example_000.snap",
        dtype="<f4",
        mode="w+",
        shape=shape + (8,),
        order="F",
    )
    for field in range(8):
        a[..., field] = np.float32(field + 1)
    if periodic_axis == 0:
        a[..., 5] += np.sin(2 * np.pi * coords[0][:, None, None] / nx)
    else:
        a[..., 6] += np.cos(2 * np.pi * coords[1][None, :, None] / ny)
    a.flush()
    del a
    dd = BifrostData("example", fdir=tmp_path)
    full = dd.get_var("modcurlb", 0, br=True, units="bifrost")
    kwargs = {"ix": slice(0, 500)} if periodic_axis == 0 else {"iy": slice(0, 500)}
    sub = dd.get_var("modcurlb", 0, br=True, units="bifrost", **kwargs)
    expected = full[:500, :, :] if periodic_axis == 0 else full[:, :500, :]
    np.testing.assert_array_equal(sub, expected)


def test_empty_declared_meshfile_is_invalid(tmp_path):
    dd = _binary_run(tmp_path, n=8)
    path = tmp_path / "example_000.idl"
    path.write_text(path.read_text().replace("meshfile='mesh.mesh'", "meshfile=''"))
    with pytest.raises(ValueError, match="empty meshfile"):
        dd.get_var("r", 0)
