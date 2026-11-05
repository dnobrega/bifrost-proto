"""End-to-end regression tests for v0.1.33 changes."""

from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from bifrost_proto import BifrostData
from bifrost_proto.eos import EOSTable, read_idl_ascii
from test_api_io_v127 import _write_mesh
from test_eos_numba_v114 import make_eos


def _binary_run(tmp_path, n=96):
    """Build a minimal on-disk run with large enough axes to test wrapped halos."""
    x = np.arange(n, dtype=np.float32)
    _write_mesh(tmp_path / "mesh.mesh", x, x, x)
    (tmp_path / "example_000.idl").write_text(
        f"mx={n}\nmy={n}\nmz={n}\ndx=1\ndy=1\ndz=1\n"
        "do_mhd=.true.\ndo_hion=.false.\ndo_helium=.false.\n"
        "aux=''\nmeshfile='mesh.mesh'\nu_l=1e8\nu_t=1e2\nu_r=1e-7\n"
    )
    shape = (n, n, n)
    mm = np.memmap(tmp_path / "example_000.snap", dtype="<f4",
                   mode="w+", shape=shape + (8,), order="F")
    for i in range(8):
        mm[..., i] = i + 1
    xx = x[:, None, None]
    yy = x[None, :, None]
    zz = x[None, None, :]
    mm[..., 5] = np.sin(2 * np.pi * xx / n) + 0 * yy + 0 * zz
    mm[..., 6] = np.cos(2 * np.pi * yy / n) + 0 * xx + 0 * zz
    mm[..., 7] = (zz + 0 * xx + 0 * yy) / n
    mm.flush()
    del mm
    return BifrostData("example", fdir=tmp_path)


def test_wrapped_halo_matches_full_crops_at_periodic_edges(tmp_path):
    dd = _binary_run(tmp_path)
    full = dd.get_var("divb", 0, br=True, units="bifrost")
    for axis in ("x", "y"):
        for index in (0, 2, 94, 95):
            selection = {"i" + axis: index}
            plane = dd.get_var("divb", 0, br=True, units="bifrost", **selection)
            np.testing.assert_array_equal(plane, np.take(full, index, axis="xyz".index(axis)))
    # Nested xup/ddxdn and Z operators retain crop correctness.
    for index in (0, 48, 95):
        plane = dd.get_var("curlb_z", 0, ix=index, br=True, units="bifrost")
        np.testing.assert_array_equal(plane, np.take(
            dd.get_var("curlb_z", 0, br=True, units="bifrost"), index, axis=0
        ))


def test_overwrite_requires_explicit_flag_and_preserves_memmap_inputs(tmp_path, monkeypatch):
    dd = _binary_run(tmp_path, n=8)
    path = tmp_path / "example_000.snap"
    fields = [dd.get_var(name, 0, br=True, units="bifrost") for name in
              ("r", "px", "py", "pz", "e", "bx", "by", "bz")]
    assert not fields[0].flags.writeable
    before = fields[0].copy()
    # No prompt must be displayed in either case (notebook-safe).
    monkeypatch.setattr("builtins.input", lambda *args: pytest.fail("unexpected input()"))
    with pytest.raises(FileExistsError, match="overwrite=True"):
        dd.write_snap(path, *fields, overwrite=False)
    changed = list(fields)
    changed[4] = fields[4].copy() + np.float32(3.0)
    with pytest.warns(UserWarning, match="REPLACED"):
        dd.write_snap(path, *changed, overwrite=True)
    np.testing.assert_array_equal(dd.get_var("r", 0, br=True, units="bifrost"), before)
    np.testing.assert_array_equal(dd.get_var("e", 0, br=True, units="bifrost"), changed[4])


def test_xyz_point_returns_float32_and_public_cache_is_protected(tmp_path):
    dd = _binary_run(tmp_path, n=8)
    for br in (True, False):
        x, y, z = dd.get_grid(br=br)
        for method in ("linear", "nearest"):
            point = dd.get_var("r", 0, x=float(x[2]), y=float(y[2]),
                               z=float(z[2]), method=method, br=br)
            assert isinstance(point, np.float32)
    with dd.cache():
        b = dd.get_var("r", 0, br=True, units="bifrost")
        with pytest.raises(ValueError):
            b *= 2
        assert np.all(dd.get_var("r", 0, br=True, units="bifrost") == 1.0)


def test_eos_bounds_and_gamma_scale_equivalence():
    eos = make_eos()
    rho = np.asfortranarray(np.full((6, 6, 6), 2e-11, dtype=np.float32))
    ee = np.asfortranarray(np.full((6, 6, 6), 1e13, dtype=np.float32))
    with pytest.raises(ValueError, match="outside its table range"):
        eos.interpolate(rho * 1e9, ee)
    rho[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        eos.interpolate(rho, ee)
    rho[0, 0, 0] = 2e-11
    direct = eos.gamma1(rho, ee)
    scaled = eos.gamma1(rho / 1e3, ee / 1e4, rho_scale=1e3, ee_scale=1e4)
    np.testing.assert_allclose(direct, scaled, rtol=1e-6, atol=1e-6)


def test_idl_parser_accepts_fortran_exponents(tmp_path):
    file = tmp_path / "params.idl"
    file.write_text("u_l = 1.0D+08\nu_r = 1.0d-7\nfoo='abc'\n")
    values = read_idl_ascii(file)
    assert values["u_l"] == 1e8
    assert values["u_r"] == 1e-7
    assert values["foo"] == "abc"


def test_saved_description_is_nonsemantic(tmp_path):
    dd = _binary_run(tmp_path, n=10)
    dd.savevar("modb", 0)
    jsonfile = tmp_path / "saved/modb_0000.json"
    saved = json.loads(jsonfile.read_text())
    saved["description"] = "Documentation changed, numerical recipe did not"
    jsonfile.write_text(json.dumps(saved))
    assert dd.has_savedvar("modb", 0)
