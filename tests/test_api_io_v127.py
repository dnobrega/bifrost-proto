"""Regression tests for the v0.1.27 public API and native writers."""

from __future__ import annotations

import os

import numpy as np
import pytest

from bifrost_proto import BifrostData
from test_engine import make_data


def _write_mesh(path, x, y, z):
    """Write a minimal Bifrost mesh file for an I/O regression test."""
    with open(path, "w", encoding="utf-8") as fp:
        for q in (x, y, z):
            d = q[1] - q[0]
            qdn = q - 0.5 * d
            inv = np.full(q.size, 1.0 / d)
            fp.write(f"{q.size}\n")
            fp.write(" ".join(map(str, q)) + "\n")
            fp.write(" ".join(map(str, qdn)) + "\n")
            fp.write(" ".join(map(str, inv)) + "\n")
            fp.write(" ".join(map(str, inv)) + "\n")


def test_snapshot_zero_prefers_numbered_000_files(tmp_path):
    """Snapshot zero must use _000 filenames when that run convention is present."""
    n = 6
    root = "numbered"
    x = np.arange(n, dtype=np.float32)
    _write_mesh(tmp_path / "mesh.mesh", x, x, x)
    (tmp_path / f"{root}_000.idl").write_text(
        "mx=6\nmy=6\nmz=6\n"
        "dx=1.0\ndy=1.0\ndz=1.0\n"
        "do_mhd=.true.\ndo_hion=.false.\ndo_helium=.false.\n"
        "aux=''\nmeshfile='mesh.mesh'\n"
        "u_l=1e8\nu_t=1e2\nu_r=1e-7\n",
        encoding="utf-8",
    )
    shape = (n, n, n)
    fields = [np.full(shape, i + 1, dtype=np.float32, order="F") for i in range(8)]
    mm = np.memmap(
        tmp_path / f"{root}_000.snap",
        dtype="<f4",
        mode="w+",
        order="F",
        shape=shape + (8,),
    )
    for i, field in enumerate(fields):
        mm[..., i] = field
    mm.flush()
    del mm

    dd = BifrostData(root, fdir=tmp_path)
    assert dd._param_filename(0).endswith(f"{root}_000.idl")
    assert dd._data_filename(0, "snap").endswith(f"{root}_000.snap")
    np.testing.assert_array_equal(
        dd.get_var("r", 0, units="bifrost", br=True), fields[0]
    )


def test_snapshot_zero_keeps_legacy_unnumbered_fallback(tmp_path):
    """Legacy root.idl/root.snap snapshot-zero runs remain readable."""
    n = 5
    root = "legacy"
    x = np.arange(n, dtype=np.float32)
    _write_mesh(tmp_path / "mesh.mesh", x, x, x)
    (tmp_path / f"{root}.idl").write_text(
        "mx=5\nmy=5\nmz=5\n"
        "dx=1.0\ndy=1.0\ndz=1.0\n"
        "do_mhd=.true.\ndo_hion=.false.\ndo_helium=.false.\n"
        "aux=''\nmeshfile='mesh.mesh'\n",
        encoding="utf-8",
    )
    shape = (n, n, n)
    mm = np.memmap(
        tmp_path / f"{root}.snap",
        dtype="<f4",
        mode="w+",
        order="F",
        shape=shape + (8,),
    )
    for i in range(8):
        mm[..., i] = i + 1
    mm.flush()
    del mm

    dd = BifrostData(root, fdir=tmp_path)
    assert dd._param_filename(0).endswith(f"{root}.idl")
    assert dd._data_filename(0, "snap").endswith(f"{root}.snap")
    assert np.all(dd.get_var("r", 0, units="bifrost", br=True) == 1.0)


def test_canonical_get_api_face_coordinates_and_units():
    """Canonical get_* helpers expose center/face coordinates and length units."""
    dd = make_data(n=12)

    np.testing.assert_array_equal(dd.get_grid(location="center")[0], dd.get_x())
    np.testing.assert_array_equal(dd.get_grid(location="xface")[0], dd.get_xf())
    np.testing.assert_array_equal(dd.get_grid(location="yface")[1], dd.get_yf())
    np.testing.assert_array_equal(dd.get_grid(location="zface")[2], dd.get_zf())

    np.testing.assert_allclose(dd.get_x(units="cgs"), dd.get_x() * 1.0e8)
    np.testing.assert_allclose(dd.get_x(units="si"), dd.get_x() * 1.0e6)
    np.testing.assert_allclose(dd.get_x(units="bifrost"), dd.get_x())
    np.testing.assert_allclose(dd.get_dx(units="cgs"), dd.get_dx() * 1.0e8)

    xedge = dd.get_grid(location="xedge")
    np.testing.assert_array_equal(xedge[0], dd.get_x())
    np.testing.assert_array_equal(xedge[1], dd.get_yf())
    np.testing.assert_array_equal(xedge[2], dd.get_zf())


def test_center_is_explicit_alias_for_centered():
    """center=True must be equivalent to centered=True and reject conflicts."""
    dd = make_data(n=16)
    short = dd.get_var("bz", 0, center=True, units="bifrost")
    long = dd.get_var("bz", 0, centered=True, units="bifrost")
    np.testing.assert_array_equal(short, long)

    with pytest.raises(ValueError, match="conflicting"):
        dd.get_var("bz", 0, center=True, centered=False, units="bifrost")


def test_native_write_snap_and_aux_roundtrip(tmp_path):
    """Native writers preserve block order and round-trip through the reader."""
    n = 7
    root = "roundtrip"
    shape = (n, n, n)
    x = np.arange(n, dtype=np.float32)
    _write_mesh(tmp_path / "mesh.mesh", x, x, x)
    (tmp_path / f"{root}_000.idl").write_text(
        "mx=7\nmy=7\nmz=7\n"
        "dx=1.0\ndy=1.0\ndz=1.0\n"
        "do_mhd=.true.\ndo_hion=.false.\ndo_helium=.false.\n"
        "aux='p tg foo'\nmeshfile='mesh.mesh'\n"
        "u_l=1e8\nu_t=1e2\nu_r=1e-7\n",
        encoding="utf-8",
    )
    dd = BifrostData(root, fdir=tmp_path)

    fields = [
        np.asfortranarray(
            (i + 1)
            + np.arange(np.prod(shape), dtype=np.float32).reshape(shape, order="F")
        )
        for i in range(8)
    ]
    p = np.full(shape, 21.0, dtype=np.float32, order="F")
    tg = np.full(shape, 22.0, dtype=np.float32, order="F")
    foo = np.full(shape, 23.0, dtype=np.float32, order="F")

    snap_path = dd.write_snap(tmp_path / f"{root}_000.snap", *fields, overwrite=False)
    aux_path = dd.write_aux(tmp_path / f"{root}_000.aux", p, tg, extra=[("foo", foo)])
    assert os.path.isfile(snap_path)
    assert os.path.isfile(aux_path)

    for name, expected in zip(("r", "px", "py", "pz", "e", "bx", "by", "bz"), fields):
        got = dd.get_var(name, 0, units="bifrost", br=True)
        np.testing.assert_array_equal(got, expected)

    np.testing.assert_array_equal(dd.get_var("p", 0, units="bifrost", br=True), p)
    np.testing.assert_array_equal(dd.get_var("tg", 0, units="bifrost", br=True), tg)
    np.testing.assert_array_equal(dd.get_var("foo", 0, units="bifrost", br=True), foo)

    with pytest.raises(FileExistsError):
        dd.write_snap(tmp_path / f"{root}_000.snap", *fields)
    with pytest.raises(ValueError, match="Duplicate AUX"):
        dd.write_aux(tmp_path / "bad.aux", p, tg, extra=[("p", foo)])


def test_public_get_api_has_no_compact_aliases():
    """Only the underscored get_* API should appear on BifrostData."""
    dd = make_data(n=8)
    dd._meta_cache[0].params["tsnap"] = 2.0
    assert dd.get_var_name("bx", latex=True) == r"$B_x$"
    assert dd.get_t(0) == 200.0
    for name in (
        "getvar",
        "getx",
        "gety",
        "getz",
        "gett",
        "getdx",
        "getdy",
        "getdz",
        "getdx1d",
        "getdy1d",
        "getdz1d",
        "getgrid",
        "getvarname",
    ):
        assert not hasattr(dd, name)
