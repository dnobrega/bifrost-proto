"""Regression tests for automatic IDL snapname run-root selection."""

from __future__ import annotations

import builtins
import numpy as np
import pytest

from bifrost_proto import BifrostData
from bifrost_proto._idl_detection import detect_snapname


def _idl(directory, name, snapname=None):
    content = (
        "! comment not a parameter\n"
        "; snapname = 'commented_out'\n"
        "mx = 12\n"
        + (f"SNAPNAME = '{snapname}   ' ; trailing comment\n" if snapname is not None else "")
        + "my = 12\n"
    )
    (directory / name).write_text(content, encoding="utf-8")


def test_unique_snapname_does_not_prompt(tmp_path, monkeypatch):
    for i in range(5):
        _idl(tmp_path, f"sim_{i:03d}.idl", "sim")
    monkeypatch.setattr(builtins, "input", lambda _: pytest.fail("unexpected prompt"))
    dd = BifrostData(folder=tmp_path)
    assert dd.root_name == "sim"
    assert dd.fdir == str(tmp_path)
    assert dd._param_filename(4) == str(tmp_path / "sim_004.idl")


def test_detect_from_fdir_when_root_omitted(tmp_path):
    _idl(tmp_path, "sim_000.idl", "sim")
    assert BifrostData(fdir=tmp_path).root_name == "sim"


def test_case_insensitive_key_and_unquoted_name(tmp_path):
    (tmp_path / "sim.idl").write_text("SnApNaMe = sim # comment\n", encoding="utf-8")
    assert detect_snapname(tmp_path) == "sim"


def test_multiple_snapnames_are_counted_and_prompted(tmp_path, monkeypatch, capsys):
    for i in range(4):
        _idl(tmp_path, f"a_{i:03d}.idl", "alpha")
    for i in range(2):
        _idl(tmp_path, f"b_{i:03d}.idl", "beta")
    choices = iter(["bad", "2"])
    monkeypatch.setattr(builtins, "input", lambda _: next(choices))
    dd = BifrostData(folder=tmp_path)
    assert dd.root_name == "beta"
    output = capsys.readouterr().out
    assert "[1] alpha (4 .idl files)" in output
    assert "[2] beta (2 .idl files)" in output
    assert "Invalid selection" in output


def test_choose_by_exact_name(tmp_path, monkeypatch):
    _idl(tmp_path, "first.idl", "first")
    _idl(tmp_path, "second.idl", "second")
    monkeypatch.setattr(builtins, "input", lambda _: "second")
    assert detect_snapname(tmp_path) == "second"


def test_empty_files_are_ignored(tmp_path, monkeypatch):
    _idl(tmp_path, "ignore.idl")
    _idl(tmp_path, "sim_000.idl", "sim")
    monkeypatch.setattr(builtins, "input", lambda _: pytest.fail("unexpected prompt"))
    assert detect_snapname(tmp_path) == "sim"


def test_no_snapname_reports_error(tmp_path):
    _idl(tmp_path, "sim_000.idl")
    with pytest.raises(ValueError, match="No snapname found"):
        BifrostData(folder=tmp_path)


def test_nonexistent_folder_reports_error(tmp_path):
    with pytest.raises(NotADirectoryError, match="does not exist"):
        BifrostData(folder=tmp_path / "missing")


def test_no_recursion(tmp_path, monkeypatch):
    _idl(tmp_path, "sim.idl", "sim")
    nested = tmp_path / "nested"
    nested.mkdir()
    _idl(nested, "other.idl", "other")
    monkeypatch.setattr(builtins, "input", lambda _: pytest.fail("unexpected prompt"))
    assert detect_snapname(tmp_path) == "sim"


@pytest.mark.parametrize("failure", [EOFError, OSError])
def test_noninteractive_selection_fails_helpfully(tmp_path, monkeypatch, failure):
    _idl(tmp_path, "a.idl", "alpha")
    _idl(tmp_path, "b.idl", "beta")

    def fail(_):
        raise failure("no input")

    monkeypatch.setattr(builtins, "input", fail)
    with pytest.raises(ValueError, match="Specify file_root or idlparam explicitly"):
        BifrostData(folder=tmp_path)


def test_explicit_file_root_skips_scan_and_prompt(tmp_path, monkeypatch):
    _idl(tmp_path, "a.idl", "alpha")
    _idl(tmp_path, "b.idl", "beta")
    monkeypatch.setattr(builtins, "input", lambda _: pytest.fail("unexpected prompt"))
    assert BifrostData("manual", fdir=tmp_path).root_name == "manual"
    assert BifrostData(folder=tmp_path, idlparam="manual").root_name == "manual"
    assert BifrostData(file_root="manual", fdir=tmp_path).root_name == "manual"


def test_conflicting_roots_or_folders_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="one run root"):
        BifrostData("a", idlparam="b", folder=tmp_path)
    with pytest.raises(ValueError, match="same directory"):
        BifrostData("a", fdir=tmp_path, folder=tmp_path / "elsewhere")


def test_original_constructor_unchanged(tmp_path):
    dd = BifrostData("legacy", fdir=tmp_path, low_memory=True)
    assert dd.root_name == "legacy"
    assert dd.low_memory is True
    assert dd.file_root == str(tmp_path / "legacy")


def test_discovered_root_reads_real_snapshot_data(tmp_path):
    """Integration: found snapname is genuinely usable by the native reader."""
    n = 6
    mesh = tmp_path / "mesh.mesh"
    coord = np.arange(n, dtype=np.float32)
    with mesh.open("w", encoding="utf-8") as stream:
        for _ in range(3):
            stream.write(f"{n}\n")
            for values in (coord, coord - 0.5, np.ones(n), np.ones(n)):
                stream.write(" ".join(map(str, values)) + "\n")

    (tmp_path / "demo_000.idl").write_text(
        "snapname='demo'\n"
        "mx=6\nmy=6\nmz=6\n"
        "dx=1.0\ndy=1.0\ndz=1.0\n"
        "do_mhd=.true.\ndo_hion=.false.\ndo_helium=.false.\n"
        "aux=''\nmeshfile='mesh.mesh'\n",
        encoding="utf-8",
    )
    data = np.memmap(
        tmp_path / "demo_000.snap", dtype="<f4", mode="w+",
        shape=(n, n, n, 8), order="F",
    )
    data[..., 0] = 3.25  # r
    data.flush()
    del data

    dd = BifrostData(folder=tmp_path)
    actual = dd.get_var("r", 0, br=True, units="bifrost")
    assert actual.shape == (6, 6, 6)
    np.testing.assert_array_equal(actual, 3.25)
