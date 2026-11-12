import json

import numpy as np
import pytest

from bifrost_proto import BifrostData


def write_mesh(path, x, y, z):
    with open(path, "w") as f:
        for q in (x, y, z):
            d = q[1] - q[0]
            qdn = q - 0.5 * d
            inv = np.full(q.size, 1.0 / d)
            f.write(f"{q.size}\n")
            f.write(" ".join(map(str, q)) + "\n")
            f.write(" ".join(map(str, qdn)) + "\n")
            f.write(" ".join(map(str, inv)) + "\n")
            f.write(" ".join(map(str, inv)) + "\n")


def make_run(tmp_path, n=12):
    root = "testrun_saved"
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    write_mesh(tmp_path / "mesh.mesh", x, y, z)

    (tmp_path / f"{root}.idl").write_text(
        f"mx={n}\nmy={n}\nmz={n}\n"
        "dx=1.0\ndy=1.0\ndz=1.0\n"
        "do_mhd=.true.\n"
        "do_hion=.false.\ndo_helium=.false.\n"
        "aux=''\n"
        "meshfile='mesh.mesh'\n"
        "u_l=1e8\nu_t=1e2\nu_r=1e-7\n"
    )

    shape = (n, n, n)
    r = np.ones(shape, dtype=np.float32, order="F")
    fields = [
        r,
        r,
        r,
        r,
        r * 3.0,
        np.full(shape, 3.0, dtype=np.float32, order="F"),
        np.full(shape, 4.0, dtype=np.float32, order="F"),
        np.zeros(shape, dtype=np.float32, order="F"),
    ]
    mm = np.memmap(
        tmp_path / f"{root}.snap",
        dtype="<f4",
        mode="w+",
        order="F",
        shape=shape + (8,),
    )
    for i, field in enumerate(fields):
        mm[..., i] = field
    mm.flush()
    del mm
    return root, shape


def rewrite_b(tmp_path, root, shape, bx, by, bz):
    mm = np.memmap(
        tmp_path / f"{root}.snap",
        dtype="<f4",
        mode="r+",
        order="F",
        shape=shape + (8,),
    )
    mm[..., 5] = bx
    mm[..., 6] = by
    mm[..., 7] = bz
    mm.flush()
    del mm


def test_saved_npy_debug_path_and_memmap(tmp_path):
    root, shape = make_run(tmp_path)
    dd = BifrostData(root, fdir=tmp_path)

    original = dd.get_var("modb", 0, units="bifrost", br=True, use_saved=False)
    path = dd.savevar("modb", 0)

    assert path.endswith("saved/modb_0000.npy")
    assert (tmp_path / "saved" / "modb_0000.npy").is_file()
    assert (tmp_path / "saved" / "modb_0000.json").is_file()
    assert dd.has_savedvar("modb", 0)

    mm = np.load(path, mmap_mode="r", allow_pickle=False)
    assert isinstance(mm, np.memmap)
    assert mm.flags.f_contiguous
    assert mm.dtype == np.float32
    assert np.array_equal(mm, original)

    # A changed source snapshot invalidates the saved cache; the default
    # resolver must recompute rather than silently returning stale data.
    rewrite_b(tmp_path, root, shape, 6.0, 8.0, 0.0)
    with pytest.warns(UserWarning, match="Source files changed"):
        from_saved = dd.get_var("modb", 0, units="bifrost", br=True)
    recalculated = dd.get_var("modb", 0, units="bifrost", br=True, use_saved=False)

    assert not np.array_equal(from_saved, original)
    np.testing.assert_array_equal(from_saved, recalculated)
    interior = (slice(5, -5), slice(5, -5), slice(5, -5))
    assert np.allclose(recalculated[interior], 10.0, atol=1e-5)

    # A partial recomputation must match the updated source, not stale NPY.
    # This is a second request for the same stale saved array, so another
    # invalidation warning is expected. Verify it instead of leaking it into
    # pytest's warnings summary or suppressing real user-facing warnings.
    with pytest.warns(UserWarning, match="Source files changed"):
        sub = dd.get_var(
            "modb",
            0,
            ix=slice(2, 9),
            iy=slice(1, 8),
            iz=slice(3, 10),
            units="bifrost",
            br=True,
        )
    np.testing.assert_array_equal(sub, from_saved[2:9, 1:8, 3:10])


def test_saved_json_metadata_and_mismatch_detection(tmp_path):
    root, _ = make_run(tmp_path)
    dd = BifrostData(root, fdir=tmp_path)
    dd.savevar("modb", 0)

    json_path = tmp_path / "saved" / "modb_0000.json"
    metadata = json.loads(json_path.read_text())
    assert metadata["variable"] == "modb"
    assert metadata["units"] == "bifrost"
    assert metadata["orientation"] == "bifrost"
    assert metadata["location"] == "center"
    assert metadata["unit_type"] == "magnetic_field"
    assert metadata["dtype"] == "float32"
    assert metadata["order"] == "F"

    metadata["location"] = "zface"
    json_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata mismatch"):
        dd.has_savedvar("modb", 0)


def test_savevar_requires_explicit_overwrite(tmp_path):
    root, _ = make_run(tmp_path)
    dd = BifrostData(root, fdir=tmp_path)
    dd.savevar("modb", 0)
    with pytest.raises(FileExistsError):
        dd.savevar("modb", 0)
    dd.savevar("modb", 0, overwrite=True)


def test_units_and_info_api(tmp_path):
    root, _ = make_run(tmp_path)
    dd = BifrostData(root, fdir=tmp_path)

    assert dd.get_unit_type("eb") == "energy_density"
    assert dd.get_units("eb") == "erg cm^-3"
    assert dd.get_units("eb", units="si") == "J m^-3"
    assert dd.get_units("qjoule", latex=True) == "erg cm$^{-3}$ s$^{-1}$"
    assert dd.get_units("beta") == ""
    assert dd.get_units("bx") == dd.get_units("b_x") == "G"

    raw_info = dd.info("b_x", 0)
    assert raw_info["source"] == "snap"
    assert raw_info["units"] == "G"

    derived_info = dd.info("modb", 0)
    assert derived_info["source"] == "computed"

    dd.savevar("modb", 0)
    saved_info = dd.info("modb", 0)
    assert saved_info["source"] == "npy"
