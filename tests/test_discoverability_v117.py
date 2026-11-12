from test_diagnostics_v18 import constant_data


def test_info_is_concise_and_alias_resolution_is_user_facing():
    dd = constant_data(efield=(2.0, 3.0, 4.0))
    info = dd.info("ex")

    assert info["name"] == "e_x"
    assert "ex" in info["aliases"]
    assert info["location"] == "xedge"
    assert info["unit_type"] == "electric_field"
    assert info["description"]

    # Resolver implementation details are intentionally not part of info().
    assert "requested" not in info
    assert "storage_name" not in info
    assert "derived" not in info
    assert "component" not in info


def test_variables_returns_sorted_canonical_names():
    dd = constant_data()
    names = dd.variables()

    assert names == sorted(names)
    assert "e_x" in names
    assert "ex" not in names
    assert "pg" in names
    assert "p" not in names


def test_variables_searches_names_aliases_descriptions_and_unit_type():
    dd = constant_data()

    assert dd.variables("ex") == ["e_x"]
    assert "pg" in dd.variables("gas pressure")
    assert "qjoule" in dd.variables("energy_rate_density")
    assert dd.variables("") == dd.variables()
    assert dd.variables("THIS_WILL_NOT_MATCH_ANY_VARIABLE") == []


def test_info_pg_reports_aux_source_for_bifrost_p(tmp_path):
    import numpy as np
    from bifrost_proto import BifrostData
    from test_io import write_mesh

    n = 6
    root = "testrun_info_pg"
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    write_mesh(tmp_path / "mesh.mesh", x, y, z)

    (tmp_path / f"{root}.idl").write_text(
        f"mx={n}\nmy={n}\nmz={n}\n"
        "dx=1.0\ndy=1.0\ndz=1.0\n"
        "do_mhd=.true.\n"
        "do_hion=.false.\ndo_helium=.false.\n"
        "aux='p'\n"
        "meshfile='mesh.mesh'\n"
        "u_l=1e8\nu_t=1e2\nu_r=1e-7\n"
    )

    shape = (n, n, n)
    snap = np.memmap(
        tmp_path / f"{root}.snap",
        dtype="<f4",
        mode="w+",
        order="F",
        shape=shape + (8,),
    )
    snap[...] = 1.0
    snap.flush()
    del snap

    aux = np.memmap(
        tmp_path / f"{root}.aux",
        dtype="<f4",
        mode="w+",
        order="F",
        shape=shape + (1,),
    )
    aux[..., 0] = 7.0
    aux.flush()
    del aux

    dd = BifrostData(root, fdir=tmp_path)
    for name in ("pg", "p"):
        info = dd.info(name, 0)
        assert info["name"] == "pg"
        assert info["source"] == "aux"
        assert info["available"] is True
        assert "requested" not in info
        assert "storage_name" not in info
        assert "derived" not in info
