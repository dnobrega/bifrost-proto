import numpy as np

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


def test_real_memmap_layout_and_unknown_aux(tmp_path):
    n = 8
    root = "testrun"
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    write_mesh(tmp_path / "mesh.mesh", x, y, z)

    (tmp_path / f"{root}.idl").write_text(
        "mx=8\nmy=8\nmz=8\n"
        "dx=1.0\ndy=1.0\ndz=1.0\n"
        "do_mhd=.true.\n"
        "do_hion=.false.\ndo_helium=.false.\n"
        "aux='foo'\n"
        "meshfile='mesh.mesh'\n"
        "u_l=1e8\nu_t=1e2\nu_r=1e-7\nu_b=1121.0\n"
    )

    shape = (n, n, n)
    fields = [np.full(shape, i + 1, dtype=np.float32, order="F") for i in range(8)]
    mm = np.memmap(
        tmp_path / f"{root}.snap", dtype="<f4", mode="w+", order="F", shape=shape + (8,)
    )
    for i, field in enumerate(fields):
        mm[..., i] = field
    mm.flush()

    aux = np.memmap(
        tmp_path / f"{root}.aux", dtype="<f4", mode="w+", order="F", shape=shape + (1,)
    )
    aux[..., 0] = 42.0
    aux.flush()

    dd = BifrostData(root, fdir=tmp_path)
    r = dd.get_var("r", 0, units="bifrost")
    bz = dd.get_var("bz", 0, units="bifrost")
    foo = dd.get_var("foo", 0, units="bifrost")

    assert np.all(r == 1.0)
    assert np.all(bz == -8.0)  # physical z-component changes sign
    assert np.all(foo == 42.0)

    # Unknown AUX metadata is deliberately required before solar conversion.
    try:
        dd.get_var("foo", 0)
    except ValueError as err:
        assert "units are not registered" in str(err)
    else:
        raise AssertionError("unknown AUX should require units='bifrost'")

    dd.register_variable("foo", location="center", unit_type="temperature")
    assert np.all(dd.get_var("foo", 0) == 42.0)


def test_getvar_br_returns_native_bifrost_bytes(tmp_path):
    n = 8
    root = "testrun_br"
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    write_mesh(tmp_path / "mesh.mesh", x, y, z)

    (tmp_path / f"{root}.idl").write_text(
        "mx=8\nmy=8\nmz=8\n"
        "dx=1.0\ndy=1.0\ndz=1.0\n"
        "do_mhd=.true.\n"
        "do_hion=.false.\ndo_helium=.false.\n"
        "aux=''\n"
        "meshfile='mesh.mesh'\n"
        "u_l=1e8\nu_t=1e2\nu_r=1e-7\nu_b=1121.0\n"
    )

    shape = (n, n, n)
    # Use non-trivial values so orientation/sign changes are obvious.
    ii, jj, kk = np.meshgrid(
        np.arange(n, dtype=np.float32),
        np.arange(n, dtype=np.float32),
        np.arange(n, dtype=np.float32),
        indexing="ij",
    )
    fields = [
        1.0 + ii + 10.0 * jj + 100.0 * kk,
        2.0 + ii + 10.0 * jj + 100.0 * kk,
        3.0 + ii + 10.0 * jj + 100.0 * kk,
        4.0 + ii + 10.0 * jj + 100.0 * kk,
        5.0 + ii + 10.0 * jj + 100.0 * kk,
        6.0 + ii + 10.0 * jj + 100.0 * kk,
        7.0 + ii + 10.0 * jj + 100.0 * kk,
        8.0 + ii + 10.0 * jj + 100.0 * kk,
    ]
    fields = [np.asfortranarray(f.astype(np.float32)) for f in fields]

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

    dd = BifrostData(root, fdir=tmp_path)

    for var, expected in (("bx", fields[5]), ("by", fields[6]), ("bz", fields[7])):
        got = dd.get_var(var, 0, units="bifrost", br=True)
        assert np.array_equal(got, expected)
        # Stronger regression check for the user's IDL-style raw comparison.
        assert got.tobytes(order="F") == expected.tobytes(order="F")

    # Normal output is still reoriented physically.
    by_phys = dd.get_var("by", 0, units="bifrost")
    assert np.array_equal(by_phys, -fields[6][:, ::-1, ::-1])


def test_getvar_br_and_public_stagger_share_native_frame():
    from bifrost_proto.core import EvaluationContext
    from bifrost_proto.selection import Selection

    # Reuse the synthetic in-memory reader from test_engine without importing it.
    from test_engine import make_data

    dd = make_data()
    bx_br = dd.get_var("bx", 0, units="bifrost", br=True)
    manual = dd.stagger.xup(bx_br, br=True)

    full = Selection(slice(0, 24), slice(0, 24), slice(0, 24))
    context = EvaluationContext(cache={})
    automatic_native = dd._ops(("xup",), "bx", 0, full, context)

    assert np.allclose(manual, automatic_native, atol=2e-5)
