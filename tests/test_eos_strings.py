from bifrost_proto.eos import read_idl_ascii


def test_fixed_width_idl_strings_are_trimmed(tmp_path):
    f = tmp_path / "params.idl"
    f.write_text(
        "tabinputfile = ' tabparam.in  '\n"
        "meshfile = ' mesh.mesh   '\n"
        "aux = ' tg ne  '\n"
    )
    p = read_idl_ascii(f)
    assert p["tabinputfile"] == "tabparam.in"
    assert p["meshfile"] == "mesh.mesh"
    assert p["aux"] == "tg ne"
