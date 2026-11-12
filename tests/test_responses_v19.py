import json
from pathlib import Path

import numpy as np

from bifrost_proto.core import AMU_CGS
from bifrost_proto.responses import INSTRUMENT_RESPONSE_NAMES, InstrumentResponseTable
from test_engine import make_data


def write_tabparam(path):
    path.write_text(
        "EOSTableFile = 'eostable.dat'\n"
        "cel = 'H  HE C  O  FE '\n"
        "abund = '12.00 10.93 8.43 8.69 7.50'\n"
        "aweight = '1.01 4.00 12.01 16.00 55.85'\n"
    )


def attach_composition(dd, tmp_path):
    write_tabparam(tmp_path / "tabparam.in")
    dd.fdir = str(tmp_path)
    dd._meta_cache[0].params["tabinputfile"] = "tabparam.in"
    dd._composition_cache = {}


def write_response(root, name, table, logne, logt, **metadata):
    folder = Path(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    np.save(
        folder / "table.npy", np.asarray(table, dtype=np.float64), allow_pickle=False
    )
    np.save(
        folder / "logne.npy", np.asarray(logne, dtype=np.float64), allow_pickle=False
    )
    np.save(folder / "logt.npy", np.asarray(logt, dtype=np.float64), allow_pickle=False)
    payload = {
        "format_version": 1,
        "name": name,
        "fill_value": 0.0,
        "abundance": "test abundance",
        **metadata,
    }
    (folder / "metadata.json").write_text(json.dumps(payload))


def test_composition_public_properties_and_nh(tmp_path):
    dd = make_data()
    # Put the synthetic test cube at non-negative physical heights so the
    # instrument validity mask does not alter tests unrelated to masking.
    dd._meta_cache[0].z[:] = -np.arange(dd._meta_cache[0].shape[2], dtype=np.float32)
    attach_composition(dd, tmp_path)

    assert dd.elements == ("H", "HE", "C", "O", "FE")
    assert np.allclose(dd.abundances, [12.00, 10.93, 8.43, 8.69, 7.50])
    assert np.allclose(dd.atomic_weights, [1.01, 4.00, 12.01, 16.00, 55.85])
    assert np.isclose(np.sum(dd.mass_fractions), 1.0)

    abund = np.asarray([12.00, 10.93, 8.43, 8.69, 7.50])
    weights = np.asarray([1.01, 4.00, 12.01, 16.00, 55.85])
    grph = np.sum((10.0 ** (abund - 12.0)) * weights) * AMU_CGS
    assert np.isclose(dd.grams_per_hydrogen, grph)

    nh = dd.get_var("nh", 0, units="bifrost", br=True)
    expected = (2.0 * dd._meta(0).units.u_r) / grph
    assert np.allclose(nh, expected, rtol=2e-6)


def test_response_bilinear_interpolation_and_fill(tmp_path):
    logne = np.array([8.0, 9.0, 10.0])
    logt = np.array([5.0, 6.0, 7.0, 8.0])
    # Bilinear function is reproduced exactly by bilinear interpolation.
    table = 2.0 * logne[:, None] + 3.0 * logt[None, :] + 4.0
    write_response(tmp_path, "aia193", table, logne, logt, fill_value=-1.0)

    response = InstrumentResponseTable("aia193", base_dir=tmp_path)
    ne = np.array([10.0**8.5, 10.0**9.25, 10.0**12])
    tg = np.array([10.0**5.5, 10.0**7.25, 10.0**6])
    got = response.evaluate(tg, ne)
    expected = np.array(
        [
            2.0 * 8.5 + 3.0 * 5.5 + 4.0,
            2.0 * 9.25 + 3.0 * 7.25 + 4.0,
            -1.0,
        ]
    )
    assert np.allclose(got, expected)


def test_aia_recipe_uses_ne_nh_g_and_metadata(tmp_path):
    dd = make_data()
    # Put the synthetic test cube at non-negative physical heights so the
    # instrument validity mask does not alter tests unrelated to masking.
    dd._meta_cache[0].z[:] = -np.arange(dd._meta_cache[0].shape[2], dtype=np.float32)
    attach_composition(dd, tmp_path)
    shape = dd.arrays["r"].shape
    dd.arrays["ne"] = np.full(shape, 1.0e9, dtype=np.float32)
    dd.arrays["tg"] = np.full(shape, 1.0e6, dtype=np.float32)

    response_root = tmp_path / "responses"
    write_response(
        response_root,
        "aia193",
        np.full((2, 2), 2.5e-27),
        [8.0, 10.0],
        [5.0, 7.0],
        instrument="SDO/AIA",
        channel="193",
    )
    dd.responses_dir = str(response_root)
    dd._response_cache = {}

    nh = dd.get_var("nh", 0, units="bifrost", br=True)
    out = dd.get_var("aia193", 0, units="solar", br=True)
    assert np.allclose(out, 1.0e9 * nh * 2.5e-27, rtol=2e-6)
    assert dd.get_units("aia193") == "DN cm^-1 s^-1 pix^-1"
    assert dd.get_units("aia193", units="si") == "DN m^-1 s^-1 pix^-1"

    info = dd.info("aia193")
    assert info["response_table_available"] is True
    assert info["response_metadata"]["channel"] == "193"
    assert dd.response_info("aia193")["instrument"] == "SDO/AIA"


def test_all_instrument_response_names_are_registered_and_documented():
    dd = make_data()
    for name in INSTRUMENT_RESPONSE_NAMES:
        info = dd.info(name)
        assert info["location"] == "center"
        assert info["unit_type"] == "instrument_emissivity"
        assert info["description"]


def test_aia_xrt_eui_share_temperature_and_height_emissivity_mask(tmp_path):
    dd = make_data(n=6)
    attach_composition(dd, tmp_path)
    shape = dd.arrays["r"].shape

    # Native Bifrost z is ascending. In the package's physical convention,
    # physical height in native array order is -z_native.
    dd._meta_cache[0].z[:] = np.asarray(
        [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32
    )

    dd.arrays["ne"] = np.full(shape, 1.0e9, dtype=np.float32, order="F")
    tg = np.full(shape, 1.0e6, dtype=np.float32, order="F")
    tg[:, :, 1] = 9999.0  # z_phys=+2 Mm: removed only by T mask
    tg[:, :, 2] = 1.0e4  # exact threshold remains valid
    dd.arrays["tg"] = tg

    response_root = tmp_path / "responses"
    for name in ("aia193", "xrt_alpoly", "eui174"):
        write_response(
            response_root,
            name,
            np.full((2, 2), 2.5e-27),
            [8.0, 10.0],
            [3.0, 7.0],
        )
    dd.responses_dir = str(response_root)
    dd._response_cache = {}

    # Native-order physical heights are [+3,+2,+1,0,-1,-2] Mm.
    # Expected validity by k: [yes, no(T), yes(T=1e4), yes(z=0), no(z), no(z)].
    for name in ("aia193", "xrt_alpoly", "eui174"):
        out = dd.get_var(name, 0, units="solar", br=True)
        profile = np.mean(out, axis=(0, 1))
        assert profile[0] > 0.0
        assert profile[1] == 0.0
        assert profile[2] > 0.0
        assert profile[3] > 0.0
        assert profile[4] == 0.0
        assert profile[5] == 0.0

    # A native subdomain crossing both cuts must match cropping the full result.
    full = dd.get_var("aia193", 0, units="solar", br=True)
    sub = dd.get_var("aia193", 0, iz_br=slice(1, 5), units="solar", br=True)
    np.testing.assert_array_equal(sub, full[:, :, 1:5])

    # The public physical orientation reverses the same physical mask; it does
    # not change which plasma is excluded.
    physical = dd.get_var("aia193", 0, units="solar", br=False)
    np.testing.assert_allclose(
        np.mean(physical, axis=(0, 1)),
        np.mean(full, axis=(0, 1))[::-1],
        rtol=3e-7,
        atol=0.0,
    )
