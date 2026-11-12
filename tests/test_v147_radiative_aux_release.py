"""Radiative-source AUX requirements, derived diagnostics, plot defaults, license."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from matplotlib.colors import LogNorm, SymLogNorm

from bifrost_proto import BifrostData, UnknownVariableError
from bifrost_proto.labels import latex_varname
from bifrost_proto.plot_defaults import PLOT_PRESETS, get_plot_preset
from bifrost_proto.registry import GLOBAL_REGISTRY
from bifrost_proto.units import BifrostUnits
from test_io import write_mesh


NEW_RATES = ("qrad", "qgenrad")


@pytest.fixture
def sample_aux(tmp_path):
    n = 8
    name = "radiation"
    x = np.arange(n, dtype=np.float32)
    write_mesh(tmp_path / "mesh.mesh", x, x, x)
    (tmp_path / f"{name}.idl").write_text(
        "mx=8\nmy=8\nmz=8\n"
        "dx=1.0\ndy=1.0\ndz=1.0\n"
        "do_mhd=.true.\ndo_hion=.false.\ndo_helium=.false.\n"
        "aux='qrad qgenrad'\nmeshfile='mesh.mesh'\n"
        "u_l=1e8\nu_t=1e2\nu_r=1e-7\nu_b=1121.0\ntsnap=0\n"
    )
    # r=2, e=12, 6 remaining native snap fields.
    snap = np.memmap(
        tmp_path / f"{name}.snap",
        dtype="<f4",
        mode="w+",
        order="F",
        shape=(n, n, n, 8),
    )
    snap[...] = 0
    snap[..., 0] = 2
    snap[..., 4] = 12
    snap.flush()
    del snap
    aux = np.memmap(
        tmp_path / f"{name}.aux",
        dtype="<f4",
        mode="w+",
        order="F",
        shape=(n, n, n, 2),
    )
    aux[..., 0] = -4
    aux[..., 1] = 6
    aux[1, 2, 3, 1] = 0
    aux.flush()
    del aux
    return BifrostData(name, fdir=tmp_path), tmp_path


def test_native_aux_signs_derived_rates_and_timescales(sample_aux):
    dd, _ = sample_aux
    for name, rate, rate_over_r, tau in (
        ("qrad", -4, -2, 3),
        ("qgenrad", 6, 3, 2),
    ):
        assert dd.info(name, 0)["source"] == "aux"
        assert dd.info(name, 0)["available"]
        assert dd.info(name, 0)["location"] == "center"
        assert dd.get_unit_type(name) == "energy_rate_density"
        assert dd.get_units(name) == "erg cm^-3 s^-1"
        assert dd.get_units(name + "or") == "erg g^-1 s^-1"
        assert dd.get_units("tau" + name) == "s"
        q = dd.get_var(name, 0, units="bifrost", br=True)
        qor = dd.get_var(name + "or", 0, units="bifrost", br=True)
        t = dd.get_var("tau" + name, 0, units="bifrost", br=True)
        assert q[2, 2, 2] == pytest.approx(rate)
        assert qor[2, 2, 2] == pytest.approx(rate_over_r)
        assert t[2, 2, 2] == pytest.approx(tau)
    assert np.isposinf(dd.get_var("tauqgenrad", 0, units="bifrost", br=True)[1, 2, 3])
    assert dd.get_var("qrad", 0, units="solar", br=True)[2, 2, 2] == pytest.approx(
        -4 * BifrostUnits(u_l=1e8, u_t=1e2, u_r=1e-7).u_q
    )


def test_source_must_exist_in_aux_even_if_saved_cache_exists(sample_aux):
    dd, tmp_path = sample_aux
    # Save generated data, then remove the original AUX. Saved data may not
    # stand in for radiation source fields; the recorded source is mandatory.
    dd.savevar("qrad", 0)
    dd.savevar("qgenrad", 0)
    Path(tmp_path / "radiation.aux").unlink()
    for name in NEW_RATES:
        assert dd.info(name, 0)["source"] == "unavailable"
        assert not dd.info(name, 0)["available"]
        with pytest.raises(FileNotFoundError, match="AUX file"):
            dd.get_var(name, 0, units="bifrost", br=True)
        with pytest.raises(FileNotFoundError, match="AUX file"):
            dd.get_var("tau" + name, 0, units="bifrost", br=True)


def test_source_not_declared_raises_even_with_other_aux_fields(sample_aux):
    dd, tmp_path = sample_aux
    # Changing IDL requires a fresh reader to re-read its AUX list.
    p = tmp_path / "radiation.idl"
    p.write_text(p.read_text().replace("aux='qrad qgenrad'", "aux='qrad'"))
    dd2 = BifrostData("radiation", fdir=tmp_path)
    assert not dd2.info("qgenrad", 0)["available"]
    with pytest.raises(UnknownVariableError, match="not listed"):
        dd2.get_var("qgenrad", 0, units="bifrost", br=True)


def test_registry_presets_latex_and_los():
    assert set(PLOT_PRESETS) == set(GLOBAL_REGISTRY)
    # The source and specific-rate presets exactly mirror qspitz and qspitzor;
    # the internal-energy timescale mirrors tauqspitz.
    for prefix in NEW_RATES:
        for suffix, analog in (("", "qspitz"), ("or", "qspitzor")):
            assert PLOT_PRESETS[prefix + suffix] == PLOT_PRESETS[analog]
        assert PLOT_PRESETS["tau" + prefix] == PLOT_PRESETS["tauqspitz"]
        assert GLOBAL_REGISTRY[prefix].getter is None
        for name, unit_type in (
            (prefix, "energy_rate_density"),
            (prefix + "or", "specific_energy_rate"),
            ("tau" + prefix, "time"),
        ):
            assert GLOBAL_REGISTRY[name].unit_type == unit_type
            assert latex_varname(name).startswith("$")
    from test_v144_plot_presets_colormaps import DummyData

    dd = DummyData()
    cmap, norm = get_plot_preset(dd, "qrad")
    assert cmap == "RdBu_r" and isinstance(norm, SymLogNorm)
    assert (norm.vmin, norm.vmax, norm.linthresh) == (-1, 1, 1e-3)
    _, specific = get_plot_preset(dd, "qgenrador")
    assert (specific.vmin, specific.vmax, specific.linthresh) == (-1e13, 1e13, 2e12)
    _, tau = get_plot_preset(dd, "tauqrad")
    assert isinstance(tau, LogNorm) and (tau.vmin, tau.vmax) == (0.1, 1000)
    _, los = get_plot_preset(dd, "qgenrad", los="z")
    assert los.vcenter == 0


def test_public_release_metadata():
    repo = Path(__file__).resolve().parents[1]
    for filename in ("LICENSE", "CITATION.cff", "THIRD_PARTY_NOTICES.md"):
        assert (repo / filename).is_file()
    assert "BSD 3-Clause" in (repo / "LICENSE").read_text()
    from bifrost_proto import __version__

    assert f'version: "{__version__}"' in (repo / "CITATION.cff").read_text()


def test_plot_new_radiative_source_uses_default_divergent_preset(sample_aux):
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    dd, _ = sample_aux
    fig, ax, im = dd.plot("qrad", 0, z=-3)
    try:
        assert im.cmap.name == "RdBu_r"
        assert isinstance(im.norm, SymLogNorm)
        assert (im.norm.vmin, im.norm.vmax, im.norm.linthresh) == (-1, 1, 0.001)
        assert "rad" in dd.get_var_name("qrad", latex=True)
    finally:
        plt.close(fig)
