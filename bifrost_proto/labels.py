"""Display labels for registered Bifrost quantities.

The registry stores computational metadata (location, units, storage name and
recipe).  Plotting notation is kept separate here so a variable can retain a
stable programmatic name while exposing a compact physical symbol for figures.

``latex_varname`` always receives a *canonical* registry name and returns a
Matplotlib-mathtext-ready string including the surrounding ``$...$``.  Explicit
labels are used for quantities whose notation is not obvious; regular vector
families are handled by small deterministic rules.
"""

from __future__ import annotations

from .quantities import _ABSOLUTE_SOURCES

_COMPONENT_SYMBOL = {
    "b": r"B",
    "u": r"u",
    "p": r"p",
    "j": r"J",
    "e": r"E",
}

_Q_SYMBOL = {
    "qjoule": r"Q_J",
    "qvisc": r"Q_{\mathrm{visc}}",
    "qthin": r"Q_{\mathrm{thin}}",
    "qspitz": r"Q_{\mathrm{Spitzer}}",
    "qrad": r"Q_{\mathrm{rad}}",
    "qgenrad": r"Q_{\mathrm{genrad}}",
    "qpdv": r"Q_{p\nabla\!\cdot\!u}",
    "qlf": r"Q_L",
}

_XRT_LABELS = {
    "almed": r"\mathrm{Al\!\!-med}",
    "almesh": r"\mathrm{Al\!\!-mesh}",
    "alpoly": r"\mathrm{Al\!\!-poly}",
    "althick": r"\mathrm{Al\!\!-thick}",
    "bemed": r"\mathrm{Be\!\!-med}",
    "bethick": r"\mathrm{Be\!\!-thick}",
    "bethin": r"\mathrm{Be\!\!-thin}",
    "cpoly": r"\mathrm{C\!\!-poly}",
    "tipoly": r"\mathrm{Ti\!\!-poly}",
}

_EXPLICIT = {
    "r": r"\rho",
    "e": r"e",
    "ee": r"\epsilon",
    "tg": r"T",
    "tg_eos": r"T_{\mathrm{EOS}}",
    "ne": r"n_e",
    "ne_eos": r"n_{e,\mathrm{EOS}}",
    "nh": r"n_{\mathrm{H}}",
    "nh2": r"n_{\mathrm{H}_2}",
    "hionne": r"n_{e,\mathrm{HION}}",
    "hiontg": r"T_{\mathrm{HION}}",
    "nhe1": r"n_{\mathrm{He\,I}}",
    "nhe2": r"n_{\mathrm{He\,II}}",
    "nhe3": r"n_{\mathrm{He\,III}}",
    "pg": r"p_g",
    "pg_eos": r"p_{g,\mathrm{EOS}}",
    "pb": r"p_B",
    "pt": r"p_{\mathrm{tot}}",
    "kr": r"\kappa_R",
    "kr_eos": r"\kappa_{R,\mathrm{EOS}}",
    "beta": r"\beta",
    "gamma1": r"\Gamma_1",
    "cs": r"c_s",
    "cs_gamma": r"c_{s,\gamma}",
    "va": r"v_A",
    "mach_a": r"M_A",
    "mach_s": r"M_s",
    "mach_s_gamma": r"M_{s,\gamma}",
    "modb": r"|\mathbf{B}|",
    "b_h": r"B_h",
    "modu": r"|\mathbf{u}|",
    "u_h": r"u_h",
    "modj": r"|\mathbf{J}|",
    "mode": r"|\mathbf{E}|",
    "sp_h": r"S_h",
    "modsp": r"|\mathbf{S}|",
    "sp_ideal_h": r"S^{\mathrm{ideal}}_h",
    "modsp_ideal": r"|\mathbf{S}^{\mathrm{ideal}}|",
    "sp_ideal_emerg_z": r"S^{\mathrm{ideal}}_{z,\mathrm{emerg}}",
    "sp_ideal_shear_z": r"S^{\mathrm{ideal}}_{z,\mathrm{shear}}",
    "theta_jb": r"\theta_{JB}",
    "divb": r"\nabla\!\cdot\!\mathbf{B}",
    "divu": r"\nabla\!\cdot\!\mathbf{u}",
    "modcurlb": r"|\nabla\!\times\!\mathbf{B}|",
    "modcurlu": r"|\nabla\!\times\!\mathbf{u}|",
    "ldivb": r"|\mathbf{B}|/|\nabla\!\cdot\!\mathbf{B}|",
    "lcurlb": r"|\mathbf{B}|/|\nabla\!\times\!\mathbf{B}|",
    "ldivu": r"|\mathbf{u}|/|\nabla\!\cdot\!\mathbf{u}|",
    "lcurlu": r"|\mathbf{u}|/|\nabla\!\times\!\mathbf{u}|",
    "uparb": r"u_{\parallel B}",
    "uperpb": r"u_{\perp B}",
    "eparb": r"E_{\parallel B}",
    "eperpb": r"E_{\perp B}",
    "gradpg_h": r"|\nabla_h p_g|",
    "modgradpg": r"|\nabla p_g|",
    "gradpgor_h": r"|\nabla_h p_g|/\rho",
    "modgradpgor": r"|\nabla p_g|/\rho",
    "lf_h": r"|\mathbf{f}_{L,h}|",
    "modlf": r"|\mathbf{f}_L|",
    "lfor_h": r"|\mathbf{f}_{L,h}|/\rho",
    "modlfor": r"|\mathbf{f}_L|/\rho",
    "eb": r"e_B",
    "ek": r"e_K",
    "hpg_hydro": r"H_{p,\mathrm{hydro}}",
    "hpg_grad": r"H_{p,\nabla}",
    "tau500": r"\tau_{500}",
    "keparb": r"\mathcal{K}",
    "taurec3d": r"\tau_{\mathrm{rec,3D}}",
    "taudif3d": r"\tau_{\mathrm{dif,3D}}",
}


def _component_label(name: str) -> str | None:
    """Return a LaTeX body for regular vector-component families."""
    if len(name) >= 3 and name[-2] == "_" and name[-1] in "xyz":
        axis = name[-1]
        stem = name[:-2]
        if stem in _COMPONENT_SYMBOL:
            return rf"{_COMPONENT_SYMBOL[stem]}_{axis}"
        if stem == "bhat":
            return rf"\hat{{B}}_{axis}"
        if stem == "curlb":
            return rf"(\nabla\!\times\!\mathbf{{B}})_{axis}"
        if stem == "curlu":
            return rf"(\nabla\!\times\!\mathbf{{u}})_{axis}"
        if stem == "e_ideal":
            return rf"E^{{\mathrm{{ideal}}}}_{axis}"
        if stem == "sp":
            return rf"S_{axis}"
        if stem == "sp_ideal":
            return rf"S^{{\mathrm{{ideal}}}}_{axis}"
        if stem == "uparb":
            return rf"u_{{\parallel B,{axis}}}"
        if stem == "uperpb":
            return rf"u_{{\perp B,{axis}}}"
        if stem == "eparb":
            return rf"E_{{\parallel B,{axis}}}"
        if stem == "eperpb":
            return rf"E_{{\perp B,{axis}}}"
        if stem == "gradpg":
            return rf"-(\nabla p_g)_{axis}"
        if stem == "gradpgor":
            return rf"-(\nabla p_g)_{{{axis}}}/\rho"
        if stem == "lf":
            return rf"f_{{L,{axis}}}"
        if stem == "lfor":
            return rf"f_{{L,{axis}}}/\rho"
    return None


def _instrument_label(name: str) -> str | None:
    """Return a compact emissivity label for AIA, XRT and EUI variables."""
    if name.startswith("aia") and name[3:].isdigit():
        return rf"\epsilon_{{\mathrm{{AIA\,{name[3:]}\,\AA}}}}"
    if name == "eui174":
        return r"\epsilon_{\mathrm{EUI\,174\,\AA}}"
    if name.startswith("xrt_"):
        filt = _XRT_LABELS.get(name[4:])
        if filt is not None:
            return rf"\epsilon_{{\mathrm{{XRT}},\,{filt}}}"
    return None


def _heating_label(name: str) -> str | None:
    """Return labels for volumetric/specific heating rates and timescales."""
    if name in _Q_SYMBOL:
        return _Q_SYMBOL[name]
    if name.endswith("or") and name[:-2] in _Q_SYMBOL:
        return rf"{_Q_SYMBOL[name[:-2]]}/\rho"
    if name.startswith("tau"):
        qname = name[3:]
        if qname in _Q_SYMBOL:
            return rf"\tau_{{{_Q_SYMBOL[qname]}}}"
    return None


def latex_varname(name: str) -> str:
    """Return a Matplotlib-mathtext label for a canonical variable name.

    Unknown names deliberately fall back to a roman rendering of the canonical
    identifier instead of guessing a physical meaning.
    """
    source = _ABSOLUTE_SOURCES.get(name)
    if source is not None:
        source_label = latex_varname(source)[1:-1]
        return rf"$\left|{source_label}\right|$"
    if name in _EXPLICIT:
        body = _EXPLICIT[name]
    elif name in {f"n{i}" for i in range(1, 7)}:
        body = rf"n_{name[1:]}"
    else:
        escaped_name = name.replace("_", r"\_")
        body = (
            _component_label(name)
            or _heating_label(name)
            or _instrument_label(name)
            or rf"\mathrm{{{escaped_name}}}"
        )
    return f"${body}$"
