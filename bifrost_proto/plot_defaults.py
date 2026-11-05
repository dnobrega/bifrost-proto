"""User-approved plot defaults, separate from physical variable metadata.

The 169 catalog entries are solar-unit presets. The current reader registry
resolves aliases, and optional physical-unit conversion changes the norm limits
without changing the underlying variable data. Passing a user ``cmap`` or
``norm`` always overrides that field. Unknown/custom variables keep Matplotlib
behavior. Instrument LOS integrations have separate integrated-data limits.
"""

from __future__ import annotations

from matplotlib.colors import CenteredNorm, LogNorm, Normalize, SymLogNorm, TwoSlopeNorm

from ._plot_preset_data import PLOT_PRESETS

# Preset palettes with a meaningful neutral midpoint. Only use an automatic
# zero-centered range when the *quantity is signed*, not for a positive quantity
# that happens to use a diverging palette (such as beta).
_DIVERGENT_CMAPS = frozenset({
    "RdBu", "BrBG", "PiYG", "PRGn", "PuOr", "RdGy", "RdYlBu",
    "RdYlGn", "Spectral", "bwr", "seismic", "coolwarm",
    "divergent_terrain",
})

# LOS instrument values are integrated DN s^-1 pix^-1, not local emissivity.
INSTRUMENT_LOS = {"aia": (1.0, 2e4), "eui": (1.0, 2e4), "xrt": (0.1, 100.0)}


def _limits(name, values):
    """Construct the Matplotlib Normalize instance from one preset tuple."""
    norm, low, high, threshold, center, _cmap = values
    if norm == "Normalize":
        return Normalize(vmin=low, vmax=high)
    if norm == "LogNorm":
        return LogNorm(vmin=low, vmax=high)
    if norm == "SymLogNorm":
        return SymLogNorm(linthresh=threshold, vmin=low, vmax=high)
    if norm == "TwoSlopeNorm":
        return TwoSlopeNorm(vmin=low, vcenter=center, vmax=high)
    raise ValueError(f"Unsupported normalization {norm!r} for {name!r}")


def _factor(data, var, units, *, los):
    """Ratio of plotted values in the target system to the solar system."""
    units = units.lower()
    if units == "solar":
        return 1.0
    unit_type = data.get_unit_type(var)
    factors = data._grid_meta().units
    ratio = factors.factor(unit_type, units) / factors.factor(unit_type, "solar")
    if los:
        # plot() uses transverse coordinates in Mm and physical LOS weights
        # with cm (solar/cgs), m (SI), or code length (bifrost).
        length = {"cgs": 1.0, "si": 1e-2, "bifrost": 1e-8}[units]
        ratio *= length
    return ratio


def get_plot_preset(data, var, *, units="solar", los=None):
    """Return (colormap-name, Normalize-or-None) for a registered variable.

    No snapshot values are read. For non-instrumental LOS, keep automatic
    scaling. Signed quantities with diverging preset colormaps use CenteredNorm
    so the neutral midpoint always represents zero.
    """
    registry = getattr(data, "registry", None)
    canonical = registry.resolve(var) if registry is not None else var
    entry = PLOT_PRESETS.get(canonical)
    if entry is None:
        return None, None
    cmap = entry[5]
    if los is not None:
        if canonical.startswith(("aia", "eui")):
            family = "aia" if canonical.startswith("aia") else "eui"
            low, high = INSTRUMENT_LOS[family]
            entry = ("LogNorm", low, high, None, None, cmap)
        elif canonical.startswith("xrt_"):
            low, high = INSTRUMENT_LOS["xrt"]
            entry = ("LogNorm", low, high, None, None, cmap)
        else:
            base = cmap.removesuffix("_r")
            if entry[1] < 0 < entry[2] and base in _DIVERGENT_CMAPS:
                # imshow autosizes symmetrically to the integrated data.
                # movie() reuses that same norm, freezing the scale after frame 0.
                return cmap, CenteredNorm(vcenter=0)
            return cmap, None
    if units.lower() != "solar":
        ratio = _factor(data, canonical, units, los=los)
        entry = (entry[0], entry[1] * ratio, entry[2] * ratio,
                 entry[3] * ratio if entry[3] is not None else None,
                 entry[4] * ratio if entry[4] is not None else None,
                 entry[5])
    return cmap, _limits(canonical, entry)
