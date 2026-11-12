"""Quick-look 2-D scalar plotting on physical Bifrost coordinates.

A single scalar cut coordinate selects the plane. Optional physical slices on
its two displayed axes restrict the data read instead of merely zooming the
image. All plotting uses the existing ``get_var``/``get_grid`` mesh logic.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LogNorm, Normalize
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .geometry import LOCATION_SHIFTED_AXES
from .plot_naming import _coordinate_label, _roi_suffix

AxisName = Literal["x", "y", "z"]
_LOS_SLAB = 64  # maximum transverse planes per streamed LOS slab
_LOS_TARGET_MIB = 64  # approximate native input bytes per slab, not peak RAM
ColorbarPosition = Literal["top", "bottom", "left", "right"]


def plot(
    data: Any,
    var: str,
    snap: int,
    *,
    x: float | slice | None = None,
    y: float | slice | None = None,
    z: float | slice | None = None,
    ax: Any = None,
    cmap: str | Colormap | None = None,
    norm: Normalize | None = None,
    aspect: str | float = "equal",
    cbar: bool = True,
    cbar_position: ColorbarPosition = "top",
    figsize: tuple[float, float] | None = None,
    title: bool = True,
    minor_ticks: bool = True,
    units: str = "solar",
    contours: dict[str, Any] | None = None,
    low_memory: bool | None = None,
    los: AxisName | None = None,
    los_stream: bool = False,
    png: bool | str | Path = False,
    dpi: int = 300,
    overwrite: bool = False,
):
    """Plot a physical 2-D cut or a line-of-sight integral.

    Without ``los``, one physical coordinate must be scalar and the other
    axes may be restricted by slices. With ``los`` set to x/y/z, no coordinate
    may be scalar: the LOS axis (and displayed axes) can be cropped by slices.
    Center-grid samples are integrated with physical cell widths; face samples
    use periodic full-cell quadrature for complete periodic X/Y axes, and
    trapezoidal integration otherwise.
    The integration always uses the original Z mesh, before 2-D remeshing.

    ``los_stream=True`` integrates small 3-D slabs spanning the entire LOS,
    with an adaptive transverse slab size, instead of loading the full cube.
    On centered images, contour overlays default to cell-centered values,
    while ``contours={"var": ..., "centered": False}`` uses native staggering.

    ``png=True`` writes a 300-DPI PNG in ``<data.fdir>/plots`` with a descriptive name;
    an explicit PNG path may also be supplied. The file is not overwritten
    unless ``overwrite=True``. ``dpi`` affects PNG output only.
    """
    selection = {"x": x, "y": y, "z": z}
    if los is None and los_stream:
        raise ValueError("los_stream requires los='x', 'y' or 'z'")
    if los is None:
        cut_axis, cut_coord, selection = _parse_cut(**selection)
    else:
        if los not in ("x", "y", "z"):
            raise ValueError("los must be 'x', 'y', or 'z'")
        _validate_los_selection(selection)
        cut_axis, cut_coord = los, None
        if contours is not None:
            raise ValueError("contours are not supported with los integration")
    destination = None
    if png is not False:
        destination = _png_path(
            data,
            var,
            snap,
            png=png,
            axis=cut_axis,
            coordinate=cut_coord,
            selections=selection,
            los=los,
        )
        if (
            not isinstance(dpi, (int, np.integer))
            or isinstance(dpi, (bool, np.bool_))
            or dpi < 1
        ):
            raise ValueError("dpi must be a positive integer")
        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"PNG already exists: {destination} (set overwrite=True)"
            )

    if contours is not None and (
        not isinstance(contours, dict)
        or not isinstance(contours.get("var"), str)
        or not contours["var"]
    ):
        raise TypeError("contours must be a dict containing a nonempty 'var' name")
    uniform_z = cut_axis != "z"
    if los is None:
        values = np.asarray(
            data.get_var(
                var,
                snap,
                **selection,
                units=units,
                uniform_z=uniform_z,
                low_memory=low_memory,
            )
        )
        plane = np.squeeze(values)
        if plane.ndim != 2:
            raise ValueError(
                f"Expected a 2-D plane for {cut_axis}={cut_coord!r}, "
                f"but get_var returned shape {values.shape}."
            )
        coords = _get_plot_coordinates(
            data,
            _get_location(data, var),
            selection=selection,
            uniform_z=uniform_z,
        )
    else:
        plane, coords = _integrated_plane(
            data,
            var,
            snap,
            los=los,
            selection=selection,
            units=units,
            low_memory=low_memory,
            stream=los_stream,
        )
    horizontal, vertical = _plane_axes(cut_axis)
    _validate_plane_shape(plane, coords[horizontal], coords[vertical], cut_axis)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    # Presets are applied independently; an explicit cmap or norm wins.
    if cmap is None or norm is None:
        from .plot_defaults import get_plot_preset

        default_cmap, default_norm = get_plot_preset(data, var, units=units, los=los)
        if cmap is None:
            cmap = default_cmap
        if norm is None:
            norm = default_norm
            if (
                isinstance(norm, LogNorm)
                and norm.vmin is None
                and not np.any(np.isfinite(plane) & (plane > 0))
            ):
                # Give all-masked logarithmic LOS images a valid colorbar.
                norm = LogNorm(vmin=0.1, vmax=1.0)
    from .colormaps import register_colormaps

    register_colormaps()

    im = ax.imshow(
        plane.T,
        origin="lower",
        extent=_extent(coords[horizontal], coords[vertical]),
        cmap=cmap,
        norm=norm,
        aspect=aspect,
    )
    _format_axes(ax, horizontal=horizontal, vertical=vertical, minor_ticks=minor_ticks)
    if title:
        _set_titles(
            ax,
            cut_axis=cut_axis,
            cut_coord=float(coords[cut_axis]) if los is None else None,
            time_min=_snapshot_time_min(data, snap),
        )
    if contours is not None:
        ax._bifrost_contours = _plot_contours(
            ax,
            data,
            contours,
            snap=snap,
            background_var=var,
            selection=selection,
            cut_axis=cut_axis,
            uniform_z=uniform_z,
            units=units,
            low_memory=low_memory,
        )
    if cbar:
        _add_colorbar(
            fig,
            ax,
            im,
            label=_colorbar_label(data, var, units, los=los),
            position=cbar_position,
        )
    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    return fig, ax, im


def _validate_physical_selector(axis: str, selector: Any) -> bool:
    """Validate a physical coordinate and return whether it selects a plane."""
    if selector is None:
        return False
    if isinstance(selector, slice):
        if selector.step is not None:
            raise ValueError(f"{axis} slice.step is not supported for physical plots")
        for bound in (selector.start, selector.stop):
            if bound is not None and (
                not isinstance(bound, (int, float, np.integer, np.floating))
                or isinstance(bound, (bool, np.bool_))
                or not np.isfinite(bound)
            ):
                raise TypeError(f"{axis} slice bounds must be finite numbers in Mm")
        if (
            selector.start is not None
            and selector.stop is not None
            and selector.start > selector.stop
        ):
            raise ValueError(f"{axis} physical slice start must not exceed stop")
        return False
    if isinstance(selector, (int, float, np.integer, np.floating)) and not isinstance(
        selector, (bool, np.bool_)
    ):
        if not np.isfinite(selector):
            raise ValueError(f"{axis} cut coordinate must be finite")
        return True
    raise TypeError(f"{axis} must be a scalar physical coordinate or slice in Mm")


def _validate_los_selection(selection: dict[str, Any]) -> None:
    """A LOS integration requires physical slices or full axes, not planes."""
    for axis, selector in selection.items():
        if _validate_physical_selector(axis, selector):
            raise ValueError(f"With los=..., {axis} must be a physical slice or None")


def _los_geometry_weights(
    data: Any,
    var: str,
    *,
    los: AxisName,
    selection: dict[str, Any],
    units: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Shared physical LOS geometry/quadrature for volume and plane-wise reads."""
    location = _get_location(data, var)
    coords = _get_plot_coordinates(data, location, selection=selection, uniform_z=False)
    axis_index = "xyz".index(los)
    positions = np.asarray(coords[los], dtype=np.float64)
    if positions.ndim != 1 or positions.size < 2:
        raise ValueError(f"LOS axis {los} must contain at least two points")
    if not np.all(np.diff(positions) > 0):
        raise ValueError(f"LOS axis {los} must be strictly increasing")

    spacing_function = getattr(data, f"get_d{los}1d")
    shifted = los in LOCATION_SHIFTED_AXES[location]
    periodic = los in ("x", "y") and data._grid_meta().periodic[axis_index]
    full_period = False
    if shifted and periodic:
        full_faces = np.asarray(
            data.get_grid(location=location, units="solar")[axis_index],
            dtype=np.float64,
        )
        full_period = np.array_equal(positions, full_faces)

    if shifted and not full_period:
        differences = np.diff(positions)
        weights = np.empty_like(positions)
        weights[0] = 0.5 * differences[0]
        weights[-1] = 0.5 * differences[-1]
        weights[1:-1] = 0.5 * (differences[:-1] + differences[1:])
    else:
        spacing_selector = None if full_period else selection[los]
        weights = np.asarray(
            spacing_function(
                **{los: spacing_selector},
                units="solar",
            ),
            dtype=np.float64,
        )
        if weights.shape != positions.shape or not np.all(weights > 0):
            raise ValueError("Invalid cell widths for LOS integration")

    length_factor = {"solar": 1e8, "cgs": 1e8, "si": 1e6, "bifrost": 1.0}[units.lower()]
    weights *= length_factor
    if units.lower() == "solar":
        unit_plain = data.get_units(var, units=units, latex=False)
        _, metadata_factor = _integrated_unit_plain(str(unit_plain), "cm", solar=True)
        weights *= metadata_factor
    return coords, weights


def _integrated_plane(
    data: Any,
    var: str,
    snap: int,
    *,
    los: AxisName,
    selection: dict[str, Any],
    units: str,
    low_memory: bool | None,
    stream: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Integrate a native-Z LOS, optionally using memory-bounded 3-D slabs.

    Each streamed slab spans the entire LOS, but only a short interval along
    a transverse axis. The chosen size caps *native input array* bytes, not
    the total peak memory of derived quantities and their temporary arrays.
    Both paths share native-grid quadrature and post-integration Z remeshing.
    """
    coords, weights = _los_geometry_weights(
        data,
        var,
        los=los,
        selection=selection,
        units=units,
    )
    axis_index = "xyz".index(los)
    h, v = _plane_axes(los)
    expected = (coords[h].size, coords[v].size)
    if stream:
        # Accumulate over slabs of a non-LOS axis that is contiguous in the
        # Fortran-ordered files (y for LOS x/z, z for LOS y). Each slab
        # integrates the complete LOS, avoiding the repeated random-access
        # reads and most repeated stagger halos of one-plane-at-a-time mode.
        cut = "z" if los == "y" else "y"
        cut_index = "xyz".index(cut)
        positions = np.asarray(coords[cut])
        full = np.asarray(
            data.get_grid(location=_get_location(data, var), units="solar")[cut_index]
        )
        offset = int(np.searchsorted(full, positions[0]))
        if not np.array_equal(full[offset : offset + positions.size], positions):
            raise ValueError(f"Cannot map the selected {cut} samples to native indices")
        out_spec = {0: "ijk,i->jk", 1: "ijk,j->ik", 2: "ijk,k->ij"}[axis_index]
        out_axis = (h, v).index(cut)
        others = {key: val for key, val in selection.items() if key != cut}
        plane = np.empty(expected, dtype=np.float64)
        dtype = np.dtype(getattr(data, "dtype", "f4"))
        other_axis = next(axis for axis in "xyz" if axis not in (los, cut))
        bytes_per_plane = (
            int(coords[los].size) * int(coords[other_axis].size) * dtype.itemsize
        )
        target_bytes = int(_LOS_TARGET_MIB * 1024 * 1024)
        slab_size = min(int(_LOS_SLAB), max(1, target_bytes // max(1, bytes_per_plane)))
        for start in range(0, positions.size, slab_size):
            stop = min(start + slab_size, positions.size)
            slab = np.asarray(
                data.get_var(
                    var,
                    snap,
                    **others,
                    **{f"i{cut}": slice(offset + start, offset + stop)},
                    units=units,
                    uniform_z=False,
                    low_memory=low_memory,
                )
            )
            part = np.einsum(out_spec, slab, weights, dtype=np.float64, optimize=False)
            if out_axis == 0:
                plane[start:stop, :] = part
            else:
                plane[:, start:stop] = part
    else:
        values = np.asarray(
            data.get_var(
                var,
                snap,
                **selection,
                units=units,
                uniform_z=False,
                low_memory=low_memory,
            )
        )
        if values.ndim != 3 or values.shape != tuple(coords[a].size for a in "xyz"):
            raise ValueError(
                f"Expected a selected 3-D cube for LOS integration, got {values.shape}"
            )
        out_spec = {0: "ijk,i->jk", 1: "ijk,j->ik", 2: "ijk,k->ij"}[axis_index]
        plane = np.einsum(out_spec, values, weights, dtype=np.float64, optimize=False)
    if los != "z":
        plane, z_uniform = data._uniformize_z(plane, (h, v), coords["z"])
        coords["z"] = np.asarray(z_uniform, dtype=float)
    return plane, coords


def _integrated_unit_plain(
    original: str, length_unit: str, *, solar: bool = False
) -> tuple[str, float]:
    """Multiply a plain unit label by one length and simplify exact units.

    Solar metadata may contain Mm powers while the integration length is cm.
    Convert those existing powers to cm, returning the numeric conversion as
    well, so e.g. (G Mm^-1) integrated in cm returns G (not G cm/Mm).
    """
    tokens = str(original).split()
    correction = 1.0
    transformed = []
    for token in tokens:
        match = re.fullmatch(r"(Mm|cm|m)(?:\^(-?\d+))?", token)
        if solar and match and match.group(1) == "Mm":
            power = int(match.group(2) or 1)
            correction *= 1e8**power
            token = f"cm^{power}" if power != 1 else "cm"
        transformed.append(token)

    length_power = 1
    first_length = None
    for i, token in enumerate(transformed):
        match = re.fullmatch(rf"{re.escape(length_unit)}(?:\^(-?\d+))?", token)
        if match:
            length_power += int(match.group(1) or 1)
            if first_length is None:
                first_length = i
    replacement = length_unit if length_power == 1 else f"{length_unit}^{length_power}"
    result = []
    for i, token in enumerate(transformed):
        if re.fullmatch(rf"{re.escape(length_unit)}(?:\^(-?\d+))?", token):
            if i == first_length and length_power:
                result.append(replacement)
        else:
            result.append(token)
    if first_length is None:
        result.append(length_unit)
    return " ".join(result), correction


def _latex_unit(plain: str) -> str:
    """Convert a plain registered unit label into Matplotlib mathtext."""
    return re.sub(r"\^(-?\d+)", lambda m: f"$^{{{m.group(1)}}}$", plain)


def _png_path(
    data: Any,
    var: str,
    snap: int,
    *,
    png: bool | str | Path,
    axis: AxisName,
    coordinate: float | None,
    selections: dict[str, Any],
    los: AxisName | None,
) -> Path:
    """Resolve PNG output under the simulation directory, like movies."""
    run_dir = Path(data.fdir).expanduser().resolve()
    directory = run_dir / "plots"
    if png is True:
        plane = {"x": "YZ", "y": "XZ", "z": "XY"}[axis]
        mode = (
            f"{axis}={_coordinate_label(coordinate)}" if los is None else f"los={los}"
        )
        suffix = _roi_suffix(axis if los is None else "", selections)
        path = directory / f"{var}_{int(snap):04d}_{plane}_{mode}{suffix}.png"
    elif isinstance(png, (str, Path)):
        candidate = Path(png).expanduser()
        if candidate.name in {"", ".", ".."}:
            raise ValueError("png must specify a filename")
        path = (
            candidate
            if candidate.is_absolute()
            else (
                directory / candidate
                if candidate.parent == Path(".")
                else run_dir / candidate
            )
        )
    else:
        raise TypeError("png must be False, True, or a .png filename/path")
    if path.suffix.lower() != ".png":
        raise ValueError("png must specify a .png filename")
    return path.resolve()


def _parse_cut(
    *,
    x: float | slice | None,
    y: float | slice | None,
    z: float | slice | None,
) -> tuple[AxisName, float, dict[str, float | slice | None]]:
    """Require exactly one scalar cut; the other axes may be cropped by slices."""
    selections = {"x": x, "y": y, "z": z}
    fixed = []
    for axis, selector in selections.items():
        if _validate_physical_selector(axis, selector):
            fixed.append((axis, float(selector)))
    if len(fixed) != 1:
        raise ValueError(
            "Pass exactly one scalar cut coordinate: x=..., y=..., or z=... ."
        )
    return fixed[0][0], fixed[0][1], selections


def _plot_contours(
    ax: Any,
    data: Any,
    spec: dict[str, Any],
    *,
    snap: int,
    background_var: str,
    selection: dict[str, float | slice | None],
    cut_axis: AxisName,
    uniform_z: bool,
    units: str,
    low_memory: bool | None,
) -> Any:
    """Overlay a registered variable on the same physical cut and ROI."""
    if (
        not isinstance(spec, dict)
        or not isinstance(spec.get("var"), str)
        or not spec["var"]
    ):
        raise TypeError("contours must be a dict containing a nonempty 'var' name")
    contour_var = spec["var"]
    # Overlay contours on a centered image by default: otherwise two fields
    # appear half a stagger interval apart when the background is cell-centered.
    # A face/edge background keeps native contour coordinates by default.
    # Explicit centered=False always requests the physically native grid.
    centered = spec.get("centered", _get_location(data, background_var) == "center")
    if not isinstance(centered, (bool, np.bool_)):
        raise TypeError("contours['centered'] must be a boolean")
    opts = {
        key: val for key, val in spec.items() if key not in ("var", "centered", "units")
    }
    values = np.asarray(
        data.get_var(
            contour_var,
            snap,
            **selection,
            uniform_z=uniform_z,
            centered=bool(centered),
            units=spec.get("units", units),
            low_memory=low_memory,
        )
    )
    contour_plane = np.squeeze(values)
    if contour_plane.ndim != 2:
        raise ValueError(f"Expected a 2-D contour plane, got shape {values.shape}")
    h, v = _plane_axes(cut_axis)
    coords = _get_plot_coordinates(
        data,
        "center" if centered else _get_location(data, contour_var),
        selection=selection,
        uniform_z=uniform_z,
    )
    _validate_plane_shape(contour_plane, coords[h], coords[v], cut_axis)
    return ax.contour(coords[h], coords[v], contour_plane.T, **opts)


def _get_location(data: Any, var: str) -> str:
    """Return the registered stagger location for a quantity."""
    return str(data.get_location(var))


def _get_plot_coordinates(
    data: Any,
    location: str,
    *,
    selection: dict[str, float | slice | None],
    uniform_z: bool,
) -> dict[AxisName, np.ndarray]:
    """Return coordinate arrays/scalars matching exactly the selected data."""
    x, y, z = data.get_grid(
        location=location,
        units="solar",
        br=False,
        uniform_z=uniform_z,
        **selection,
    )
    return {
        "x": np.asarray(x, dtype=float),
        "y": np.asarray(y, dtype=float),
        "z": np.asarray(z, dtype=float),
    }


def _plane_axes(cut_axis: AxisName) -> tuple[AxisName, AxisName]:
    """Return plotted horizontal and vertical axes for a fixed cut."""
    if cut_axis == "x":
        return "y", "z"
    if cut_axis == "y":
        return "x", "z"
    return "x", "y"


def _validate_plane_shape(
    plane: np.ndarray,
    horizontal: np.ndarray,
    vertical: np.ndarray,
    cut_axis: AxisName,
) -> None:
    """Require consistent plotting geometry for an ``imshow`` plane."""
    expected = (horizontal.size, vertical.size)
    if plane.shape != expected:
        raise ValueError(
            f"Coordinate/data mismatch for a {cut_axis.upper()} cut: "
            f"plane shape is {plane.shape}, expected {expected}. "
            "Check the variable location and grid selectors."
        )


def _extent(horizontal: np.ndarray, vertical: np.ndarray) -> list[float]:
    """Return outer pixel edges from first/last sampled center positions."""

    def edges(values):
        """Extrapolate half a grid spacing at each end of an image axis."""
        centers = np.asarray(values, dtype=float)
        if centers.size < 2:
            raise ValueError("Image axis needs at least two coordinate points")
        return (
            float(centers[0] - 0.5 * (centers[1] - centers[0])),
            float(centers[-1] + 0.5 * (centers[-1] - centers[-2])),
        )

    left, right = edges(horizontal)
    bottom, top = edges(vertical)
    return [left, right, bottom, top]


def _format_axes(
    ax: Any, *, horizontal: AxisName, vertical: AxisName, minor_ticks: bool
) -> None:
    """Apply physical labels and optional minor ticks."""
    ax.set_xlabel(f"{horizontal} (Mm)")
    ax.set_ylabel(f"{vertical} (Mm)")
    if minor_ticks:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which="both", direction="out")


def _set_titles(
    ax: Any, *, cut_axis: AxisName, cut_coord: float | None, time_min: float
) -> None:
    """Show selected physical plane and physical snapshot time."""
    label = (
        f"{cut_axis.upper()}={cut_coord:.2f} Mm"
        if cut_coord is not None
        else f"LOS={cut_axis.upper()}"
    )
    ax.set_title(label, loc="left")
    ax.set_title(f"t={time_min:.2f} min", loc="right")


def _snapshot_time_min(data: Any, snap: int) -> float:
    """Return physical snapshot time in minutes."""
    return float(np.asarray(data.get_t(snap, units="solar")).squeeze()) / 60.0


def _colorbar_label(
    data: Any, var: str, units: str, *, los: AxisName | None = None
) -> str:
    """Build the registered physical symbol and optional LOS-integrated unit."""
    try:
        name = data.get_var_name(var, latex=True)
    except (AttributeError, KeyError):
        name = var
    if los is not None:
        symbol = str(name)
        if symbol.startswith("$") and symbol.endswith("$"):
            symbol = symbol[1:-1]
        name = rf"$\int {symbol}\,\mathrm{{d}}{los}$"
    try:
        if los is None:
            unit = data.get_units(var, units=units, latex=True)
        else:
            plain = data.get_units(var, units=units, latex=False)
            length = {"solar": "cm", "cgs": "cm", "si": "m", "bifrost": "Mm"}[
                units.lower()
            ]
            text, _ = _integrated_unit_plain(
                str(plain), length, solar=(units.lower() == "solar")
            )
            unit = _latex_unit(text)
    except (AttributeError, KeyError):
        unit = ""
    if unit is None:
        unit = ""
    unit = str(unit).strip()
    if not unit or unit in {"1", "dimensionless"}:
        return str(name)
    return f"{name} ({unit})"


def _add_colorbar(
    fig: Any, ax: Any, im: Any, *, label: str, position: ColorbarPosition
) -> Any:
    """Attach a colorbar to the supplied axes, preserving its parent figure."""
    if position not in {"top", "bottom", "left", "right"}:
        raise ValueError("cbar_position must be 'top', 'bottom', 'left', or 'right'.")
    if fig.get_constrained_layout():
        # Axes from make_axes_locatable do not take part in constrained layout.
        colorbar = fig.colorbar(
            im, ax=ax, location=position, fraction=0.05, pad=0.02, aspect=30
        )
        colorbar.set_label(label)
        return colorbar
    divider = make_axes_locatable(ax)
    if position in {"top", "bottom"}:
        cax = divider.append_axes(position, size="3.5%", pad=0.55)
        colorbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        if position == "top":
            cax.xaxis.set_ticks_position("top")
            cax.xaxis.set_label_position("top")
        else:
            cax.xaxis.set_ticks_position("bottom")
            cax.xaxis.set_label_position("bottom")
    else:
        cax = divider.append_axes(position, size="3.5%", pad=0.12)
        colorbar = fig.colorbar(im, cax=cax, orientation="vertical")
        if position == "left":
            cax.yaxis.set_ticks_position("left")
            cax.yaxis.set_label_position("left")
        else:
            cax.yaxis.set_ticks_position("right")
            cax.yaxis.set_label_position("right")
    colorbar.set_label(label)
    return colorbar
