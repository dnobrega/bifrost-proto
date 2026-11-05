"""Shared descriptive filenames for 2-D plots and MP4 movies."""

from __future__ import annotations

import math
from typing import Any


def _coordinate_label(coordinate: float) -> str:
    """Format physical filename coordinates to exactly two decimals (Mm)."""
    value = float(coordinate)
    if not math.isfinite(value):
        raise ValueError("Filename coordinates must be finite")
    if round(value, 2) == 0:
        value = 0.0  # Suppress float32 -0.00 labels.
    return f"{value:.2f}"


def _roi_suffix(axis: str, selections: dict[str, Any] | None) -> str:
    """Describe selected physical ROI axes without changing legacy full-plane names."""
    if not selections:
        return ""
    fragments = []
    for key in "xyz":
        if key == axis or not isinstance(selections.get(key), slice):
            continue
        region = selections[key]
        if region.start is None and region.stop is None:
            continue  # Same as selecting the complete axis.
        start = "min" if region.start is None else _coordinate_label(float(region.start))
        stop = "max" if region.stop is None else _coordinate_label(float(region.stop))
        fragments.append(f"_{key}={start}-{stop}")
    return "".join(fragments)


