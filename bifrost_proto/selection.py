"""Selection and sampling plans for partial-domain Bifrost evaluation.

Selections distinguish public physical coordinates from native array indices.
Sampling plans retain the metadata needed to crop halo-expanded computations,
reorient arrays, and optionally resample the selected Z extent uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

AXES = ("x", "y", "z")


@dataclass(frozen=True)
class Selection:
    """Contiguous 3-D selection in native Bifrost index space."""

    ix: slice
    iy: slice
    iz: slice

    @property
    def slices(self):
        """Return the native ``(ix, iy, iz)`` slice tuple."""
        return (self.ix, self.iy, self.iz)

    @property
    def key(self):
        """Return a hashable cache key for this selection."""
        return tuple((s.start, s.stop, s.step) for s in self.slices)

    def normalized(self, shape):
        """Resolve open bounds against *shape* and require unit stride."""
        out = []
        for s, n in zip(self.slices, shape):
            # Internally expanded periodic windows may use unwrapped indices
            # (e.g. [-5, 6) for a cut at zero). Public plans are in range.
            start = 0 if s.start is None else int(s.start)
            stop = n if s.stop is None else int(s.stop)
            step = 1 if s.step is None else s.step
            if step != 1:
                raise ValueError("Internal Selection objects must be contiguous (step=1)")
            out.append(slice(start, stop, 1))
        return Selection(*out)

    def expand(self, axis: int, halo: int, shape, periodic: bool):
        """Expand a stencil halo; represent wrapped X/Y by unwrapped bounds."""

        sel = list(self.normalized(shape).slices)
        s = sel[axis]
        n = shape[axis]
        start = s.start - halo
        stop = s.stop + halo

        if periodic and n <= 64 and (start < 0 or stop > n):
            # Legacy full-domain fallback for small synthetic grids, whose
            # test in-memory readers use ordinary slices rather than wrapping.
            sel[axis] = slice(0, n, 1)
        elif periodic:
            width = s.stop - s.start
            if width >= n:
                # A complete periodic window already supplies every sample;
                # preserve its unwrapped origin so nested crops stay valid.
                sel[axis] = slice(s.start, s.stop, 1)
            elif stop - start >= n:
                # The halo would exceed a full period. Keep exactly n
                # samples, *including* the original unwrapped child. The
                # periodic stencil handles the boundary of this full window.
                extra_left = min(halo, n - width)
                begin = s.start - extra_left
                sel[axis] = slice(begin, begin + n, 1)
            else:
                sel[axis] = slice(start, stop, 1)
        else:
            sel[axis] = slice(max(0, start), min(n, stop), 1)

        return Selection(*sel)

    def relative_to(self, parent, shape):
        """Return slices that crop parent data back to this selection."""

        child = self.normalized(shape)
        parent = parent.normalized(shape)
        rel = []
        for cs, ps in zip(child.slices, parent.slices):
            if cs.start < ps.start or cs.stop > ps.stop:
                raise ValueError(
                    "Expanded parent does not contain the requested selection: "
                    f"child={cs}, parent={ps}"
                )
            rel.append(slice(cs.start - ps.start, cs.stop - ps.start, 1))
        return tuple(rel)


@dataclass(frozen=True)
class AxisPlan:
    """One user-facing axis request expressed in physical-index space."""

    pstart: int
    pstop: int
    action: str = "keep"       # keep, indices, scalar, linear
    indices: tuple[int, ...] | None = None
    weight: float | None = None
    target: float | None = None


@dataclass(frozen=True)
class SamplingPlan:
    """Bridge between user selectors and one contiguous native Selection."""

    native: Selection
    axes: tuple[AxisPlan, AxisPlan, AxisPlan]
    physical_coords: tuple[np.ndarray, np.ndarray, np.ndarray]

    def apply(self, value):
        """Apply stride/scalar/linear post-processing after physical reorientation."""

        arr = value
        names = ["x", "y", "z"]
        coords = {
            axis: np.asarray(c[p.pstart:p.pstop])
            for axis, c, p in zip(AXES, self.physical_coords, self.axes)
        }

        # Work high -> low axis so removing an axis never changes lower axis numbers.
        for axis in (2, 1, 0):
            plan = self.axes[axis]
            name = AXES[axis]

            if plan.action == "keep":
                continue

            if plan.action == "indices":
                idx = np.asarray(plan.indices, dtype=np.int64)
                arr = np.take(arr, idx, axis=axis)
                coords[name] = coords[name][idx]
                continue

            if plan.action == "scalar":
                arr = np.take(arr, 0, axis=axis)
                coords.pop(name)
                names.remove(name)
                continue

            if plan.action == "linear":
                w = float(plan.weight)
                v0 = np.take(arr, 0, axis=axis)
                v1 = np.take(arr, 1, axis=axis)
                arr = (1.0 - w) * v0 + w * v1
                coords.pop(name)
                names.remove(name)
                continue

            raise ValueError(f"Unknown AxisPlan action: {plan.action}")

        return arr, tuple(names), coords
