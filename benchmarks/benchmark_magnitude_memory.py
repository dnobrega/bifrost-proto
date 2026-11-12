"""Benchmark vector-magnitude speed and temporary memory.

The ``numba`` mode calls the exact ``_norm_arrays`` implementation used by
bifrost_proto diagnostics such as modb/modj/modlf.  JIT compilation is warmed
up before the large arrays are allocated and before timing begins.

Examples
--------
python benchmarks/benchmark_magnitude_memory.py naive --n 512 --order F
python benchmarks/benchmark_magnitude_memory.py hypot --n 512 --order F
python benchmarks/benchmark_magnitude_memory.py numba --n 512 --order F
python benchmarks/benchmark_magnitude_memory.py numba --n 512 --order F --stride 2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import resource
import sys
import time

import numpy as np

# Allow running this file directly from the source checkout without installing
# the package first.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bifrost_proto.quantities import _norm_arrays


def maxrss_mib():
    """Return process maximum resident set size in MiB."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return value / (1024.0 * 1024.0) if value > 10_000_000 else value / 1024.0


def _warm_numba(order, stride):
    """Internal helper for benchmark magnitude memory."""
    shape = (8, 8, 8)
    bases = tuple(
        np.full(shape, value, dtype=np.float32, order=order)
        for value in (1.25, -2.5, 4.0)
    )
    parts = (
        bases if stride == 1 else tuple(a[::stride, ::stride, ::stride] for a in bases)
    )
    _norm_arrays(parts)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("naive", "hypot", "numba"))
    parser.add_argument("--n", type=int, default=220)
    parser.add_argument(
        "--order",
        choices=("C", "F"),
        default="F",
        help="Array memory order. Bifrost arrays are typically F-contiguous.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help=(
            "Use a non-contiguous [::stride,::stride,::stride] view. "
            "Values >1 exercise the stride-aware Numba path."
        ),
    )
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")

    if args.mode == "numba":
        _warm_numba(args.order, args.stride)

    shape = (args.n, args.n, args.n)
    xb = np.full(shape, 1.25, dtype=np.float32, order=args.order)
    yb = np.full(shape, -2.5, dtype=np.float32, order=args.order)
    zb = np.full(shape, 4.0, dtype=np.float32, order=args.order)
    if args.stride == 1:
        x, y, z = xb, yb, zb
    else:
        x = xb[:: args.stride, :: args.stride, :: args.stride]
        y = yb[:: args.stride, :: args.stride, :: args.stride]
        z = zb[:: args.stride, :: args.stride, :: args.stride]

    baseline = maxrss_mib()
    t0 = time.perf_counter()
    if args.mode == "naive":
        out = np.sqrt(x * x + y * y + z * z)
    elif args.mode == "hypot":
        out = np.empty_like(x)
        np.hypot(x, y, out=out)
        np.hypot(out, z, out=out)
    else:
        out = _norm_arrays((x, y, z))
    elapsed = time.perf_counter() - t0
    peak = maxrss_mib()

    checksum = float(out[0, 0, 0])
    print(
        f"mode={args.mode} n={args.n} order={args.order} stride={args.stride} "
        f"shape={x.shape} elapsed_s={elapsed:.6f} baseline_mib={baseline:.1f} "
        f"peak_mib={peak:.1f} delta_peak_mib={peak - baseline:.1f} "
        f"checksum={checksum:.8g}"
    )


if __name__ == "__main__":
    main()
