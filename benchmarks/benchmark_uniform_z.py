"""Compare the v1.17 vectorized uniform-Z path with the current parallel kernel."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bifrost_proto.core import BifrostData


def old_uniform_z(value, z):
    """Helper used by benchmark uniform z."""
    z = np.asarray(z, dtype=np.float64)
    znew = np.linspace(z[0], z[-1], z.size)
    hi = np.searchsorted(z, znew, side="right")
    hi = np.clip(hi, 1, z.size - 1)
    lo = hi - 1
    w = (znew - z[lo]) / (z[hi] - z[lo])
    v0 = np.take(value, lo, axis=2)
    v1 = np.take(value, hi, axis=2)
    out = (1.0 - w[None, None, :]) * v0 + w[None, None, :] * v1
    return out.astype(value.dtype, copy=False)


def best_time(fn, repeat=3):
    """Helper used by benchmark uniform z."""
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), result


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--order", choices=("C", "F"), default="C")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(1234)
    value = np.array(
        rng.random((args.n, args.n, args.n), dtype=np.float32),
        order=args.order,
    )
    z = np.linspace(0.0, 1.0, args.n, dtype=np.float64) ** 1.3

    # Warm up Numba before timing.
    BifrostData._uniformize_z(value, ("x", "y", "z"), z)

    told, old = best_time(lambda: old_uniform_z(value, z), args.repeat)
    tnew, new = best_time(
        lambda: BifrostData._uniformize_z(value, ("x", "y", "z"), z)[0],
        args.repeat,
    )

    np.testing.assert_array_equal(new, old)
    gib = value.nbytes / 1024**3
    print(f"shape       : {value.shape}")
    print(f"layout      : {args.order}")
    print(f"one cube    : {gib:.3f} GiB")
    print(f"v1.17 numpy : {told:.4f} s")
    print(f"current numba : {tnew:.4f} s")
    print(f"speed-up    : {told / tnew:.2f}x")
    print(
        "The current kernel allocates only the output cube; the old path also materialized"
    )
    print("two np.take() cubes and float64 broadcast temporaries.")


if __name__ == "__main__":
    main()
