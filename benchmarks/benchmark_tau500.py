#!/usr/bin/env python3
"""Benchmark the fused native-order tau500 integration kernel."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bifrost_proto.quantities import _integrate_tau500_native


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--order", choices=("C", "F"), default="F")
    args = parser.parse_args()

    shape = (args.n, args.n, args.n)
    ne = np.full(shape, 2.0e12, dtype=np.float32, order=args.order)
    nh = np.full(shape, 1.0e17, dtype=np.float32, order=args.order)
    tg = np.empty(shape, dtype=np.float32, order=args.order)
    for k in range(args.n):
        tg[:, :, k] = 5000.0 + 20.0 * k
    z_cm = np.linspace(0.0, 2.0e8, args.n, dtype=np.float64)
    out = np.empty_like(tg, dtype=np.float32, order="K")

    # Compile outside the timed region.
    tiny = np.asfortranarray(np.ones((2, 2, 2), dtype=np.float32))
    _integrate_tau500_native(tiny, tiny, tiny * 6000.0, z_cm[:2], np.empty_like(tiny))

    t0 = time.perf_counter()
    _integrate_tau500_native(ne, nh, tg, z_cm, out)
    elapsed = time.perf_counter() - t0

    print(
        f"n={args.n} order={args.order} elapsed_s={elapsed:.6f} "
        f"tau_bottom={float(out[0, 0, -1]):.8g}"
    )


if __name__ == "__main__":
    main()
