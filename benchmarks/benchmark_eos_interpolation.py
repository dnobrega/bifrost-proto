#!/usr/bin/env python3
"""Benchmark the legacy SciPy reference and runtime Numba rho-e EOS interpolator."""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bifrost_proto.eos import EOSTable


def make_eos(nei=1000, nrho=1000):
    """Helper used by benchmark eos interpolation."""
    eos = EOSTable.__new__(EOSTable)
    eos.lnei = np.linspace(np.log(2.03742577e12), np.log(2.07685234e15), nei)
    eos.lnrho = np.linspace(np.log(1.0e-18), np.log(1.0e-4), nrho)
    eos.dlnei = eos.lnei[1] - eos.lnei[0]
    eos.dlnrho = eos.lnrho[1] - eos.lnrho[0]
    x = np.arange(nei, dtype=np.float32)[:, None]
    y = np.arange(nrho, dtype=np.float32)[None, :]
    eos.tg = np.asfortranarray(4000.0 + 4.0 * x + 2.0 * y, dtype=np.float32)
    eos.lnpg = np.asfortranarray(8.0 + 0.002 * x + 0.001 * y, dtype=np.float32)
    eos.lnne = np.asfortranarray(24.0 + 0.0015 * x + 0.0007 * y, dtype=np.float32)
    eos.lnkr = np.asfortranarray(-6.0 + 0.001 * x + 0.0005 * y, dtype=np.float32)
    return eos


def scipy_reference(eos, rho_cgs, ee_cgs, out="tg"):
    """Pre-v1.14 SciPy definition kept only inside this benchmark."""
    table = eos._table(out)
    x = (np.log(ee_cgs) - eos.lnei[0]) / eos.dlnei
    y = (np.log(rho_cgs) - eos.lnrho[0]) / eos.dlnrho
    value = map_coordinates(table, [x, y], order=1, mode="nearest")
    if out != "tg":
        value = np.exp(value)
    return value.astype(np.float32, copy=False)


def peak_mib():
    """Helper used by benchmark eos interpolation."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes; Linux reports KiB.
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("scipy", "numba"))
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--order", choices=("C", "F"), default="F")
    parser.add_argument("--output", choices=("tg", "ne", "pg", "kr"), default="ne")
    args = parser.parse_args()

    eos = make_eos()
    rng = np.random.default_rng(123)
    shape = (args.n, args.n, args.n)
    lr = rng.uniform(eos.lnrho[0] + 0.02, eos.lnrho[-1] - 0.02, size=shape)
    le = rng.uniform(eos.lnei[0] + 0.02, eos.lnei[-1] - 0.02, size=shape)
    rho = np.array(np.exp(lr), dtype=np.float32, order=args.order)
    ee = np.array(np.exp(le), dtype=np.float32, order=args.order)
    del lr, le

    if args.mode == "numba":
        # Warm up compilation on a tiny array outside the timed section.
        eos.interpolate(
            rho[:2, :2, :2].copy(order=args.order),
            ee[:2, :2, :2].copy(order=args.order),
            out=args.output,
        )
        fn = lambda r, e: eos.interpolate(r, e, out=args.output)
    else:
        fn = lambda r, e: scipy_reference(eos, r, e, out=args.output)

    baseline = peak_mib()
    t0 = time.perf_counter()
    result = fn(rho, ee)
    elapsed = time.perf_counter() - t0
    peak = peak_mib()

    print(
        f"mode={args.mode} output={args.output} n={args.n} order={args.order} "
        f"elapsed_s={elapsed:.6f} baseline_mib={baseline:.1f} "
        f"peak_mib={peak:.1f} delta_peak_mib={peak - baseline:.1f} "
        f"checksum={float(result.flat[result.size // 2]):.8g}"
    )


if __name__ == "__main__":
    main()
