"""Benchmark synthetic instrument-response evaluation.

The ``numpy`` mode reproduces the v1.11 path::

    G = response.evaluate(T, ne)
    emissivity = ne * nH * G

The ``numba`` mode calls the fused ``evaluate_emissivity`` implementation used
by v1.12 instrument variables such as ``aia193``. JIT compilation is warmed up
before timing.

Examples
--------
python benchmarks/benchmark_instrument_response.py numpy --n 256 --order F
python benchmarks/benchmark_instrument_response.py numba --n 256 --order F
"""

from __future__ import annotations

import argparse
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bifrost_proto.responses import InstrumentResponseTable


def maxrss_mib():
    """Helper used by benchmark instrument response."""
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024.0 * 1024.0) if value > 10_000_000 else value / 1024.0


def make_response(root: Path):
    """Helper used by benchmark instrument response."""
    folder = root / "aia193"
    folder.mkdir(parents=True, exist_ok=True)
    logt = np.linspace(4.0, 8.0, 161, dtype=np.float64)
    logne = np.linspace(7.0, 13.0, 121, dtype=np.float64)
    # Smooth strictly positive synthetic table; only interpolation performance matters.
    table = np.exp(-(((logt[None, :] - 6.2) / 0.45) ** 2)) * (
        1.0 + 0.05 * (logne[:, None] - 10.0)
    )
    np.save(folder / "table.npy", table)
    np.save(folder / "logt.npy", logt)
    np.save(folder / "logne.npy", logne)
    (folder / "metadata.json").write_text(
        json.dumps({"name": "aia193", "fill_value": 0.0})
    )
    return InstrumentResponseTable("aia193", base_dir=root)


def warm(response):
    """Helper used by benchmark instrument response."""
    shape = (8, 8, 8)
    tg = np.full(shape, 1.2e6, dtype=np.float32, order="F")
    ne = np.full(shape, 2e9, dtype=np.float32, order="F")
    nh = np.full(shape, 1e9, dtype=np.float32, order="F")
    response.evaluate_emissivity(tg, ne, nh)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("numpy", "numba"))
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--order", choices=("C", "F"), default="F")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        response = make_response(Path(tmp))
        if args.mode == "numba":
            warm(response)

        shape = (args.n, args.n, args.n)
        tg = np.full(shape, 1.2e6, dtype=np.float32, order=args.order)
        ne = np.full(shape, 2.0e9, dtype=np.float32, order=args.order)
        nh = np.full(shape, 1.0e9, dtype=np.float32, order=args.order)

        baseline = maxrss_mib()
        t0 = time.perf_counter()
        if args.mode == "numpy":
            out = ne * nh * response.evaluate(tg, ne)
        else:
            out = response.evaluate_emissivity(tg, ne, nh)
        elapsed = time.perf_counter() - t0
        peak = maxrss_mib()

        print(
            f"mode={args.mode} n={args.n} order={args.order} "
            f"elapsed_s={elapsed:.6f} baseline_mib={baseline:.1f} "
            f"peak_mib={peak:.1f} delta_peak_mib={peak - baseline:.1f} "
            f"checksum={float(out[0, 0, 0]):.8g}"
        )


if __name__ == "__main__":
    main()
