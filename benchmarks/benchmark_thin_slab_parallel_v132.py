"""Microbenchmark for v0.1.32 pointwise thin-slab parallelism.

Run from the repository root, for example:
    NUMBA_NUM_THREADS=16 python benchmarks/benchmark_thin_slab_parallel_v132.py
"""

from functools import partial
from time import perf_counter

import numpy as np
from numba import get_num_threads

from bifrost_proto import quantities


def timed(call, repeat=20):
    """Return the mean wall-clock time per warmed kernel call."""
    call()  # compile/warm cache
    t0 = perf_counter()
    for _ in range(repeat):
        call()
    return (perf_counter() - t0) / repeat


def main():
    """Benchmark flat and collapsed-row dot kernels for several Z thicknesses."""
    rng = np.random.default_rng(132)
    print(f"Numba threads: {get_num_threads()}")
    for nz in (1, 2, 4, 16, 64):
        shape = (512, 512, nz)
        arrays = [
            np.array(rng.normal(size=shape), dtype=np.float32, order="F")
            for _ in range(6)
        ]
        ax, ay, az, bx, by, bz = arrays
        out = np.empty_like(ax)

        flat = tuple(quantities._flat_memory_view(a) for a in arrays)
        flat_out = quantities._flat_memory_view(out)
        t_flat = timed(partial(quantities._dot3_flat, *flat, flat_out))
        t_rows = timed(partial(quantities._dot3_native, ax, ay, az, bx, by, bz, out))
        print(
            f"512x512x{nz:<2d}: flat={1e3 * t_flat:8.3f} ms  "
            f"row-parallel={1e3 * t_rows:8.3f} ms"
        )


if __name__ == "__main__":
    main()
