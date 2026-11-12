#!/usr/bin/env python3
"""Microbenchmark representative serial vs parallel kernels from v0.1.22."""

from __future__ import annotations

import argparse
import time

import numpy as np
from numba import get_num_threads, njit, prange


@njit
def serial_norm3(x, y, z, out):
    """Serial three-component norm used as the benchmark reference."""
    for i in range(out.size):
        out[i] = np.sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i])


@njit(parallel=True)
def parallel_norm3(x, y, z, out):
    """Parallel three-component norm with unchanged per-element arithmetic."""
    for i in prange(out.size):
        out[i] = np.sqrt(x[i] * x[i] + y[i] * y[i] + z[i] * z[i])


@njit
def serial_tauq(e, q, out):
    """Serial heating/cooling timescale kernel used as the reference."""
    for i in range(out.size):
        qv = q[i]
        out[i] = np.inf if qv == 0.0 else e[i] / abs(qv)


@njit(parallel=True)
def parallel_tauq(e, q, out):
    """Parallel timescale kernel preserving exact zero/NaN semantics."""
    for i in prange(out.size):
        qv = q[i]
        out[i] = np.inf if qv == 0.0 else e[i] / abs(qv)


def best(fn, args, repeat):
    """Return the best warmed execution time over *repeat* runs."""
    fn(*args)
    values = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn(*args)
        values.append(time.perf_counter() - t0)
    return min(values)


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(122)
    size = args.n**3
    x = rng.normal(size=size).astype(np.float32)
    y = rng.normal(size=size).astype(np.float32)
    z = rng.normal(size=size).astype(np.float32)
    e = np.abs(rng.normal(size=size)).astype(np.float32) + np.float32(1.0)
    q = rng.normal(size=size).astype(np.float32)

    out_s = np.empty(size, dtype=np.float32)
    out_p = np.empty(size, dtype=np.float32)

    print(f"Numba threads: {get_num_threads()}")
    for label, serial, parallel, inputs in (
        ("norm3", serial_norm3, parallel_norm3, (x, y, z)),
        ("tauq", serial_tauq, parallel_tauq, (e, q)),
    ):
        ts = best(serial, (*inputs, out_s), args.repeat)
        tp = best(parallel, (*inputs, out_p), args.repeat)
        assert np.array_equal(out_s, out_p)
        print(f"{label:8s} serial={ts:.6f}s parallel={tp:.6f}s speedup={ts / tp:.2f}x")


if __name__ == "__main__":
    main()
