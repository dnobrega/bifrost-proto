"""Microbenchmark the v0.1.19 fused algebra helpers on float32 cubes."""

from __future__ import annotations

import argparse
import time
import numpy as np

from bifrost_proto import quantities


def timed(label, func, repeat=3):
    """Helper used by benchmark algebra v119."""
    times = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = func()
        times.append(time.perf_counter() - t0)
    print(f"{label:24s}: {min(times):.4f} s  {times}")
    return out


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=192)
    args = parser.parse_args()
    n = args.n
    shape = (n, n, n)
    rng = np.random.default_rng(1234)
    a = [
        np.array(rng.normal(size=shape), dtype=np.float32, order="F") for _ in range(6)
    ]
    ax, ay, az, bx, by, bz = a

    # Warm Numba before timing.
    quantities._dot3_arrays(ax, ay, az, bx, by, bz)
    quantities._negative_cross_term_arrays(ax, ay, az, bx)
    quantities._timescale_from_energy(np.abs(ax) + 1.0, ay)
    cosine = np.empty_like(ax)
    quantities._acute_cosine_native(ax, ay, az, bx, by, bz, cosine)

    timed("dot3 NumPy reference", lambda: ax * bx + ay * by + az * bz)
    timed("dot3 fused Numba", lambda: quantities._dot3_arrays(ax, ay, az, bx, by, bz))
    timed("E term NumPy", lambda: -(ax * ay - az * bx))
    timed(
        "E term fused Numba",
        lambda: quantities._negative_cross_term_arrays(ax, ay, az, bx),
    )

    def theta_numpy():
        """Helper used by benchmark algebra v119."""
        dot = ax * bx + ay * by + az * bz
        modj = np.sqrt(ax * ax + ay * ay + az * az)
        modb = np.sqrt(bx * bx + by * by + bz * bz)
        with np.errstate(invalid="ignore", divide="ignore"):
            cosine = np.abs(dot) / (modj * modb)
        np.clip(cosine, 0.0, 1.0, out=cosine)
        np.arccos(cosine, out=cosine)
        return cosine

    def theta_fused():
        """Helper used by benchmark algebra v119."""
        cosine = np.empty_like(ax)
        quantities._acute_cosine_native(ax, ay, az, bx, by, bz, cosine)
        np.arccos(cosine, out=cosine)
        return cosine

    timed("theta NumPy reference", theta_numpy)
    timed("theta fused+ufunc", theta_fused)

    denom = np.abs(bx) + np.float32(0.25)
    timed("uparb expression", lambda: (ax + ay + az) / denom)

    def uparb_out():
        """Helper used by benchmark algebra v119."""
        out = np.empty_like(ax)
        np.add(ax, ay, out=out)
        np.add(out, az, out=out)
        np.divide(out, denom, out=out)
        return out

    timed("uparb out=", uparb_out)
    timed("qlf expression", lambda: ax + ay + az)

    def qlf_out():
        """Helper used by benchmark algebra v119."""
        out = np.empty_like(ax)
        np.add(ax, ay, out=out)
        np.add(out, az, out=out)
        return out

    timed("qlf out=", qlf_out)

    e = np.abs(ax) + np.float32(1.0)
    q = ay.copy(order="F")
    q[0, 0, 0] = 0.0
    timed(
        "timescale NumPy",
        lambda: np.divide(e, np.abs(q), out=np.full_like(e, np.inf), where=(q != 0.0)),
    )
    timed("timescale fused", lambda: quantities._timescale_from_energy(e, q))


if __name__ == "__main__":
    main()
