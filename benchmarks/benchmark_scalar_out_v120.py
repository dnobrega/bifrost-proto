"""Benchmark helper for bifrost-proto development and validation."""

import argparse
import time
import numpy as np


def bench(fn, repeat=5):
    """Helper used by benchmark scalar out v120."""
    vals = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        vals.append(time.perf_counter() - t0)
    return min(vals), out


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=192)
    ap.add_argument("--order", choices=["C", "F"], default="F")
    args = ap.parse_args()
    n = args.n
    order = args.order
    rng = np.random.default_rng(20)
    shape = (n, n, n)
    r = np.array(np.abs(rng.normal(size=shape)).astype(np.float32) + 0.25, order=order)
    a = np.array(rng.normal(size=shape).astype(np.float32), order=order)
    b = np.array(np.abs(rng.normal(size=shape)).astype(np.float32) + 0.25, order=order)
    u = np.array(np.abs(rng.normal(size=shape)).astype(np.float32) + 0.25, order=order)
    pg = np.array(np.abs(rng.normal(size=shape)).astype(np.float32) + 0.25, order=order)
    gamma = np.float32(5.0 / 3.0)
    grav = np.float32(2.74)

    cases = []

    cases.append(("pb", lambda: np.float32(0.5) * b * b, lambda: _pb_out(b)))
    cases.append(("ek", lambda: np.float32(0.5) * r * u * u, lambda: _ek_out(r, u)))
    cases.append(("qpdv", lambda: -pg * a, lambda: _qpdv_out(pg, a)))
    cases.append(("ldivu", lambda: np.abs(a) / u, lambda: _ldivu_out(a, u)))
    cases.append(("va", lambda: b / np.sqrt(r), lambda: _va_out(b, r)))
    cases.append(
        ("cs_gamma", lambda: np.sqrt(gamma * pg / r), lambda: _cs_out(pg, r, gamma))
    )
    cases.append(("hpg_hydro", lambda: pg / (r * grav), lambda: _hpg_out(pg, r, grav)))

    print(f"shape={shape}, dtype=float32, order={order}")
    for name, old, new in cases:
        told, o = bench(old)
        tnew, nout = bench(new)
        np.testing.assert_array_equal(o, nout)
        print(
            f"{name:10s} old={told:.5f}s  out={tnew:.5f}s  speedup={told / tnew:.2f}x"
        )


def _pb_out(b):
    """Internal helper for benchmark scalar out v120."""
    out = np.empty_like(b)
    np.multiply(b, np.float32(0.5), out=out)
    np.multiply(out, b, out=out)
    return out


def _ek_out(r, u):
    """Internal helper for benchmark scalar out v120."""
    out = np.empty_like(r)
    np.multiply(r, np.float32(0.5), out=out)
    np.multiply(out, u, out=out)
    np.multiply(out, u, out=out)
    return out


def _qpdv_out(pg, d):
    """Internal helper for benchmark scalar out v120."""
    out = np.empty_like(pg)
    np.negative(pg, out=out)
    np.multiply(out, d, out=out)
    return out


def _ldivu_out(d, u):
    """Internal helper for benchmark scalar out v120."""
    out = np.empty_like(d)
    np.abs(d, out=out)
    np.divide(out, u, out=out)
    return out


def _va_out(b, r):
    """Internal helper for benchmark scalar out v120."""
    out = np.empty_like(r)
    np.sqrt(r, out=out)
    np.divide(b, out, out=out)
    return out


def _cs_out(pg, r, g):
    """Internal helper for benchmark scalar out v120."""
    out = np.empty_like(pg)
    np.multiply(pg, g, out=out)
    np.divide(out, r, out=out)
    np.sqrt(out, out=out)
    return out


def _hpg_out(pg, r, g):
    """Internal helper for benchmark scalar out v120."""
    out = np.empty_like(pg)
    np.multiply(r, g, out=out)
    np.divide(pg, out, out=out)
    return out


if __name__ == "__main__":
    main()
