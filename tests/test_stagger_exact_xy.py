import numpy as np

from bifrost_proto.stagger import StaggerGrid


def _manual_xup(f):
    a = np.float32(150.0 / 256.0)
    b = np.float32(-25.0 / 256.0)
    c = np.float32(3.0 / 256.0)
    nx = f.shape[0]
    out = np.empty_like(f)
    for i in range(nx):
        idx = np.array([(i - 2 + l) % nx for l in range(6)])
        out[i] = (
            a * (f[idx[3]] + f[idx[2]])
            + b * (f[idx[4]] + f[idx[1]])
            + c * (f[idx[5]] + f[idx[0]])
        )
    return out


def test_xup_uses_legacy_grouped_float32_arithmetic():
    rng = np.random.default_rng(1234)
    shape = (16, 7, 9)
    f = np.asfortranarray(rng.normal(size=shape).astype(np.float32))
    x = np.arange(shape[0], dtype=np.float32) * np.float32(0.1)
    y = np.arange(shape[1], dtype=np.float32) * np.float32(0.2)
    z = np.linspace(0.0, 1.0, shape[2], dtype=np.float32)
    g = StaggerGrid(
        x, x - 0.05, y, y - 0.1, z, z - 0.01, dx=np.float32(0.1), dy=np.float32(0.2)
    )
    got = g.xup(f)
    ref = _manual_xup(f)
    assert np.array_equal(got, ref)
