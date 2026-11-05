import numpy as np

from bifrost_proto import quantities


class DummyData:
    def __init__(self, arrays, gamma=np.float32(5.0/3.0), grav=np.float32(2.74)):
        self.arrays = arrays
        self._gamma = gamma
        self._grav = grav

    def _getvar(self, name, snap, selection, context):
        return self.arrays[name]

    class _Meta:
        class _Units:
            u_l = 1e8
            u_t = 1e2
        units = _Units()
        params = {}

    def _meta(self, snap):
        m = self._Meta()
        m.params = {"gamma": float(self._gamma), "grav": float(self._grav)}
        return m


def _farray(rng, shape=(13, 11, 9), positive=False):
    a = rng.normal(size=shape).astype(np.float32)
    if positive:
        a = np.abs(a) + np.float32(0.25)
    return np.array(a, order="F", copy=True)


def test_scalar_out_paths_match_v119_expressions():
    rng = np.random.default_rng(120)
    r = _farray(rng, positive=True)
    b = _farray(rng, positive=True)
    u = _farray(rng, positive=True)
    pg = _farray(rng, positive=True)
    divu = _farray(rng)
    dd = DummyData({"r": r, "modb": b, "modu": u, "pg": pg, "divu": divu})

    np.testing.assert_allclose(
        quantities.get_pb(dd, 0, None, None), np.float32(0.5) * b * b, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        quantities.get_eb(dd, 0, None, None), np.float32(0.5) * b * b, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        quantities.get_ek(dd, 0, None, None), np.float32(0.5) * r * u * u, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        quantities.get_qpdv(dd, 0, None, None), -pg * divu, rtol=0, atol=0
    )
    np.testing.assert_allclose(
        quantities.get_ldivu(dd, 0, None, None), u / np.abs(divu), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        quantities.get_va(dd, 0, None, None), b / np.sqrt(r), rtol=0, atol=0
    )
    gamma = np.float32(5.0 / 3.0)
    np.testing.assert_allclose(
        quantities.get_cs_gamma(dd, 0, None, None), np.sqrt(gamma * pg / r), rtol=0, atol=0
    )
    grav = np.float32(2.74)
    np.testing.assert_allclose(
        quantities.get_hpg_hydro(dd, 0, None, None), pg / (r * grav), rtol=0, atol=0
    )
