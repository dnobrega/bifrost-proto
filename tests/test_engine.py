import numpy as np

from bifrost_proto.core import BifrostData, EvaluationContext, SnapshotMeta
from bifrost_proto.registry import VariableRegistry, GLOBAL_REGISTRY
from bifrost_proto.selection import Selection
from bifrost_proto.units import BifrostUnits


class ArrayData(BifrostData):
    def __init__(self, arrays, x, y, z):
        self.arrays = arrays
        self.registry = VariableRegistry(GLOBAL_REGISTRY)
        self._meta_cache = {}
        self._eos_cache = {}
        self._shared_context = None
        self.read_counts = {}
        self.dtype = "<f4"
        self.base_dtype = "f4"
        self.big_endian = False
        self.verbose = False

        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        shape = arrays["r"].shape
        raw = {name: object() for name in arrays}
        self._meta_cache[0] = SnapshotMeta(
            snap=0,
            params={},
            shape=shape,
            x=x.astype(np.float32),
            xdn=(x - 0.5 * dx).astype(np.float32),
            y=y.astype(np.float32),
            ydn=(y - 0.5 * dy).astype(np.float32),
            z=z.astype(np.float32),
            zdn=(z - 0.5 * dz).astype(np.float32),
            units=BifrostUnits(),
            raw=raw,
            periodic=(True, True, False),
        )

    def _variable_exists(self, var, snap):
        return var in self.arrays

    def _read_variable(self, var, snap, selection):
        self.read_counts[var] = self.read_counts.get(var, 0) + 1
        return np.asfortranarray(self.arrays[var][selection.slices])


def make_data(n=24):
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r = np.ones((n, n, n), dtype=np.float32) * 2.0
    arrays = {
        "r": r,
        "e": r * 3.0,
        "px": r * 4.0,
        "py": r * 5.0,
        "pz": r * 6.0,
        "bx": xx.astype(np.float32),
        "by": yy.astype(np.float32),
        "bz": zz.astype(np.float32),
    }
    return ArrayData(arrays, x, y, z)


def test_velocity_and_centering_shapes():
    dd = make_data()
    ux = dd.get_var("ux", 0, units="bifrost")
    assert ux.shape == (24, 24, 24)
    assert np.allclose(ux[5:-5, 5:-5, 5:-5], 4.0, atol=2e-5)

    plane = dd.get_var("ux", 0, iz=10, centered=True, units="bifrost")
    assert plane.shape == (24, 24)


def test_subdomain_halo_keeps_requested_shape():
    dd = make_data()
    out = dd.get_var(
        "modb",
        0,
        ix=slice(7, 17),
        iy=slice(8, 18),
        iz=slice(6, 16),
        units="bifrost",
    )
    assert out.shape == (10, 10, 10)


def test_cache_reuses_identical_operation():
    dd = make_data()

    @dd.register_variable("twice_bx", location="center", unit_type="magnetic_field")
    def get_twice_bx(dd, snap, selection, context):
        a = dd._op("xup", "bx", snap, selection, context)
        b = dd._op("xup", "bx", snap, selection, context)
        return a + b

    dd.get_var("twice_bx", 0, ix=slice(6, 18), units="bifrost")
    assert dd.read_counts["bx"] == 1


def test_physical_index_reverses_z():
    dd = make_data()
    # iz=0 is the first physical (bottom) index, corresponding to native nz-1
    plane = dd.get_var("r", 0, iz=0, units="bifrost")
    plane_br = dd.get_var("r", 0, iz_br=23, units="bifrost")
    assert np.array_equal(plane, plane_br)


def test_linear_coordinate_and_uniform_z():
    dd = make_data()
    # Bifrost z is converted to physical z=-z[::-1]. Pick an exact midpoint
    # between two physical levels and check linear interpolation.
    zc = dd.get_z()
    target = 0.5 * (zc[10] + zc[11])
    out = dd.get_var("bz", 0, z=target, method="linear", centered=True, units="bifrost")
    assert out.shape == (24, 24)

    # A non-uniform synthetic z grid is resampled with the same number of points.
    n = 24
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32) ** 1.2
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    r = np.ones((n, n, n), dtype=np.float32)
    arrays = {
        "r": r,
        "e": r,
        "px": r,
        "py": r,
        "pz": r,
        "bx": xx.astype(np.float32),
        "by": yy.astype(np.float32),
        "bz": zz.astype(np.float32),
    }
    dd2 = ArrayData(arrays, x, y, z)
    native = dd2.get_var("r", 0, x=3.0, units="bifrost")
    uniform = dd2.get_var("r", 0, x=3.0, uniform_z=True, units="bifrost")
    assert uniform.shape == native.shape
    assert np.allclose(uniform, 1.0)


def test_mesh_api_uses_single_location_convention():
    dd = make_data()

    assert np.array_equal(dd.get_grid(location="center")[0], dd.get_x())
    assert np.array_equal(dd.get_grid(location="yface")[0], dd.get_x())
    assert not np.array_equal(dd.get_xf(), dd.get_x())

    # Physical Y/Z coordinates are reversed and sign-flipped relative to Bifrost.
    # Public Y is additionally translated so that the centered grid starts at 0.
    y_unshifted = -dd.get_y(br=True)[::-1]
    assert np.allclose(dd.get_y(), y_unshifted - y_unshifted[0])
    assert np.isclose(dd.get_y()[0], 0.0)
    assert np.allclose(dd.get_z(), -dd.get_z(br=True)[::-1])

    assert np.isclose(dd.get_dx(), np.mean(dd.get_dx1d()))
    assert np.isclose(dd.get_dy(), np.mean(dd.get_dy1d()))
    assert np.isclose(dd.get_dz(), np.mean(dd.get_dz1d()))

    x_bz, y_bz, z_bz = dd.get_grid(location="zface")
    assert np.array_equal(x_bz, dd.get_x())
    assert np.array_equal(y_bz, dd.get_y())
    assert np.array_equal(z_bz, dd.get_zf())


def test_public_stagger_matches_centered_getvar():
    dd = make_data()

    for var, opname in (("bx", "xup"), ("by", "yup"), ("bz", "zup")):
        raw = dd.get_var(var, 0, units="bifrost")
        manual = getattr(dd.stagger, opname)(raw)
        automatic = dd.get_var(var, 0, centered=True, units="bifrost")
        assert np.allclose(manual, automatic, atol=2e-5)


def test_public_stagger_derivative_signs_in_physical_orientation():
    dd = make_data()
    shape = dd._meta(0).shape

    xface = dd.get_xf()
    yface = dd.get_yf()
    zface = dd.get_zf()

    fx = np.broadcast_to(xface[:, None, None], shape).astype(np.float32).copy()
    fy = np.broadcast_to(yface[None, :, None], shape).astype(np.float32).copy()
    fz = np.broadcast_to(zface[None, None, :], shape).astype(np.float32).copy()

    dfx = dd.stagger.ddxup(fx)
    dfy = dd.stagger.ddyup(fy)
    dfz = dd.stagger.ddzup(fz)

    interior = (slice(4, -4), slice(4, -4), slice(4, -4))
    assert np.allclose(dfx[interior], 1.0, atol=2e-4)
    assert np.allclose(dfy[interior], 1.0, atol=2e-4)
    assert np.allclose(dfz[interior], 1.0, atol=2e-4)


def test_public_y_coordinate_selection_uses_shifted_origin():
    dd = make_data()

    # Build a center-located diagnostic whose native value is the Bifrost Y
    # index.  The public coordinate y=7 corresponds to output index iy=7 even
    # though internally that is native index ny-1-7.
    n = 24
    probe = np.broadcast_to(
        np.arange(n, dtype=np.float32)[None, :, None], (n, n, n)
    ).copy()
    dd.arrays["probe_y"] = probe
    dd.register_variable("probe_y", location="center", unit_type="temperature")

    assert np.isclose(dd.get_y()[0], 0.0)
    by_coord = dd.get_var("probe_y", 0, y=7.0, units="bifrost")
    by_index = dd.get_var("probe_y", 0, iy=7, units="bifrost")
    assert np.array_equal(by_coord, by_index)


def test_total_electric_field_is_stored_only_but_ideal_field_is_derived():
    n = 24
    x = np.arange(n, dtype=np.float32)
    y = np.arange(n, dtype=np.float32)
    z = np.arange(n, dtype=np.float32)
    shape = (n, n, n)
    r = np.full(shape, 2.0, dtype=np.float32)
    arrays = {
        "r": r,
        "e": np.full(shape, 6.0, dtype=np.float32),
        "px": r * 1.0,
        "py": r * 2.0,
        "pz": r * 3.0,
        "bx": np.full(shape, 4.0, dtype=np.float32),
        "by": np.full(shape, 5.0, dtype=np.float32),
        "bz": np.full(shape, 6.0, dtype=np.float32),
    }
    dd = ArrayData(arrays, x, y, z)

    import pytest
    from bifrost_proto.core import UnknownVariableError

    with pytest.raises(UnknownVariableError):
        dd.get_var("ex", 0, units="bifrost", br=True)

    interior = (slice(5, -5), slice(5, -5), slice(5, -5))
    ex = dd.get_var("ex_ideal", 0, units="bifrost", br=True)
    ey = dd.get_var("ey_ideal", 0, units="bifrost", br=True)
    ez = dd.get_var("ez_ideal", 0, units="bifrost", br=True)

    # -u x B for u=(1,2,3), B=(4,5,6) is (3,-6,3).
    assert np.allclose(ex[interior], 3.0, atol=2e-5)
    assert np.allclose(ey[interior], -6.0, atol=2e-5)
    assert np.allclose(ez[interior], 3.0, atol=2e-5)

    # If the total E field is stored, the raw value wins automatically.
    dd.arrays["ex"] = np.full(shape, 11.0, dtype=np.float32)
    stored = dd.get_var("ex", 0, units="bifrost", br=True)
    assert np.all(stored == 11.0)
