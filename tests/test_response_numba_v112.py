import numpy as np
from test_engine import make_data
from test_responses_v19 import attach_composition, write_response

from bifrost_proto.responses import InstrumentResponseTable


def _make_response(tmp_path, *, fill_value=0.0):
    logne = np.linspace(8.0, 11.0, 7)
    logt = np.linspace(4.0, 8.0, 9)
    table = (
        0.25 * logne[:, None]
        + 1.75 * logt[None, :]
        + 0.1 * logne[:, None] * logt[None, :]
    )
    write_response(
        tmp_path,
        "aia193",
        table,
        logne,
        logt,
        fill_value=fill_value,
    )
    return InstrumentResponseTable("aia193", base_dir=tmp_path)


def _reference(response, tg, ne, nh):
    return (ne * nh * response.evaluate(tg, ne)).astype(np.float32, copy=False)


def test_fused_response_matches_numpy_reference_c_and_f(tmp_path):
    response = _make_response(tmp_path)
    rng = np.random.default_rng(1234)

    for order in ("C", "F"):
        logt = rng.uniform(4.05, 7.95, size=(17, 13, 11))
        logne = rng.uniform(8.05, 10.95, size=(17, 13, 11))
        tg = np.array(10.0**logt, dtype=np.float32, order=order)
        ne = np.array(10.0**logne, dtype=np.float32, order=order)
        nh = np.array(
            rng.uniform(1e8, 1e12, size=tg.shape), dtype=np.float32, order=order
        )

        expected = _reference(response, tg, ne, nh)
        got = response.evaluate_emissivity(tg, ne, nh)

        assert got.dtype == np.float32
        np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0.0)


def test_fused_response_strided_view_matches_reference_without_modifying_inputs(
    tmp_path,
):
    response = _make_response(tmp_path)
    rng = np.random.default_rng(2)
    shape = (28, 24, 20)
    tg0 = np.asfortranarray(10.0 ** rng.uniform(4.1, 7.9, size=shape), dtype=np.float32)
    ne0 = np.asfortranarray(
        10.0 ** rng.uniform(8.1, 10.9, size=shape), dtype=np.float32
    )
    nh0 = np.asfortranarray(rng.uniform(1e8, 1e12, size=shape), dtype=np.float32)

    tg = tg0[1:-1:2, 2:-2:2, 1:-1:2]
    ne = ne0[1:-1:2, 2:-2:2, 1:-1:2]
    nh = nh0[1:-1:2, 2:-2:2, 1:-1:2]
    assert not tg.flags.c_contiguous and not tg.flags.f_contiguous

    tg_before = tg.copy()
    ne_before = ne.copy()
    nh_before = nh.copy()

    expected = _reference(response, tg, ne, nh)
    got = response.evaluate_emissivity(tg, ne, nh)

    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0.0)
    np.testing.assert_array_equal(tg, tg_before)
    np.testing.assert_array_equal(ne, ne_before)
    np.testing.assert_array_equal(nh, nh_before)


def test_fused_response_preserves_fill_semantics(tmp_path):
    response = _make_response(tmp_path, fill_value=-3.0)
    tg = np.array([1e6, 1e2, np.nan, 1e6], dtype=np.float32)
    ne = np.array([1e9, 1e9, 1e9, -2.0], dtype=np.float32)
    nh = np.array([2e9, 3e9, 4e9, 5e9], dtype=np.float32)

    expected = _reference(response, tg, ne, nh)
    got = response.evaluate_emissivity(tg, ne, nh)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0.0, equal_nan=True)


def test_nonuniform_axes_fall_back_to_general_evaluate(tmp_path):
    logne = np.array([8.0, 8.7, 10.0])
    logt = np.array([5.0, 5.2, 6.5, 8.0])
    table = 2.0 * logne[:, None] + 3.0 * logt[None, :] + 4.0
    write_response(tmp_path, "aia193", table, logne, logt)
    response = InstrumentResponseTable("aia193", base_dir=tmp_path)

    tg = np.array([10**5.1, 10**7.0], dtype=np.float32)
    ne = np.array([10**8.4, 10**9.5], dtype=np.float32)
    nh = np.array([1e10, 2e10], dtype=np.float32)
    expected = _reference(response, tg, ne, nh)
    got = response.evaluate_emissivity(tg, ne, nh)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0.0)


def test_instrument_recipe_uses_fused_response_method(tmp_path, monkeypatch):
    dd = make_data()
    # Put the synthetic test cube at non-negative physical heights so the
    # instrument validity mask does not alter tests unrelated to masking.
    dd._meta_cache[0].z[:] = -np.arange(dd._meta_cache[0].shape[2], dtype=np.float32)
    attach_composition(dd, tmp_path)
    shape = dd.arrays["r"].shape
    dd.arrays["ne"] = np.full(shape, 1.0e9, dtype=np.float32, order="F")
    dd.arrays["tg"] = np.full(shape, 1.0e6, dtype=np.float32, order="F")

    response_root = tmp_path / "responses"
    write_response(
        response_root,
        "aia193",
        np.full((2, 2), 2.5e-27),
        [8.0, 10.0],
        [5.0, 7.0],
    )
    dd.responses_dir = str(response_root)
    dd._response_cache = {}

    response = dd._instrument_response("aia193")

    def fail_evaluate(*args, **kwargs):
        raise AssertionError(
            "slow evaluate() path should not be used for uniform tables"
        )

    monkeypatch.setattr(response, "evaluate", fail_evaluate)
    out = dd.get_var("aia193", 0, units="solar", br=True)
    assert np.all(np.isfinite(out))
    assert np.all(out > 0.0)


def test_float32_quantized_uniform_axes_use_fast_path_without_hardcoded_origin(
    tmp_path, monkeypatch
):
    # Deliberately use non-round origins so a hidden 7.0/4.0 assumption would fail.
    logne32 = np.asarray(8.35 + 0.05 * np.arange(41), dtype=np.float32)
    logt32 = np.asarray(4.65 + 0.05 * np.arange(61), dtype=np.float32)
    table = 0.2 * logne32[:, None].astype(np.float64) + 1.3 * logt32[None, :].astype(
        np.float64
    )
    write_response(tmp_path, "aia193", table, logne32, logt32)
    response = InstrumentResponseTable("aia193", base_dir=tmp_path)

    assert response._uniform_t
    assert response._uniform_ne
    assert response.logt[0] == float(logt32[0])
    assert response.logne[0] == float(logne32[0])
    assert response._dlogt == float(np.diff(logt32.astype(np.float64))[0])
    assert response._dlogne == float(np.diff(logne32.astype(np.float64))[0])

    # Compute the slow reference before disabling evaluate().
    tg = np.asfortranarray(
        10.0 ** np.array([[[4.70, 5.25], [6.10, 7.20]]], dtype=np.float32)
    )
    ne = np.asfortranarray(
        10.0 ** np.array([[[8.40, 8.95], [9.50, 10.10]]], dtype=np.float32)
    )
    nh = np.asfortranarray(np.full(tg.shape, 2.0e9, dtype=np.float32))
    expected = _reference(response, tg, ne, nh)

    def fail_evaluate(*args, **kwargs):
        raise AssertionError(
            "float32-quantized uniform axes must use the fused fast path"
        )

    monkeypatch.setattr(response, "evaluate", fail_evaluate)
    got = response.evaluate_emissivity(tg, ne, nh)
    np.testing.assert_allclose(got, expected, rtol=3e-5, atol=0.0)


def test_genuinely_nonuniform_float32_axis_is_not_misclassified(tmp_path):
    logne = np.asarray([8.35, 8.40, 8.45, 8.53, 8.58], dtype=np.float32)
    logt = np.asarray(4.65 + 0.05 * np.arange(7), dtype=np.float32)
    table = logne[:, None].astype(np.float64) + logt[None, :].astype(np.float64)
    write_response(tmp_path, "aia193", table, logne, logt)
    response = InstrumentResponseTable("aia193", base_dir=tmp_path)

    assert response._uniform_t
    assert not response._uniform_ne


def test_fused_response_negative_stride_view_matches_reference(tmp_path):
    """Instrument emissivity interpolation must accept reversed strided cubes."""
    response = _make_response(tmp_path)
    rng = np.random.default_rng(20260924)
    shape = (17, 15, 13)
    tg0 = np.asfortranarray(10.0 ** rng.uniform(4.1, 7.9, size=shape), dtype=np.float32)
    ne0 = np.asfortranarray(
        10.0 ** rng.uniform(8.1, 10.9, size=shape), dtype=np.float32
    )
    nh0 = np.asfortranarray(rng.uniform(1e8, 1e12, size=shape), dtype=np.float32)

    tg = tg0[:, :, ::-1]
    ne = ne0[:, :, ::-1]
    nh = nh0[:, :, ::-1]
    assert not tg.flags.c_contiguous and not tg.flags.f_contiguous
    assert tg.strides[2] < 0

    expected = _reference(response, tg, ne, nh)
    got = response.evaluate_emissivity(tg, ne, nh)
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=0.0)
