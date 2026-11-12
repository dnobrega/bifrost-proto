"""v0.1.46: signed-velocity palettes, direct lengths, safe movie publication."""

from __future__ import annotations

import errno
import logging
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from matplotlib import colormaps
from matplotlib.colors import LogNorm

from bifrost_proto import BifrostData  # noqa: F401 - populate registry
from bifrost_proto import quantities
from bifrost_proto.movie import _new_partial_movie, _publish_movie
from bifrost_proto.plot_defaults import get_plot_preset
from bifrost_proto.registry import GLOBAL_REGISTRY
from bifrost_proto.units import BifrostUnits
from test_v144_plot_presets_colormaps import DummyData


VELOCITY_COMPONENTS = (
    "u_x",
    "u_y",
    "u_z",
    "uparb_x",
    "uparb_y",
    "uparb_z",
    "uperpb_x",
    "uperpb_y",
    "uperpb_z",
)
LENGTHS = {
    "ldivb": ("modb", "divb", quantities.get_ldivb),
    "lcurlb": ("modb", "modcurlb", quantities.get_lcurlb),
    "ldivu": ("modu", "divu", quantities.get_ldivu),
    "lcurlu": ("modu", "modcurlu", quantities.get_lcurlu),
}


def test_velocity_colormaps_and_signed_los_midpoint():
    dd = DummyData()
    assert np.allclose(colormaps["bwr_r"](0), (1, 0, 0, 1))
    assert np.allclose(colormaps["bwr_r"](1.0), (0, 0, 1, 1))
    for name in VELOCITY_COMPONENTS:
        cmap, norm = get_plot_preset(dd, name)
        assert cmap == "bwr_r"
        assert norm.vmin == -10 and norm.vmax == 10
        cmap_los, norm_los = get_plot_preset(dd, name, los="z")
        assert cmap_los == "bwr_r"
        assert norm_los(np.array([-2.0, 0.0, 2.0]))[1] == pytest.approx(0.5)
    assert get_plot_preset(dd, "uperpb")[0] == "jet"
    assert get_plot_preset(dd, "uparb")[0] == "RdBu_r"


class FakeGetter:
    def __init__(self, **values):
        self.values = values

    def _getvar(self, name, snap, selection, context):
        return self.values[name]


@pytest.mark.parametrize(
    "name,numerator,denominator,getter",
    [(name, *inputs) for name, inputs in LENGTHS.items()],
)
def test_characteristic_length_values_and_zero_limits(
    name, numerator, denominator, getter
):
    values = {
        numerator: np.array([4, 4, 0, 0], dtype="f4"),
        denominator: np.array(
            [-2 if name.startswith("ldiv") else 2, 0, 3, 0], dtype="f4"
        ),
    }
    # The magnitude of a curl is non-negative; divergence can be signed.
    data = FakeGetter(**values)
    with np.errstate(divide="raise", invalid="raise"):
        got = getter(data, 0, None, None)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got[:3], [2, np.inf, 0])
    assert np.isnan(got[3])
    spec = GLOBAL_REGISTRY[name]
    assert spec.unit_type == "length"
    assert "length" in spec.description.lower()


def test_length_unit_factors_and_latex_labels():
    units = BifrostUnits(u_l=2.5e8)
    assert units.factor("length", "bifrost") == pytest.approx(1.0)
    assert units.factor("length", "solar") == pytest.approx(2.5)
    assert units.factor("length", "cgs") == pytest.approx(2.5e8)
    assert units.factor("length", "si") == pytest.approx(2.5e6)
    for name in LENGTHS:
        assert units.label("length", "solar") == "Mm"
        from bifrost_proto.labels import latex_varname

        label = latex_varname(name)
        assert "nabla" in label and "mathbf" in label
    # The norm now scales as length rather than inverse length.
    cmap, norm = get_plot_preset(DummyData(), "lcurlb")
    assert cmap == "green_terrain_r" and isinstance(norm, LogNorm)


def test_publication_interrupted_copy_keeps_source(tmp_path, monkeypatch):
    destination = tmp_path / "interrupt.mp4"
    partial = _new_partial_movie(destination)
    partial.write_bytes(b"complete encoded data")

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "links unsupported")

    def interrupted(source, dest, length):
        dest.write(b"incomplete")
        raise KeyboardInterrupt

    monkeypatch.setattr(os, "link", unsupported)
    monkeypatch.setattr(shutil, "copyfileobj", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _publish_movie(partial, destination, overwrite=False)
    assert not destination.exists()
    assert partial.read_bytes() == b"complete encoded data"
    partial.unlink()


def test_published_movie_cleanup_failure_is_logged_not_raised(
    tmp_path, monkeypatch, caplog
):
    destination = tmp_path / "published.mp4"
    partial = _new_partial_movie(destination)
    partial.write_bytes(b"complete encoded data")
    original_unlink = Path.unlink

    def fail_cleanup(self, *args, **kwargs):
        if self == partial:
            raise PermissionError(errno.EACCES, "cannot remove temporary")
        return original_unlink(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_cleanup)
        with caplog.at_level(logging.WARNING, logger="bifrost_proto.movie"):
            _publish_movie(partial, destination, overwrite=False)
    assert destination.read_bytes() == b"complete encoded data"
    assert partial.exists()
    assert str(partial) in caplog.text and str(destination) in caplog.text
    partial.unlink()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_movie_keyboard_interrupt_during_publication_keeps_encoded(
    tmp_path, monkeypatch
):
    from test_v142_review_fixes import _run

    dd = _run(tmp_path)

    def stop_publish(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("bifrost_proto.movie._publish_movie", stop_publish)
    with pytest.raises(KeyboardInterrupt):
        dd.movie(
            "r",
            [0],
            y=1.0,
            cbar=False,
            title=False,
            progress=False,
            output="stopped.mp4",
            dpi=72,
        )
    dest = tmp_path / "movies" / "stopped.mp4"
    assert not dest.exists()
    temps = list(dest.parent.glob(".*.partial.mp4"))
    assert len(temps) == 1 and temps[0].stat().st_size > 100
    temps[0].unlink()
