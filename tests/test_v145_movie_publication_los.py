"""v0.1.45: ffmpeg publication, permissions, centered LOS, cmap compatibility."""

from __future__ import annotations

import errno
import os
import shutil
import stat
import warnings

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import CenteredNorm, LogNorm
from test_engine import make_data
from test_v144_plot_presets_colormaps import DummyData

from bifrost_proto.colormaps import make_colormaps
from bifrost_proto.movie import _new_partial_movie, _publish_movie
from bifrost_proto.plot_defaults import get_plot_preset


def test_colormaps_are_valid_with_warnings_as_errors():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        maps = make_colormaps()
    assert len(maps) == 24
    assert len(maps["aia94"].colors) == 256


def test_b_h_logarithmic():
    cmap, norm = get_plot_preset(DummyData(), "b_h")
    assert cmap == "jet"
    assert isinstance(norm, LogNorm)
    assert (norm.vmin, norm.vmax) == (1, 1000)


@pytest.mark.parametrize("var", ["u_z", "j_z", "p_x", "lf_z", "hpg_grad"])
def test_signed_divergent_los_is_centered_at_zero(var):
    cmap, norm = get_plot_preset(DummyData(), var, los="z")
    assert cmap
    assert isinstance(norm, CenteredNorm)
    np.testing.assert_allclose(norm(np.array([-17.0, 0.0, 17.0])), [0.0, 0.5, 1.0])


def test_non_diverging_and_positive_los_keep_autoscale():
    dd = DummyData()
    # b_z has an explicitly user-selected Greys_r *cut* preset. Preserve it.
    assert get_plot_preset(dd, "b_z", los="z") == ("Greys_r", None)
    assert get_plot_preset(dd, "beta", los="z") == ("BrBG", None)
    assert get_plot_preset(dd, "r", los="z") == ("terrain_r", None)


def test_real_signed_los_uses_zero_midpoint_and_explicit_norm_override():
    dd = make_data(n=12)
    fig, _, im = dd.plot("u_z", 0, los="z", cbar=False, title=False)
    try:
        assert isinstance(im.norm, CenteredNorm)
        assert im.norm(0) == pytest.approx(0.5)
    finally:
        plt.close(fig)
    manual = LogNorm(1e-3, 1e4)
    fig, _, im = dd.plot("u_z", 0, los="z", norm=manual, cbar=False, title=False)
    try:
        assert im.norm is manual
    finally:
        plt.close(fig)


def test_partial_movie_respects_umask(tmp_path):
    target = tmp_path / "demo.mp4"
    old = os.umask(0o022)
    try:
        partial = _new_partial_movie(target)
    finally:
        os.umask(old)
    assert stat.S_IMODE(partial.stat().st_mode) == 0o644
    partial.write_bytes(b"encoded film")
    _publish_movie(partial, target, overwrite=False)
    assert target.read_bytes() == b"encoded film"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert not partial.exists()


def test_partial_movie_respects_restrictive_umask(tmp_path):
    target = tmp_path / "private.mp4"
    old = os.umask(0o077)
    try:
        partial = _new_partial_movie(target)
    finally:
        os.umask(old)
    assert stat.S_IMODE(partial.stat().st_mode) == 0o600
    partial.unlink()


def test_link_unsupported_falls_back_without_clobber(tmp_path, monkeypatch):
    target = tmp_path / "fallback.mp4"
    partial = _new_partial_movie(target)
    partial.write_bytes(b"complete MP4")

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "filesystem does not support hardlinks")

    monkeypatch.setattr(os, "link", unsupported)
    _publish_movie(partial, target, overwrite=False)
    assert target.read_bytes() == b"complete MP4"
    assert not partial.exists()
    second_partial = _new_partial_movie(target)
    second_partial.write_bytes(b"different")
    with pytest.raises(FileExistsError):
        _publish_movie(second_partial, target, overwrite=False)
    assert target.read_bytes() == b"complete MP4"
    second_partial.unlink()


def test_failed_fallback_preserves_completed_partial(tmp_path, monkeypatch):
    target = tmp_path / "broken.mp4"
    partial = _new_partial_movie(target)
    partial.write_bytes(b"finished movie, should survive")

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "unsupported")

    def interrupted_copy(src, dst, length):
        dst.write(b"incomplete")
        raise OSError(errno.ENOSPC, "simulated full disk")

    monkeypatch.setattr(os, "link", unsupported)
    monkeypatch.setattr(shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(OSError, match="simulated full disk"):
        _publish_movie(partial, target, overwrite=False)
    assert not target.exists()
    assert partial.read_bytes() == b"finished movie, should survive"
    partial.unlink()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_actual_mp4_permissions_and_unsupported_link(tmp_path, monkeypatch):
    from test_v142_review_fixes import _run

    dd = _run(tmp_path)

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "emulated SMB mount")

    monkeypatch.setattr(os, "link", unsupported)
    old = os.umask(0o022)
    try:
        target = dd.movie(
            "r",
            [0],
            y=1.0,
            cbar=False,
            title=False,
            progress=False,
            output="perms.mp4",
            dpi=72,
        )
    finally:
        os.umask(old)
    assert target.is_file() and target.stat().st_size > 100
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
    assert not list(tmp_path.glob(".*.partial.mp4"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_actual_movie_preserves_encoded_file_if_publish_fails(tmp_path, monkeypatch):
    from test_v142_review_fixes import _run

    dd = _run(tmp_path)

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "emulated FUSE mount")

    def out_of_space(*args, **kwargs):
        raise OSError(errno.ENOSPC, "emulated disk full")

    monkeypatch.setattr(os, "link", unsupported)
    monkeypatch.setattr(shutil, "copyfileobj", out_of_space)
    with pytest.raises(OSError, match="Encoded movie preserved at"):
        dd.movie(
            "r",
            [0],
            y=1.0,
            cbar=False,
            title=False,
            progress=False,
            output="will-fail.mp4",
            dpi=72,
        )
    assert not (tmp_path / "movies" / "will-fail.mp4").exists()
    encoded = list((tmp_path / "movies").glob(".*.partial.mp4"))
    assert len(encoded) == 1 and encoded[0].stat().st_size > 100
    encoded[0].unlink()


def test_overwrite_uses_original_temporary_mode(tmp_path):
    target = tmp_path / "present.mp4"
    target.write_bytes(b"old film")
    old = os.umask(0o022)
    try:
        partial = _new_partial_movie(target)
    finally:
        os.umask(old)
    partial.write_bytes(b"new film")
    _publish_movie(partial, target, overwrite=True)
    assert target.read_bytes() == b"new film"
    assert stat.S_IMODE(target.stat().st_mode) == 0o644
