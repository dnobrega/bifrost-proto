"""Fast-movie regressions; synthetic input, no Bifrost snapshot files required."""

import shutil
import subprocess

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import LogNorm

from bifrost_proto.movie import (
    _coordinate_label,
    _output_path,
    _physical_cut,
    _snapshots_and_step,
    movie,
)


class FakeData:
    def __init__(self, folder):
        self.fdir = str(folder)
        self.calls = []

    def get_var(
        self,
        var,
        snap,
        *,
        x=None,
        y=None,
        z=None,
        uniform_z=False,
        units="solar",
        low_memory=None,
    ):
        assert (
            sum(
                not isinstance(val, slice)
                for _, val in (("x", x), ("y", y), ("z", z))
                if val is not None
            )
            == 1
        )
        axis = next(
            a
            for a, v in (("x", x), ("y", y), ("z", z))
            if v is not None and not isinstance(v, slice)
        )
        self.calls.append(("get_var", int(snap), axis, uniform_z, units))
        if axis == "z":
            a = np.arange(42).reshape(7, 6) + 1.0
        else:
            a = np.arange(42).reshape(7, 6) + 1.0
        return a + snap * 0.25

    def get_t(self, snap, *, units="solar"):
        assert units == "solar"
        self.calls.append(("get_t", int(snap)))
        return 11.0 * snap

    def plot(
        self,
        var,
        snap,
        *,
        x=None,
        y=None,
        z=None,
        cmap=None,
        norm=None,
        aspect="equal",
        cbar=True,
        cbar_position="top",
        figsize=None,
        title=True,
        minor_ticks=True,
        units="solar",
        low_memory=None,
    ):
        plane = self.get_var(
            var,
            snap,
            x=x,
            y=y,
            z=z,
            uniform_z=(z is None),
            units=units,
            low_memory=low_memory,
        )
        self.calls.append(("plot", int(snap)))
        fig, ax = plt.subplots(figsize=figsize or (5, 3))
        im = ax.imshow(plane.T, origin="lower", cmap=cmap, norm=norm, aspect=aspect)
        ax.set_xlabel("x (Mm)")
        ax.set_ylabel("z (Mm)")
        if title:
            ax.set_title("Y=3.25 Mm", loc="left")
            ax.set_title(f"t={self.get_t(snap) / 60:.2f} min", loc="right")
        if cbar:
            fig.colorbar(im, ax=ax)
        return fig, ax, im


def test_snapshot_validation():
    assert _snapshots_and_step(np.arange(0, 231, 2))[1] == 2
    assert _snapshots_and_step([7])[1] == 1
    for seq in ([], [0, 2, 5], [0, 2, 2], [4, 3, 2], [-1, 0]):
        with pytest.raises(ValueError):
            _snapshots_and_step(seq)
    with pytest.raises(TypeError):
        _snapshots_and_step([0.0, 2.0])


def test_cut_and_name(tmp_path):
    data = FakeData(tmp_path)
    seq, step = _snapshots_and_step(np.arange(0, 231, 2))
    assert _physical_cut(x=None, y=25.5, z=None) == ("y", 25.5, "XZ")
    path = _output_path(data, "modb", seq, step, "y", 25.5, "XZ", None, None)
    assert path == tmp_path / "movies" / "modb_0000_0230_0002_XZ_y=25.50.mp4"
    explicit = tmp_path / "out.mp4"
    assert (
        _output_path(data, "modb", seq, step, "y", 1.0, "XZ", explicit, None)
        == explicit
    )
    assert (
        _output_path(data, "modb", seq, step, "y", 1.0, "XZ", "v.mp4", "other")
        == tmp_path / "other" / "v.mp4"
    )


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
@pytest.mark.parametrize("axis, coordinate", [("x", 2.2), ("y", 3.25), ("z", 0.5)])
def test_mp4_real_frames_title_and_uniform_z(tmp_path, axis, coordinate):
    data = FakeData(tmp_path)
    filename = tmp_path / f"test_{axis}.mp4"
    result = movie(
        data,
        "modb",
        np.arange(0, 6, 2),
        **{axis: coordinate},
        cmap="turbo",
        norm=LogNorm(vmin=1, vmax=100),
        fps=12,
        dpi=100,
        output=filename,
        progress=False,
    )
    assert result == filename
    assert filename.exists() and filename.stat().st_size > 300
    req = [x for x in data.calls if x[0] == "get_var"]
    assert len(req) == 3  # First frame reused from initial plot, not reread.
    assert all(x[3] == (axis != "z") for x in req)
    time_reads = [x[1] for x in data.calls if x[0] == "get_t"]
    assert time_reads == [0, 2, 4]
    # Check encoded frame count via ffprobe if available.
    if shutil.which("ffprobe"):
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=nokey=1:noprint_wrappers=1",
            str(filename),
        ]
        assert int(subprocess.check_output(cmd, text=True).strip()) == 3
    with pytest.raises(FileExistsError):
        movie(
            data,
            "modb",
            [0, 2, 4],
            **{axis: coordinate},
            output=filename,
            progress=False,
        )


def test_validation_before_creating_dirs(tmp_path):
    dd = FakeData(tmp_path)
    with pytest.raises(ValueError, match="equispaced"):
        movie(dd, "modb", [0, 2, 5], y=25.5)
    with pytest.raises(ValueError, match="exactly one"):
        movie(dd, "modb", [0, 2], x=1, y=2)
    with pytest.raises(ValueError, match="exactly one scalar"):
        movie(dd, "modb", [0, 2], y=slice(1, 2))
    with pytest.raises(ValueError, match="fps"):
        movie(dd, "modb", [0, 2], y=25.5, fps=0)
    assert not (tmp_path / "movies").exists()


def test_filename_uses_two_decimals_for_all_coordinates():
    assert _coordinate_label(25.5) == "25.50"
    assert _coordinate_label(25.505) == "25.50"
    assert _coordinate_label(-3.1234) == "-3.12"
    assert _coordinate_label(float(np.float32(51.1))) == "51.10"
    assert _coordinate_label(-0.0002) == "0.00"


def test_movie_roi_filename_and_validation(tmp_path):
    """Movie names distinguish cropped regions but retain old full-plane names."""
    data = FakeData(tmp_path)
    seq, step = _snapshots_and_step([0, 2, 4])
    selections = {"x": slice(5.0, 20.0), "y": 25.5, "z": slice(0.0, 10.0)}
    path = _output_path(data, "tg", seq, step, "y", 25.5, "XZ", None, None, selections)
    assert path.name == "tg_0000_0004_0002_XZ_y=25.50_x=5.00-20.00_z=0.00-10.00.mp4"
    assert _physical_cut(**selections) == ("y", 25.5, "XZ")
    with pytest.raises(ValueError, match="slice.step"):
        movie(data, "tg", [0, 2], y=25.5, x=slice(0, 4, 1))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_movie_roi_passed_to_first_plot_and_frames(tmp_path):
    """A crop must be forwarded to both initial plot and every movie frame."""
    data = FakeData(tmp_path)
    original = data.get_var
    selected = []

    def spy(var, snap, **kwargs):
        selected.append(
            (snap, kwargs["x"], kwargs["y"], kwargs["z"], kwargs["low_memory"])
        )
        return original(var, snap, **kwargs)

    data.get_var = spy
    roi = {"x": slice(5, 20), "y": 25.5, "z": slice(0, 10)}
    path = movie(
        data,
        "tg",
        np.arange(0, 6, 2),
        **roi,
        low_memory=True,
        output=tmp_path / "cropped.mp4",
        progress=False,
        cbar=False,
    )
    assert path.exists()
    assert [s[0] for s in selected] == [0, 2, 4]
    assert all(s[1:] == (roi["x"], roi["y"], roi["z"], True) for s in selected)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg unavailable")
def test_movie_roi_with_bifrost_data_reader(tmp_path):
    """Full integration: plot/get_var/get_grid select identical real-reader ROI."""
    from test_engine import make_data

    reader = make_data(n=20)
    reader.fdir = str(tmp_path)
    reader._meta_cache[0].params["tsnap"] = 3.0
    x, y, z = reader.get_grid()
    kwargs = {
        "y": float(y[9]),
        "x": slice(float(x[4]), float(x[14])),
        "z": slice(float(z[3]), float(z[15])),
    }
    target = tmp_path / "bifrost_real_roi.mp4"
    result = movie(
        reader,
        "r",
        [0],
        **kwargs,
        output=target,
        dpi=72,
        fps=12,
        figsize=(4, 3),
        progress=False,
        low_memory=True,
    )
    assert result == target
    assert target.stat().st_size > 400
    if shutil.which("ffprobe"):
        result = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                str(target),
            ],
            text=True,
        )
        assert int(result.strip()) == 1
