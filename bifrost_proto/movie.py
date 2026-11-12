"""Fast MP4 movies of scalar physical cuts, reusing the existing plotting API.

This module intentionally delegates all spatial geometry, registered units,
normalization, and Z remeshing to :meth:`BifrostData.plot` and
:meth:`BifrostData.get_var` rather than implementing a second plotting stack.
"""

from __future__ import annotations

import logging
import os
import secrets
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.image import AxesImage

from .plot_naming import _coordinate_label, _roi_suffix

_LOG = logging.getLogger(__name__)


def _snapshots_and_step(snapshots: Any) -> tuple[np.ndarray, int]:
    """Require a nonempty 1-D, strictly increasing, equispaced integer sequence."""
    seq = np.asarray(snapshots)
    if seq.ndim != 1 or seq.size == 0:
        raise ValueError("snapshots must be a nonempty one-dimensional sequence")
    if not np.issubdtype(seq.dtype, np.integer):
        raise TypeError("snapshots must contain integers, e.g. np.arange(0, 231, 2)")
    if np.any(seq < 0):
        raise ValueError("snapshots must be nonnegative")
    if seq.size == 1:
        step = 1  # A one-frame movie has no measurable cadence.
    else:
        gaps = np.diff(seq.astype(np.int64))
        if np.any(gaps <= 0) or np.any(gaps != gaps[0]):
            raise ValueError("snapshots must be strictly increasing and equispaced")
        step = int(gaps[0])
    return seq.astype(np.int64), step


def _physical_cut(*, x: Any, y: Any, z: Any) -> tuple[str, float, str]:
    """Get the fixed physical plane, allowing ROI slices on the other axes."""
    from .plotting import _parse_cut

    axis, coordinate, _selection = _parse_cut(x=x, y=y, z=z)
    plane = {"x": "YZ", "y": "XZ", "z": "XY"}[axis]
    return axis, coordinate, plane


def _output_path(
    data: Any,
    var: str,
    seq: np.ndarray,
    step: int,
    axis: str,
    coordinate: float,
    plane: str,
    output: Any,
    output_dir: Any,
    selections: dict[str, Any] | None = None,
    los: str | None = None,
) -> Path:
    """Resolve a movie path under the simulation, including ROI if requested."""
    run_dir = Path(data.fdir).expanduser().resolve()
    directory = (
        run_dir / "movies" if output_dir is None else Path(output_dir).expanduser()
    )
    if not directory.is_absolute():
        directory = run_dir / directory
    if output is None:
        mode = (
            f"los={los}"
            if los is not None
            else f"{axis}={_coordinate_label(coordinate)}"
        )
        name = (
            f"{var}_{int(seq[0]):04d}_{int(seq[-1]):04d}_{step:04d}_"
            f"{plane}_{mode}{_roi_suffix(axis if los is None else '', selections)}.mp4"
        )
        destination = directory / name
    else:
        candidate = Path(output).expanduser()
        destination = (
            candidate
            if candidate.is_absolute()
            else (
                run_dir / candidate
                if output_dir is None and candidate.parent != Path(".")
                else directory / candidate
            )
        )
    if destination.suffix.lower() != ".mp4":
        raise ValueError("output must be an .mp4 filename")
    return destination.resolve()


def _scan_positions(
    data: Any, var: str, snap: int, scan: str, selections: dict[str, Any], every: int
) -> np.ndarray:
    """Select original stagger-aware spatial samples, with no interpolation."""
    from .plotting import _validate_physical_selector

    if scan not in ("x", "y", "z"):
        raise ValueError("scan must be 'x', 'y', or 'z'")
    if (
        not isinstance(every, (int, np.integer))
        or isinstance(every, (bool, np.bool_))
        or every < 1
    ):
        raise ValueError("every must be a positive integer")
    for axis, selector in selections.items():
        if _validate_physical_selector(axis, selector):
            raise ValueError(
                "Spatial scans cannot have a second fixed axis; "
                "use None or physical slices for all selections"
            )

    # The selected coordinate array belongs to the registered quantity,
    # including face/edge staggering and the original nonuniform Z mesh.
    coords = data.get_grid(
        location=data.get_location(var),
        units="solar",
        br=False,
        uniform_z=False,
        **selections,
    )
    positions = np.asarray(coords["xyz".index(scan)], dtype=np.float64)[::every]
    if positions.ndim != 1 or positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError(f"scan={scan!r} selects no valid spatial positions")
    return positions


def _scan_output_path(
    data: Any,
    var: str,
    snap: int,
    scan: str,
    positions: np.ndarray,
    every: int,
    output: Any,
    output_dir: Any,
    selections: dict[str, Any],
) -> Path:
    """Resolve a scan MP4 path with its actual first/last coordinates."""
    run_dir = Path(data.fdir).expanduser().resolve()
    directory = (
        run_dir / "movies" if output_dir is None else Path(output_dir).expanduser()
    )
    if not directory.is_absolute():
        directory = run_dir / directory
    plane = {"x": "YZ", "y": "XZ", "z": "XY"}[scan]
    if output is None:
        name = (
            f"{var}_{int(snap):04d}_{plane}_scan-{scan}="
            f"{_coordinate_label(positions[0])}-{_coordinate_label(positions[-1])}"
            f"_every={every:04d}{_roi_suffix(scan, selections)}.mp4"
        )
        destination = directory / name
    else:
        candidate = Path(output).expanduser()
        destination = (
            candidate
            if candidate.is_absolute()
            else (
                run_dir / candidate
                if output_dir is None and candidate.parent != Path(".")
                else directory / candidate
            )
        )
    if destination.suffix.lower() != ".mp4":
        raise ValueError("output must be an .mp4 filename")
    return destination.resolve()


def _check_snapshot_files(data: Any, seq: np.ndarray) -> None:
    """Preflight native files; reject missing or truncated .snap before ffmpeg.

    Synthetic readers without a file_root intentionally opt out. Metadata is
    read from .idl directly, avoiding a full mesh and variable-cache warmup.
    """
    if not getattr(data, "file_root", None):
        return
    from .eos import read_idl_ascii

    missing = []
    incomplete = []
    for snap in seq:
        snap = int(snap)
        idl = Path(data._param_filename(snap))
        binary = Path(data._data_filename(snap, "snap"))
        if not idl.is_file() or not binary.is_file():
            missing.append((snap, str(idl) if not idl.is_file() else str(binary)))
            continue
        params = read_idl_ascii(str(idl))
        shape = (int(params["mx"]), int(params["my"]), int(params["mz"]))
        mhd = data._as_bool(params.get("do_mhd", True))
        expected = (
            int(np.prod(shape, dtype=np.int64))
            * (8 if mhd else 5)
            * np.dtype(data.dtype).itemsize
        )
        actual = binary.stat().st_size
        if actual < expected:
            incomplete.append((snap, str(binary), actual, expected))
    if missing:
        preview = ", ".join(f"{snap}: {path}" for snap, path in missing[:8])
        raise FileNotFoundError(
            f"Missing .idl/.snap snapshot files: {preview}"
            + (f" (and {len(missing) - 8} more)" if len(missing) > 8 else "")
        )
    if incomplete:
        preview = ", ".join(
            f"{snap}: {path} ({actual}/{expected} bytes)"
            for snap, path, actual, expected in incomplete[:8]
        )
        raise ValueError("Incomplete .snap snapshot files: " + preview)


def _figure_and_image(plot_result: Any):
    """Get the figure and image from the established scalar dd.plot contract."""
    if isinstance(plot_result, tuple) and len(plot_result) == 3:
        fig, ax, im = plot_result
    else:
        fig = getattr(plot_result, "fig", None)
        ax = getattr(plot_result, "ax", None)
        im = getattr(plot_result, "im", None)
    if fig is None or ax is None or not isinstance(im, AxesImage):
        raise TypeError(
            "movie currently requires dd.plot() to return a scalar AxesImage cut"
        )
    return fig, ax, im


def _frame_data(
    data: Any,
    var: str,
    snap: int,
    axis: str,
    selections: dict[str, Any],
    units: str,
    low_memory: bool | None,
) -> np.ndarray:
    """Load one selected plane with the exact geometry and transpose of dd.plot."""
    plane = np.squeeze(
        data.get_var(
            var,
            int(snap),
            **selections,
            uniform_z=(axis != "z"),
            units=units,
            low_memory=low_memory,
        )
    )
    if np.ndim(plane) != 2:
        raise ValueError(f"Expected a 2-D cut; got shape {np.shape(plane)}")
    return plane.T


def _contour_artists(contour_set: Any) -> list[Any]:
    """Render contour overlays on both recent and older Matplotlib versions.

    Newer ContourSet objects are Artists; older Matplotlib exposes separate
    LineCollection artists under the ``collections`` attribute.
    """
    if contour_set is None:
        return []
    if hasattr(contour_set, "draw") and hasattr(contour_set, "set_animated"):
        return [contour_set]
    return list(contour_set.collections)


def _remove_contour(contour_set: Any) -> None:
    """Remove an overlay without accumulating previous-frame collections."""
    if hasattr(contour_set, "remove"):
        contour_set.remove()
    else:
        for item in _contour_artists(contour_set):
            item.remove()


def _pipe_command(
    path: Path,
    width: int,
    height: int,
    *,
    fps: float,
    overwrite: bool,
    crf: int,
    preset: str,
) -> list[str]:
    """Build an FFmpeg command accepting raw RGBA frames on stdin."""
    return [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgba",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        str(fps),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        str(path),
    ]


def _new_partial_movie(destination: Path) -> Path:
    """Securely create a random same-directory temporary honoring process umask.

    ``tempfile.mkstemp()`` forces 0600 regardless of umask. Published MP4s
    should instead use the same 0666 & ~umask rule as PNGs/normal files.
    O_EXCL and cryptographic randomness prevent pre-creation/symlink races.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for _ in range(10):
        partial = destination.with_name(
            f".{destination.stem}.{secrets.token_hex(12)}.partial.mp4"
        )
        try:
            fd = os.open(partial, flags, 0o666)
        except FileExistsError:
            continue
        else:
            os.close(fd)
            return partial
    raise FileExistsError("Unable to allocate a unique temporary movie filename")


def _copy_no_overwrite(source: Path, destination: Path) -> None:
    """Fallback for filesystems with no hardlinks (exFAT, SMB, FUSE).

    Exclusive creation still prevents overwriting a competing writer, although
    unlike a hardlink this fallback cannot provide atomic *visibility* of all
    bytes. On I/O errors the incomplete destination is removed; the complete
    encoded source is left in place for recovery by the movie() caller.
    """
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    inode = os.fstat(fd).st_ino
    try:
        with os.fdopen(fd, "wb") as output, source.open("rb") as encoded:
            shutil.copyfileobj(encoded, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        # Only clean up the exclusive file we created, not a replacement a
        # concurrent writer might have installed in the meantime.
        try:
            if destination.stat().st_ino == inode:
                destination.unlink()
        except FileNotFoundError:
            pass
        raise


def _publish_movie(source: Path, destination: Path, *, overwrite: bool) -> None:
    """Publish an encoded MP4 without losing it if hardlinks are unsupported."""
    if overwrite:
        os.replace(source, destination)
        return
    try:
        os.link(source, destination)  # Atomic and no-clobber where supported.
    except FileExistsError:
        raise
    except OSError:
        _copy_no_overwrite(source, destination)  # Exclusive, but non-atomic.
    # The destination is already published. An unremoved temporary is not a
    # failed movie; log it instead of raising a warning (which could become
    # an exception with python -W error).
    try:
        source.unlink()
    except OSError as exc:
        _LOG.warning(
            "Published %s, but could not remove the temporary %s: %s",
            destination,
            source,
            exc,
            stacklevel=3,
        )


def movie(
    data: Any,
    var: str,
    snapshots: Any,
    *,
    x: float | slice | None = None,
    y: float | slice | None = None,
    z: float | slice | None = None,
    scan: str | None = None,
    every: int = 1,
    los: str | None = None,
    contours: dict[str, Any] | None = None,
    cmap: Any = None,
    norm: Any = None,
    aspect: str | float = "equal",
    cbar: bool = True,
    cbar_position: str = "top",
    figsize: tuple[float, float] | None = None,
    title: bool = True,
    minor_ticks: bool = True,
    units: str = "solar",
    low_memory: bool | None = None,
    fps: float = 24,
    dpi: int = 144,
    output: str | Path | None = None,
    output_dir: str | Path | None = None,
    overwrite: bool = False,
    crf: int = 18,
    preset: str = "medium",
    progress: bool = True,
) -> Path:
    """Encode a temporal or spatial MP4 using 2-D reads and direct FFmpeg piping.

    Without ``scan``, ``snapshots`` is a strictly increasing, equispaced
    sequence and exactly one spatial selector is a fixed scalar plane.
    With ``scan='x'/'y'/'z'``, ``snapshots`` is one integer snapshot;
    ``every`` selects every Nth native spatial position along the scan axis.
    Spatial slices on the other axes crop every frame. Positions follow the
    variable's face/center staggering and native Z sampling. Neither mode
    loads a 3-D cube or writes intermediate PNGs.

    ``los='x'/'y'/'z'`` animates temporal LOS integrations only, using
    memory-bounded transverse slabs with float64 integration on the native
    grid, without a full-cube read. LOS cannot be combined with ``scan`` or ``contours``.
    The optional ``contours`` mapping follows ``dd.plot`` (including
    ``centered=True`` for interpolation to cell centers); contours are
    recomputed from a 2-D read in each temporal or scan frame.

    Requires FFmpeg with libx264; writes in ``<data.fdir>/movies`` by default
    and refuses to overwrite an existing output unless explicitly requested.
    """
    selections = {"x": x, "y": y, "z": z}
    spatial = scan is not None
    if los is not None:
        if los not in ("x", "y", "z"):
            raise ValueError("los must be 'x', 'y', or 'z'")
        if spatial:
            raise ValueError("scan and los are incompatible movie modes")
        if contours is not None:
            raise ValueError("contours are not supported with LOS integration")
    if contours is not None and (
        not isinstance(contours, dict)
        or not isinstance(contours.get("var"), str)
        or not contours["var"]
    ):
        raise TypeError("contours must contain a nonempty 'var' name")
    if spatial:
        if (
            not isinstance(snapshots, (int, np.integer))
            or isinstance(snapshots, (bool, np.bool_))
            or snapshots < 0
        ):
            raise ValueError("Spatial scans require one nonnegative integer snapshot")
        snap = int(snapshots)
        positions = _scan_positions(data, var, snap, scan, selections, every)
        axis = scan
        coordinate = float(positions[0])
        frames = positions
        seq = None
    else:
        if every != 1:
            raise ValueError("every is only supported for spatial scans")
        seq, step = _snapshots_and_step(snapshots)
        if los is None:
            axis, coordinate, plane = _physical_cut(**selections)
        else:
            from .plotting import _validate_los_selection

            _validate_los_selection(selections)
            axis, coordinate = los, None
            plane = {"x": "YZ", "y": "XZ", "z": "XY"}[los]
        frames = seq
    if not isinstance(var, str) or not var:
        raise ValueError("var must be a nonempty string")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not isinstance(dpi, (int, np.integer)) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    if not isinstance(crf, (int, np.integer)) or not 0 <= crf <= 51:
        raise ValueError("crf must be an integer between 0 and 51")
    if preset not in {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }:
        raise ValueError("Unsupported H.264 encoding preset")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; install FFmpeg to create MP4 movies"
        )

    if spatial:
        destination = _scan_output_path(
            data,
            var,
            snap,
            scan,
            positions,
            every,
            output,
            output_dir,
            selections,
        )
    else:
        destination = _output_path(
            data,
            var,
            seq,
            step,
            axis,
            coordinate,
            plane,
            output,
            output_dir,
            selections,
            los=los,
        )
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"Movie already exists: {destination} (set overwrite=True)"
        )
    _check_snapshot_files(data, [snap] if spatial else seq)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # A unique same-directory temporary protects independent encoders and
    # preserves atomic publication even when encoding or reads fail.
    partial = _new_partial_movie(destination)

    from matplotlib import pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    from .plotting import _integrated_plane, _plot_contours, _snapshot_time_min

    fig = None
    proc = None
    exit_status = None
    keep_encoded_partial = False
    try:
        first_selections = dict(selections)
        if spatial:
            first_selections[scan] = coordinate
        initial_kwargs = {
            "cmap": cmap,
            "norm": norm,
            "aspect": aspect,
            "cbar": cbar,
            "cbar_position": cbar_position,
            "figsize": figsize,
            "title": title,
            "minor_ticks": minor_ticks,
            "units": units,
            "low_memory": low_memory,
        }
        if los is not None:
            initial_kwargs.update(los=los, los_stream=True)
        if contours is not None:
            initial_kwargs["contours"] = contours
        result = data.plot(
            var,
            snap if spatial else int(seq[0]),
            **first_selections,
            **initial_kwargs,
        )
        fig, ax, artist = _figure_and_image(result)
        fig.set_dpi(int(dpi))
        if not fig.canvas.supports_blit or not hasattr(fig.canvas, "buffer_rgba"):
            FigureCanvasAgg(fig)
        if not fig.canvas.supports_blit:
            raise RuntimeError("Matplotlib canvas does not support blitting")

        # Blitting draws animated artists on top of the saved static background.
        # Spines must therefore ALSO be animated, to retain correct z-order.
        contour_artist = getattr(ax, "_bifrost_contours", None)
        if contours is not None and contour_artist is None:
            raise RuntimeError("Unable to locate contour overlay from dd.plot()")
        moving = [artist, *ax.spines.values()]
        moving.extend(_contour_artists(contour_artist))
        right_title = None
        left_title = None
        if title:
            if spatial:
                left_title = getattr(ax, "_left_title", None)
                if left_title is None:
                    raise RuntimeError(
                        "Unable to locate Matplotlib's left-hand plane title"
                    )
                moving.append(left_title)
            else:
                right_title = getattr(ax, "_right_title", None)
                if right_title is None:
                    raise RuntimeError(
                        "Unable to locate Matplotlib's right-hand time title"
                    )
                moving.append(right_title)
        for obj in moving:
            obj.set_animated(True)
        fig.canvas.draw()
        background = fig.canvas.copy_from_bbox(fig.bbox)
        width, height = fig.canvas.get_width_height()

        proc = subprocess.Popen(
            _pipe_command(
                partial,
                width,
                height,
                fps=fps,
                overwrite=True,
                crf=int(crf),
                preset=preset,
            ),
            stdin=subprocess.PIPE,
        )

        iterable = frames
        if progress:
            try:
                from tqdm.auto import tqdm

                iterable = tqdm(frames, desc=f"movie {var}")
            except ImportError:
                pass
        initial_shape = artist.get_array().shape
        for frame_idx, frame in enumerate(iterable):
            if frame_idx:
                frame_selections = dict(selections)
                if spatial:
                    frame_selections[scan] = float(frame)
                if los is None:
                    image = _frame_data(
                        data,
                        var,
                        snap if spatial else int(frame),
                        axis,
                        frame_selections,
                        units,
                        low_memory,
                    )
                else:
                    los_plane, _ = _integrated_plane(
                        data,
                        var,
                        int(frame),
                        los=los,
                        selection=frame_selections,
                        units=units,
                        low_memory=low_memory,
                        stream=True,
                    )
                    image = los_plane.T
                if image.shape != initial_shape:
                    raise ValueError(
                        f"Frame {frame_idx}: cut shape {image.shape} differs from "
                        f"the initial plotted shape {initial_shape}"
                    )
                artist.set_data(image)
                if contours is not None:
                    _remove_contour(contour_artist)
                    contour_artist = _plot_contours(
                        ax,
                        data,
                        contours,
                        snap=snap if spatial else int(frame),
                        background_var=var,
                        selection=frame_selections,
                        cut_axis=axis,
                        uniform_z=(axis != "z"),
                        units=units,
                        low_memory=low_memory,
                    )
                    for overlay_item in _contour_artists(contour_artist):
                        overlay_item.set_animated(True)
            # Frame zero is already present in the AxesImage from data.plot().
            if left_title is not None:
                left_title.set_text(f"{scan.upper()}={float(frame):.2f} Mm")
            if right_title is not None and frame_idx:
                right_title.set_text(
                    f"t={_snapshot_time_min(data, int(frame)):.2f} min"
                )
            fig.canvas.restore_region(background)
            ax.draw_artist(artist)
            for overlay_item in _contour_artists(contour_artist):
                ax.draw_artist(overlay_item)
            for spine in ax.spines.values():
                ax.draw_artist(spine)
            if right_title is not None:
                ax.draw_artist(right_title)
            if left_title is not None:
                ax.draw_artist(left_title)
            fig.canvas.blit(fig.bbox)
            buffer = memoryview(fig.canvas.buffer_rgba()).cast("B")
            sent = proc.stdin.write(buffer)
            if sent != len(buffer):
                raise OSError(
                    f"Incomplete FFmpeg pipe write ({sent}/{len(buffer)} bytes)"
                )
        proc.stdin.close()
        exit_status = proc.wait()
        proc = None
        if exit_status != 0:
            raise RuntimeError(
                f"FFmpeg returned exit code {exit_status}; see its error output"
            )
        try:
            _publish_movie(partial, destination, overwrite=overwrite)
        except FileExistsError:
            # A concurrent writer won; never replace it or retain this temp.
            raise
        except OSError as exc:
            # An intact, encoded movie must survive unexpected publication
            # errors, even if the fallback copy also fails.
            keep_encoded_partial = True
            raise OSError(
                f"Could not publish MP4 to {destination}: {exc}. "
                f"Encoded movie preserved at {partial}"
            ) from exc
        except BaseException:
            # A Ctrl-C during publication must not delete a completed movie.
            keep_encoded_partial = True
            raise
        return destination
    finally:
        if proc is not None:
            if proc.stdin is not None and not proc.stdin.closed:
                try:
                    proc.stdin.close()
                except BrokenPipeError:
                    pass
            # On exceptions, do not leave a possibly blocked child running.
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        if fig is not None:
            plt.close(fig)
        if not keep_encoded_partial:
            partial.unlink(missing_ok=True)  # no temp for failed encodes/collisions
