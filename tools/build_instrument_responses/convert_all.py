#!/usr/bin/env python3
"""Instrument-response build utility for bifrost-proto development and validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "bifrost_proto" / "resources" / "instrument_responses"

AIA = {
    "aia94": "aia_94_response_ne_tg.sav",
    "aia131": "aia_131_response_ne_tg.sav",
    "aia171": "aia_171_response_ne_tg.sav",
    "aia193": "aia_193_response_ne_tg.sav",
    "aia211": "aia_211_response_ne_tg.sav",
    "aia304": "aia_304_response_ne_tg.sav",
    "aia335": "aia_335_response_ne_tg.sav",
}

XRT = {
    "xrt_almed": "xrt_al-med_response_ne_tg.sav",
    "xrt_almesh": "xrt_al-mesh_response_ne_tg.sav",
    "xrt_alpoly": "xrt_al-poly_response_ne_tg.sav",
    "xrt_althick": "xrt_al-thick_response_ne_tg.sav",
    "xrt_bemed": "xrt_be-med_response_ne_tg.sav",
    "xrt_bethick": "xrt_be-thick_response_ne_tg.sav",
    "xrt_bethin": "xrt_be-thin_response_ne_tg.sav",
    "xrt_cpoly": "xrt_c-poly_response_ne_tg.sav",
    "xrt_tipoly": "xrt_ti-poly_response_ne_tg.sav",
}

EUI174_FITS = (
    "goft_hri_174_sun_coronal_1992_feldman_ext.abund_chianti.ioneq_synthetic.fits"
)

REQUIRED_OUTPUTS = {"table.npy", "logt.npy", "logne.npy", "metadata.json"}


def _run(cmd):
    """Internal helper for convert all."""
    print("\n>>", " ".join(str(item) for item in cmd))
    subprocess.run([str(item) for item in cmd], check=True)


def _preflight(source: Path, include_eui: bool, nh_over_ne: Path | None):
    """Internal helper for convert all."""
    missing = []
    for name, filename in {**AIA, **XRT}.items():
        path = source / filename
        if not path.is_file():
            missing.append((name, path))

    if include_eui:
        fits_path = source / EUI174_FITS
        if not fits_path.is_file():
            missing.append(("eui174 FITS", fits_path))
        if nh_over_ne is None or not nh_over_ne.is_file():
            missing.append(("eui174 proton_dens(T)", nh_over_ne))

    if missing:
        lines = ["Missing response inputs:"]
        for name, path in missing:
            lines.append(f"  {name:24s} -> {path}")
        raise FileNotFoundError("\n".join(lines))


def _verify(output_root: Path, names):
    """Internal helper for convert all."""
    problems = []
    for name in names:
        folder = output_root / name
        if not folder.is_dir():
            problems.append(f"{name}: directory missing")
            continue
        existing = {path.name for path in folder.iterdir()}
        missing = REQUIRED_OUTPUTS - existing
        if missing:
            problems.append(f"{name}: missing {sorted(missing)}")
    if problems:
        raise RuntimeError(
            "Incomplete response conversion:\n  " + "\n  ".join(problems)
        )


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert the complete legacy AIA/XRT response set and optionally "
            "the Solar Orbiter/EUI HRI 174 response."
        )
    )
    parser.add_argument(
        "source", type=Path, help="directory containing the legacy response files"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"destination instrument_responses directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--nh-over-ne",
        type=Path,
        default=None,
        help="proton_dens(T) helper for EUI 174 (.sav, .npy or text); omit to skip EUI",
    )
    parser.add_argument(
        "--nh-var",
        default="nhone",
        help="variable name inside --nh-over-ne when it is an IDL .sav (default: nhone)",
    )
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    nh_over_ne = args.nh_over_ne.expanduser().resolve() if args.nh_over_ne else None
    include_eui = nh_over_ne is not None

    if not source.is_dir():
        raise NotADirectoryError(source)

    # Check every requested source before writing anything.
    _preflight(source, include_eui, nh_over_ne)
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Source directory : {source}")
    print(f"Output directory : {output_root}")

    for name, filename in AIA.items():
        _run(
            [
                sys.executable,
                HERE / "convert_aia.py",
                source / filename,
                "--name",
                name,
                "--output-root",
                output_root,
            ]
        )

    for name, filename in XRT.items():
        _run(
            [
                sys.executable,
                HERE / "convert_xrt.py",
                source / filename,
                "--name",
                name,
                "--output-root",
                output_root,
            ]
        )

    converted = list(AIA) + list(XRT)

    if include_eui:
        _run(
            [
                sys.executable,
                HERE / "convert_eui.py",
                source / EUI174_FITS,
                "--nh-over-ne",
                nh_over_ne,
                "--nh-var",
                args.nh_var,
                "--output-root",
                output_root,
            ]
        )
        converted.append("eui174")
    else:
        print(
            "\nEUI 174 skipped: pass --nh-over-ne to reproduce the legacy proton_dens(T) normalization."
        )

    _verify(output_root, converted)
    print("\nAll requested response tables converted successfully:")
    for name in converted:
        print(f"  {name}")


if __name__ == "__main__":
    main()
