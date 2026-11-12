#!/usr/bin/env python3
"""Instrument-response build utility for bifrost-proto development and validation."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import readsav_response, sha256sum, write_response


VALID_NAMES = {
    "xrt_almed",
    "xrt_almesh",
    "xrt_alpoly",
    "xrt_althick",
    "xrt_bemed",
    "xrt_bethick",
    "xrt_bethin",
    "xrt_cpoly",
    "xrt_tipoly",
}


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Convert one legacy XRT ne-T IDL response table."
    )
    parser.add_argument(
        "source", help="Legacy .sav containing table, density and temperature"
    )
    parser.add_argument("--name", required=True, choices=sorted(VALID_NAMES))
    parser.add_argument(
        "--output-root", required=True, help="instrument_responses directory"
    )
    parser.add_argument("--abundance", default="Asplund 2021 coronal")
    parser.add_argument(
        "--ionization-equilibrium", default="legacy table; see original source"
    )
    parser.add_argument("--fill-value", type=float, default=0.0)
    args = parser.parse_args()

    table, logne, logt = readsav_response(args.source)
    target = write_response(
        args.output_root,
        args.name,
        table,
        logne,
        logt,
        {
            "instrument": "Hinode/XRT",
            "filter": args.name.removeprefix("xrt_"),
            "abundance": args.abundance,
            "ionization_equilibrium": args.ionization_equilibrium,
            "fill_value": args.fill_value,
            "source_file": Path(args.source).name,
            "source_sha256": sha256sum(args.source),
            "conversion": "Legacy IDL table used directly; axes are log10(ne/cm^-3) and log10(T/K).",
        },
    )
    print(target)


if __name__ == "__main__":
    main()
