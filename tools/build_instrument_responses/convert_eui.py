#!/usr/bin/env python3
"""Instrument-response build utility for bifrost-proto development and validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import canonicalize_table, load_vector, sha256sum, write_response


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert the legacy Solar Orbiter/EUI HRI 174 FITS response. "
            "The --nh-over-ne array must be the proton_dens(T) output used by the IDL recipe."
        )
    )
    parser.add_argument("source", help="Legacy HRI 174 response FITS")
    parser.add_argument("--nh-over-ne", required=True, help="proton_dens(T) helper: .npy, text, or IDL .sav")
    parser.add_argument("--nh-var", default="nhone", help="variable name when --nh-over-ne is an IDL .sav")
    parser.add_argument("--output-root", required=True, help="instrument_responses directory")
    parser.add_argument("--abundance", default="sun_coronal_1992_feldman_ext.abund")
    parser.add_argument("--ionization-equilibrium", default="legacy FITS response; see original source")
    parser.add_argument("--fill-value", type=float, default=1.0e-32)
    args = parser.parse_args()

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise RuntimeError(
            "astropy is required only for this conversion tool, not by bifrost_proto at runtime"
        ) from exc

    raw, header = fits.getdata(args.source, header=True)
    nt = int(header["NAXIS1"])
    nne = int(header["NAXIS2"])
    logt = np.linspace(np.log10(float(header["TMIN"])), np.log10(float(header["TMAX"])), nt)
    # Legacy FITS density limits are m^-3; runtime response axes use cm^-3.
    logne = np.linspace(np.log10(float(header["NMIN"])) - 6.0, np.log10(float(header["NMAX"])) - 6.0, nne)

    table, logne, logt = canonicalize_table(raw, logne, logt)
    nh_over_ne = load_vector(args.nh_over_ne, sav_variable=args.nh_var)
    if nh_over_ne.size != nt:
        raise ValueError(f"--nh-over-ne has {nh_over_ne.size} values; expected NAXIS1={nt}")
    if np.any(nh_over_ne <= 0.0) or not np.all(np.isfinite(nh_over_ne)):
        raise ValueError("--nh-over-ne must contain finite positive values")

    ne = 10.0 ** logne
    # Reproduce the IDL normalization exactly:
    #   goft /= 1e2          (DN m^-1 -> DN cm^-1)
    #   goft /= ne^2
    #   goft /= proton_dens(T)
    table = table / 1.0e2
    table = table / (ne[:, None] ** 2)
    table = table / nh_over_ne[None, :]

    target = write_response(
        args.output_root,
        "eui174",
        table,
        logne,
        logt,
        {
            "instrument": "Solar Orbiter/EUI",
            "detector": "HRI_EUV",
            "channel": "174",
            "abundance": args.abundance,
            "ionization_equilibrium": args.ionization_equilibrium,
            "fill_value": args.fill_value,
            "source_file": Path(args.source).name,
            "source_sha256": sha256sum(args.source),
            "nh_over_ne_file": Path(args.nh_over_ne).name,
            "nh_over_ne_sha256": sha256sum(args.nh_over_ne),
            "conversion": (
                "Reproduces the legacy IDL normalization: FITS DN m^-1 s^-1 pix^-1 -> "
                "DN cm^-1 s^-1 pix^-1, then divide by ne^2 and proton_dens(T), so runtime "
                "multiplication by ne*nH yields the local synthetic signal."
            ),
        },
    )
    print(target)


if __name__ == "__main__":
    main()
