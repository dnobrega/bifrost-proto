"""Instrument-response build utility for bifrost-proto development and validation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def sha256sum(path):
    """Helper used by common."""
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonicalize_table(table, logne, logt):
    """Helper used by common."""
    table = np.asarray(table, dtype=np.float64)
    logne = np.asarray(logne, dtype=np.float64).reshape(-1)
    logt = np.asarray(logt, dtype=np.float64).reshape(-1)
    canonical = (logne.size, logt.size)
    if table.shape == canonical:
        pass
    elif table.shape == canonical[::-1]:
        table = table.T
    else:
        raise ValueError(
            f"Cannot map input table shape {table.shape!r} onto canonical "
            f"(n_logne,n_logt)={canonical!r}"
        )
    if not np.all(np.diff(logne) > 0):
        raise ValueError("logne axis is not strictly increasing")
    if not np.all(np.diff(logt) > 0):
        raise ValueError("logt axis is not strictly increasing")
    return table, logne, logt


def write_response(output_root, name, table, logne, logt, metadata):
    """Helper used by common."""
    output_root = Path(output_root)
    target = output_root / name
    target.mkdir(parents=True, exist_ok=True)

    table, logne, logt = canonicalize_table(table, logne, logt)
    np.save(
        target / "table.npy",
        np.asfortranarray(table, dtype=np.float64),
        allow_pickle=False,
    )
    np.save(target / "logne.npy", logne.astype(np.float64), allow_pickle=False)
    np.save(target / "logt.npy", logt.astype(np.float64), allow_pickle=False)

    payload = {
        "format_version": 1,
        "name": name,
        "canonical_table_axes": ["log10_ne_cm-3", "log10_T_K"],
        "response_definition": "G such that local synthetic signal = ne * nH * G",
        "response_units": "DN cm^5 s^-1 pix^-1",
        "local_output_units": "DN cm^-1 s^-1 pix^-1",
        **metadata,
    }
    with (target / "metadata.json").open("w") as fp:
        json.dump(payload, fp, indent=2, sort_keys=True)
        fp.write("\n")
    return target


def readsav_response(path):
    """Helper used by common."""
    try:
        from scipy.io import readsav
    except ImportError as exc:
        raise RuntimeError(
            "scipy is required to convert legacy IDL .sav tables"
        ) from exc

    sav = readsav(path, python_dict=True, verbose=False)
    lower = {str(k).lower(): v for k, v in sav.items()}
    for required in ("table", "temperature", "density"):
        if required not in lower:
            raise KeyError(f"{path!r} does not contain IDL variable {required!r}")
    return lower["table"], lower["density"], lower["temperature"]


def load_vector(path, *, sav_variable=None):
    """Load a 1-D helper vector from .npy, text, or IDL .sav."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = np.load(path, allow_pickle=False)
    elif suffix == ".sav":
        try:
            from scipy.io import readsav
        except ImportError as exc:
            raise RuntimeError(
                "scipy is required to read IDL .sav helper arrays"
            ) from exc
        payload = readsav(path, python_dict=True, verbose=False)
        lower = {str(k).lower(): v for k, v in payload.items()}
        if sav_variable is not None:
            key = sav_variable.lower()
            if key not in lower:
                raise KeyError(f"{path!s} does not contain {sav_variable!r}")
            value = lower[key]
        else:
            arrays = [v for v in lower.values() if np.asarray(v).ndim >= 1]
            if len(arrays) != 1:
                raise ValueError(
                    "Specify --nh-var when the .sav contains more than one array"
                )
            value = arrays[0]
    else:
        value = np.loadtxt(path)
    return np.asarray(value, dtype=np.float64).reshape(-1)
